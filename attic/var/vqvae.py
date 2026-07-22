"""
VAR-mini 的多尺度殘差 VQ-VAE。

把一張 28x28 的 MNIST 影像編碼成 7x7xD 的 latent 特徵，再透過共用 codebook 的
residual quantize，將其分解成 K=4 張逐步變細的解析度 [1, 2, 4, 7] 的 token map。
decoder 由所有尺度貢獻的累加總和重建出影像。

參考文獻：
  - VAR (Tian et al., NeurIPS 2024 best paper, arXiv:2404.02905) — 多尺度
    residual quantize 的方案。
  - VQ-VAE-2 (Razavi et al., 2019) — EMA codebook 更新、dead-code 重新初始化。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """GroupNorm + SiLU + Conv 做兩次，並帶一條 skip connection。"""

    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return x + h


class Encoder(nn.Module):
    """28x28x1 -> 14x14x32 -> 7x7x64 -> 7x7xD（latent 特徵 f）。"""

    def __init__(self, in_channels=1, embedding_dim=64):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.res_in = ResidualBlock(32)
        self.down1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 28 -> 14
        self.res1 = ResidualBlock(64)
        self.down2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # 14 -> 7
        self.res2 = ResidualBlock(64)
        self.norm_out = nn.GroupNorm(8, 64)
        self.conv_out = nn.Conv2d(64, embedding_dim, 1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res_in(x)
        x = self.down1(x)
        x = self.res1(x)
        x = self.down2(x)
        x = self.res2(x)
        x = F.silu(self.norm_out(x))
        return self.conv_out(x)


class Decoder(nn.Module):
    """7x7xD -> 14x14x32 -> 28x28x1。"""

    def __init__(self, out_channels=1, embedding_dim=64):
        super().__init__()
        self.conv_in = nn.Conv2d(embedding_dim, 64, 1)
        self.res_in = ResidualBlock(64)
        self.up1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)  # 7 -> 14
        self.res1 = ResidualBlock(32)
        self.up2 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)  # 14 -> 28
        self.res2 = ResidualBlock(16)
        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res_in(x)
        x = self.up1(x)
        x = self.res1(x)
        x = self.up2(x)
        x = self.res2(x)
        x = F.silu(self.norm_out(x))
        return self.conv_out(x)


class VectorQuantizerEMA(nn.Module):
    """
    帶 EMA codebook 更新與 dead-code 重新初始化的 vector quantizer。

    Buffers（非參數）：
      embedding   (V, D)  codebook 本身
      cluster_size (V,)   每個 entry 被指派到多少向量的 running EMA
      embed_avg   (V, D)  每個 entry 輸入向量總和的 running EMA

    EMA 更新取代了標準的 codebook loss；只有 commitment loss 會回傳給
    optimizer。
    """

    def __init__(self, num_embeddings, embedding_dim,
                 commitment_cost=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        embed = torch.randn(num_embeddings, embedding_dim) * 0.1
        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z):
        """z: (B, D, H, W)。回傳 (q_st, idx, commit_loss, perplexity)。"""
        B, D, H, W = z.shape
        z_perm = z.permute(0, 2, 3, 1).contiguous()  # (B, H, W, D)
        flat = z_perm.reshape(-1, D)  # (N, D), N = B*H*W

        # 到每個 codebook entry 的平方距離
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.embedding.t()
                + self.embedding.pow(2).sum(1))  # (N, V)
        idx = dist.argmin(1)  # (N,)

        onehot = F.one_hot(idx, self.num_embeddings).type_as(flat)  # (N, V)
        quant_flat = onehot @ self.embedding  # (N, D)
        quantized = quant_flat.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        if self.training:
            with torch.no_grad():
                cluster_new = onehot.sum(0)
                self.cluster_size.mul_(self.decay).add_(cluster_new, alpha=1 - self.decay)

                embed_sum = onehot.t() @ flat  # (V, D)
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

                n = self.cluster_size.sum()
                cluster_smooth = (
                    (self.cluster_size + self.eps)
                    / (n + self.num_embeddings * self.eps) * n
                )
                self.embedding.copy_(self.embed_avg / cluster_smooth.unsqueeze(1))

        commit_loss = self.commitment_cost * F.mse_loss(quantized.detach(), z)
        # Straight-through：forward 使用 quantized，backward 則視為 identity
        q_st = z + (quantized - z).detach()

        avg_probs = onehot.mean(0)
        perplexity = torch.exp(-(avg_probs * torch.log(avg_probs + 1e-10)).sum())

        idx_map = idx.view(B, H, W)
        return q_st, idx_map, commit_loss, perplexity

    def lookup(self, idx_map):
        """idx_map: (B, H, W) -> (B, D, H, W)。"""
        return F.embedding(idx_map, self.embedding).permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def reinit_dead_codes(self, threshold=1.0):
        """把 EMA cluster size 已衰減到低於 `threshold` 的 codebook entry，從仍
        存活的 entry 重新取樣（並加上小量雜訊）。回傳被重新初始化的 entry 數量。"""
        dead = self.cluster_size < threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return 0
        alive_mask = ~dead
        device = self.embedding.device
        if alive_mask.sum() == 0:
            self.embedding[dead] = torch.randn(n_dead, self.embedding_dim, device=device) * 0.1
        else:
            alive = self.embedding[alive_mask]
            pick = torch.randint(0, alive.shape[0], (n_dead,), device=device)
            jitter = torch.randn(n_dead, self.embedding_dim, device=device) * 0.01
            self.embedding[dead] = alive[pick] + jitter
        # 重設 EMA 統計量，讓新的 code 有公平的機會。
        self.cluster_size[dead] = 1.0
        self.embed_avg[dead] = self.embedding[dead]
        return n_dead


class MultiScaleRVQ(nn.Module):
    """
    多尺度 Residual Vector Quantizer（VAR 風格）。

    對 `scales` 中的每個尺度 k，把目前的 residual 下採樣到該空間大小，用共用
    codebook quantize，再把 quantize 後的 map 上採樣回最細解析度，並通過該尺度
    專屬的 post-quant 平滑器 phi[k]。在進入下一個尺度前，先從 residual 減去這個
    平滑後的貢獻。
    """

    def __init__(self, scales=(1, 2, 4, 7), codebook_size=256, embedding_dim=64,
                 commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.scales = tuple(scales)
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.finest = scales[-1]
        self.quantizer = VectorQuantizerEMA(
            codebook_size, embedding_dim, commitment_cost, decay,
        )
        self.phi = nn.ModuleList([
            nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1)
            for _ in scales
        ])

    @property
    def scale_token_counts(self):
        return [s * s for s in self.scales]

    @property
    def total_tokens(self):
        return sum(self.scale_token_counts)

    def _resize_to_finest(self, q):
        if q.shape[-1] == self.finest:
            return q
        return F.interpolate(q, size=(self.finest, self.finest),
                             mode="bicubic", align_corners=False)

    def forward(self, f):
        """f: (B, D, finest, finest)。回傳 (f_hat, indices_list, commit, perplexities)。"""
        B, D, H, W = f.shape
        assert H == self.finest and W == self.finest, \
            f"expected {self.finest}x{self.finest}, got {H}x{W}"

        f_hat = torch.zeros_like(f)
        r = f
        indices_list = []
        commit_terms = []
        perp_terms = []

        for k, size in enumerate(self.scales):
            z_k = r if size == H else F.adaptive_avg_pool2d(r, size)
            q_k_st, idx_k, commit, ppl = self.quantizer(z_k)
            q_k_up = self.phi[k](self._resize_to_finest(q_k_st))
            f_hat = f_hat + q_k_up
            r = r - q_k_up
            indices_list.append(idx_k)
            commit_terms.append(commit)
            perp_terms.append(ppl)

        commit_loss = torch.stack(commit_terms).mean()
        perplexity = torch.stack(perp_terms)  # (K,)
        return f_hat, indices_list, commit_loss, perplexity

    @torch.no_grad()
    def reconstruct_from_indices(self, indices_list):
        """用於 inference：僅從 token indices 重建 f_hat。"""
        assert len(indices_list) == len(self.scales)
        device = indices_list[0].device
        B = indices_list[0].shape[0]
        f_hat = torch.zeros(B, self.embedding_dim, self.finest, self.finest, device=device)
        for k, idx_k in enumerate(indices_list):
            q_k = self.quantizer.lookup(idx_k)
            q_k_up = self.phi[k](self._resize_to_finest(q_k))
            f_hat = f_hat + q_k_up
        return f_hat

    @torch.no_grad()
    def add_scale(self, f_hat, idx_k, scale_idx):
        """用於 autoregressive 取樣：逐步加入尺度 k 的貢獻。"""
        q_k = self.quantizer.lookup(idx_k)
        q_k_up = self.phi[scale_idx](self._resize_to_finest(q_k))
        return f_hat + q_k_up

    @torch.no_grad()
    def cumulative_f_hat_per_scale(self, indices_list):
        """用於 Stage-2 transformer 的 teacher-forcing。

        回傳一個含 K 個 tensor 的 list，其中第 k 個元素是「已加入尺度 0..k-1 之後」
        的累積 f_hat（也就是 transformer 在預測尺度 k 時應該看到的 context）。
        第 [0] 個元素為全零。
        """
        assert len(indices_list) == len(self.scales)
        device = indices_list[0].device
        B = indices_list[0].shape[0]
        f_hat = torch.zeros(B, self.embedding_dim, self.finest, self.finest, device=device)
        cumulative = [f_hat.clone()]
        for k in range(len(self.scales) - 1):
            f_hat = self.add_scale(f_hat, indices_list[k], k)
            cumulative.append(f_hat.clone())
        return cumulative


class VARVQVAE(nn.Module):
    """端到端的多尺度殘差 VQ-VAE。"""

    def __init__(self, in_channels=1, embedding_dim=64, codebook_size=256,
                 scales=(1, 2, 4, 7), commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.scales = tuple(scales)
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.encoder = Encoder(in_channels, embedding_dim)
        self.decoder = Decoder(in_channels, embedding_dim)
        self.rvq = MultiScaleRVQ(scales, codebook_size, embedding_dim,
                                 commitment_cost, decay)

    def forward(self, x):
        f = self.encoder(x)
        f_hat, indices, commit, perplexity = self.rvq(f)
        x_hat = self.decoder(f_hat)
        return x_hat, indices, commit, perplexity

    @torch.no_grad()
    def encode_to_indices(self, x):
        """Stage 2 的輔助函式：回傳一個由 (B, H_k, W_k) index tensor 組成的 list。"""
        was_training = self.training
        self.eval()
        f = self.encoder(x)
        _, indices, _, _ = self.rvq(f)
        if was_training:
            self.train()
        return indices

    @torch.no_grad()
    def decode_from_indices(self, indices_list):
        f_hat = self.rvq.reconstruct_from_indices(indices_list)
        return self.decoder(f_hat)
