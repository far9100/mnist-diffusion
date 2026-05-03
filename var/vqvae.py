"""
Multi-scale Residual VQ-VAE for VAR-mini.

Encodes a 28x28 MNIST image to a 7x7xD latent feature, then decomposes it
into K=4 token maps at progressively finer resolutions [1, 2, 4, 7] via
residual quantization with a shared codebook. The decoder reconstructs the
image from the cumulative sum of all scales' contributions.

References:
  - VAR (Tian et al., NeurIPS 2024 best paper, arXiv:2404.02905) — multi-scale
    residual quantization scheme.
  - VQ-VAE-2 (Razavi et al., 2019) — EMA codebook update, dead-code reinit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """GroupNorm + SiLU + Conv twice with a skip connection."""

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
    """28x28x1 -> 14x14x32 -> 7x7x64 -> 7x7xD (latent feature f)."""

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
    """7x7xD -> 14x14x32 -> 28x28x1."""

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
    Vector quantizer with EMA codebook update and dead-code reinitialisation.

    Buffers (not parameters):
      embedding   (V, D)  the codebook itself
      cluster_size (V,)   running EMA of how many vectors are assigned to each entry
      embed_avg   (V, D)  running EMA of the sum of input vectors per entry

    The EMA update replaces the standard codebook loss; only the commitment
    loss is returned for the optimiser.
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
        """z: (B, D, H, W). Returns (q_st, idx, commit_loss, perplexity)."""
        B, D, H, W = z.shape
        z_perm = z.permute(0, 2, 3, 1).contiguous()  # (B, H, W, D)
        flat = z_perm.reshape(-1, D)  # (N, D), N = B*H*W

        # Squared distance to each codebook entry
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
        # Straight-through: forward uses quantized, backward sees identity
        q_st = z + (quantized - z).detach()

        avg_probs = onehot.mean(0)
        perplexity = torch.exp(-(avg_probs * torch.log(avg_probs + 1e-10)).sum())

        idx_map = idx.view(B, H, W)
        return q_st, idx_map, commit_loss, perplexity

    def lookup(self, idx_map):
        """idx_map: (B, H, W) -> (B, D, H, W)."""
        return F.embedding(idx_map, self.embedding).permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def reinit_dead_codes(self, threshold=1.0):
        """Resample codebook entries whose EMA cluster size has decayed below
        `threshold` from the surviving entries (with small noise). Returns the
        number of entries that were reinitialised."""
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
        # Reset EMA stats so the new code gets a fair chance.
        self.cluster_size[dead] = 1.0
        self.embed_avg[dead] = self.embedding[dead]
        return n_dead


class MultiScaleRVQ(nn.Module):
    """
    Multi-scale Residual Vector Quantizer (VAR-style).

    For each scale k in `scales`, downsample the running residual to that
    spatial size, quantize with the shared codebook, upsample the quantized
    map back to the finest resolution and pass through a per-scale post-quant
    smoother phi[k]. Subtract the smoothed contribution from the residual
    before moving to the next scale.
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
        """f: (B, D, finest, finest). Returns (f_hat, indices_list, commit, perplexities)."""
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
        """For inference: rebuild f_hat from token indices alone."""
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
        """For autoregressive sampling: incrementally add scale k's contribution."""
        q_k = self.quantizer.lookup(idx_k)
        q_k_up = self.phi[scale_idx](self._resize_to_finest(q_k))
        return f_hat + q_k_up

    @torch.no_grad()
    def cumulative_f_hat_per_scale(self, indices_list):
        """For Stage-2 transformer teacher-forcing.

        Returns a list of K tensors where element k is the cumulative f_hat
        AFTER scales 0..k-1 have been added (i.e. the context the transformer
        should see when predicting scale k). Element [0] is zeros.
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
    """End-to-end multi-scale residual VQ-VAE."""

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
        """Stage 2 helper: returns list of (B, H_k, W_k) index tensors."""
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
