"""
VAR-mini 的 scale-wise transformer。

實作 next-scale 預測：
  - 序列排列：位置 0..L-1 對應到 K 個尺度的所有 token，由粗到細串接。以我們預設
    的尺度 [1, 2, 4, 7] 為例：
        pos 0     -> scale 0 (1 token)
        pos 1..4  -> scale 1 (4 tokens)
        pos 5..20 -> scale 2 (16 tokens)
        pos 21..69 -> scale 3 (49 tokens)
    總長 L = 70。
  - Block-causal attention：尺度 k 的 token 可以 attend 到尺度 0..k（含）的所有
    token，也就是在同一尺度內完全雙向，跨尺度之間則為 causal。
  - Teacher-forcing 的輸入建構：對尺度 k，輸入 embedding 是由 GT 尺度 0..k-1
    重建出的累積 f_hat 得來（下採樣到尺度 k 的解析度後再做線性投影）。第一個尺度
    的輸入則被替換成 class-conditioned 的 SOS embedding。
  - DiT 風格的 AdaLN modulation，以 class embedding 作為 conditioning 訊號。
    最後一層做 zero-initialize，讓模型一開始等同 identity。

參考文獻：
  - VAR (Tian et al., NeurIPS 2024 best paper, arXiv:2404.02905)
  - DiT (Peebles & Xie, ICCV 2023) — AdaLN modulation 的模式
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """標準 MHSA，使用 F.scaled_dot_product_attention 搭配預先算好的 mask。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, attn_mask):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, head_dim)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


def _modulate(x, shift, scale):
    """AdaLN modulation：x * (1 + scale) + shift，沿序列方向做 broadcast。"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TransformerBlock(nn.Module):
    """
    帶 AdaLN modulation 的 Pre-LN block：
      x' = x + gate_msa * MSA(modulate(LN(x), shift_msa, scale_msa))
      x'' = x' + gate_mlp * MLP(modulate(LN(x'), shift_mlp, scale_mlp))
    AdaLN 的 linear 層做 zero-initialize，讓每個 block 一開始等同 identity
    （DiT 風格的穩定初始化）。
    """

    def __init__(self, d_model: int, n_heads: int,
                 mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        hidden = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, d_model),
        )
        # 每個 block 有 6 個 modulation 參數：MSA 與 MLP 各自的 shift/scale/gate
        self.adaln = nn.Linear(d_model, 6 * d_model, bias=True)
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)

    def forward(self, x, cond, attn_mask):
        s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = self.adaln(cond).chunk(6, dim=-1)
        h = _modulate(self.norm1(x), s_msa, sc_msa)
        x = x + g_msa.unsqueeze(1) * self.attn(h, attn_mask)
        h = _modulate(self.norm2(x), s_mlp, sc_mlp)
        x = x + g_mlp.unsqueeze(1) * self.mlp(h)
        return x


class VARTransformer(nn.Module):
    """
    Next-scale 的 autoregressive transformer。

    參數：
        scales: 每個尺度的空間大小 tuple（與 MultiScaleRVQ 對應）
        embedding_dim: VQ-VAE codebook 的 latent 維度 D（輸入特徵維度）
        num_classes: conditioning 類別數（MNIST 為 10）
        codebook_size: V（輸出詞彙表大小）
        d_model, n_heads, n_layers, mlp_ratio, dropout: 標準 transformer 的
            超參數
    """

    def __init__(self, scales=(1, 2, 4, 7), embedding_dim: int = 64,
                 num_classes: int = 10, codebook_size: int = 256,
                 d_model: int = 384, n_heads: int = 6,
                 n_layers: int = 8, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.scales = tuple(scales)
        self.scale_token_counts = [s * s for s in scales]
        self.total_tokens = sum(self.scale_token_counts)
        self.scale_offsets = [0]
        for c in self.scale_token_counts:
            self.scale_offsets.append(self.scale_offsets[-1] + c)
        # latent 特徵圖最細的空間大小（用於累積 f_hat）
        self.finest = scales[-1]

        self.num_classes = num_classes
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.d_model = d_model

        # Class embedding：index 0..num_classes-1 為真實類別；
        # index num_classes 則是 classifier-free guidance 用的 null 類別。
        self.class_emb = nn.Embedding(num_classes + 1, d_model)
        nn.init.normal_(self.class_emb.weight, std=0.02)

        # 驅動所有 AdaLN 層的 conditioning MLP
        self.cond_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # 對非 SOS 的位置，把攤平的 latent 特徵 (D) 投影到 d_model
        self.input_proj = nn.Linear(embedding_dim, d_model)

        # 可學習的 position embedding（每個序列位置各一個）
        self.pos_emb = nn.Parameter(torch.zeros(self.total_tokens, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        # 可學習的各尺度 level embedding（在同一尺度內的 token 間做 broadcast）
        self.level_emb = nn.Parameter(torch.zeros(len(scales), d_model))
        nn.init.normal_(self.level_emb, std=0.02)
        # 每個位置對應的 scale id，供 level 查表使用
        level_ids = []
        for k, count in enumerate(self.scale_token_counts):
            level_ids.extend([k] * count)
        self.register_buffer("level_ids", torch.tensor(level_ids, dtype=torch.long))

        # Block-causal 的 attention mask，只計算一次
        self.register_buffer("attn_mask", self._build_block_causal_mask())

        # Transformer 堆疊
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])

        # 最後的 AdaLN 加上分類 head
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.final_adaln = nn.Linear(d_model, 2 * d_model)
        nn.init.zeros_(self.final_adaln.weight)
        nn.init.zeros_(self.final_adaln.bias)
        # Head：採用小尺度的 normal init，讓梯度從第一步就能回傳。（若把 head 做
        # zero-init，雖然會讓 logits 變平坦，卻會擋住穿過它的 backward 訊號；我們
        # 寧可用極小的隨機 init，使 logits 幾乎均勻，同時仍能傳遞梯度。）
        self.head = nn.Linear(d_model, codebook_size)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------ utils

    def _build_block_causal_mask(self):
        """(L, L) 的 bool mask。當且僅當 scale(j) <= scale(i) 時 mask[i, j] = True。"""
        scale_of = []
        for k, count in enumerate(self.scale_token_counts):
            scale_of.extend([k] * count)
        scale_of = torch.tensor(scale_of)
        return scale_of.unsqueeze(0) <= scale_of.unsqueeze(1)

    def scale_slice(self, scale_idx: int):
        """回傳指定尺度所在位置的 (start, end) 切片索引。"""
        return self.scale_offsets[scale_idx], self.scale_offsets[scale_idx + 1]

    # --------------------------------------------------------------- forward

    def _build_input_sequence(self, sos_token, cumulative_per_scale):
        """建構 (B, L, d_model) 的輸入 embedding（在加上 pos/level 之前）。

        參數：
            sos_token: (B, d_model)，SOS 位置用的 class embedding
            cumulative_per_scale: 含 K 個 tensor 的 list (B, D, finest, finest)，
                其中第 k 個元素是尺度 k 的輸入 context
                （尺度 0..k-1 之後的累積 f_hat）。
                第 0 個元素依慣例為全零，但不會被用到，因為該位置的輸入會被 SOS
                取代。
        """
        B = sos_token.shape[0]
        # 尺度 0：SOS token 取代原本會是投影後之全零 context 的位置。
        assert self.scales[0] == 1, \
            "First scale must be 1x1 so SOS occupies exactly one position"
        scale_inputs = [sos_token.unsqueeze(1)]  # (B, 1, d_model)

        for k in range(1, len(self.scales)):
            size = self.scales[k]
            cum = cumulative_per_scale[k]  # (B, D, finest, finest)
            if cum.shape[-1] != size:
                pooled = F.adaptive_avg_pool2d(cum, size)
            else:
                pooled = cum
            feat = pooled.permute(0, 2, 3, 1).reshape(B, size * size, self.embedding_dim)
            scale_inputs.append(self.input_proj(feat))

        return torch.cat(scale_inputs, dim=1)  # (B, L, d_model)

    def forward(self, class_labels, cumulative_per_scale, target_indices=None):
        """
        參數：
            class_labels: (B,) long，範圍 [0, num_classes]（num_classes 為 null）
            cumulative_per_scale: 含 K 個 (B, D, finest, finest) 輸入 context 的 list
            target_indices: 選用的 list，含 K 個 (B, H_k, W_k) 目標 token map
                （僅用於訓練，若有提供則一併回傳 loss）

        回傳：
            logits: (B, L, V)
            loss:   純量 cross-entropy（僅在有給 target_indices 時），否則為 None
        """
        B = class_labels.shape[0]
        c = self.class_emb(class_labels)  # (B, d_model)
        cond = self.cond_mlp(c)

        x = self._build_input_sequence(c, cumulative_per_scale)
        x = x + self.level_emb[self.level_ids].unsqueeze(0)
        x = x + self.pos_emb.unsqueeze(0)

        for block in self.blocks:
            x = block(x, cond, self.attn_mask)

        shift, scale = self.final_adaln(cond).chunk(2, dim=-1)
        x = _modulate(self.final_norm(x), shift, scale)
        logits = self.head(x)  # (B, L, V)

        loss = None
        if target_indices is not None:
            targets = torch.cat(
                [idx.reshape(B, -1) for idx in target_indices], dim=1,
            )  # (B, L)
            loss = F.cross_entropy(
                logits.reshape(-1, self.codebook_size),
                targets.reshape(-1),
            )
        return logits, loss
