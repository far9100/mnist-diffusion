"""
Scale-wise transformer for VAR-mini.

Implements next-scale prediction:
  - Sequence layout: positions 0..L-1 correspond to all tokens across K scales,
    concatenated coarse-to-fine. For our default scales [1, 2, 4, 7]:
        pos 0     -> scale 0 (1 token)
        pos 1..4  -> scale 1 (4 tokens)
        pos 5..20 -> scale 2 (16 tokens)
        pos 21..69 -> scale 3 (49 tokens)
    Total L = 70.
  - Block-causal attention: a token at scale k can attend to all tokens at
    scales 0..k (inclusive, i.e. fully bidirectional within scale, causal
    across scales).
  - Teacher-forcing input construction: for scale k, the input embedding is
    derived from the cumulative f_hat reconstructed from GT scales 0..k-1
    (downsampled to scale k's resolution and linearly projected). The first
    scale's input is replaced by the class-conditioned SOS embedding.
  - DiT-style AdaLN modulation with the class embedding as the conditioning
    signal. Final layer is zero-initialised so the model starts as identity.

References:
  - VAR (Tian et al., NeurIPS 2024 best paper, arXiv:2404.02905)
  - DiT (Peebles & Xie, ICCV 2023) — AdaLN modulation pattern
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Standard MHSA using F.scaled_dot_product_attention with a precomputed mask."""

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
    """AdaLN modulation: x * (1 + scale) + shift, broadcasting along sequence."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TransformerBlock(nn.Module):
    """
    Pre-LN block with AdaLN modulation:
      x' = x + gate_msa * MSA(modulate(LN(x), shift_msa, scale_msa))
      x'' = x' + gate_mlp * MLP(modulate(LN(x'), shift_mlp, scale_mlp))
    The AdaLN linear is zero-initialised so each block starts as identity
    (DiT-style stable initialisation).
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
        # 6 modulation params per block: shift/scale/gate for MSA and MLP
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
    Next-scale autoregressive transformer.

    Args:
        scales: tuple of spatial sizes for each scale (matches MultiScaleRVQ)
        embedding_dim: latent dim D of the VQ-VAE codebook (input feature dim)
        num_classes: number of conditioning classes (for MNIST: 10)
        codebook_size: V (output vocabulary size)
        d_model, n_heads, n_layers, mlp_ratio, dropout: standard transformer
            hyperparameters
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
        # finest spatial size of the latent feature map (for cumulative f_hat)
        self.finest = scales[-1]

        self.num_classes = num_classes
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.d_model = d_model

        # Class embedding: indices 0..num_classes-1 are real classes;
        # index num_classes is the null class for classifier-free guidance.
        self.class_emb = nn.Embedding(num_classes + 1, d_model)
        nn.init.normal_(self.class_emb.weight, std=0.02)

        # Conditioning MLP that drives all AdaLN layers
        self.cond_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Project flat latent feature (D) to d_model for non-SOS positions
        self.input_proj = nn.Linear(embedding_dim, d_model)

        # Learned position embedding (per sequence position)
        self.pos_emb = nn.Parameter(torch.zeros(self.total_tokens, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        # Learned per-scale level embedding (broadcast over tokens within a scale)
        self.level_emb = nn.Parameter(torch.zeros(len(scales), d_model))
        nn.init.normal_(self.level_emb, std=0.02)
        # Per-position scale id for level lookup
        level_ids = []
        for k, count in enumerate(self.scale_token_counts):
            level_ids.extend([k] * count)
        self.register_buffer("level_ids", torch.tensor(level_ids, dtype=torch.long))

        # Block-causal attention mask, computed once
        self.register_buffer("attn_mask", self._build_block_causal_mask())

        # Transformer stack
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])

        # Final AdaLN + classification head
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.final_adaln = nn.Linear(d_model, 2 * d_model)
        nn.init.zeros_(self.final_adaln.weight)
        nn.init.zeros_(self.final_adaln.bias)
        # Head: small-scale normal init so gradients can flow back from the
        # very first step. (Zero-init the head would flatten logits but block
        # backward signal through it; we prefer a tiny random init that
        # produces nearly-uniform logits while still passing gradient.)
        self.head = nn.Linear(d_model, codebook_size)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------ utils

    def _build_block_causal_mask(self):
        """(L, L) bool mask. mask[i, j] = True iff scale(j) <= scale(i)."""
        scale_of = []
        for k, count in enumerate(self.scale_token_counts):
            scale_of.extend([k] * count)
        scale_of = torch.tensor(scale_of)
        return scale_of.unsqueeze(0) <= scale_of.unsqueeze(1)

    def scale_slice(self, scale_idx: int):
        """Return (start, end) slice indices for a given scale's positions."""
        return self.scale_offsets[scale_idx], self.scale_offsets[scale_idx + 1]

    # --------------------------------------------------------------- forward

    def _build_input_sequence(self, sos_token, cumulative_per_scale):
        """Construct (B, L, d_model) input embeddings (before pos/level).

        Args:
            sos_token: (B, d_model) class embedding for SOS position
            cumulative_per_scale: list of K tensors (B, D, finest, finest)
                where element k is the input context for scale k
                (cumulative f_hat after scales 0..k-1).
                Element 0 is conventionally zeros — not used since SOS replaces
                that position's input.
        """
        B = sos_token.shape[0]
        # Scale 0: SOS token replaces what would be the projected zero context.
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
        Args:
            class_labels: (B,) long, in [0, num_classes] (num_classes is null)
            cumulative_per_scale: list of K (B, D, finest, finest) input contexts
            target_indices: optional list of K (B, H_k, W_k) target token maps
                (only for training — if provided, returns loss too)

        Returns:
            logits: (B, L, V)
            loss:   scalar cross-entropy (only if target_indices given), else None
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
