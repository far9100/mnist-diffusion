"""
Autoregressive sampling for VAR-mini with classifier-free guidance.

The generation loop iterates over scales; within a scale, all tokens are
sampled in parallel (next-scale prediction). For CFG, the conditional and
unconditional forward passes are batched together by doubling the batch
dimension. Both passes share the same sampled tokens to keep the cumulative
f_hat consistent.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_top_k_top_p(logits, top_k: int = 0, top_p: float = 1.0):
    """Filter logits with top-k and/or top-p, then sample one token per row.

    Args:
        logits: (..., V) raw logits (already temperature-scaled).
        top_k:  keep only the top-k logits (0 = disabled).
        top_p:  keep the smallest set of tokens whose cumulative probability
                exceeds top_p (1.0 = disabled).

    Returns:
        sampled token indices of shape (...,).
    """
    orig_shape = logits.shape[:-1]
    V = logits.shape[-1]
    logits = logits.reshape(-1, V).clone()

    if top_k > 0:
        top_k = min(top_k, V)
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits.masked_fill_(logits < kth, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_remove = cum_probs > top_p
        # Shift right by one so we always keep at least one token (the first
        # token that crosses the threshold should remain).
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove = torch.zeros_like(logits, dtype=torch.bool)
        remove.scatter_(-1, sorted_idx, sorted_remove)
        logits.masked_fill_(remove, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return idx.reshape(orig_shape)


@torch.no_grad()
def generate(
    transformer,
    vqvae,
    class_labels,
    *,
    cfg_scale: float = 2.0,
    top_k: int = 50,
    top_p: float = 0.95,
    temperature: float = 1.0,
):
    """Sample images by scale-wise AR over the multi-scale token grid.

    Args:
        transformer: trained VARTransformer.
        vqvae:       trained VARVQVAE (provides codebook + decoder).
        class_labels: (B,) target classes (0..num_classes-1).
        cfg_scale:    classifier-free guidance scale. 1.0 disables CFG.
        top_k, top_p, temperature: sampling controls.

    Returns:
        images: (B, 1, 28, 28) ∈ [-1, 1]
        all_indices: list of K (B, H_k, W_k) sampled token maps
    """
    device = class_labels.device
    B = class_labels.shape[0]
    K = len(transformer.scales)
    finest = transformer.finest
    D = transformer.embedding_dim
    num_classes = transformer.num_classes

    transformer.eval()
    vqvae.eval()

    use_cfg = cfg_scale != 1.0
    null_labels = torch.full_like(class_labels, num_classes)

    f_hat = torch.zeros(B, D, finest, finest, device=device)
    # cumulative[k] is the input context for scale k (k=0 => zeros).
    cumulative = [f_hat.clone()]
    all_indices = []

    for k in range(K):
        # Pad to full K-length list (zeros for scales not yet sampled).
        cum_full = cumulative + [torch.zeros_like(f_hat) for _ in range(K - len(cumulative))]

        if use_cfg:
            cum_double = [torch.cat([c, c], dim=0) for c in cum_full]
            labels_double = torch.cat([class_labels, null_labels], dim=0)
            logits_all, _ = transformer(labels_double, cum_double)
            logits_cond, logits_uncond = logits_all.chunk(2, dim=0)
            logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
        else:
            logits, _ = transformer(class_labels, cum_full)

        start, end = transformer.scale_slice(k)
        scale_logits = logits[:, start:end] / max(temperature, 1e-6)
        idx_k_flat = sample_top_k_top_p(scale_logits, top_k=top_k, top_p=top_p)
        size = transformer.scales[k]
        idx_k = idx_k_flat.view(B, size, size)
        all_indices.append(idx_k)

        f_hat = vqvae.rvq.add_scale(f_hat, idx_k, k)
        if k + 1 < K:
            cumulative.append(f_hat.clone())

    images = vqvae.decoder(f_hat).clamp(-1, 1)
    return images, all_indices
