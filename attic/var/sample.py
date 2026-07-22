"""
VAR-mini 的自迴歸取樣，含 classifier-free guidance。

生成迴圈會逐一走過各個 scale；在同一個 scale 內，所有 token 會平行取樣
（next-scale prediction）。為了做 CFG，會把條件與非條件的前向傳遞在 batch
維度上加倍後一起批次計算。兩條路徑共用相同的取樣 token，以維持累積的
f_hat 一致。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_top_k_top_p(logits, top_k: int = 0, top_p: float = 1.0):
    """用 top-k 與／或 top-p 過濾 logits，然後每一列取樣一個 token。

    Args:
        logits: (..., V) 原始 logits（已經做過溫度縮放）。
        top_k:  只保留前 top-k 個 logits（0 = 停用）。
        top_p:  保留累積機率超過 top_p 的最小 token 集合
                （1.0 = 停用）。

    Returns:
        形狀為 (...,) 的取樣 token 索引。
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
        # 向右位移一格，讓我們永遠至少保留一個 token（第一個
        # 跨過門檻的 token 應該被保留）。
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
    """在多尺度 token 網格上，以 scale-wise AR 的方式取樣影像。

    Args:
        transformer: 訓練好的 VARTransformer。
        vqvae:       訓練好的 VARVQVAE（提供 codebook 與 decoder）。
        class_labels: (B,) 目標類別（0..num_classes-1）。
        cfg_scale:    classifier-free guidance 的強度。1.0 表示停用 CFG。
        top_k, top_p, temperature: 取樣控制參數。

    Returns:
        images: (B, 1, 28, 28) ∈ [-1, 1]
        all_indices: 由 K 個 (B, H_k, W_k) 取樣 token map 組成的 list
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
    # cumulative[k] 是 scale k 的輸入 context（k=0 => 全為零）。
    cumulative = [f_hat.clone()]
    all_indices = []

    for k in range(K):
        # 補齊成長度 K 的完整 list（尚未取樣的 scale 以零填補）。
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
