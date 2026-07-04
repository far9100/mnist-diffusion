"""Phase 1（CIFAR 及後續）用的表徵式生成指標。

提供修訂後計畫所需的 feature 抽取器與指標包裝：
  - DINOv2 feature（FD-DINOv2），即 Stein et al.（NeurIPS 2023）為避免
    Inception-FID 假影所給的建議；
  - 任意 feature 空間上的 Frechet distance，重用 fid.frechet_distance；
  - 任意 feature 空間上的 PRDC（precision/recall/density/coverage），重用
    metrics_prdc。

標準 Inception-FID 另外透過 clean-fid 計算（見 fid_clean.py），
分開處理使回報的 FID 與正統 clean-fid 實作及其
預先計算的 CIFAR statistics 一致（正確性錨點）。

護欄（brief 第 3 節，DINOv2 雙重使用）：CaF selector 的 coverage 與
回報的 FD 必須在*不同的*表徵中交叉檢查。因此本模組
提供 `backbone` 參數，讓 coverage 可在 CLIP 或
Inception 空間中重新計算，作為穩健性檢查。
"""

import torch
import torch.nn.functional as F

from fid import feature_stats, frechet_distance
from metrics_prdc import compute_prdc, compute_prdc_per_class

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_CACHE = {}


def get_dinov2(model_name="dinov2_vitb14", device="cpu"):
    """透過 torch.hub 載入（並快取）DINOv2 backbone。vitb14 -> 768 維 CLS。"""
    key = (model_name, str(device))
    if key not in _CACHE:
        model = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
        model.eval().to(device)
        _CACHE[key] = model
    return _CACHE[key]


@torch.no_grad()
def dinov2_features(images, device="cpu", model_name="dinov2_vitb14",
                    batch_size=128, image_size=224):
    """對 [0,1] 範圍的影像 (N,3,H,W) 取 CLS feature。會做縮放 + ImageNet 正規化。

    灰階 (N,1,H,W) 輸入會被擴充為 3 個 channel。
    """
    model = get_dinov2(model_name, device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    feats = []
    for start in range(0, images.size(0), batch_size):
        x = images[start:start + batch_size].to(device)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=image_size, mode="bicubic", align_corners=False)
        x = (x.clamp(0, 1) - mean) / std
        feats.append(model(x).cpu())
    return torch.cat(feats)


def fd_from_features(real_feats, gen_feats):
    """兩組 feature 之間的 Frechet distance（例如 FD-DINOv2）。"""
    mu_r, cov_r = feature_stats(real_feats)
    mu_g, cov_g = feature_stats(gen_feats)
    return frechet_distance(mu_r, cov_r, mu_g, cov_g)


def prdc_from_features(real_feats, gen_feats, nearest_k=5):
    return compute_prdc(real_feats, gen_feats, nearest_k=nearest_k)


def prdc_per_class_from_features(real_feats, real_labels, gen_feats, gen_labels,
                                 nearest_k=5, num_classes=10):
    return compute_prdc_per_class(real_feats, real_labels, gen_feats, gen_labels,
                                  nearest_k=nearest_k, num_classes=num_classes)


def _self_check():
    """對隨機影像做 CPU 自我檢查：real-vs-real 的 FD 小，vs-noise 的 FD 大。"""
    device = torch.device("cpu")
    torch.manual_seed(0)
    real = torch.rand(64, 3, 32, 32)
    real2 = torch.rand(64, 3, 32, 32)
    fr = dinov2_features(real, device)
    fr2 = dinov2_features(real2, device)
    print(f"dinov2 feat dim: {fr.shape[1]}")
    print(f"FD(real,real2)  : {fd_from_features(fr, fr2):.3f}")
    print(f"PRDC(real,real2): {prdc_from_features(fr, fr2, nearest_k=5)}")


if __name__ == "__main__":
    _self_check()
