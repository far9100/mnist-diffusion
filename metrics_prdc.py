"""Precision / Recall / Density / Coverage（PRDC）——純 torch 實作，不依賴 sklearn。

實作以下文獻中基於流形（manifold）的 fidelity/diversity 度量：
  - Kynkaanniemi et al., "Improved Precision and Recall Metric for Assessing
    Generative Models," NeurIPS 2019，以及
  - Naeem et al., "Reliable Fidelity and Diversity Metrics for Generative
    Models"（Density & Coverage），ICML 2020。

定義（real R、fake F、特徵維度 D；k = nearest_k）：
  real_nn[i]  = real i 到其第 k 個最近的 real 鄰居的距離（排除自身）
  fake_nn[j]  = fake j 到其第 k 個最近的 fake 鄰居的距離（排除自身）
  d[i,j]      = ||real_i - fake_j||_2
  precision = mean_j  [ exists i : d[i,j] < real_nn[i] ]        # fidelity
  recall    = mean_i  [ exists j : d[i,j] < fake_nn[j] ]        # fake 流形對 real 的覆蓋
  density   = mean_j  ( (1/k) * sum_i [ d[i,j] < real_nn[i] ] ) # 穩健的 fidelity
  coverage  = mean_i  [ min_j d[i,j] < real_nn[i] ]             # 穩健的 diversity（我們主要的多樣性把手）

對本專案的論點而言，**coverage**（Naeem）是 CaF 選擇器所最大化的主要類別內
diversity 訊號；**precision/density** 則是 fidelity 下限。

Usage (self-check):
    uv run python metrics_prdc.py
"""

import torch


def _pairwise_dist(x, y):
    """x (N,D) 與 y (M,D) 各列之間的歐氏距離矩陣。"""
    # 對我們的特徵量級而言，torch.cdist 在數值上沒問題；維持 float32。
    return torch.cdist(x, y)


def kth_nn_distance(features, nearest_k):
    """集合內每一列到其第 k 個最近鄰居的距離。

    自身的距離為 0（排序後位於第 0 欄），因此排除自身後的第 k 個鄰居落在排序索引
    `nearest_k` 處。
    """
    n = features.size(0)
    k = min(nearest_k, n - 1)
    d = _pairwise_dist(features, features)
    # 由小到大排序；索引 0 是自身（距離 0）
    knn = torch.topk(d, k=k + 1, dim=1, largest=False).values[:, k]
    return knn


@torch.no_grad()
def compute_prdc(real_features, fake_features, nearest_k=5):
    """回傳含 precision、recall、density、coverage 的 dict（大致都在 [0,1] 範圍）。

    輸入為 (N, D) 的 float tensor（任意 device）。計算在輸入所在的 device 上進行。
    """
    real_features = real_features.float()
    fake_features = fake_features.float()

    real_nn = kth_nn_distance(real_features, nearest_k)     # (R,)
    fake_nn = kth_nn_distance(fake_features, nearest_k)     # (F,)
    d = _pairwise_dist(real_features, fake_features)         # (R, F)

    # 若 d[i,j] < real_nn[i]，則 fake j 落在 real i 的球內
    inside_real = d < real_nn.unsqueeze(1)                   # (R, F) bool
    precision = inside_real.any(dim=0).float().mean().item()
    density = (inside_real.sum(dim=0).float() / nearest_k).mean().item()

    inside_fake = d < fake_nn.unsqueeze(0)                   # (R, F) bool
    recall = inside_fake.any(dim=1).float().mean().item()

    nearest_fake = d.min(dim=1).values                      # (R,)
    coverage = (nearest_fake < real_nn).float().mean().item()

    return {"precision": precision, "recall": recall,
            "density": density, "coverage": coverage}


@torch.no_grad()
def compute_prdc_per_class(real_features, real_labels, fake_features, fake_labels,
                           nearest_k=5, num_classes=10, min_per_class=None):
    """各類別 PRDC 再對類別取平均（類別內 fidelity/diversity）。

    類別內 coverage 是與 downstream 訓練相關的 diversity 軸：它問的是某類別的
    生成樣本是否覆蓋了該類別真實的散布範圍，而非類別之間的結構。

    回傳 (mean_dict, per_class_list)。樣本數過少的類別會被略過。
    """
    if min_per_class is None:
        min_per_class = nearest_k + 1
    per_class = []
    for c in range(num_classes):
        rf = real_features[real_labels == c]
        ff = fake_features[fake_labels == c]
        if rf.size(0) < min_per_class or ff.size(0) < min_per_class:
            continue
        m = compute_prdc(rf, ff, nearest_k)
        m["class"] = c
        per_class.append(m)
    if not per_class:
        return {"precision": float("nan"), "recall": float("nan"),
                "density": float("nan"), "coverage": float("nan")}, []
    mean = {key: sum(pc[key] for pc in per_class) / len(per_class)
            for key in ("precision", "recall", "density", "coverage")}
    return mean, per_class


def _self_check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    # 兩個重疊的高斯分布 -> 高 precision/recall/coverage。
    real = torch.randn(2000, 64, device=device)
    fake_good = torch.randn(2000, 64, device=device)
    # 平移並縮小的 fake -> 低 coverage/recall（類比 mode collapse）。
    fake_collapsed = 0.2 * torch.randn(2000, 64, device=device) + 3.0

    good = compute_prdc(real, fake_good, nearest_k=5)
    bad = compute_prdc(real, fake_collapsed, nearest_k=5)
    print("PRDC self-check (nearest_k=5):")
    print(f"  real vs iid-real   : {good}")
    print(f"  real vs collapsed  : {bad}")
    assert good["coverage"] > 0.5, "expected high coverage for matched dist"
    assert bad["coverage"] < good["coverage"], "collapsed should reduce coverage"
    assert bad["precision"] < 0.5, "far-shifted fake should have low precision"
    print("  OK")


if __name__ == "__main__":
    _self_check()
