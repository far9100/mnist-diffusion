"""機制分析（C2）：guidance 是否會抽乾決策邊界附近的樣本支持度？

論文機制：classifier-free guidance 會把取樣分布銳化
朝向類別原型（Ho & Salimans 2022；Bradley & Nakkiran 2024 將其表述為 denoise+sharpen）。
原型集中會移除那些低 margin、
接近決策邊界的樣本，而下游分類器正需要這些樣本來擺放它的
邊界（margin 理論：Bartlett 2017；near-boundary 範例才是有用的
那些：Sorscher 2022）。所以較高的 guidance -> 較少的 near-boundary 合成
樣本 -> 較弱的下游 margin -> 較低的 TSTR。

我們以一個用 REAL 資料訓練的分類器（mnist_cnn.pt）來量化「near-boundary」：
對每張合成影像，margin = p(top-1) - p(top-2)。margin 小 = 真實
分類器覺得它模稜兩可 = 接近某條決策邊界。

兩項分析：
  (a) near-boundary 比例與 margin 分布相對於 guidance 的變化；
  (b) coverage 受控的檢查（護欄：partial correlation 不代表因果）——
      透過子取樣使 coverage 相等，再在相同 coverage 下比較 margin，如此
      在 coverage 對齊後仍存在的 margin 落差就不只是「樣本較少」而已。

可重用的函式由掃描的驅動程式匯入；CLI 則將單一
產生的資料集對照真實的 margin 分布進行計分。

Usage:
    uv run python mechanism.py --generated generated/dataset.pt
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

from fid import load_cnn
from analyze_distribution import load_real_per_class, load_generated
from metrics_prdc import compute_prdc


@torch.no_grad()
def compute_margins(cnn, images, device, batch_size=512):
    """在以真實資料訓練的分類器下，機率 margin p(top1) - p(top2)。"""
    margins, preds = [], []
    for start in range(0, images.size(0), batch_size):
        batch = images[start:start + batch_size].to(device)
        probs = F.softmax(cnn(batch), dim=1)
        top2 = probs.topk(2, dim=1).values
        margins.append((top2[:, 0] - top2[:, 1]).cpu())
        preds.append(probs.argmax(1).cpu())
    return torch.cat(margins), torch.cat(preds)


def near_boundary_fraction(margins, threshold=0.5):
    """真實分類器覺得模稜兩可（margin < threshold）的樣本比例。"""
    return (margins < threshold).float().mean().item()


def margin_summary(margins, thresholds=(0.1, 0.3, 0.5)):
    q = torch.tensor([0.05, 0.25, 0.50])
    quantiles = torch.quantile(margins, q).tolist()
    return {
        "mean_margin": margins.mean().item(),
        "median_margin": quantiles[2],
        "p05_margin": quantiles[0],
        "p25_margin": quantiles[1],
        "near_boundary_frac": {str(t): near_boundary_fraction(margins, t)
                               for t in thresholds},
    }


def analyze_dataset(cnn, images, labels, device, threshold=0.5):
    """整個資料集的完整 margin 摘要 + 一個 label-noise 代理指標（top-1 != 給定標籤）。

    label-noise 代理指標之所以重要，是因為低 guidance 也會產生離類別（off-class）
    的樣本；若 near-boundary 比例上升其實是因為標錯的樣本，
    那是一個競爭的機制，所以我們兩者都回報。
    """
    margins, preds = compute_margins(cnn, images, device)
    labels = labels.to(preds.device)
    label_noise = (preds != labels).float().mean().item()
    summary = margin_summary(margins, thresholds=(0.1, 0.3, threshold))
    summary["label_noise_frac"] = label_noise
    return summary, margins, preds


@torch.no_grad()
def coverage_of(real_features, fake_features, nearest_k=5):
    """該 fake 集合相對 real 的 coverage（重用 metrics_prdc.compute_prdc）。"""
    return compute_prdc(real_features, fake_features, nearest_k)["coverage"]


@torch.no_grad()
def _subsample_coverage(real_features, fake_features, n, nearest_k, repeats):
    """隨機抽 n 個 fake，回傳（平均 coverage, 最接近平均那次抽樣的 CPU 索引）。"""
    n_full = fake_features.size(0)
    covs, idxs = [], []
    for _ in range(repeats):
        perm = torch.randperm(n_full)[:n]  # CPU 索引，供特徵與 margin 共用
        c = coverage_of(real_features, fake_features[perm.to(fake_features.device)], nearest_k)
        covs.append(c)
        idxs.append(perm)
    mean_c = sum(covs) / len(covs)
    rep = min(range(len(covs)), key=lambda i: abs(covs[i] - mean_c))
    return mean_c, idxs[rep]


@torch.no_grad()
def match_coverage_by_subsampling(real_features, fake_features, target_coverage,
                                  nearest_k=5, repeats=5, tol=0.01, max_iter=18):
    """對 fake 隨機子取樣，使其 coverage 逼近 target_coverage。

    coverage 隨 fake 樣本數（近似）單調遞增，故可對「子取樣數」做二分搜尋。每個候選數
    以 repeats 次隨機抽樣取平均以降變異。回傳（n, achieved_coverage, CPU 索引）；即使未達
    tol 也回傳目前為止最接近 target 的一組。
    """
    n_full = fake_features.size(0)
    lo, hi = nearest_k + 1, n_full
    best = None
    for _ in range(max_iter):
        mid = (lo + hi) // 2
        c, idx = _subsample_coverage(real_features, fake_features, mid, nearest_k, repeats)
        if best is None or abs(c - target_coverage) < abs(best[1] - target_coverage):
            best = (mid, c, idx)
        if abs(c - target_coverage) <= tol or hi - lo <= 1:
            break
        if c < target_coverage:
            lo = mid
        else:
            hi = mid
    return best


@torch.no_grad()
def coverage_controlled_margin(low_feats, low_margins, high_feats, high_margins,
                               real_feats, nearest_k=5, threshold=0.5, repeats=5):
    """介入式證據：對齊 coverage 後，比較低/高 guidance 的 near-boundary 佔比。

    護欄動機：partial correlation（utility ~ coverage | precision）只是相關、非因果。
    高 guidance 通常同時 coverage 較低、near-boundary 也較少，故單看單調曲線無法區分
    「因為樣本/coverage 較少」與「因為 guidance 真的抽乾了邊界樣本」兩種解釋。作法是把
    coverage 較高的一方（通常是低 guidance）隨機子取樣到與另一方相同 coverage，再比對
    near-boundary 佔比。若對齊 coverage 之後，低 guidance 仍有較高的 near-boundary 佔比，
    即為 guidance 抽乾邊界支持度的介入式證據，而不只是「樣本較少」的假象。

    low_feats/high_feats 為 coverage 特徵空間的特徵；low_margins/high_margins 為對應的
    margin。回傳含原始與對齊後 coverage、對齊後兩側 near-boundary 佔比、以及其差
    （low - high）的 dict。
    """
    cov_low = coverage_of(real_feats, low_feats, nearest_k)
    cov_high = coverage_of(real_feats, high_feats, nearest_k)
    nb_low_full = near_boundary_fraction(low_margins, threshold)
    nb_high_full = near_boundary_fraction(high_margins, threshold)

    if cov_low >= cov_high:
        # 低 guidance coverage 較高：把它降到與高 guidance 相同
        _, cov_matched, idx = match_coverage_by_subsampling(
            real_feats, low_feats, cov_high, nearest_k, repeats)
        nb_low_matched = near_boundary_fraction(low_margins[idx], threshold)
        nb_high_matched = nb_high_full
        matched_low, matched_high = cov_matched, cov_high
    else:
        _, cov_matched, idx = match_coverage_by_subsampling(
            real_feats, high_feats, cov_low, nearest_k, repeats)
        nb_high_matched = near_boundary_fraction(high_margins[idx], threshold)
        nb_low_matched = nb_low_full
        matched_low, matched_high = cov_low, cov_matched

    return {
        "coverage_low": cov_low,
        "coverage_high": cov_high,
        "nb_frac_low_full": nb_low_full,
        "nb_frac_high_full": nb_high_full,
        "matched_coverage_low": matched_low,
        "matched_coverage_high": matched_high,
        "nb_frac_low_matched": nb_low_matched,
        "nb_frac_high_matched": nb_high_matched,
        "nb_gap_matched": nb_low_matched - nb_high_matched,
    }


def _self_check():
    """以合成資料驗證 coverage 受控介入分析的機制流程（非科學主張）。"""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 16
    real = torch.randn(1500, dim, device=device)
    # 低 guidance：與 real 同分布（高 coverage），margin 全部偏小（near-boundary 多）
    low_feats = torch.randn(1500, dim, device=device)
    low_margins = torch.rand(1500) * 0.45                      # 全部 < 0.5
    # 高 guidance：集中於原型附近（低 coverage），margin 全部偏大（near-boundary 少）
    high_feats = 0.5 * torch.randn(1500, dim, device=device) + 0.8
    high_margins = 0.55 + torch.rand(1500) * 0.45              # 全部 > 0.5

    out = coverage_controlled_margin(low_feats, low_margins, high_feats, high_margins,
                                     real, nearest_k=5, threshold=0.5, repeats=3)
    print("coverage-controlled margin self-check:")
    for key, val in out.items():
        print(f"  {key:<24} {val:.4f}")
    assert out["coverage_low"] > out["coverage_high"], "低 guidance 應有較高 coverage"
    assert abs(out["matched_coverage_low"] - out["matched_coverage_high"]) < 0.06, \
        "對齊後兩側 coverage 應接近"
    assert out["nb_gap_matched"] > 0, "對齊 coverage 後低 guidance 仍應有較多 near-boundary"
    print("  OK（合成資料 margin 與特徵獨立，僅驗證流程，非科學主張）")


def main():
    parser = argparse.ArgumentParser(description="Near-boundary support analysis (C2).")
    parser.add_argument("--generated", default="generated/dataset.pt")
    parser.add_argument("--cnn", default="mnist_cnn.pt")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--per-class", type=int, default=1000,
                        help="Real samples per class for the reference margin distribution")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-check", action="store_true",
                        help="Run the synthetic self-check for coverage-controlled margin analysis and exit.")
    args = parser.parse_args()

    if args.self_check:
        _self_check()
        return

    for path in (args.cnn, args.generated):
        if not os.path.exists(path):
            print(f"ERROR: file not found: {path}")
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    cnn = load_cnn(args.cnn, device)

    real_images, real_labels = load_real_per_class(args.data_dir, args.per_class, args.seed)
    gen_images, gen_labels = load_generated(args.generated)

    real_summary, _, _ = analyze_dataset(cnn, real_images, real_labels, device, args.threshold)
    gen_summary, _, _ = analyze_dataset(cnn, gen_images, gen_labels, device, args.threshold)

    print("\nMargin / near-boundary analysis (real-trained CNN as judge)")
    print(f"  {'metric':<24} {'real':>10} {'generated':>12}")
    print("  " + "-" * 48)
    print(f"  {'mean margin':<24} {real_summary['mean_margin']:>10.4f} {gen_summary['mean_margin']:>12.4f}")
    print(f"  {'median margin':<24} {real_summary['median_margin']:>10.4f} {gen_summary['median_margin']:>12.4f}")
    print(f"  {'p05 margin':<24} {real_summary['p05_margin']:>10.4f} {gen_summary['p05_margin']:>12.4f}")
    tkey = str(args.threshold)
    print(f"  {'near-boundary frac':<24} {real_summary['near_boundary_frac'][tkey]:>10.4f} "
          f"{gen_summary['near_boundary_frac'][tkey]:>12.4f}")
    print(f"  {'label-noise frac':<24} {real_summary['label_noise_frac']:>10.4f} "
          f"{gen_summary['label_noise_frac']:>12.4f}")
    print("\n  (generated << real near-boundary frac => guidance depleted boundary support;")
    print("   check label-noise frac is not the driver.)")


if __name__ == "__main__":
    main()
