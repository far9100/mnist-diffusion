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


def main():
    parser = argparse.ArgumentParser(description="Near-boundary support analysis (C2).")
    parser.add_argument("--generated", default="generated/dataset.pt")
    parser.add_argument("--cnn", default="mnist_cnn.pt")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--per-class", type=int, default=1000,
                        help="Real samples per class for the reference margin distribution")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

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
