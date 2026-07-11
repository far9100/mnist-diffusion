"""P1.1：在真實 CIFAR-10 上對量測堆疊做數值驗證。

在信任 Phase-1 量測堆疊處理 generated 樣本之前，先驗證它能算出
合理的數字：
  - 兩個不相交的真實切分之間的 FD-DINOv2 與 PRDC -> FD 小、P/R/coverage 高；
  - FD-DINOv2 real-vs-高斯 noise -> 大（有鑑別力）；
  - clean-fid（Inception）真實子集 vs 內建 CIFAR-10 stats -> 小（盡力而為）。

已發表模型的 FID 錨點（EDM 約 1.79 / diffusers ddpm-cifar10 約 3.17）是
另一個、更強的 gate，就在 sweep 之前執行；本腳本把關的是
CaF selector 所仰賴的 feature 指標基礎設施。

用法：
    uv run python validate_metrics.py
"""

import argparse

import torch

from datasets.cifar import load_cifar_01
from metrics_features import dinov2_features, fd_from_features, prdc_from_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="images per split")
    ap.add_argument("--nearest-k", type=int, default=5)
    ap.add_argument("--clean-fid", action="store_true", help="also run clean-fid anchor check")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_imgs, _ = load_cifar_01("cifar10", train=True)   # feature 抽取器預期 [0,1]
    test_imgs, _ = load_cifar_01("cifar10", train=False)
    n = min(args.n, train_imgs.size(0), test_imgs.size(0))
    real_a = train_imgs[:n]           # 不相交的真實切分
    real_b = test_imgs[:n]
    noise = torch.rand(n, 3, 32, 32)  # 均勻 noise 對照組
    print(f"real_a(train)={real_a.shape}  real_b(test)={real_b.shape}  n={n}")

    print("Extracting DINOv2 features ...")
    fa = dinov2_features(real_a, device)
    fb = dinov2_features(real_b, device)
    fn = dinov2_features(noise, device)
    print(f"feature dim: {fa.shape[1]}")

    fd_rr = fd_from_features(fa, fb)
    fd_rn = fd_from_features(fa, fn)
    prdc_rr = prdc_from_features(fa, fb, nearest_k=args.nearest_k)
    prdc_rn = prdc_from_features(fa, fn, nearest_k=args.nearest_k)

    print("\n=== FD-DINOv2 ===")
    print(f"  real vs real  : {fd_rr:10.3f}   (expect small)")
    print(f"  real vs noise : {fd_rn:10.3f}   (expect >> real-vs-real)")
    print("=== PRDC (DINOv2 space) ===")
    print(f"  real vs real  : {prdc_rr}")
    print(f"  real vs noise : {prdc_rn}")

    ok = (fd_rr < fd_rn / 5) and (prdc_rr["coverage"] > prdc_rn["coverage"]) \
        and (prdc_rr["precision"] > 0.5)
    print(f"\n  feature-metric sanity: {'PASS' if ok else 'FAIL'}")

    if args.clean_fid:
        try:
            from fid_clean import clean_fid_vs_dataset
            sub = real_b[:2000]
            fid_val = clean_fid_vs_dataset(sub, dataset_name="cifar10",
                                           dataset_split="test", dataset_res=32)
            print(f"\n=== clean-fid (Inception) ===")
            print(f"  2k real-test vs built-in cifar10-test stats: {fid_val:.3f} (expect small)")
        except Exception as e:
            print(f"\n  clean-fid check skipped: {repr(e)[:200]}")


if __name__ == "__main__":
    main()
