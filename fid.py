"""MNIST-FID：不使用 scipy 計算的生成 MNIST Fréchet distance。

標準 FID 使用 Inception-v3 feature。MNIST 數字不在 Inception 的
領域內，而且引入笨重的相依套件與本專案 from-scratch 的精神相衝突，
因此我們改用以真實資料訓練的裁判 CNN 倒數第二層的 256 維 feature
（``mnist_cnn.pt``，與 analyze_distribution.py 使用的抽取器相同）。

我們因此將此指標稱為 **MNIST-FID（classifier-Frechet distance）**。它是一個
*相對*指標——用於將不同 sampler / guidance 尺度互相排名時有意義，
無法與已發表的 Inception-FID 數字比較。

兩個高斯分布之間的 Frechet distance：

    FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2 (Sigma_r Sigma_g)^{1/2})

唯一不平凡的項是 Tr((Sigma_r Sigma_g)^{1/2})。我們避免計算完整的矩陣
平方根：構造對稱 PSD 的夾層 M = Sigma_r^{1/2} Sigma_g Sigma_r^{1/2}
（與 Sigma_r Sigma_g 有相同的 eigenvalue），並透過 torch.linalg.eigvalsh
將其 eigenvalue 的平方根加總。所有計算皆以 float64 進行。

用法（自我檢查）：
    uv run python fid.py                       # noise-vs-real and real-vs-real sanity
    uv run python fid.py --generated generated/dataset.pt
"""

import argparse
import os
import sys

import torch

from evaluate import MNISTClassifier
from analyze_distribution import extract_features, load_real_per_class, load_generated


def load_cnn(ckpt="mnist_cnn.pt", device="cpu"):
    """載入作為 feature 抽取器、以真實資料訓練的裁判 CNN。"""
    model = MNISTClassifier().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model


def feature_stats(feats):
    """(N, D) feature 矩陣的平均值與（經 ridge 穩定化的）covariance。"""
    feats = feats.double()
    mu = feats.mean(dim=0)
    centered = feats - mu
    cov = (centered.T @ centered) / (feats.size(0) - 1)
    # Ridge 項：用少於 1000 個樣本估出的 256 維 covariance 接近奇異矩陣。
    cov = cov + 1e-6 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device)
    return mu, cov


def _matrix_sqrt_psd(cov):
    """透過 eigendecomposition 計算對稱 PSD 的平方根（穩定，float64）。"""
    s, u = torch.linalg.eigh(cov)
    s = s.clamp(min=0)
    return (u * s.sqrt()) @ u.T


def frechet_distance(mu_r, cov_r, mu_g, cov_g):
    """給定兩個高斯分布的平均值/covariance，計算其 Frechet distance。"""
    mu_r, cov_r = mu_r.double(), cov_r.double()
    mu_g, cov_g = mu_g.double(), cov_g.double()

    mean_term = (mu_r - mu_g).pow(2).sum()
    # 透過對稱夾層 M 計算 Tr((Sigma_r Sigma_g)^{1/2})，其 eigenvalue
    # 等於 Sigma_r Sigma_g 的 eigenvalue，且為實數且非負（兩者皆 PSD）。
    cov_r_sqrt = _matrix_sqrt_psd(cov_r)
    m = cov_r_sqrt @ cov_g @ cov_r_sqrt
    eig = torch.linalg.eigvalsh(m).clamp(min=0)
    tr_sqrt = eig.sqrt().sum()

    fid = mean_term + torch.trace(cov_r) + torch.trace(cov_g) - 2.0 * tr_sqrt
    return float(fid.item())


def real_feature_stats(real_images, cnn_ckpt="mnist_cnn.pt", device="cpu", model=None):
    """計算一次真實影像的 (mu, cov)，以便在多次呼叫間快取重用。"""
    if model is None:
        model = load_cnn(cnn_ckpt, device)
    return feature_stats(extract_features(model, real_images, device))


def compute_fid(gen_images, real_images=None, cnn_ckpt="mnist_cnn.pt", device="cpu",
                model=None, real_stats=None):
    """generated 與真實影像之間的 MNIST-FID。

    傳入 `real_images` 或預先計算好的 `real_stats` (mu, cov) 其中之一。傳入
    已載入的 `model` 與已快取的 `real_stats` 可避免在 sweep 中重複運算。
    """
    if model is None:
        model = load_cnn(cnn_ckpt, device)
    gen_feats = extract_features(model, gen_images, device)
    mu_g, cov_g = feature_stats(gen_feats)
    if real_stats is None:
        if real_images is None:
            raise ValueError("Provide either real_images or real_stats.")
        real_stats = real_feature_stats(real_images, device=device, model=model)
    mu_r, cov_r = real_stats
    return frechet_distance(mu_r, cov_r, mu_g, cov_g)


def main():
    parser = argparse.ArgumentParser(description="MNIST-FID (classifier-Frechet distance).")
    parser.add_argument("--generated", default=None,
                        help="Optional generated .pt to score against real MNIST")
    parser.add_argument("--checkpoint", default="mnist_cnn.pt",
                        help="Judge CNN checkpoint (feature extractor)")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--per-class", type=int, default=1000,
                        help="Real samples per class (default: 1000)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: judge CNN not found: {args.checkpoint}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = load_cnn(args.checkpoint, device)

    real_images, real_labels = load_real_per_class(args.data_dir, args.per_class, args.seed)
    n = real_images.size(0)
    half = n // 2
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    split_a = real_images[perm[:half]]
    split_b = real_images[perm[half:]]

    stats_a = real_feature_stats(split_a, device=device, model=model)

    # 檢查 1：real-vs-real 的不相交切分 -> 約為 0
    fid_rr = compute_fid(split_b, device=device, model=model, real_stats=stats_a)
    # 檢查 2：純高斯 noise -> 很大
    noise = torch.randn(2000, 1, 28, 28).clamp(-1, 1)
    fid_noise = compute_fid(noise, device=device, model=model, real_stats=stats_a)
    # 檢查 3：完全相同的統計量 -> 恰好為 0
    mu, cov = stats_a
    fid_same = frechet_distance(mu, cov, mu, cov)

    print(f"\nMNIST-FID sanity checks (real per-class={args.per_class}):")
    print(f"  real vs real (disjoint split) : {fid_rr:10.4f}   (expect small)")
    print(f"  random noise vs real          : {fid_noise:10.4f}   (expect large)")
    print(f"  identical stats               : {fid_same:10.4f}   (expect ~0)")

    if args.generated:
        gen_images, _ = load_generated(args.generated)
        full_stats = real_feature_stats(real_images, device=device, model=model)
        fid_gen = compute_fid(gen_images, device=device, model=model, real_stats=full_stats)
        print(f"\n  {args.generated}")
        print(f"  generated vs real             : {fid_gen:10.4f}")


if __name__ == "__main__":
    main()
