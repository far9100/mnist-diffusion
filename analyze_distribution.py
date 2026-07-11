"""
比對生成的 MNIST .pt 資料集與真實 MNIST，檢查分佈是否合理、是否有 mode collapse 徵兆。

計算的 per-class 指標：
  1. 像素空間的平均成對 L2 —— 在原始像素層級的類別內多樣性。
  2. CNN 特徵空間的平均成對 L2 —— 在分類器倒數第二層（256 維）特徵空間中的
     類別內多樣性。
  3. 對全部 784 個像素取平均的像素 std —— 分散程度是否與真實相符？
  4. 最大 softmax confidence（mean ± std）—— 若飽和在接近 1.0 且沒有分散，
     表示生成器只會產生「教科書式」的樣本。
  5. 生成與真實類別 centroid 在特徵空間中的 L2 距離 —— 生成的數字是否落在
     真實類別 manifold 附近？

mode collapse 的判斷法則（在最後印出）：
  - 生成的成對多樣性 << 真實（gen/real 比值 < ~0.6），無論是像素或特徵空間
    都值得懷疑
  - 生成的像素 std << 真實（比值 < ~0.6）值得懷疑
  - 生成的 confidence ≈ 1.0 且 std ≈ 0，而真實有明顯分散 =>
    生成器避開了困難／模稜兩可的樣本

Usage:
    uv run python analyze_distribution.py
    uv run python analyze_distribution.py --generated generated/dataset.pt
    uv run python analyze_distribution.py --per-class 100 --seed 0
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from evaluate import MNISTClassifier
from train import download_mnist


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare a generated MNIST .pt dataset against real MNIST."
    )
    parser.add_argument("--generated", default="generated/dataset.pt",
                        help="Path to generated .pt file (default: generated/dataset.pt)")
    parser.add_argument("--checkpoint", default="mnist_cnn.pt",
                        help="CNN checkpoint to use as feature extractor")
    parser.add_argument("--data-dir", default="./data",
                        help="MNIST data directory (default: ./data)")
    parser.add_argument("--per-class", type=int, default=100,
                        help="Real samples per class to sample (default: 100, "
                             "matches generated/dataset.pt)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Sampling seed for real MNIST subset (default: 0)")
    parser.add_argument("--diversity-ratio-warn", type=float, default=0.6,
                        help="Warn if gen/real diversity ratio falls below this "
                             "(default: 0.6)")
    return parser.parse_args()


@torch.no_grad()
def extract_features(model, images, device, batch_size=256):
    feats = []
    flatten = nn.Flatten()
    fc1 = model.classifier[1]
    relu = model.classifier[2]
    for start in range(0, images.size(0), batch_size):
        batch = images[start:start + batch_size].to(device)
        h = model.features(batch)
        h = relu(fc1(flatten(h)))
        feats.append(h.cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def softmax_confidence(model, images, device, batch_size=256):
    confs = []
    for start in range(0, images.size(0), batch_size):
        batch = images[start:start + batch_size].to(device)
        logits = model(batch)
        confs.append(F.softmax(logits, dim=1).max(dim=1).values.cpu())
    return torch.cat(confs, dim=0)


def mean_pairwise_l2(x):
    n = x.size(0)
    if n < 2:
        return float("nan")
    d = torch.cdist(x, x)
    mask = torch.triu(torch.ones_like(d), diagonal=1).bool()
    return d[mask].mean().item()


def load_real_per_class(data_dir, per_class, seed):
    download_mnist(data_dir)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    test = datasets.MNIST(data_dir, train=False, transform=transform)
    targets = test.targets.tolist()
    by_class = {d: [] for d in range(10)}
    for i, t in enumerate(targets):
        by_class[int(t)].append(i)

    g = torch.Generator().manual_seed(seed)
    images_list, labels_list = [], []
    for d in range(10):
        idx = torch.tensor(by_class[d])
        k = min(per_class, idx.numel())
        perm = torch.randperm(idx.numel(), generator=g)[:k]
        for s in idx[perm].tolist():
            img, lab = test[s]
            images_list.append(img)
            labels_list.append(lab)
    return torch.stack(images_list), torch.tensor(labels_list)


def load_generated(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    return data["images"], data["labels"]


def main():
    args = parse_args()
    for path in (args.checkpoint, args.generated):
        if not os.path.exists(path):
            print(f"ERROR: file not found: {path}")
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MNISTClassifier().to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"Loaded CNN: {args.checkpoint}")

    real_images, real_labels = load_real_per_class(
        args.data_dir, args.per_class, args.seed
    )
    gen_images, gen_labels = load_generated(args.generated)
    print(f"Real     : {real_images.shape[0]} images "
          f"({args.per_class}/class, seed={args.seed})")
    print(f"Generated: {gen_images.shape[0]} images  ({args.generated})")

    real_feats = extract_features(model, real_images, device)
    gen_feats = extract_features(model, gen_images, device)
    real_conf = softmax_confidence(model, real_images, device)
    gen_conf = softmax_confidence(model, gen_images, device)

    pix_ratios, feat_ratios, std_ratios = [], [], []

    print()
    print("=" * 92)
    print("  Intra-class diversity  (mean pairwise L2; lower in gen = collapse)")
    print("=" * 92)
    print(f"  {'Digit':<6} {'pixel real':>11} {'pixel gen':>11} {'gen/real':>9}    "
          f"{'feat real':>10} {'feat gen':>10} {'gen/real':>9}")
    print(f"  {'-'*6} {'-'*11} {'-'*11} {'-'*9}    {'-'*10} {'-'*10} {'-'*9}")
    for d in range(10):
        rm = real_labels == d
        gm = gen_labels == d
        r_pix = real_images[rm].view(rm.sum().item(), -1)
        g_pix = gen_images[gm].view(gm.sum().item(), -1)
        r_pd = mean_pairwise_l2(r_pix)
        g_pd = mean_pairwise_l2(g_pix)
        r_fd = mean_pairwise_l2(real_feats[rm])
        g_fd = mean_pairwise_l2(gen_feats[gm])
        pr = g_pd / r_pd if r_pd > 0 else float("nan")
        fr = g_fd / r_fd if r_fd > 0 else float("nan")
        pix_ratios.append(pr)
        feat_ratios.append(fr)
        print(f"  {d:<6} {r_pd:>11.3f} {g_pd:>11.3f} {pr:>9.3f}    "
              f"{r_fd:>10.3f} {g_fd:>10.3f} {fr:>9.3f}")

    print()
    print("=" * 92)
    print("  Per-class pixel std  (averaged over all 784 pixels)")
    print("=" * 92)
    print(f"  {'Digit':<6} {'real':>10} {'gen':>10} {'gen/real':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
    for d in range(10):
        r = real_images[real_labels == d]
        g = gen_images[gen_labels == d]
        r_std = r.std(dim=0).mean().item()
        g_std = g.std(dim=0).mean().item()
        ratio = g_std / r_std if r_std > 0 else float("nan")
        std_ratios.append(ratio)
        print(f"  {d:<6} {r_std:>10.4f} {g_std:>10.4f} {ratio:>10.3f}")

    print()
    print("=" * 92)
    print("  CNN max-softmax confidence  (mean ± std; saturated gen w/ no spread = canonical-only)")
    print("=" * 92)
    print(f"  {'Digit':<6} {'real (mean ± std)':>22} {'gen (mean ± std)':>22}")
    print(f"  {'-'*6} {'-'*22} {'-'*22}")
    for d in range(10):
        rc = real_conf[real_labels == d]
        gc = gen_conf[gen_labels == d]
        print(f"  {d:<6}    {rc.mean().item():.4f} ± {rc.std().item():.4f}    "
              f"   {gc.mean().item():.4f} ± {gc.std().item():.4f}")

    print()
    print("=" * 92)
    print("  Centroid distance in feature space  (gen class-mean vs real class-mean)")
    print("=" * 92)
    print(f"  {'Digit':<6} {'L2(real_mean, gen_mean)':>26} {'real intra-class L2':>22}  "
          f"{'centroid/diversity':>20}")
    print(f"  {'-'*6} {'-'*26} {'-'*22}  {'-'*20}")
    for d in range(10):
        rm = real_labels == d
        gm = gen_labels == d
        rc = real_feats[rm].mean(dim=0)
        gc = gen_feats[gm].mean(dim=0)
        cdist = (rc - gc).norm().item()
        intra = mean_pairwise_l2(real_feats[rm])
        ratio = cdist / intra if intra > 0 else float("nan")
        print(f"  {d:<6} {cdist:>26.3f} {intra:>22.3f}  {ratio:>20.3f}")
    print(
        "  (centroid/diversity > ~1.0 means generated centroid sits outside the\n"
        "   typical real-class spread — i.e., the generator drifted off the manifold.)"
    )

    print()
    print("=" * 92)
    print("  Summary heuristics")
    print("=" * 92)
    warn = args.diversity_ratio_warn
    pix_avg = sum(pix_ratios) / len(pix_ratios)
    feat_avg = sum(feat_ratios) / len(feat_ratios)
    std_avg = sum(std_ratios) / len(std_ratios)
    print(f"  Avg pixel-space gen/real diversity  : {pix_avg:.3f}  "
          f"({'OK' if pix_avg >= warn else 'LOW — possible mode collapse'})")
    print(f"  Avg feature-space gen/real diversity: {feat_avg:.3f}  "
          f"({'OK' if feat_avg >= warn else 'LOW — possible mode collapse'})")
    print(f"  Avg pixel std gen/real ratio        : {std_avg:.3f}  "
          f"({'OK' if std_avg >= warn else 'LOW — possible mode collapse'})")
    overall_conf_real = real_conf.mean().item()
    overall_conf_gen = gen_conf.mean().item()
    print(f"  Mean CNN confidence  real / gen     : "
          f"{overall_conf_real:.4f} / {overall_conf_gen:.4f}  "
          f"(gen >> real with tiny std => 'canonical-only' bias)")
    print("=" * 92)


if __name__ == "__main__":
    main()
