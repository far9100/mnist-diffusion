"""Phase 1 正確性 gate：重現 EDM CIFAR-10 條件式 FID（約 1.79）。

在任何 CIFAR sweep 之前，我們必須證明自己的 EDM 取樣路徑能重現論文的
FID，否則下游數字都不可信（重新產生的計畫、Phase 1 步驟 1
以及不變的正確性 gate）。

設計：單一行程，不使用 torch.distributed（在 Windows 上較穩健）。我們原封不動地
重用 vendored NVlabs repo（third_party/edm）中的官方 EDM sampler +
StackedRandomGenerator，載入官方預訓練 pkl，並以官方 Inception-v3 偵測器
對官方 CIFAR-10 參考 stats 計算 FID。自訂 CUDA op（bias_act/upfirdn2d）
若無法 JIT 編譯，會退回較慢的純 torch 路徑——對 32x32 而言可接受。

用法：
    uv run python phase1_edm_repro.py --smoke                 # 64 imgs -> grid PNG
    uv run python phase1_edm_repro.py --num 50000 --batch 256 # full FID gate
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch

EDM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "edm")
sys.path.insert(0, EDM_DIR)

import dnnlib  # noqa: E402  (from vendored third_party/edm)
from generate import edm_sampler, StackedRandomGenerator  # noqa: E402

DETECTOR_URL = ("https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/"
                "versions/1/files/metrics/inception-2015-12-05.pkl")


def load_net(network_pkl, device):
    with dnnlib.util.open_url(network_pkl, verbose=True) as f:
        net = pickle.load(f)["ema"].to(device)
    net.eval()
    return net


@torch.no_grad()
def generate_images(net, seeds, device, batch, class_idx=None, progress=True):
    """以官方 EDM sampler 產生影像（uint8, N x C x H x W, [0,255]）。"""
    out = []
    for i in range(0, len(seeds), batch):
        bs = seeds[i:i + batch]
        rnd = StackedRandomGenerator(device, bs)
        latents = rnd.randn([len(bs), net.img_channels, net.img_resolution,
                             net.img_resolution], device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[
                rnd.randint(net.label_dim, size=[len(bs)], device=device)]
            if class_idx is not None:
                class_labels[:, :] = 0
                class_labels[:, class_idx] = 1
        x = edm_sampler(net, latents, class_labels, randn_like=rnd.randn_like)
        img = (x * 127.5 + 128).clip(0, 255).to(torch.uint8)
        out.append(img.cpu())
        if progress:
            print(f"  generated {min(i + batch, len(seeds))}/{len(seeds)}", flush=True)
    return torch.cat(out)


@torch.no_grad()
def inception_stats(images_uint8, device, batch=64):
    """官方 Inception-v3 feature -> (mu, sigma)。images_uint8: N x C x H x W [0,255]。"""
    with dnnlib.util.open_url(DETECTOR_URL, verbose=True) as f:
        detector = pickle.load(f).to(device)
    feature_dim = 2048
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    n = images_uint8.shape[0]
    for i in range(0, n, batch):
        im = images_uint8[i:i + batch].to(device)
        if im.shape[1] == 1:
            im = im.repeat([1, 3, 1, 1])
        feat = detector(im, return_features=True).to(torch.float64)
        mu += feat.sum(0)
        sigma += feat.T @ feat
        if i % (batch * 50) == 0:
            print(f"  inception {min(i + batch, n)}/{n}", flush=True)
    mu /= n
    sigma -= mu.ger(mu) * n
    sigma /= n - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()


def compute_fid(mu, sigma, mu_ref, sigma_ref):
    import scipy.linalg
    m = np.square(mu - mu_ref).sum()
    # scipy >=1.13 移除了 `disp` 參數；sqrtm 現在只回傳矩陣本身。
    covmean = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref))
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(m + np.trace(sigma + sigma_ref - 2 * covmean))


def save_grid(images_uint8, path, nrow=8):
    from PIL import Image
    n = images_uint8.shape[0]
    c, h, w = images_uint8.shape[1:]
    ncol = (n + nrow - 1) // nrow
    grid = np.zeros((ncol * h, nrow * w, 3), dtype=np.uint8)
    imgs = images_uint8.permute(0, 2, 3, 1).numpy()
    if c == 1:
        imgs = np.repeat(imgs, 3, axis=3)
    for idx in range(n):
        r, cc = divmod(idx, nrow)
        grid[r * h:(r + 1) * h, cc * w:(cc + 1) * w] = imgs[idx]
    Image.fromarray(grid).save(path)


def main():
    p = argparse.ArgumentParser(description="EDM CIFAR-10 FID reproduction gate.")
    p.add_argument("--network", default="checkpoints/edm-cifar10-32x32-cond-vp.pkl")
    p.add_argument("--ref", default="checkpoints/cifar10-32x32.npz")
    p.add_argument("--num", type=int, default=50000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="Generate 64 imgs, save a grid, no FID")
    p.add_argument("--stats-cache", default="results/edm_cifar_stats.npz",
                   help="Where to cache generated Inception mu/sigma")
    p.add_argument("--from-stats", action="store_true",
                   help="Skip generation; compute FID from cached --stats-cache")
    p.add_argument("--output", default="results/edm_cifar_fid.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs("results", exist_ok=True)

    # 快速路徑：從已快取的 stats 重新計算 FID（不重新產生影像）。
    if args.from_stats:
        cached = np.load(args.stats_cache)
        mu, sigma = cached["mu"], cached["sigma"]
        print(f"Loaded cached inception stats from {args.stats_cache}")
    else:
        net = load_net(args.network, device)
        print(f"Loaded net: img {net.img_resolution}x{net.img_resolution}x{net.img_channels}, "
              f"label_dim={net.label_dim}, sigma[{net.sigma_min},{net.sigma_max}]")

        if args.smoke:
            imgs = generate_images(net, list(range(64)), device, batch=64)
            save_grid(imgs, "results/edm_cifar_smoke.png")
            print(f"Wrote results/edm_cifar_smoke.png  (imgs {tuple(imgs.shape)}, "
                  f"range [{imgs.min()},{imgs.max()}])")
            return

        seeds = list(range(args.seed_start, args.seed_start + args.num))
        print(f"Generating {len(seeds)} images (batch {args.batch})...")
        imgs = generate_images(net, seeds, device, batch=args.batch, progress=True)
        print(f"Generated {tuple(imgs.shape)}. Computing Inception stats...")
        mu, sigma = inception_stats(imgs, device, batch=args.batch)
        # 在計算 FID 前先快取，如此下游的數值問題絕不會迫使重新產生影像。
        np.savez(args.stats_cache, mu=mu, sigma=sigma)
        print(f"Cached inception stats -> {args.stats_cache}")

    with dnnlib.util.open_url(args.ref) as f:
        ref = dict(np.load(f))
    fid = compute_fid(mu, sigma, ref["mu"], ref["sigma"])
    print(f"\n=== EDM CIFAR-10 (cond, VP) FID over {args.num} images: {fid:.4f} ===")
    print(f"    (paper reference ~1.79; gate PASS if within ~0.1)")

    import json
    os.makedirs("results", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"fid": fid, "num_images": args.num, "network": args.network,
                   "ref": args.ref, "paper_reference": 1.79}, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
