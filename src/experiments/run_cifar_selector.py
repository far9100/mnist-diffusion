"""Phase 1 主體：在 CIFAR-10 的 CFG-guidance 軸上量測 CaF 選擇器訊號。

將 MNIST 的 Gate-A 實驗延伸到 CIFAR-10，使用經驗證的 EDM backbone
（FID 1.848，見 R-2026-07-03-03_test_edm-fid-gate.md）。這裡的「sampler config」
軸是 classifier-free guidance 強度 w，由 cond + uncond 兩個 EDM checkpoint 建構
（原廠 cond 模型並非以 CFG 訓練，因此我們在 denoised 空間結合兩個
模型：D_w = D_uncond + w*(D_cond - D_uncond)）。

每個 seed、每個 guidance w：產生各類別平衡的 CIFAR，於 Inception 特徵空間計算各類別
PRDC（重用快取的 FID detector——這是最先取得、免下載的特徵空間；DINOv2/CLIP 交叉檢查
是後續的護欄），以及 TSTR（從頭訓練的 ResNet-18）。接著由 CaF（自 real-vs-real 參考自動決定 tau）
選出一個 config；我們在各 seed 上計分 regret@selected / rank / top-k。

問題（Phase 1 go/no-go）：MNIST 的結果——CaF 免訓練即挑出
TSTR 最佳的 config、coverage 驅動效用、precision 不驅動——是否能在
CIFAR-10（以及後續 near-boundary 機制尚未飽和的 CIFAR-100）上
重現？

Usage:
    uv run python run_cifar_selector.py --quick
    uv run python run_cifar_selector.py --seeds 0 1 2 --per-class 1000
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

EDM_DIR = os.path.join(_pathfix.ROOT, "third_party", "edm")  # vendored EDM 在專案根，改用墊片提供的 ROOT
sys.path.insert(0, EDM_DIR)

import dnnlib  # noqa: E402
from generate import edm_sampler, StackedRandomGenerator  # noqa: E402
from phase1_edm_repro import load_net, DETECTOR_URL  # noqa: E402
from metrics_prdc import compute_prdc_per_class  # noqa: E402
from cifar_classifier import run_tstr  # noqa: E402
from datasets.cifar import load_real_per_class, build_test_loader, NUM_CLASSES  # noqa: E402
from selector import select_and_report  # noqa: E402

FULL_GUIDANCE = [1.0, 1.5, 2.0, 3.0]
QUICK_GUIDANCE = [1.0, 3.0]


def summarize(values):
    """對一組純量計算 mean/std/sem/min/max（忽略 None）。"""
    vals = [float(v) for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return None
    mean = sum(vals) / n
    std = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return {"mean": mean, "std": std, "sem": std / (n ** 0.5) if n else 0.0,
            "min": min(vals), "max": max(vals), "n": n, "values": vals}


class CFGNet:
    """以兩個 EDM 模型在 EDM denoised 空間進行 classifier-free guidance。

    對外提供 edm_sampler 所需的屬性，並結合 conditional 與
    unconditional 的 denoiser：D_w = D_uncond + w*(D_cond - D_uncond)。w=1 => cond。
    """

    def __init__(self, cond, uncond, w):
        self.cond, self.uncond, self.w = cond, uncond, w
        self.sigma_min = cond.sigma_min
        self.sigma_max = cond.sigma_max
        self.round_sigma = cond.round_sigma
        self.img_channels = cond.img_channels
        self.img_resolution = cond.img_resolution
        self.label_dim = cond.label_dim

    def __call__(self, x, sigma, class_labels=None):
        d_cond = self.cond(x, sigma, class_labels)
        if self.w == 1.0:
            return d_cond
        d_uncond = self.uncond(x, sigma, None)
        return d_uncond + self.w * (d_cond - d_uncond)


@torch.no_grad()
def generate_balanced(net, per_class, device, batch, seed_base, num_classes=10, num_steps=18):
    """為每個類別產生 `per_class` 張影像。回傳 float [-1,1] 的影像與標籤。"""
    imgs, labels = [], []
    cursor = seed_base
    for c in range(num_classes):
        remaining = per_class
        while remaining > 0:
            bs = min(batch, remaining)
            seeds = list(range(cursor, cursor + bs))
            cursor += bs
            rnd = StackedRandomGenerator(device, seeds)
            latents = rnd.randn([bs, net.img_channels, net.img_resolution,
                                 net.img_resolution], device=device)
            class_labels = torch.zeros([bs, net.label_dim], device=device)
            class_labels[:, c] = 1
            x = edm_sampler(net, latents, class_labels, randn_like=rnd.randn_like,
                            num_steps=num_steps)
            imgs.append(x.clip(-1, 1).float().cpu())
            labels.append(torch.full((bs,), c, dtype=torch.long))
            remaining -= bs
    return torch.cat(imgs), torch.cat(labels)


@torch.no_grad()
def inception_feats(images_float, detector, device, batch=256):
    """Inception-v3 的 2048 維特徵。images_float 位於 [-1,1] -> uint8[0,255]。"""
    feats = []
    n = images_float.size(0)
    for i in range(0, n, batch):
        im = images_float[i:i + batch]
        im = ((im + 1) * 127.5).clip(0, 255).to(torch.uint8).to(device)
        if im.shape[1] == 1:
            im = im.repeat([1, 3, 1, 1])
        feats.append(detector(im, return_features=True).float().cpu())
    return torch.cat(feats)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one_seed(seed, cond, uncond, detector, guidance_list, real_feats, real_labels,
                 real_test_loader, args, device):
    set_seed(seed)
    print(f"\n########## seed {seed} ##########", flush=True)

    # Real-vs-real 參考 precision（免 TSTR 的 tau）。
    n = real_feats.size(0)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    a_idx, b_idx = perm[:n // 2], perm[n // 2:]
    ref_mean, _ = compute_prdc_per_class(real_feats[b_idx].to(device), real_labels[b_idx].to(device),
                                         real_feats[a_idx].to(device), real_labels[a_idx].to(device),
                                         nearest_k=args.nearest_k, num_classes=args.num_classes)
    real_ref_precision = ref_mean["precision"]
    print(f"  real-vs-real ref precision: {real_ref_precision:.4f}", flush=True)

    configs = []
    for w in guidance_list:
        net = CFGNet(cond, uncond, w)
        seed_base = seed * 10_000_000 + int(w * 1000) * 10_000
        gen_imgs, gen_labels = generate_balanced(net, args.per_class, device,
                                                 args.batch, seed_base, args.num_classes,
                                                 num_steps=args.gen_steps)
        gen_feats = inception_feats(gen_imgs, detector, device, batch=args.batch)
        prdc, _ = compute_prdc_per_class(real_feats.to(device), real_labels.to(device),
                                         gen_feats.to(device), gen_labels.to(device),
                                         nearest_k=args.nearest_k, num_classes=args.num_classes)
        tstr, _ = run_tstr(gen_imgs, gen_labels, real_test_loader, device,
                           num_classes=args.num_classes, epochs=args.tstr_epochs,
                           lr=args.tstr_lr, batch_size=args.tstr_batch_size)
        cfg = {"name": f"w{w:g}", "guidance": w,
               "precision": prdc["precision"], "coverage": prdc["coverage"],
               "recall": prdc["recall"], "density": prdc["density"], "tstr": tstr}
        configs.append(cfg)
        print(f"  w={w:<4g} precision={prdc['precision']:.4f} coverage={prdc['coverage']:.4f}"
              f" recall={prdc['recall']:.4f} tstr={tstr:.2f}", flush=True)

    report = select_and_report(configs, real_ref_precision=real_ref_precision,
                               tau_fraction=args.tau_fraction, utility_key="tstr")
    print(f"  -> CaF selected {report['selected']} (oracle {report['oracle_best']}, "
          f"regret {report['regret_at_selected']}, rank {report['rank']}/{report['n_configs']})",
          flush=True)
    return {"seed": seed, "real_ref_precision": real_ref_precision,
            "configs": configs, "report": report}


def aggregate(seed_results, guidance_list):
    names = [f"w{w:g}" for w in guidance_list]
    per_config = []
    for name in names:
        prec, cov, tstr = [], [], []
        for sr in seed_results:
            c = next(c for c in sr["configs"] if c["name"] == name)
            prec.append(c["precision"]); cov.append(c["coverage"]); tstr.append(c["tstr"])
        per_config.append({"name": name, "precision": summarize(prec),
                           "coverage": summarize(cov), "tstr": summarize(tstr)})
    selected = [sr["report"]["selected"] for sr in seed_results]
    counts = {nm: selected.count(nm) for nm in sorted(set(selected))}
    modal = max(counts, key=counts.get)
    return {
        "n_seeds": len(seed_results),
        "per_config": per_config,
        "selection": {"per_seed": selected, "counts": counts, "modal": modal,
                      "modal_fraction": counts[modal] / len(selected)},
        "regret_at_selected": summarize([sr["report"]["regret_at_selected"] for sr in seed_results]),
        "rank_per_seed": [sr["report"]["rank"] for sr in seed_results],
        "topk_hit_rate": sum(bool(sr["report"]["topk_hit"]) for sr in seed_results) / len(seed_results),
        "oracle_best_per_seed": [sr["report"]["oracle_best"] for sr in seed_results],
    }


def print_summary(agg):
    print("\n" + "=" * 78)
    print(f"  CIFAR-10 CaF selector signal over {agg['n_seeds']} seeds (CFG guidance axis)")
    print("=" * 78)
    print(f"  {'config':>7} {'precision':>18} {'coverage':>18} {'TSTR%':>18}")
    print("  " + "-" * 74)
    for pc in agg["per_config"]:
        def f(s): return f"{s['mean']:.4f}+/-{s['std']:.4f}" if s else "n/a"
        def f2(s): return f"{s['mean']:.2f}+/-{s['std']:.2f}" if s else "n/a"
        print(f"  {pc['name']:>7} {f(pc['precision']):>18} {f(pc['coverage']):>18} {f2(pc['tstr']):>18}")
    print("  " + "-" * 74)
    sel = agg["selection"]
    print(f"  CaF selection per seed : {sel['per_seed']}  (modal {sel['modal']}, "
          f"{sel['modal_fraction']*100:.0f}%)")
    print(f"  oracle TSTR-best/seed  : {agg['oracle_best_per_seed']}")
    if agg["regret_at_selected"]:
        r = agg["regret_at_selected"]
        print(f"  regret@selected        : {r['mean']:.3f} +/- {r['std']:.3f} pp (max {r['max']:.3f})")
    print(f"  rank per seed          : {agg['rank_per_seed']}")
    print(f"  top-3 hit rate         : {agg['topk_hit_rate']*100:.0f}%")
    print("=" * 78)


def main():
    p = argparse.ArgumentParser(description="CIFAR-10 CaF selector signal (Phase 1).")
    p.add_argument("--cond", default="checkpoints/edm-cifar10-32x32-cond-vp.pkl")
    p.add_argument("--uncond", default="checkpoints/edm-cifar10-32x32-uncond-vp.pkl")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--per-class", type=int, default=1000)
    p.add_argument("--real-per-class", type=int, default=1000)
    p.add_argument("--gen-steps", type=int, default=18,
                   help="EDM sampling steps (18=full quality; lower=faster preview)")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--nearest-k", type=int, default=5)
    p.add_argument("--tau-fraction", type=float, default=0.9)
    p.add_argument("--tstr-epochs", type=int, default=30)
    p.add_argument("--tstr-lr", type=float, default=0.1)
    p.add_argument("--tstr-batch-size", type=int, default=128)
    p.add_argument("--output", default="results/cifar_selector.json")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    args.num_classes = NUM_CLASSES[args.dataset]

    guidance_list = QUICK_GUIDANCE if args.quick else FULL_GUIDANCE
    if args.quick:
        if args.per_class == 1000:
            args.per_class = 64
        if args.real_per_class == 1000:
            args.real_per_class = 200
        if args.tstr_epochs == 30:
            args.tstr_epochs = 2
        args.seeds = args.seeds[:1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"seeds={args.seeds} guidance={guidance_list} per_class={args.per_class} "
          f"tstr_epochs={args.tstr_epochs}", flush=True)
    os.makedirs("results", exist_ok=True)

    cond = load_net(args.cond, device)
    uncond = load_net(args.uncond, device)
    with dnnlib.util.open_url(DETECTOR_URL, verbose=True) as f:
        detector = pickle.load(f).to(device)
    print("Loaded cond + uncond EDM nets and Inception detector", flush=True)

    # Real 參考集（Inception 特徵），跨 seed 共用。
    real_imgs, real_labels = load_real_per_class(args.dataset, args.real_per_class, seed=0)
    real_feats = inception_feats(real_imgs, detector, device, batch=args.batch)
    real_test_loader = build_test_loader(args.dataset, batch_size=256)
    print(f"Real ref: {real_feats.shape[0]} imgs, feat dim {real_feats.shape[1]}", flush=True)

    seed_results = []
    for seed in args.seeds:
        seed_results.append(run_one_seed(seed, cond, uncond, detector, guidance_list,
                                         real_feats, real_labels, real_test_loader, args, device))

    agg = aggregate(seed_results, guidance_list)
    print_summary(agg)

    out = {"metadata": {"dataset": args.dataset, "axis": "CFG guidance (cond+uncond EDM)",
                        "seeds": args.seeds, "per_class": args.per_class,
                        "guidance_grid": guidance_list, "tstr_epochs": args.tstr_epochs,
                        "prdc_feature_space": "Inception-v3 (FID detector)"},
           "aggregate": agg, "per_seed": seed_results, "args": {k: v for k, v in vars(args).items()}}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
