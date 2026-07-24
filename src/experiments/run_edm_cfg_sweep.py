"""P1-1：EDM 第二 backbone 的 CFG guidance sweep（方案 A，對應規模問題）。骨架＋dry-run＋ETA。

問題設定（給第一次讀的研究生）：本專案的三判決（FID-min 近最優、CaF/coverage 選擇器、TSTR 內部峰）
全在「自訓 CFG backbone」上得出。審查問「換一個完全不同來源的 backbone，三判決還成立嗎？」——即
判決是否 backbone 相依。方案 A 用 NVlabs 官方預訓練 EDM 的兩個 checkpoint（cond 與 uncond）在取樣器
層組出 classifier-free guidance，掃與 confirmatory 相同的 10 點 grid、3 seeds、per_class 1000、TSTR
同協定，看 argmax/argmin 結構是否複製。這樣「backbone 換人、量測堆疊不變」，能乾淨隔離 backbone 因素。

兩模型 CFG（EDM denoiser 空間）：EDM 的網路回傳去噪估計 D(x;σ)。導引去噪為
    D_w(x,σ,c) = D_uncond(x,σ) + w·(D_cond(x,σ,c) − D_uncond(x,σ))
w=1 即純 cond、w=0 即純 uncond；w>1 往類別方向外插（與本專案 s-convention 的 w 一致）。`CFGWrapper`
把兩個 net 包成一個可餵給官方 `edm_sampler` 的物件，屬性（sigma_min/max、round_sigma、img_*、
label_dim）委派給 cond net。

種子：per (seed,w,class) 以 `int(sha256(...)[:15],16)` 派生無碰撞 hash 種子（§0.5），禁用舊
`seed*1e7` 公式。TSTR reps 以 T6b 的決定性衍生種子。

凍結規則（重要）：本 sweep 屬 confirmatory 等級的新跑，依 fix_tasks §3.3，**真跑前必須先 commit 一份
pre-registration amendment**（`docs/amendment_edm_backbone.md`，凍結四要件 a）。故本檔預設只做
`--dry-run`（枚舉 cell＋CFG 正確性檢查＋單 cell 計時探針→ETA），**不真跑**；真跑需明確傳 `--run` 且
amendment 已 commit。輸出一律寫新檔 `results/edm_cfg_sweep*.json`，不覆寫任何凍結檔。

Usage:
    uv run python src/experiments/run_edm_cfg_sweep.py --dry-run        # 預設：枚舉＋CFG 檢查＋ETA
    uv run python src/experiments/run_edm_cfg_sweep.py --run            # 真跑（需先 commit amendment）
    uv run python src/experiments/run_edm_cfg_sweep.py --dry-run --quick  # 極小規模冒煙
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py

import argparse
import hashlib
import json
import sys
import time

import torch

# 先匯入專案模組，讓專案的 `fid`（src/gen1_mnist/fid.py）先進 sys.modules 快取；否則稍後匯入
# phase1_edm_repro 會把 third_party/edm 插到 sys.path[0]，使 metrics_features 的 `from fid import ...`
# 誤解析到 EDM 的 third_party/edm/fid.py（模組名衝突）。
from cifar_classifier import run_tstr
from datasets.cifar import load_real_per_class, build_test_loader
from metrics_features import dinov2_features
from metrics_prdc import compute_prdc_per_class
from fid_clean import clean_fid_vs_dataset

# 官方 EDM sampler／載入器（phase1_edm_repro 於 import 時把 third_party/edm 補進 sys.path，故置於專案匯入之後）
from phase1_edm_repro import load_net  # noqa: E402
from generate import edm_sampler, StackedRandomGenerator  # noqa: E402

# 凍結規格：與 CIFAR-10 confirmatory 同 grid/seeds/per_class（amendment 只換 backbone，其餘不動）。
FROZEN_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
FROZEN_SEEDS = [10, 11, 12]
PER_CLASS = 1000
NUM_CLASSES = 10
NEAREST_K = 5


def hash_seed(*parts):
    """§0.5 hash 派生種子：int(sha256(join)[:15], 16)，無碰撞、禁用舊 seed*1e7 公式。"""
    return int(hashlib.sha256("_".join(str(p) for p in parts).encode()).hexdigest()[:15], 16)


class CFGWrapper:
    """把 cond 與 uncond 兩個 EDM net 包成單一導引去噪器，可直接餵官方 edm_sampler。

    __call__(x, sigma, class_labels) 回傳 D_uncond + w·(D_cond − D_uncond)。屬性委派給 cond net，
    使 edm_sampler 的 net.sigma_min/max、round_sigma、img_channels/resolution、label_dim 一如單模型。
    """

    def __init__(self, cond_net, uncond_net, w):
        self.cond, self.uncond, self.w = cond_net, uncond_net, w
        # edm_sampler 直接讀取的屬性一律取自 cond net
        self.sigma_min = cond_net.sigma_min
        self.sigma_max = cond_net.sigma_max
        self.img_channels = cond_net.img_channels
        self.img_resolution = cond_net.img_resolution
        self.label_dim = cond_net.label_dim

    def round_sigma(self, sigma):
        return self.cond.round_sigma(sigma)

    def __call__(self, x, sigma, class_labels=None, **kwargs):
        d_cond = self.cond(x, sigma, class_labels, **kwargs)
        d_uncond = self.uncond(x, sigma, None, **kwargs)   # uncond net label_dim=0，不吃 labels
        return d_uncond + self.w * (d_cond - d_uncond)


@torch.no_grad()
def generate_cell(cond_net, uncond_net, w, per_class, device, batch, seed):
    """對 (seed,w) 逐類生成 per_class 張導引影像，回傳 (imgs[-1,1] float32, labels)。"""
    wrapper = CFGWrapper(cond_net, uncond_net, w)
    imgs, labels = [], []
    for c in range(NUM_CLASSES):
        # 每類的 per-image 種子（hash 派生、無碰撞）
        seeds = [hash_seed("edm_cfg", seed, f"{w:g}", c, i) % (2 ** 31) for i in range(per_class)]
        for i in range(0, len(seeds), batch):
            bs = seeds[i:i + batch]
            rnd = StackedRandomGenerator(device, bs)
            latents = rnd.randn([len(bs), cond_net.img_channels, cond_net.img_resolution,
                                 cond_net.img_resolution], device=device)
            class_labels = torch.zeros([len(bs), cond_net.label_dim], device=device)
            class_labels[:, c] = 1
            x = edm_sampler(wrapper, latents, class_labels, randn_like=rnd.randn_like)
            imgs.append((x.float().clip(-1, 1)).cpu())   # EDM 輸出約 [-1,1]，夾限後餵量測
            labels.append(torch.full((len(bs),), c, dtype=torch.long))
    return torch.cat(imgs), torch.cat(labels)


def measure_cell(gen, gen_labels, real_dino, real_labels, test_loader, device, args, seed, w):
    """對一個生成 cell 量三判決所需：TSTR（reps 均值）、DINOv2 coverage/precision、char clean-fid。"""
    gen_dino = dinov2_features((gen + 1) / 2, device)
    prdc, _ = compute_prdc_per_class(real_dino.to(device), real_labels.to(device),
                                     gen_dino.to(device), gen_labels.to(device),
                                     nearest_k=NEAREST_K, num_classes=NUM_CLASSES)
    tstr_reps = []
    for rep in range(args.reps):
        rep_seed = hash_seed("tstr_edm_cifar10", seed, f"{w:g}", rep)
        t, _ = run_tstr(gen, gen_labels, test_loader, device, num_classes=NUM_CLASSES,
                        epochs=args.tstr_epochs, seed=rep_seed)
        tstr_reps.append(t)
    char_fid = float(clean_fid_vs_dataset((gen + 1) / 2, dataset_name="cifar10", dataset_split="train"))
    return {"name": f"w{w:g}", "guidance": w, "seed": seed,
            "tstr": sum(tstr_reps) / len(tstr_reps), "tstr_reps": tstr_reps,
            "precision": prdc["precision"], "coverage": prdc["coverage"],
            "recall": prdc["recall"], "char_clean_fid": char_fid}


def dry_run(cond_net, uncond_net, args, device):
    """枚舉 cell、驗 CFG 正確性、單 cell 計時探針→ETA。不做全量、不寫承重結果。"""
    cells = [(s, w) for s in args.seeds for w in args.grid]
    print(f"[dry-run] 枚舉 {len(cells)} cells：seeds={args.seeds} × grid={args.grid}，per_class={args.per_class}")

    # CFG 正確性：w=1 應等於純 cond；w=0 應等於純 uncond；w=3 介於外插方向。
    probe_n = 8
    seeds = [hash_seed("edm_cfg_probe", i) % (2 ** 31) for i in range(probe_n)]
    rnd = StackedRandomGenerator(device, seeds)
    latents = rnd.randn([probe_n, cond_net.img_channels, cond_net.img_resolution,
                         cond_net.img_resolution], device=device)
    labels = torch.zeros([probe_n, cond_net.label_dim], device=device); labels[:, 0] = 1
    with torch.no_grad():
        sig = torch.full([probe_n, 1, 1, 1], 1.0, device=device)
        d_cond = cond_net(latents, sig, labels)
        d_uncond = uncond_net(latents, sig, None)
        d_w1 = CFGWrapper(cond_net, uncond_net, 1.0)(latents, sig, labels)
        d_w0 = CFGWrapper(cond_net, uncond_net, 0.0)(latents, sig, labels)
        d_w3 = CFGWrapper(cond_net, uncond_net, 3.0)(latents, sig, labels)
    ok_w1 = torch.allclose(d_w1, d_cond, atol=1e-4)
    ok_w0 = torch.allclose(d_w0, d_uncond, atol=1e-4)
    ok_w3_diff = not torch.allclose(d_w3, d_cond, atol=1e-4)
    print(f"[dry-run] CFG 正確性：w=1==cond {ok_w1}；w=0==uncond {ok_w0}；w=3≠cond {ok_w3_diff}；"
          f"輸出 shape {tuple(d_w3.shape)} finite {bool(torch.isfinite(d_w3).all())}")
    assert ok_w1 and ok_w0 and ok_w3_diff, "CFG wrapper 未通過正確性檢查"

    # 計時探針：生成一小批（probe_gen 張），量單張生成秒數→外插 ETA。
    probe_gen = args.probe_gen
    t0 = time.time()
    _imgs, _labs = generate_cell(cond_net, uncond_net, 2.0, max(1, probe_gen // NUM_CLASSES),
                                 device, args.batch, seed=args.seeds[0])
    dt = time.time() - t0
    per_img = dt / max(1, _imgs.size(0))
    total_imgs = len(cells) * args.per_class * NUM_CLASSES
    gen_hours = per_img * total_imgs / 3600
    print(f"[dry-run] 計時探針：{_imgs.size(0)} 張生成 {dt:.1f}s（{per_img*1000:.1f} ms/張）")
    print(f"[dry-run] ETA（僅生成）：{total_imgs} 張 ≈ {gen_hours:.1f} GPU 小時"
          f"（不含 TSTR {args.reps} reps×{args.tstr_epochs} epochs×{len(cells)} cells 與 DINOv2/clean-fid 量測）")
    return {"cells": len(cells), "cfg_check": {"w1_eq_cond": ok_w1, "w0_eq_uncond": ok_w0,
            "w3_ne_cond": ok_w3_diff}, "probe_gen_imgs": _imgs.size(0), "probe_seconds": dt,
            "ms_per_image": per_img * 1000, "eta_gen_hours": gen_hours, "total_images": total_imgs}


def main():
    p = argparse.ArgumentParser(description="EDM 第二 backbone CFG sweep（P1-1 方案A）。預設 dry-run。")
    p.add_argument("--cond", default="checkpoints/edm-cifar10-32x32-cond-vp.pkl")
    p.add_argument("--uncond", default="checkpoints/edm-cifar10-32x32-uncond-vp.pkl")
    p.add_argument("--grid", type=float, nargs="+", default=FROZEN_GRID)
    p.add_argument("--seeds", type=int, nargs="+", default=FROZEN_SEEDS)
    p.add_argument("--per-class", type=int, default=PER_CLASS)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--reps", type=int, default=5)          # 同 CIFAR-10 confirmatory v2
    p.add_argument("--tstr-epochs", type=int, default=15)  # 同 CIFAR-10 confirmatory（消 σ_cls）
    p.add_argument("--probe-gen", type=int, default=20, help="dry-run 計時探針的生成張數")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--output", default="results/edm_cfg_sweep.json")
    p.add_argument("--dry-run", action="store_true", help="只枚舉＋CFG 檢查＋ETA（預設行為）")
    p.add_argument("--run", action="store_true", help="真跑全量（需先 commit amendment；否則拒跑）")
    p.add_argument("--quick", action="store_true", help="冒煙：grid {1,3}、seeds {10}、per_class 20")
    args = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if args.quick:
        args.grid, args.seeds, args.per_class, args.reps, args.tstr_epochs = [1.0, 3.0], [10], 20, 1, 2
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} 載入 EDM cond={args.cond} uncond={args.uncond}")
    cond_net = load_net(args.cond, device)
    uncond_net = load_net(args.uncond, device)
    print(f"cond label_dim={cond_net.label_dim} uncond label_dim={uncond_net.label_dim} "
          f"img={cond_net.img_resolution}x{cond_net.img_resolution}x{cond_net.img_channels}")

    meta = {"analysis": "edm_cfg_sweep", "backbone": "NVlabs EDM CIFAR-10 (cond+uncond, CFG at sampler)",
            "cfg_formula": "D_uncond + w*(D_cond - D_uncond)", "grid": args.grid, "seeds": args.seeds,
            "per_class": args.per_class, "reps": args.reps, "tstr_epochs": args.tstr_epochs,
            "nearest_k": NEAREST_K, "seed_formula": "sha256[:15] hash (§0.5)",
            "prereg_amendment": "docs/amendment_edm_backbone.md", "status": "exploratory",
            "start_timestamp": start_timestamp, "argv": " ".join(sys.argv),
            "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version()}}

    if not args.run:
        # 預設：dry-run（不真跑）。
        report = dry_run(cond_net, uncond_net, args, device)
        out = {"metadata": {**meta, "mode": "dry_run"}, "dry_run": report}
        with open(args.output.replace(".json", "_dryrun.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.output.replace('.json', '_dryrun.json')}（dry-run，未真跑）")
        print("真跑請先 commit docs/amendment_edm_backbone.md，再加 --run。")
        return

    # 真跑路徑（需 amendment 已 commit；本會話不執行）。
    print("[run] 真跑模式：載入真實參考、逐 cell 量測。")
    real_imgs, real_labels = load_real_per_class("cifar10", args.per_class, seed=0, data_dir=args.data_dir)
    real_dino = dinov2_features((real_imgs + 1) / 2, device)
    test_loader = build_test_loader("cifar10", batch_size=256, num_workers=0)
    cells = []
    for seed in args.seeds:
        for w in args.grid:
            g_seed = hash_seed("edm_cfg_cell", seed, f"{w:g}")
            gen, gen_labels = generate_cell(cond_net, uncond_net, w, args.per_class, device,
                                            args.batch, seed=g_seed)
            row = measure_cell(gen, gen_labels, real_dino, real_labels, test_loader, device, args, seed, w)
            cells.append(row)
            print(f"  seed {seed} w{w:g}: tstr={row['tstr']:.2f} cov={row['coverage']:.4f} "
                  f"prec={row['precision']:.4f} clean_fid={row['char_clean_fid']:.2f}", flush=True)
    out = {"metadata": {**meta, "mode": "run", "end_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
           "cells": cells}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
