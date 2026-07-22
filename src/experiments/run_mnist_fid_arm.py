"""T1a：MNIST 真 FID-min 臂（對應審查 A1(a)）。

問題設定（給第一次讀的研究生）：論文表 5.3／圖 5.2 的 MNIST「保真代理」欄，過去用 precision-argmax
的 regret 代打，從未實測 FID-min——這與 CIFAR 兩尺度（實測 clean-fid 的 FID-min）不同構，構成缺口。
本檔補上 MNIST 每個 guidance 組態的**實測 FID**，讓三個資料集的 FID-min 臂同構：
  - MNIST-FID（classifier-Fréchet，`fid.py`）：judge CNN 256 維特徵空間的 Fréchet distance；本專案自製，
    僅供 MNIST sandbox 內相對排名。
  - clean-fid（Inception 空間，`fid_clean.py`）：標準 Inception-FID，作第二讀數，兩空間並列，避免
    「自製 FID 恰好有利」的質疑。

對每 (seed, guidance) 生成 per_digit*10 張影像（與 `run_selector_signal.py` 同路徑：DDIM、steps=50、
eta=0），影像快取於 `results/mnist_gen_cache/`（供 T1b DINOv2 堆疊共用同一批影像）。每個 config 以
決定性子種子 `seed*1000+int(g*10)` 生成，故各 config 可獨立重現、快取自洽。

量的東西（逐 seed，兩 FID 空間各一）：
  - oracle        = argmax_g TSTR（取自凍結 `selector_signal_multiseed.json` 的 per-seed TSTR）
  - fidmin        = argmin_g FID（MNIST-FID 或 clean-fid）
  - fidmin_regret = TSTR(oracle) - TSTR(fidmin)
  - sep_step      = FID-argmin 與 TSTR-argmax 在 MNIST grid 上相隔幾格

驗收關鍵：若 FID-argmin 落在 oracle（g1），代表「FID-min 在 MNIST 選錯」的舊敘事須改寫（T7）；若確實
選錯，把表 5.3 括註由「precision-argmax 代打」換成實測值。兩種結果都如實落地。

Usage:
    uv run python src/experiments/run_mnist_fid_arm.py               # seeds 0 1 2, per_digit 1000
    uv run python src/experiments/run_mnist_fid_arm.py --quick       # 冒煙：小規模
    uv run python src/experiments/run_mnist_fid_arm.py --no-clean-fid # 只算 MNIST-FID（省時）
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py

import argparse
import json
import os
import sys
import time

import torch

from ddpm import UNet, DiffusionSchedule
from inference import generate as gen_batches
from fid import load_cnn, compute_fid, real_feature_stats
from analyze_distribution import extract_features, load_real_per_class  # noqa: F401  extract_features 供 real_feature_stats 內部

GRID = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]   # 對齊 run_selector_signal FULL_GUIDANCE
SEEDS = [0, 1, 2]
SAMPLER, STEPS, ETA = "ddim", 50, 0.0
CACHE_DIR = "results/mnist_gen_cache"


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cfg_subseed(seed, g):
    """每個 (seed,g) 的決定性子種子，使各 config 可獨立重現、快取自洽。"""
    return seed * 1000 + int(round(g * 10))


def generate_config(model, schedule, guidance, per_digit, batch_size, device):
    imgs, labels = [], []
    for digit in range(10):
        for batch in gen_batches(model, schedule, digit, per_digit, batch_size,
                                 guidance, device, sampler=SAMPLER, steps=STEPS, eta=ETA):
            imgs.append(batch.cpu())
            labels.append(torch.full((batch.size(0),), digit, dtype=torch.long))
    return torch.cat(imgs), torch.cat(labels)


def load_or_generate(seed, g, per_digit, batch_size, device, model, schedule, cache_dir):
    """回傳 (imgs[-1,1] float32 (N,1,28,28), labels)；(seed,g) 快取於磁碟，缺則生成並快取。"""
    d = os.path.join(cache_dir, f"seed{seed}_g{g:g}")
    img_p, lab_p = os.path.join(d, "img.pt"), os.path.join(d, "labels.pt")
    if os.path.exists(img_p) and os.path.exists(lab_p):
        return torch.load(img_p), torch.load(lab_p)
    set_seed(cfg_subseed(seed, g))
    imgs, labels = generate_config(model, schedule, g, per_digit, batch_size, device)
    os.makedirs(d, exist_ok=True)
    torch.save(imgs, img_p)
    torch.save(labels, lab_p)
    return imgs, labels


def grid_index(name):
    return [f"g{g:g}" for g in GRID].index(name)


def analyse_seed(seed, per_seed_tstr, per_digit, batch_size, device, model, schedule,
                 judge, args):
    """單一 seed：對每 config 算 MNIST-FID（＋選配 clean-fid），對照凍結 TSTR 算 regret。"""
    real_img, _ = load_real_per_class(args.data_dir, args.fid_per_class, seed)
    real_stats = real_feature_stats(real_img, device=device, model=judge)
    real01 = ((real_img + 1) / 2).repeat(1, 3, 1, 1) if args.clean_fid else None

    configs = []
    for g in GRID:
        name = f"g{g:g}"
        gi, _gl = load_or_generate(seed, g, per_digit, batch_size, device, model, schedule, CACHE_DIR)
        mnist_fid = compute_fid(gi, device=device, model=judge, real_stats=real_stats)
        cfg = {"name": name, "guidance": g, "mnist_fid": round(mnist_fid, 4),
               "tstr": per_seed_tstr.get(name)}
        if args.clean_fid:
            from fid_clean import clean_fid_two_sets
            cfg["clean_fid"] = round(clean_fid_two_sets(((gi + 1) / 2).repeat(1, 3, 1, 1), real01), 4)
        configs.append(cfg)
        print(f"  seed {seed} {name:>4}: MNIST-FID={cfg['mnist_fid']:.3f}"
              f"{' clean-fid=' + format(cfg['clean_fid'], '.3f') if args.clean_fid else ''}"
              f" tstr={cfg['tstr']}")

    have_tstr = all(c["tstr"] is not None for c in configs)
    out = {"seed": seed, "configs": configs}
    if have_tstr:
        oracle = max(configs, key=lambda c: c["tstr"])
        out["oracle"] = oracle["name"]
        out["oracle_tstr"] = round(oracle["tstr"], 2)
        for space, key in [("mnist", "mnist_fid"), ("clean", "clean_fid")]:
            if not all(key in c for c in configs):
                continue
            fidmin = min(configs, key=lambda c: c[key])
            out[f"fidmin_{space}"] = fidmin["name"]
            out[f"fidmin_{space}_regret"] = round(oracle["tstr"] - fidmin["tstr"], 2)
            out[f"sep_step_{space}"] = abs(grid_index(fidmin["name"]) - grid_index(oracle["name"]))
    return out


def main():
    p = argparse.ArgumentParser(description="MNIST 真 FID-min 臂（A1a）。")
    p.add_argument("--checkpoint", default="ddpm_mnist.pt")
    p.add_argument("--cnn", default="mnist_cnn.pt")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--per-digit", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--fid-per-class", type=int, default=1000)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--selector-signal", default="results/selector_signal_multiseed.json")
    p.add_argument("--clean-fid", action=argparse.BooleanOptionalAction, default=True,
                   help="同時算 Inception 空間 clean-fid 第二讀數（預設開；--no-clean-fid 省時）")
    p.add_argument("--output", default="results/mnist_fid_arm.json")
    p.add_argument("--quick", action="store_true", help="冒煙：per_digit 60、seeds 0")
    args = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if args.quick:
        args.per_digit, args.seeds, args.fid_per_class = 60, [0], 200
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} seeds={args.seeds} grid={GRID} per_digit={args.per_digit} clean_fid={args.clean_fid}")

    model = UNet(in_channels=1, base_channels=64, num_classes=10).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
    model.eval()
    schedule = DiffusionSchedule(timesteps=1000, device=device).to(device)
    judge = load_cnn(args.cnn, device)

    # 凍結 TSTR（per-seed、per-config）供 regret
    with open(args.selector_signal, encoding="utf-8") as f:
        sig = json.load(f)
    tstr_by_seed = {b["seed"]: {c["name"]: c["tstr"] for c in b["configs"]} for b in sig["per_seed"]}

    per_seed = []
    for seed in args.seeds:
        per_seed.append(analyse_seed(seed, tstr_by_seed.get(seed, {}), args.per_digit,
                                     args.batch_size, device, model, schedule, judge, args))

    # 彙總：兩空間的 per-seed argmin/regret
    agg = {}
    for space in ("mnist", "clean"):
        regrets = [s.get(f"fidmin_{space}_regret") for s in per_seed if f"fidmin_{space}_regret" in s]
        seps = [s.get(f"sep_step_{space}") for s in per_seed if f"sep_step_{space}" in s]
        if regrets:
            agg[space] = {
                "fidmin_regret_mean": round(sum(regrets) / len(regrets), 2),
                "fidmin_regret_per_seed": regrets,
                "sep_steps": seps,
                "separated_seeds_gt1": sum(1 for s in seps if s > 1),
                "fidmin_per_seed": [s.get(f"fidmin_{space}") for s in per_seed],
                "oracle_per_seed": [s.get("oracle") for s in per_seed],
            }

    out = {"per_seed": per_seed, "aggregate": agg}

    print("\n" + "=" * 78)
    print("  T1a MNIST 真 FID-min 臂（regret@selected，pp，越低越好）")
    print("=" * 78)
    for space in ("mnist", "clean"):
        if space in agg:
            a = agg[space]
            print(f"  [{space}-FID] argmin/seed={a['fidmin_per_seed']} oracle/seed={a['oracle_per_seed']}")
            print(f"            regret 平均={a['fidmin_regret_mean']} per_seed={a['fidmin_regret_per_seed']}"
                  f"  分離(>1格) {a['separated_seeds_gt1']}/{len(a['sep_steps'])}")
    print("=" * 78)

    out["metadata"] = {
        "analysis": "mnist_fid_arm", "status": "derived",
        "sampler": SAMPLER, "steps": STEPS, "eta": ETA, "grid": GRID, "seeds": args.seeds,
        "per_digit": args.per_digit, "fid_per_class": args.fid_per_class,
        "mnist_fid_space": "mnist_cnn.pt penultimate 256-d (classifier-Fréchet)",
        "clean_fid_space": "Inception (cleanfid)" if args.clean_fid else None,
        "tstr_source": args.selector_signal, "gen_cache": CACHE_DIR,
        "gen_subseed_formula": "seed*1000+int(g*10)",
        "start_timestamp": start_timestamp, "argv": " ".join(sys.argv),
        "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version()},
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
