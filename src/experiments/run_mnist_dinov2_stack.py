"""T1b：MNIST DINOv2 堆疊版 CaF（對應審查 A1(b)）。

問題設定（給第一次讀的研究生）：MNIST 的 CaF 選擇器過去用 judge-CNN（任務對齊的 256 維）特徵算 PRDC，
CIFAR 則用 DINOv2（任務無關的自監督特徵）。這使「MNIST 選對、CIFAR 選錯」的反轉，同時混入了「資料集」
與「特徵空間」兩個變因。本檔把 MNIST 的選擇器搬到與 CIFAR 相同的 DINOv2 堆疊，拆開這兩者：對 T1a 的
同一批生成影像（`results/mnist_gen_cache/`）以 DINOv2 特徵重算 PRDC 與 CaF，回答「MNIST 換成 DINOv2
特徵後，CaF 是否仍選中 oracle？」

流程（逐 seed）：
  1. real probe（1000/class）與各 config 生成影像皆 `(x+1)/2` 後餵 `metrics_features.dinov2_features`
     （灰階自動擴 3 通道），得 768 維 DINOv2 特徵。
  2. `compute_prdc_per_class`（k=5、num_classes=10）算 per-class precision/coverage/recall。
  3. real_ref_precision 用對半切法（perm 以 seed 決定、B 為 manifold、A 為 fake），沿
     `run_cifar_cfg_multiseed.py:191-199` 與 `run_selector_signal.py:120-127` 的慣例。
  4. 以 signal_key="coverage"（CaF）與 "recall"（CaF-v2）各跑一次 `select_and_report`。
  TSTR（oracle utility）取自凍結 `selector_signal_multiseed.json` 的 per-seed 值。

驗收：能回答「MNIST 上換成 DINOv2 特徵後，CaF 是否仍選中 oracle？」並把答案寫進 §6.1 的反轉歸因段。

Usage:
    uv run python src/experiments/run_mnist_dinov2_stack.py           # seeds 0 1 2
    uv run python src/experiments/run_mnist_dinov2_stack.py --quick   # 冒煙
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py

import argparse
import json
import sys
import time

import torch

from ddpm import UNet, DiffusionSchedule
from fid import load_cnn  # noqa: F401  保留：與 T1a 同一 judge，不在此用
from analyze_distribution import load_real_per_class
from metrics_features import dinov2_features
from metrics_prdc import compute_prdc_per_class
from selector import select_and_report
# 共用 T1a 的生成／快取，確保兩臂用同一批影像
from run_mnist_fid_arm import GRID, SEEDS, load_or_generate


def dino(images, device, batch_size):
    """(N,1,28,28) [-1,1] -> (x+1)/2 -> DINOv2 768 維（灰階自動擴 3 通道）。"""
    return dinov2_features((images + 1) / 2, device=device, batch_size=batch_size)


def real_ref_precision(real_feats, real_labels, seed, nearest_k, num_classes, device):
    """對半切法的 real-vs-real 參考 precision（不依賴 TSTR 的 tau 之基礎）。"""
    n = real_feats.size(0)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    half = n // 2
    a_idx, b_idx = perm[:half], perm[half:]
    ref, _ = compute_prdc_per_class(real_feats[b_idx].to(device), real_labels[b_idx].to(device),
                                    real_feats[a_idx].to(device), real_labels[a_idx].to(device),
                                    nearest_k=nearest_k, num_classes=num_classes)
    return ref["precision"]


def analyse_seed(seed, per_seed_tstr, args, device, model, schedule):
    real_img, real_lab = load_real_per_class(args.data_dir, args.fid_per_class, seed)
    real_feat = dino(real_img, device, args.batch_size).cpu()
    ref_prec = real_ref_precision(real_feat, real_lab, seed, args.nearest_k, 10, device)

    configs = []
    for g in GRID:
        name = f"g{g:g}"
        gi, gl = load_or_generate(seed, g, args.per_digit, args.batch_size, device, model, schedule,
                                  "results/mnist_gen_cache")
        gd = dino(gi, device, args.batch_size).cpu()
        prdc, _ = compute_prdc_per_class(real_feat.to(device), real_lab.to(device),
                                         gd.to(device), gl.to(device),
                                         nearest_k=args.nearest_k, num_classes=10)
        configs.append({"name": name, "guidance": g,
                        "precision": prdc["precision"], "coverage": prdc["coverage"],
                        "recall": prdc["recall"], "density": prdc["density"],
                        "tstr": per_seed_tstr.get(name)})
        print(f"  seed {seed} {name:>4}: prec={prdc['precision']:.4f} cov={prdc['coverage']:.4f}"
              f" recall={prdc['recall']:.4f} tstr={configs[-1]['tstr']}")

    have_tstr = all(c["tstr"] is not None for c in configs)
    util = "tstr" if have_tstr else "__none__"
    out = {"seed": seed, "real_ref_precision": ref_prec, "configs": configs, "reports": {}}
    for tag, signal in [("caf", "coverage"), ("caf_v2", "recall")]:
        rep = select_and_report(configs, real_ref_precision=ref_prec,
                                tau_fraction=args.tau_fraction, utility_key=util, signal_key=signal)
        out["reports"][tag] = rep
        print(f"    [{tag}/{signal}] selected {rep['selected']} (oracle {rep['oracle_best']}, "
              f"regret {rep['regret_at_selected']}, tau {rep['tau']:.4f})")
    return out


def summarize_regret(per_seed, tag):
    regs = [s["reports"][tag]["regret_at_selected"] for s in per_seed
            if s["reports"][tag]["regret_at_selected"] is not None]
    sels = [s["reports"][tag]["selected"] for s in per_seed]
    if not regs:
        return None
    return {"selected_per_seed": sels,
            "regret_mean": round(sum(regs) / len(regs), 2), "regret_per_seed": [round(r, 2) for r in regs],
            "hit_oracle_seeds": sum(1 for s in per_seed if s["reports"][tag]["regret_at_selected"] == 0),
            "tau_stability_mean": round(sum(s["reports"][tag]["tau_robustness"]["stability"]
                                            for s in per_seed) / len(per_seed), 4)}


def main():
    p = argparse.ArgumentParser(description="MNIST DINOv2 堆疊版 CaF（A1b）。")
    p.add_argument("--checkpoint", default="ddpm_mnist.pt")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--per-digit", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--fid-per-class", type=int, default=1000)
    p.add_argument("--nearest-k", type=int, default=5)
    p.add_argument("--tau-fraction", type=float, default=0.9)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--selector-signal", default="results/selector_signal_multiseed.json")
    p.add_argument("--output", default="results/mnist_dinov2_stack.json")
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
    print(f"device={device} seeds={args.seeds} grid={GRID} per_digit={args.per_digit}")

    model = UNet(in_channels=1, base_channels=64, num_classes=10).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
    model.eval()
    schedule = DiffusionSchedule(timesteps=1000, device=device).to(device)

    with open(args.selector_signal, encoding="utf-8") as f:
        sig = json.load(f)
    tstr_by_seed = {b["seed"]: {c["name"]: c["tstr"] for c in b["configs"]} for b in sig["per_seed"]}

    per_seed = [analyse_seed(seed, tstr_by_seed.get(seed, {}), args, device, model, schedule)
                for seed in args.seeds]
    aggregate = {tag: summarize_regret(per_seed, tag) for tag in ("caf", "caf_v2")}

    print("\n" + "=" * 78)
    print("  T1b MNIST DINOv2 堆疊版 CaF（regret@selected，pp，越低越好）")
    print("=" * 78)
    for tag in ("caf", "caf_v2"):
        a = aggregate[tag]
        if a:
            print(f"  [{tag}] selected/seed={a['selected_per_seed']} regret 平均={a['regret_mean']}"
                  f" per_seed={a['regret_per_seed']} 中 oracle {a['hit_oracle_seeds']}/{len(args.seeds)}")
    print("=" * 78)

    out = {"per_seed": per_seed, "aggregate": aggregate,
           "metadata": {
               "analysis": "mnist_dinov2_stack", "status": "derived",
               "feature_space": "dinov2_vitb14 768-d (CLS)", "nearest_k": args.nearest_k,
               "num_classes": 10, "tau_fraction": args.tau_fraction, "grid": GRID, "seeds": args.seeds,
               "per_digit": args.per_digit, "fid_per_class": args.fid_per_class,
               "tstr_source": args.selector_signal, "gen_cache": "results/mnist_gen_cache",
               "start_timestamp": start_timestamp, "argv": " ".join(sys.argv),
               "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                       "cudnn": torch.backends.cudnn.version()}}}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
