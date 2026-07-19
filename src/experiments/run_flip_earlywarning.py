"""翻轉早期預警：CIFAR-10 難子集上 coverage 主導是否鬆動的方向性訊號（非 gate）。

依 Phase 1 執行計畫 2.1：在 margin 未飽和的易混類（automobile/truck、cat/dog、
deer/horse）上，跨數個 guidance 量 per-class coverage 與 per-class TSTR，看難子集上
「coverage 與效用同向」是否仍成立。此訊號單向：若鬆動（coverage 與 TSTR 在難子集
上脫鉤）是有效早期預警，指向 CIFAR-100 可能翻轉；若未鬆動，什麼都不能清除，正式的
CIFAR-100 機制 gate 仍須完整跑。

重用 run_cifar_selector 的 EDM-CFG 生成與 Inception 特徵管線（最先可得、已驗證的路徑）。
標籤噪音診斷需另訓真實 CIFAR judge，標為後續，不在此第一個結果內。

Usage:
    uv run python run_flip_earlywarning.py --quick
    uv run python run_flip_earlywarning.py --guidance 1.0 1.5 2.0 3.0 --per-class 300
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py

import argparse
import json
import os
import pickle

import torch

from run_cifar_selector import CFGNet, generate_balanced, inception_feats, set_seed
from phase1_edm_repro import load_net, DETECTOR_URL
import dnnlib
from metrics_prdc import compute_prdc_per_class
from cifar_classifier import run_tstr
from datasets.cifar import load_real_per_class, build_test_loader, NUM_CLASSES

# 易混類（margin 未飽和），CIFAR-10 索引：automobile(1) cat(3) deer(4) dog(5) horse(7) truck(9)
HARD_CLASSES = [1, 3, 4, 5, 7, 9]
CLASS_NAMES = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
               5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def _monotonic_decreasing(seq):
    """序列是否（大致）單調遞減：相鄰皆不上升則為 True。"""
    return all(seq[i + 1] <= seq[i] + 1e-9 for i in range(len(seq) - 1))


def verdict(rows):
    """依難/易子集的 coverage 與 TSTR 隨 guidance 的走向給方向性判讀。"""
    gs = [r["guidance"] for r in rows]
    hard_cov = [r["hard_coverage"] for r in rows]
    hard_tstr = [r["hard_tstr"] for r in rows]
    easy_cov = [r["easy_coverage"] for r in rows]
    easy_tstr = [r["easy_tstr"] for r in rows]

    hard_cov_dec = _monotonic_decreasing(hard_cov)
    hard_tstr_dec = _monotonic_decreasing(hard_tstr)
    easy_tstr_dec = _monotonic_decreasing(easy_tstr)

    # coverage 主導在難子集是否成立：coverage 隨 guidance 下降、TSTR 也同向下降。
    if hard_cov_dec and hard_tstr_dec:
        signal = "未鬆動：難子集上 coverage 與 TSTR 仍同向遞減（弱、單向的不翻證據）。"
    elif hard_cov_dec and not hard_tstr_dec:
        best_g = gs[max(range(len(hard_tstr)), key=lambda i: hard_tstr[i])]
        signal = (f"鬆動預警：難子集上 coverage 隨 guidance 遞減，但 TSTR 在 guidance={best_g:g} "
                  f"出現內部最優、非最低 guidance 最佳——coverage 與效用在難子集脫鉤，指向 "
                  f"CIFAR-100 可能翻轉。")
    else:
        signal = "難子集 coverage 未單調遞減，訊號不明確；需更多 guidance 點或更大樣本再判。"

    contrast = ("難子集 TSTR 走向與易子集不同" if hard_tstr_dec != easy_tstr_dec
                else "難子集與易子集 TSTR 走向一致")
    return {"signal": signal, "hard_vs_easy": contrast,
            "hard_coverage_decreasing": hard_cov_dec,
            "hard_tstr_decreasing": hard_tstr_dec,
            "easy_tstr_decreasing": easy_tstr_dec}


def main():
    p = argparse.ArgumentParser(description="CIFAR-10 hard-subset flip early-warning (directional, not a gate).")
    p.add_argument("--cond", default="checkpoints/edm-cifar10-32x32-cond-vp.pkl")
    p.add_argument("--uncond", default="checkpoints/edm-cifar10-32x32-uncond-vp.pkl")
    p.add_argument("--guidance", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    p.add_argument("--per-class", type=int, default=300)
    p.add_argument("--real-per-class", type=int, default=500)
    p.add_argument("--gen-steps", type=int, default=18)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--nearest-k", type=int, default=5)
    p.add_argument("--tstr-epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="results/flip_earlywarning.json")
    p.add_argument("--quick", action="store_true", help="小樣本快速煙霧測試，驗證管線")
    args = p.parse_args()

    if args.quick:
        args.guidance = [1.0, 3.0]
        args.per_class = 32
        args.real_per_class = 100
        args.tstr_epochs = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(f"guidance={args.guidance} per_class={args.per_class} gen_steps={args.gen_steps} "
          f"tstr_epochs={args.tstr_epochs}", flush=True)
    os.makedirs("results", exist_ok=True)
    set_seed(args.seed)

    cond = load_net(args.cond, device)
    uncond = load_net(args.uncond, device)
    with dnnlib.util.open_url(DETECTOR_URL, verbose=False) as f:
        detector = pickle.load(f).to(device)
    print("Loaded cond + uncond EDM nets and Inception detector", flush=True)

    real_imgs, real_labels = load_real_per_class("cifar10", args.real_per_class, seed=0)
    real_feats = inception_feats(real_imgs, detector, device, batch=args.batch)
    test_loader = build_test_loader("cifar10", batch_size=256)
    print(f"Real ref: {real_feats.shape[0]} imgs, feat dim {real_feats.shape[1]}", flush=True)

    easy_classes = [c for c in range(NUM_CLASSES["cifar10"]) if c not in HARD_CLASSES]
    rows = []
    for w in args.guidance:
        net = CFGNet(cond, uncond, w)
        seed_base = args.seed * 10_000_000 + int(w * 1000) * 10_000
        gen_imgs, gen_labels = generate_balanced(net, args.per_class, device, args.batch,
                                                 seed_base, num_classes=10, num_steps=args.gen_steps)
        gen_feats = inception_feats(gen_imgs, detector, device, batch=args.batch)
        _, per_class = compute_prdc_per_class(real_feats.to(device), real_labels.to(device),
                                              gen_feats.to(device), gen_labels.to(device),
                                              nearest_k=args.nearest_k, num_classes=10)
        cov_by_class = {pc["class"]: pc["coverage"] for pc in per_class}
        _, tstr_by_class = run_tstr(gen_imgs, gen_labels, test_loader, device, num_classes=10,
                                    epochs=args.tstr_epochs)

        hard_cov = _mean([cov_by_class.get(c) for c in HARD_CLASSES])
        hard_tstr = _mean([tstr_by_class.get(c) for c in HARD_CLASSES])
        easy_cov = _mean([cov_by_class.get(c) for c in easy_classes])
        easy_tstr = _mean([tstr_by_class.get(c) for c in easy_classes])
        rows.append({"guidance": w, "hard_coverage": hard_cov, "hard_tstr": hard_tstr,
                     "easy_coverage": easy_cov, "easy_tstr": easy_tstr,
                     "coverage_by_class": cov_by_class, "tstr_by_class": tstr_by_class})
        print(f"  w={w:<4g}  hard: cov={hard_cov:.3f} tstr={hard_tstr:.2f}  "
              f"easy: cov={easy_cov:.3f} tstr={easy_tstr:.2f}", flush=True)

    v = verdict(rows)
    print("\n" + "=" * 78)
    print("  翻轉早期預警（難子集 = automobile/cat/deer/dog/horse/truck）")
    print("=" * 78)
    print(f"  {'guidance':>9} {'hard_cov':>9} {'hard_TSTR':>10} {'easy_cov':>9} {'easy_TSTR':>10}")
    for r in rows:
        print(f"  {r['guidance']:>9g} {r['hard_coverage']:>9.3f} {r['hard_tstr']:>10.2f} "
              f"{r['easy_coverage']:>9.3f} {r['easy_tstr']:>10.2f}")
    print("  " + "-" * 74)
    print(f"  判讀：{v['signal']}")
    print(f"  對照：{v['hard_vs_easy']}")
    print("=" * 78)

    out = {"metadata": {"dataset": "cifar10", "axis": "CFG guidance (cond+uncond EDM proxy)",
                        "hard_classes": HARD_CLASSES, "guidance_grid": args.guidance,
                        "per_class": args.per_class, "gen_steps": args.gen_steps,
                        "tstr_epochs": args.tstr_epochs, "note": "directional warning, not a gate; "
                        "label-noise diagnosis (needs real-CIFAR judge) deferred"},
           "rows": rows, "verdict": v, "args": {k: v2 for k, v2 in vars(args).items()}}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
