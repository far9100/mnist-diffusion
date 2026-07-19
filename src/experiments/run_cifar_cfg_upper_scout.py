"""步驟 3：上緣 coverage-only scout——定位 coverage 崩點界（不訓 TSTR、不跑 judge）。

依 2026-07-05-08 協定增修規格 5：用便宜的 coverage-only 掃描（1 seed、少樣本、只量 DINOv2
coverage），從 w=8 往上掃事前定的 {8,10,12,16,20}，以事前定死的觸底判準 X=0.02（相鄰兩點
|Δcoverage| < X 即視為觸底/回穩）定崩點界。若掃到事前上限 w=20 仍未觸底，走「新登記」分支
（不臨場續掃）。coverage 是幾何量、非主結果指標，用它定 grid 掃描範圍不算對主結果 HARKing；
本 scout 只看 coverage，grid 一定死即不再依任何後續結果調整。本 scout 資料不得進入 confirmatory
統計（其 seed 與 confirmatory 的 fresh seeds 分離）。

Usage:
    uv run python run_cifar_cfg_upper_scout.py --quick
    uv run python run_cifar_cfg_upper_scout.py --guidance 8 10 12 16 20 --per-class 300
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py

import argparse
import json
import os

import torch

from cifar_cfg_sample import load_cfg_model, generate_balanced
from metrics_prdc import compute_prdc_per_class
from metrics_features import dinov2_features
from datasets.cifar import load_real_per_class


def find_bottom(rows, x):
    """回傳第一個相鄰 |Δcoverage| < x 的 w（觸底界）；未觸底回傳 None。"""
    for i in range(1, len(rows)):
        if abs(rows[i]["coverage"] - rows[i - 1]["coverage"]) < x:
            return rows[i]["guidance"]
    return None


def main():
    p = argparse.ArgumentParser(description="Upper-edge coverage-only scout to locate coverage collapse.")
    p.add_argument("--ckpt", default="checkpoints/cifar10_cfg.pt")
    p.add_argument("--guidance", type=float, nargs="+", default=[8.0, 10.0, 12.0, 16.0, 20.0])
    p.add_argument("--per-class", type=int, default=300)
    p.add_argument("--real-per-class", type=int, default=500)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--batch", type=int, default=250)
    p.add_argument("--nearest-k", type=int, default=5)
    p.add_argument("--bottom-x", type=float, default=0.02, help="事前定死的觸底判準（相鄰 |Δcoverage| < X）")
    p.add_argument("--seed", type=int, default=0, help="任意 seed；本 scout 資料不入 confirmatory 統計")
    p.add_argument("--output", default="results/cifar10_cfg_upper_scout.json")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.guidance = [8.0, 20.0]
        args.per_class = 32
        args.real_per_class = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(f"guidance={args.guidance} per_class={args.per_class} bottom_x={args.bottom_x} "
          f"steps={args.steps} eta={args.eta}", flush=True)
    os.makedirs("results", exist_ok=True)

    model, schedule, hp = load_cfg_model(args.ckpt, device)
    real_imgs, real_labels = load_real_per_class("cifar10", args.real_per_class, seed=0)
    real_dino = dinov2_features((real_imgs + 1) / 2, device)
    print(f"Loaded model + real ref {real_imgs.size(0)} imgs", flush=True)

    rows = []
    for w in args.guidance:
        gen, gen_labels = generate_balanced(model, schedule, args.per_class, device, args.steps,
                                            args.eta, guidance=w, num_classes=10, batch=args.batch,
                                            seed=args.seed)
        gen_dino = dinov2_features((gen + 1) / 2, device)
        prdc, _ = compute_prdc_per_class(real_dino.to(device), real_labels.to(device),
                                         gen_dino.to(device), gen_labels.to(device),
                                         nearest_k=args.nearest_k, num_classes=10)
        rows.append({"guidance": w, "coverage": prdc["coverage"]})
        print(f"  w={w:<4g} coverage={prdc['coverage']:.4f}", flush=True)
        del gen, gen_dino
        if device.type == "cuda":
            torch.cuda.empty_cache()

    boundary = find_bottom(rows, args.bottom_x)
    bottomed = boundary is not None
    deltas = [abs(rows[i]["coverage"] - rows[i - 1]["coverage"]) for i in range(1, len(rows))]

    print("\n" + "=" * 60)
    print("  上緣 coverage-only scout（DINOv2）")
    print("=" * 60)
    for i, r in enumerate(rows):
        d = "" if i == 0 else f"  |Δ|={deltas[i-1]:.4f}"
        print(f"  w={r['guidance']:<4g} coverage={r['coverage']:.4f}{d}")
    print("  " + "-" * 56)
    if bottomed:
        upper = boundary + 1.0
        print(f"  觸底界（首個 |Δcov|<{args.bottom_x}）= w{boundary:g}；grid 上緣 = 觸底界+1 = w{upper:g}")
    else:
        print(f"  到事前上限 w={args.guidance[-1]:g} 仍未觸底（所有 |Δcov| >= {args.bottom_x}）")
        print("  -> 走新登記分支：新增 records+commit 提高 w 上限後再掃，不臨場續掃。")
    print("=" * 60)

    out = {"metadata": {"axis": "CFG guidance (self-trained)", "coverage_feature": "DINOv2",
                        "steps": args.steps, "eta": args.eta, "per_class": args.per_class,
                        "bottom_x": args.bottom_x, "w_max": args.guidance[-1], "seed": args.seed,
                        "note": "coverage-only upper scout; data NOT used in confirmatory statistics"},
           "rows": rows, "deltas": deltas, "bottomed": bottomed,
           "collapse_boundary_w": boundary,
           "grid_upper_w": (boundary + 1.0) if bottomed else None}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
