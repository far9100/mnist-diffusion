"""T10：CIFAR-100 margin-pruning 介入（機制鏈中介量之直接介入，對應審查 B3）。

問題設定（給第一次讀的研究生）：本文機制敘事＝低 guidance 保留更多 near-boundary（低 judge margin）
樣本，這些樣本供給下游決策邊界資訊、承載 TSTR 效用。D3 的 C3 介入是「coverage-matched pruning」
（移除離真實流形最近者以降 coverage），但那是 coverage 的介入、非 near-boundary 供給的**直接**介入。
本檔直接介入中介量：移除 **judge margin 最低**（最近邊界）的 n 個樣本，對照等計數**隨機**移除，
看 TSTR 掉幅是否顯著更大。若是，支持 near-boundary 供給承載效用（介入式）。

設定（沿 D3 骨架、seed10 w2.5 資產，judge_out.pt 已含 margins）：
  - n＝13606（與 C3 同計數，便於對照）＋ n/2＝6803 兩檔位。
  - 每檔位：margin-pruned 與 random-pruned 各 N=8 次 from-scratch 重訓（T6b 決定性種子）。
  - 每類保底 ≥6（避免移空難類）。輸出差／SE／t／CI95／MDE（同 n8 檔口徑：MDE≈2.8×SE）。

凍結不動、新結果寫新檔 results/cifar100_margin_intervention.json。GPU 中、exploratory、禁因果。

Usage:
    uv run python src/experiments/run_cifar100_margin_intervention.py                 # 需 GPU
    uv run python src/experiments/run_cifar100_margin_intervention.py --n-retrain 8 --levels 13606 6803
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py
import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time

import torch

from cifar_classifier import run_tstr
from datasets.cifar import build_test_loader

ASSET = "results/p1_assets_cifar100"
CONF = "results/cifar100_cfg_confirmatory.json"
SEED = 10
NUM_CLASSES = 100
TSTR_EPOCHS = 15
PER_CLASS_FLOOR = 6          # 每類保底（≥k+1），避免剪空難類
RAND_SEED = 20260721


def load_w25():
    """seed10 w2.5：影像 [-1,1]、labels、judge margins。"""
    u8 = torch.load(f"{ASSET}/seed{SEED}_w2.5/img_uint8.pt", map_location="cpu", weights_only=True)
    labels = torch.load(f"{ASSET}/seed{SEED}_w2.5/labels.pt", map_location="cpu", weights_only=True)
    judge = torch.load(f"{ASSET}/seed{SEED}_w2.5/judge_out.pt", map_location="cpu", weights_only=True)
    return u8.float() / 255.0 * 2.0 - 1.0, labels, judge["margins"]


def prune_lowest_margin(margins, labels, n, floor=PER_CLASS_FLOOR):
    """移除 margin 最低的至多 n 個，但每類保底 floor。回 keep 遮罩與實際移除數。"""
    order = torch.argsort(margins)                       # 升冪：最低 margin（最近邊界）在前
    kept_per_class = torch.bincount(labels, minlength=NUM_CLASSES).clone()
    remove = torch.zeros(margins.size(0), dtype=torch.bool)
    removed = 0
    for idx in order.tolist():
        if removed >= n:
            break
        c = int(labels[idx])
        if kept_per_class[c] <= floor:
            continue
        remove[idx] = True
        kept_per_class[c] -= 1
        removed += 1
    return ~remove, removed


def prune_random(size, k, seed):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(size, generator=g)
    keep = torch.ones(size, dtype=torch.bool)
    keep[perm[:k]] = False
    return keep


def _seed(tag, rep):
    return int(hashlib.sha256(f"margin10_{tag}_{rep}".encode()).hexdigest()[:15], 16)


def retrain(imgs, labels, test_loader, device, n, tag):
    vals = []
    for r in range(n):
        ov, _ = run_tstr(imgs, labels, test_loader, device, num_classes=NUM_CLASSES,
                         epochs=TSTR_EPOCHS, seed=_seed(tag, r))
        vals.append(round(ov, 2))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return vals


def stats2(margin_vals, rand_vals):
    """差＝mean(rand)−mean(margin)（正＝margin-pruning 掉更多、支持 near-boundary 承載效用）。"""
    mm, mr = statistics.mean(margin_vals), statistics.mean(rand_vals)
    vm = statistics.variance(margin_vals) if len(margin_vals) > 1 else 0.0
    vr = statistics.variance(rand_vals) if len(rand_vals) > 1 else 0.0
    diff = mr - mm
    se = (vm / len(margin_vals) + vr / len(rand_vals)) ** 0.5
    t = diff / se if se > 0 else 0.0
    mde = 2.8 * se                                       # 同 n8 口徑（(1.96+0.84)×SE，80% power）
    return {"mean_margin_pruned": round(mm, 2), "mean_rand_pruned": round(mr, 2),
            "diff_rand_minus_margin": round(diff, 2), "se": round(se, 3), "t": round(t, 3),
            "ci95": [round(diff - 1.96 * se, 2), round(diff + 1.96 * se, 2)], "mde": round(mde, 3)}


def main():
    p = argparse.ArgumentParser(description="CIFAR-100 margin-pruning 介入（近邊界供給之直接介入；T10/B3）。")
    p.add_argument("--n-retrain", type=int, default=8, help="每情境重訓次數（沿 n8 慣例）")
    p.add_argument("--levels", type=int, nargs="+", default=[13606, 6803], help="移除數檔位（n、n/2）")
    p.add_argument("--output", default="results/cifar100_margin_intervention.json")
    args = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imgs, labels, margins = load_w25()
    d = json.load(open(CONF, encoding="utf-8"))
    sb = next(s for s in d["per_seed"] if s["seed"] == SEED)
    w25_tstr = next(c for c in sb["configs"] if c["name"] == "w2.5")["tstr"]
    test_loader = build_test_loader("cifar100")
    print(f"device={device} seed={SEED} w2.5 samples={imgs.size(0)} frozen w2.5 TSTR={w25_tstr} "
          f"margin[min={margins.min():.4f} p20={margins.float().quantile(0.2):.4f}]", flush=True)

    levels = {}
    for n in args.levels:
        keep_m, removed = prune_lowest_margin(margins, labels, n)
        keep_r = prune_random(imgs.size(0), removed, RAND_SEED + n)
        print(f"\n[n={n}] 實移除 {removed}（每類保底 {PER_CLASS_FLOOR}）；margin-pruned 與 random 各 "
              f"{args.n_retrain} 重訓 ...", flush=True)
        tstr_margin = retrain(imgs[keep_m], labels[keep_m], test_loader, device, args.n_retrain, f"marg_{n}")
        tstr_rand = retrain(imgs[keep_r], labels[keep_r], test_loader, device, args.n_retrain, f"rand_{n}")
        st = stats2(tstr_margin, tstr_rand)
        levels[str(n)] = {"n_requested": n, "n_removed": removed,
                          "tstr_margin_pruned": tstr_margin, "tstr_rand_pruned": tstr_rand, **st}
        print(f"  → diff(rand−margin)={st['diff_rand_minus_margin']:+}pp SE={st['se']} t={st['t']} "
              f"CI95={st['ci95']} MDE={st['mde']}", flush=True)

    out = {
        "analysis": "cifar100_margin_intervention", "seed": SEED, "config": "w2.5",
        "w2.5_frozen_tstr": w25_tstr, "levels": levels,
        "interpretation": ("差為正且 > MDE：移除近邊界（低 margin）樣本比隨機掉更多 TSTR，支持 near-boundary "
                           "供給承載效用（介入式）；差 ≈0 且有功效（|diff|<MDE）：有功效之 null，未支持。"),
        "metadata": {
            "asset": ASSET, "source": CONF, "num_classes": NUM_CLASSES, "n_retrain": args.n_retrain,
            "tstr_epochs": TSTR_EPOCHS, "per_class_floor": PER_CLASS_FLOOR, "rand_seed": RAND_SEED,
            "tstr_seeded": "sha256 決定性（T6b）", "mde_convention": "2.8×SE（(1.96+0.84)、80% power）",
            "note": "near-boundary 供給之直接介入（中介量），對照 D3 之 coverage-matched 介入；"
                    "exploratory、禁因果。§5.5 併述兩介入。",
            "start_timestamp": start_timestamp, "argv": " ".join(sys.argv),
            "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
                    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()},
        },
    }
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
