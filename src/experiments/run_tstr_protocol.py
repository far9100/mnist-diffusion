"""T9：TSTR 協定強化——real-data 上限線 + epochs 消融（對應審查 B2）。

問題設定（給第一次讀的研究生）：本文以 TSTR（train-on-synthetic）為效用黃金標準，但有兩個協定問題：
  (1) 沒有「train-on-real」上限線，讀者無法判斷合成資料離真實訓練還差多少（表 5.2/5.4 缺一行 ceiling）。
  (2) TSTR 只訓 15 epochs、每 cell reps 少，σ_cls（分類器訓練變異）可能大到把 cell 間的真實差異淹沒；
      需驗「噪聲地板是否協定內生」——加長訓練（50 epochs）、多 reps，看 σ_cls 是否縮、排序是否穩。

本檔兩模式：
  --ceiling ：對 real 資料跑 TSTR（load_real_per_class；CIFAR-100 500/class、CIFAR-10 1000/class），
              epochs∈{15,50} 各 5 reps → results/tstr_real_ceiling.json。此為 train-on-real 上限。
  --ablation：CIFAR-100 seed10 之 {w1,w1.5,w2.5} 三 cell（w1/w2.5 快取影像、w1.5 需 --generate-missing
              regen），epochs∈{15,50} 各 5 reps，量 σ_cls 與排序穩定 → results/tstr_protocol_ablation.json。

reps 以 sha256 衍生種子（T6b 決定性）。GPU 中。凍結資料不動、新結果寫新檔。

Usage:
    uv run python src/experiments/run_tstr_protocol.py --ceiling                 # real 上限（兩資料集）
    uv run python src/experiments/run_tstr_protocol.py --ablation --generate-missing   # 三 cell epochs 消融
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py
import argparse
import hashlib
import json
import os
import statistics
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from cifar_classifier import run_tstr
from datasets.cifar import build_test_loader, load_real_per_class
from datasets.cifar100_gseed import gseed as gseed_hash

NC = {"cifar10": 10, "cifar100": 100}
PC = {"cifar10": 1000, "cifar100": 500}
ASSETS = "results/p1_assets_cifar100"
CONF = "results/cifar100_cfg_confirmatory.json"


def _seed(tag, epochs, rep):
    return int(hashlib.sha256(f"tstr9_{tag}_{epochs}_{rep}".encode()).hexdigest()[:15], 16)


def tstr_block(imgs, labels, test_loader, device, nc, epochs, reps, tag):
    """跑 reps 次 TSTR（決定性種子），回 mean / std（σ_cls 代理）/ 各 rep。"""
    vals = []
    for r in range(reps):
        ov, _ = run_tstr(imgs, labels, test_loader, device, num_classes=nc,
                         epochs=epochs, seed=_seed(tag, epochs, r))
        vals.append(round(ov, 2))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    mean = sum(vals) / len(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    print(f"    {tag} epochs={epochs}: mean={mean:.2f} sigma_cls={std:.3f} reps={vals}", flush=True)
    return {"epochs": epochs, "reps": vals, "mean": round(mean, 2), "sigma_cls": round(std, 3)}


def run_ceiling(datasets, epochs_list, reps, device):
    out = {}
    for ds in datasets:
        real, lab = load_real_per_class(ds, PC[ds], seed=0)               # [-1,1]、class-ordered
        tl = build_test_loader(ds)
        print(f"  [ceiling] {ds} real {tuple(real.shape)} per_class={PC[ds]}", flush=True)
        out[ds] = {"per_class": PC[ds], "by_epochs":
                   [tstr_block(real, lab, tl, device, NC[ds], e, reps, f"ceil_{ds}") for e in epochs_list]}
    return out


def _load_cell_images(name, generate_missing, device):
    """cell 影像：快取 img_uint8.pt →[-1,1]；缺則（--generate-missing）以 hash gseed 決定性生成。"""
    img_p = os.path.join(ASSETS, f"seed10_{name}", "img_uint8.pt")
    lab_p = os.path.join(ASSETS, f"seed10_{name}", "labels.pt")
    if os.path.exists(img_p) and os.path.exists(lab_p):
        u8 = torch.load(img_p, map_location="cpu", weights_only=True)
        return u8.float() / 255.0 * 2.0 - 1.0, torch.load(lab_p, map_location="cpu", weights_only=True), "cached"
    if not generate_missing:
        return None, None, "missing"
    from cifar_cfg_sample import load_cfg_model
    torch.backends.cudnn.benchmark = True
    model, schedule, _ = load_cfg_model("checkpoints/cifar100_cfg.pt", device)
    m = json.load(open(CONF, encoding="utf-8"))["metadata"]
    w = float(name.lstrip("w"))
    per_class, steps, eta, batch = m["per_class"], m["steps"], m["eta"], m.get("batch", 250)
    gen = torch.Generator(device=device).manual_seed(gseed_hash(10, w))
    imgs, labs = [], []
    with torch.no_grad():
        for c in range(NC["cifar100"]):
            remaining = per_class
            while remaining > 0:
                bs = min(batch, remaining)
                lab = torch.full((bs,), c, device=device, dtype=torch.long)
                x = schedule.ddim_sample_loop(model, shape=(bs, 3, 32, 32), num_steps=steps, eta=eta,
                                              class_labels=lab, guidance_scale=w, generator=gen)
                imgs.append(x.clamp(-1, 1).cpu())
                labs.append(torch.full((bs,), c))
                remaining -= bs
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return torch.cat(imgs), torch.cat(labs), "regen"


def run_ablation(cells, epochs_list, reps, device, generate_missing):
    tl = build_test_loader("cifar100")
    out = {"seed": 10, "cells": {}, "missing": []}
    for name in cells:
        imgs, labs, src = _load_cell_images(name, generate_missing, device)
        if imgs is None:
            out["missing"].append(name)
            print(f"  [ablation] {name}: 缺影像（需 --generate-missing）", flush=True)
            continue
        print(f"  [ablation] {name} ({src}) {tuple(imgs.shape)}", flush=True)
        out["cells"][name] = {"source": src,
                              "by_epochs": [tstr_block(imgs, labs, tl, device, NC["cifar100"], e, reps,
                                                       f"abl_{name}") for e in epochs_list]}
    # 排序穩定：各 epochs 下 cell 的 TSTR-mean 排序是否一致
    ranks = {}
    for e_idx, e in enumerate(epochs_list):
        order = sorted((c for c in out["cells"]), key=lambda c: -out["cells"][c]["by_epochs"][e_idx]["mean"])
        ranks[str(e)] = order
    out["rank_by_epochs"] = ranks
    out["rank_stable"] = len({tuple(v) for v in ranks.values()}) == 1 if ranks else None
    return out


def main():
    p = argparse.ArgumentParser(description="TSTR 協定強化：real 上限線 + epochs 消融（T9/B2）。")
    p.add_argument("--ceiling", action="store_true", help="跑 real-data TSTR 上限線")
    p.add_argument("--ablation", action="store_true", help="跑 CIFAR-100 seed10 三 cell epochs 消融")
    p.add_argument("--datasets", nargs="+", default=["cifar100", "cifar10"])
    p.add_argument("--cells", nargs="+", default=["w1", "w1.5", "w2.5"])
    p.add_argument("--epochs", type=int, nargs="+", default=[15, 50])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--generate-missing", action="store_true", help="ablation 缺格（w1.5）以 hash gseed regen")
    p.add_argument("--ceiling-output", default="results/tstr_real_ceiling.json")
    p.add_argument("--ablation-output", default="results/tstr_protocol_ablation.json")
    args = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if not args.ceiling and not args.ablation:
        args.ceiling = args.ablation = True
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = {"start_timestamp": start_timestamp, "argv": " ".join(sys.argv), "reps": args.reps,
            "epochs": args.epochs, "tstr_seeded": "sha256 決定性（T6b）",
            "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None}}

    if args.ceiling:
        print("########## TSTR real-data 上限線 ##########")
        res = {"analysis": "tstr_real_ceiling", "by_dataset": run_ceiling(args.datasets, args.epochs, args.reps, device),
               "note": "train-on-real 上限；表 5.2/5.4 加一行 ceiling。", "metadata": meta}
        json.dump(res, open(args.ceiling_output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"Wrote {args.ceiling_output}")

    if args.ablation:
        print("########## TSTR epochs 消融（σ_cls 是否協定內生） ##########")
        res = {"analysis": "tstr_protocol_ablation", **run_ablation(args.cells, args.epochs, args.reps,
                                                                    device, args.generate_missing),
               "note": "σ_cls 隨 epochs 是否縮、排序是否穩——回應「噪聲地板為協定內生」。", "metadata": meta}
        json.dump(res, open(args.ablation_output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"Wrote {args.ablation_output}")


if __name__ == "__main__":
    main()
