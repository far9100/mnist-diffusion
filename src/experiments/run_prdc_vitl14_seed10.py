"""T11：CIFAR-100 seed10 之 ViT-L/14 PRDC 與 FD 複算（對應審查 B1）。

問題設定（給第一次讀的研究生）：本文的 which-FID 交叉裁決與 coverage 選擇器都建立在 DINOv2
ViT-B/14（768 維）特徵上。Stein et al.（2023，FD-DINOv2）建議用更大的 ViT-L/14（1024 維）以更貼近
人類品質判斷。若把 backbone 由 ViT-B 換成 ViT-L，coverage 的**跨 guidance 排序**與 FD 讀數是否改變？
若不變，則本文結論對 backbone 穩健；若變，則須在限制節註明 backbone 相依。本檔即做這個複算。

流程（單 seed 10、CIFAR-100）：
  1. 特徵 = `metrics_features.dinov2_features(model_name="dinov2_vitl14")`（[0,1] 輸入、內部縮放 224
     + ImageNet 正規化，回 1024 維 CLS）。
  2. 影像來源：p1_assets 快取 `img_uint8.pt`（seed10 僅 w1／w2.5 有）；其餘 8 格以 `--generate-missing`
     streaming 補生成（決定性 hash gseed、同 confirmatory 分佈；只留特徵、不累積影像以省 CPU RAM）。
  3. 對每格算 per-class PRDC（k=5、num_classes=100）與全域 FD（ViT-L 空間）。
  4. 對照凍結 confirmatory 之 ViT-B coverage 與 `cifar100_fd_dinov2.json` 之 ViT-B FD，比排序與 argmax。

純衍生於快取影像（DINOv2 forward，不重跑擴散）＋（選配）8 格 GPU regen。輸出
results/cifar100_prdc_vitl14_seed10.json。最小版（w1／w2.5 兩快取格 + real）免授權、可直接跑。

Usage:
    uv run python src/experiments/run_prdc_vitl14_seed10.py                    # 兩快取格 + real（最小版）
    uv run python src/experiments/run_prdc_vitl14_seed10.py --generate-missing # 補 8 格（GPU）→ 全 10 格
    uv run python src/experiments/run_prdc_vitl14_seed10.py --quick            # smoke（w1、per_class 16）
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py

import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # 決定性 cublas（同 regen；須在 import torch 前）

import torch

from metrics_features import dinov2_features, fd_from_features
from metrics_prdc import compute_prdc_per_class
from datasets.cifar import load_real_per_class
from datasets.cifar100_gseed import gseed as gseed_hash

GRID = ["w1", "w1.5", "w2", "w2.5", "w3", "w4", "w5", "w6", "w7", "w8"]
DATASET = "cifar100"
NUM_CLASSES = 100
PER_CLASS = 500
ASSETS = "results/p1_assets_cifar100"
CONF = "results/cifar100_cfg_confirmatory.json"
FD_VITB = "results/cifar100_fd_dinov2.json"       # 既有 ViT-B FD-DINOv2（對照）
MODEL = "dinov2_vitl14"                            # 1024 維 CLS


def features(imgs01, device, model_name=MODEL):
    """[0,1] 影像 → ViT-L/14 1024 維 CLS 特徵（dinov2_features 內部分批＋no_grad）。"""
    return dinov2_features(imgs01, device=device, model_name=model_name)


def stream_gen_features(cfg_model, schedule, w, gseed, per_class, device, model_name,
                        batch=250, steps=50, eta=0.0):
    """缺格：逐類生成 + 即算 ViT-L 特徵，只留特徵、不累積影像（低 CPU RAM）。
    RNG 序同 generate_balanced（seed 一次、逐類 batch、決定性），故與凍結 confirmatory 同分佈。"""
    gen = torch.Generator(device=device).manual_seed(gseed)
    feats, labs = [], []
    with torch.no_grad():
        for c in range(NUM_CLASSES):
            remaining = per_class
            while remaining > 0:
                bs = min(batch, remaining)
                lab = torch.full((bs,), c, device=device, dtype=torch.long)
                x = schedule.ddim_sample_loop(cfg_model, shape=(bs, 3, 32, 32), num_steps=steps,
                                              eta=eta, class_labels=lab, guidance_scale=w, generator=gen)
                feats.append(features((x.clamp(-1, 1) + 1) / 2, device, model_name).cpu())
                labs.append(torch.full((bs,), c))
                remaining -= bs
                del x
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return torch.cat(feats), torch.cat(labs)


def load_vitb_reference():
    """凍結 confirmatory 的 seed10 ViT-B coverage（per config）＋ cifar100_fd_dinov2.json 的 ViT-B FD。"""
    cov_b, fd_b = {}, {}
    with open(CONF, encoding="utf-8") as f:
        conf = json.load(f)
    block = next((b for b in conf["per_seed"] if b["seed"] == 10), None)
    if block:
        for c in block["configs"]:
            cov_b[c["name"]] = c.get("coverage")
    if os.path.exists(FD_VITB):
        with open(FD_VITB, encoding="utf-8") as f:
            fdj = json.load(f)
        # 盡量從常見結構取 per-config FD；結構未知時留空、僅比 coverage 排序。
        for key in ("per_config", "configs", "fd_per_config"):
            if isinstance(fdj.get(key), dict):
                fd_b.update({k: v for k, v in fdj[key].items()})
            elif isinstance(fdj.get(key), list):
                for c in fdj[key]:
                    if isinstance(c, dict) and "name" in c:
                        fd_b[c["name"]] = c.get("fd") or c.get("fd_dinov2")
    return cov_b, fd_b


def main():
    p = argparse.ArgumentParser(description="CIFAR-100 seed10 ViT-L/14 PRDC 與 FD 複算（T11、B1）。")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--nearest-k", type=int, default=5)
    p.add_argument("--model-name", default=MODEL)
    p.add_argument("--generate-missing", action="store_true",
                   help="以 streaming 補生成缺 8 格（GPU；決定性 hash gseed、同 confirmatory 分佈）")
    p.add_argument("--quick", action="store_true", help="smoke：只跑 w1、per_class 16")
    p.add_argument("--output", default="results/cifar100_prdc_vitl14_seed10.json")
    args = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_class = 16 if args.quick else PER_CLASS
    grid = ["w1"] if args.quick else GRID
    print(f"device={device} seed={args.seed} model={args.model_name} per_class={per_class} grid={grid}")

    real_imgs, real_lab = load_real_per_class(DATASET, per_class, seed=0)   # [-1,1]、class-ordered
    real_feat = features((real_imgs + 1) / 2, device, args.model_name)
    print(f"  real ViT-L 特徵 {tuple(real_feat.shape)}")
    cov_b, fd_b = load_vitb_reference()

    cfg_model = schedule = None
    gp = {"per_class": per_class, "steps": 50, "eta": 0.0, "batch": 250}
    if args.generate_missing:
        from cifar_cfg_sample import load_cfg_model
        torch.backends.cudnn.benchmark = True
        cfg_model, schedule, _ = load_cfg_model(f"checkpoints/{DATASET}_cfg.pt", device)
        with open(CONF, encoding="utf-8") as f:
            m = json.load(f)["metadata"]
        gp.update({"steps": m["steps"], "eta": m["eta"], "batch": m.get("batch", 250)})
        print(f"  補生成模式 on（steps={gp['steps']}、streaming ViT-L 特徵、seed 化分佈匹配）", flush=True)

    configs, missing, regen = [], [], []
    for name in grid:
        img_p = os.path.join(ASSETS, f"seed{args.seed}_{name}", "img_uint8.pt")
        lab_p = os.path.join(ASSETS, f"seed{args.seed}_{name}", "labels.pt")
        if os.path.exists(img_p) and os.path.exists(lab_p):
            u8 = torch.load(img_p, map_location="cpu", weights_only=True)
            gen_feat = features(u8.float() / 255.0, device, args.model_name)   # 快取 uint8 →[0,1]
            lab = torch.load(lab_p, map_location="cpu", weights_only=True)
        elif args.generate_missing and cfg_model is not None:
            w = float(name.lstrip("w"))
            gen_feat, lab = stream_gen_features(cfg_model, schedule, w, gseed_hash(args.seed, w),
                                                gp["per_class"], device, args.model_name,
                                                batch=gp["batch"], steps=gp["steps"], eta=gp["eta"])
            regen.append(name)
        else:
            missing.append(name)
            continue
        prdc, _ = compute_prdc_per_class(real_feat.to(device), real_lab.to(device),
                                         gen_feat.to(device), lab.to(device),
                                         nearest_k=args.nearest_k, num_classes=NUM_CLASSES)
        fd = float(fd_from_features(real_feat, gen_feat))
        row = {"name": name, "precision": prdc["precision"], "coverage": prdc["coverage"],
               "recall": prdc["recall"], "density": prdc["density"], "fd_vitl14": fd,
               "coverage_vitb": cov_b.get(name), "fd_vitb": fd_b.get(name)}
        configs.append(row)
        print(f"  {name:>5}{' (regen)' if name in regen else ''}: covL={prdc['coverage']:.4f}"
              f" (covB={cov_b.get(name)}) fdL={fd:.3f} prec={prdc['precision']:.4f}", flush=True)

    # backbone 相依性判讀：ViT-L 與 ViT-B 的 coverage-argmax 與 FD-argmin（which-FID）是否一致
    analysis = {}
    done = [c for c in configs if c["coverage"] is not None]
    if len(done) >= 2:
        gord = {n: i for i, n in enumerate(GRID)}
        argmax_l = max(done, key=lambda c: c["coverage"])["name"]
        vb = [c for c in done if c["coverage_vitb"] is not None]
        argmax_b = max(vb, key=lambda c: c["coverage_vitb"])["name"] if vb else None
        fdmin_l = min(done, key=lambda c: c["fd_vitl14"])["name"]
        vbfd = [c for c in done if c["fd_vitb"] is not None]
        fdmin_b = min(vbfd, key=lambda c: c["fd_vitb"])["name"] if vbfd else None
        tstr_argmax = None                                    # seed10 TSTR-argmax 自凍結 confirmatory
        with open(CONF, encoding="utf-8") as f:
            blk = next((b for b in json.load(f)["per_seed"] if b["seed"] == args.seed), None)
        if blk:
            tstr_argmax = max(blk["configs"], key=lambda c: c["tstr"])["name"]
        sep_l = (abs(gord[fdmin_l] - gord[tstr_argmax])
                 if (tstr_argmax in gord and fdmin_l in gord) else None)
        analysis = {"coverage_argmax_vitl": argmax_l, "coverage_argmax_vitb": argmax_b,
                    "coverage_argmax_agrees": (argmax_b is not None and argmax_l == argmax_b),
                    "fd_argmin_vitl": fdmin_l, "fd_argmin_vitb": fdmin_b,
                    "fd_argmin_agrees": (fdmin_b is not None and fdmin_l == fdmin_b),
                    "tstr_argmax_seed10": tstr_argmax, "vitl_fd_vs_tstr_separation_step": sep_l,
                    "vitl_which_fid_separated_gt1": (sep_l is not None and sep_l > 1),
                    "cells_compared": len(done), "complete_grid": not missing}
        print("\n" + "=" * 78)
        print(f"  T11 backbone 相依（cells {len(done)}/{len(GRID)}，完整={not missing}）：")
        print(f"    coverage-argmax  ViT-L={argmax_l}  ViT-B={argmax_b}  一致={analysis['coverage_argmax_agrees']}")
        print(f"    FD-argmin        ViT-L={fdmin_l}  ViT-B={fdmin_b}  一致={analysis['fd_argmin_agrees']}")
        print(f"    which-FID(ViT-L) FD-argmin vs TSTR-argmax({tstr_argmax}) 相隔 {sep_l} 格  "
              f"分離>1={analysis['vitl_which_fid_separated_gt1']}")
        print("=" * 78)

    out = {"analysis": "prdc_vitl14_seed10", "dataset": DATASET, "seed": args.seed,
           "feature_space": f"DINOv2 {args.model_name} (1024-d CLS, 224px)",
           "num_classes": NUM_CLASSES, "per_class": per_class,
           "missing_cells": missing, "regenerated_cells": regen,
           "complete": not missing, "configs": configs, "backbone_dependence": analysis,
           "metadata": {"status": "derived (ViT-L features on cached images)" if not regen
                        else "derived + GPU regen of missing cells",
                        "nearest_k": args.nearest_k, "model_name": args.model_name,
                        "note": "最小版＝w1／w2.5 兩快取格；全 10 格需 --generate-missing（GPU regen）。",
                        "start_timestamp": start_timestamp, "argv": " ".join(sys.argv),
                        "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                                "cudnn": torch.backends.cudnn.version()}}}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output}  (cells done: {len(configs)}, missing: {missing})")


if __name__ == "__main__":
    main()
