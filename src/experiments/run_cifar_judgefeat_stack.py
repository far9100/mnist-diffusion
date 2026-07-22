"""T1c：CIFAR judge 特徵堆疊版 CaF（對應審查 A1(b) 的反向交叉）。

問題設定（給第一次讀的研究生）：T1b 已顯示 MNIST 換到 CIFAR 用的 DINOv2 特徵後、coverage 選擇器亦失效。
要坐實「反轉沿特徵空間」的 2×2 歸因，還缺對稱的一格：CIFAR 換到 MNIST 用的**任務對齊 judge 特徵**
（ResNet-18 penultimate 512 維）後，coverage 選擇器是否反而選中 oracle？本檔補上這一格。

2×2（列＝資料集，欄＝特徵空間；值＝coverage-CaF 之 regret@selected）：
                judge（任務對齊）      DINOv2（任務無關）
    MNIST       0.00（原 selector_signal）   1.02（T1b）
    CIFAR       ← 本檔                        0.91/3.69（原 confirmatory，CaF 敗/平 FID-min）

流程（單 seed 10，先呈作者；授權後擴 8/3 seeds）：
  1. judge = ResNet18 + `checkpoints/<ds>_judge.pt`（eval）；feat_fn = chamfer.cifar_penultimate_feature_fn。
  2. 影像來源：p1_assets 快取 `img_uint8.pt`（uint8 0-255 →[-1,1] 以 u8/255*2-1；直接餵 judge，不做 (x+1)/2）。
     CIFAR-10 seed10 十格齊備；CIFAR-100 seed10 僅 w1／w2.5 有影像，其餘 8 格需 regen（GPU，標【需授權】）。
  3. real judge 特徵取自 `load_real_per_class(ds, per_class, seed=0)`（已 [-1,1]）；real_ref_precision 用對半切。
  4. `compute_prdc_per_class`（judge 512 維、k=5、num_classes=10/100）算 precision/coverage/recall。
  5. `select_and_report` 以 coverage（CaF）與 recall（CaF-v2）各跑一次；TSTR（oracle）取自凍結 confirmatory。

純衍生於快取影像（judge forward，不重跑擴散生成）；CIFAR-10 免 regen。輸出 results/cifar_judgefeat_stack.json。

Usage:
    uv run python src/experiments/run_cifar_judgefeat_stack.py                 # cifar10（全）＋cifar100（部分）
    uv run python src/experiments/run_cifar_judgefeat_stack.py --datasets cifar10
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

from cifar_classifier import ResNet18
from chamfer import cifar_penultimate_feature_fn
from metrics_prdc import compute_prdc_per_class
from selector import select_and_report
from datasets.cifar import load_real_per_class
from cifar_cfg_sample import load_cfg_model            # T1c streaming：CIFAR-100 缺格邊生成邊算 judge 特徵
from datasets.cifar100_gseed import gseed as gseed_hash

GRID = ["w1", "w1.5", "w2", "w2.5", "w3", "w4", "w5", "w6", "w7", "w8"]
DS = {
    "cifar10": {"nc": 10, "per_class": 1000, "signal": "coverage",
                "assets": "results/p1_assets", "conf": "results/cifar10_cfg_confirmatory.json"},
    "cifar100": {"nc": 100, "per_class": 500, "signal": "recall",
                 "assets": "results/p1_assets_cifar100", "conf": "results/cifar100_cfg_confirmatory.json"},
}


def judge_features(feat_fn, imgs, device, batch=512):
    """judge 512 維特徵（[-1,1] 輸入），分批＋no_grad。"""
    outs = []
    with torch.no_grad():
        for i in range(0, imgs.size(0), batch):
            outs.append(feat_fn(imgs[i:i + batch].to(device)).cpu())
    return torch.cat(outs)


def real_ref_precision(real_feat, real_lab, seed, nearest_k, nc, device):
    """對半切法之 real-vs-real 參考 precision（judge 空間）。"""
    n = real_feat.size(0)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    a, b = perm[:n // 2], perm[n // 2:]
    ref, _ = compute_prdc_per_class(real_feat[b].to(device), real_lab[b].to(device),
                                    real_feat[a].to(device), real_lab[a].to(device),
                                    nearest_k=nearest_k, num_classes=nc)
    return ref["precision"]


def stream_gen_judge(cfg_model, schedule, feat_fn, w, gseed, per_class, nc, device,
                     batch=250, steps=50, eta=0.0):
    """CIFAR-100 缺格：逐類生成 + 即算 judge 512 維特徵，只留特徵、不累積影像（低 CPU RAM，避 OOM/kill）。
    RNG 序同 `generate_balanced`（seed 一次、逐類 batch、決定性），故與凍結 confirmatory 同分佈。"""
    gen = torch.Generator(device=device).manual_seed(gseed)
    feats, labs = [], []
    with torch.no_grad():
        for c in range(nc):
            remaining = per_class
            while remaining > 0:
                bs = min(batch, remaining)
                lab = torch.full((bs,), c, device=device, dtype=torch.long)
                x = schedule.ddim_sample_loop(cfg_model, shape=(bs, 3, 32, 32), num_steps=steps,
                                              eta=eta, class_labels=lab, guidance_scale=w, generator=gen)
                feats.append(feat_fn(x.clamp(-1, 1)).cpu())     # 512 維，立即搬 CPU、釋放影像
                labs.append(torch.full((bs,), c))
                remaining -= bs
                del x
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return torch.cat(feats), torch.cat(labs)


def analyse(ds, seed, nearest_k, tau_fraction, device, generate_missing=False):
    cfg = DS[ds]
    nc = cfg["nc"]
    judge = ResNet18(num_classes=nc).to(device)
    judge.load_state_dict(torch.load(f"checkpoints/{ds}_judge.pt", map_location=device, weights_only=True))
    judge.eval()
    feat_fn = cifar_penultimate_feature_fn(judge)

    real_imgs, real_lab = load_real_per_class(ds, cfg["per_class"], seed=0)   # [-1,1]、class-ordered
    real_feat = judge_features(feat_fn, real_imgs, device)
    ref_prec = real_ref_precision(real_feat, real_lab, seed, nearest_k, nc, device)

    with open(cfg["conf"], encoding="utf-8") as f:
        conf = json.load(f)
    seed_block = next(b for b in conf["per_seed"] if b["seed"] == seed)
    tstr_by_name = {c["name"]: c["tstr"] for c in seed_block["configs"]}

    # 允許補生成缺格（CIFAR-100）：僅在確有缺格時載 CFG 模型（CIFAR-10 十格齊備、不載）
    has_missing = any(not (os.path.exists(os.path.join(cfg["assets"], f"seed{seed}_{n}", "img_uint8.pt"))
                           and os.path.exists(os.path.join(cfg["assets"], f"seed{seed}_{n}", "labels.pt")))
                      for n in GRID)
    cfg_model = schedule = None
    gp = {}
    if generate_missing and has_missing:
        # 為速度不用 strict deterministic algos：judge coverage 為 per-class PRDC 均值、對確切樣本穩健，
        # 故 seed 化（hash gseed）之分佈匹配即足，不需與 confirmatory 逐位相同（此為結構性 2×2 之補格）。
        torch.backends.cudnn.benchmark = True
        cfg_model, schedule, _ = load_cfg_model(f"checkpoints/{ds}_cfg.pt", device)
        m = conf["metadata"]
        gp = {"per_class": m["per_class"], "steps": m["steps"], "eta": m["eta"], "batch": m.get("batch", 250)}
        print(f"  {ds}: 補生成模式 on（batch={gp['batch']}、steps={gp['steps']}、streaming judge 特徵、"
              f"seed 化分佈匹配）", flush=True)

    configs, missing, regen = [], [], []
    for name in GRID:
        img_p = os.path.join(cfg["assets"], f"seed{seed}_{name}", "img_uint8.pt")
        lab_p = os.path.join(cfg["assets"], f"seed{seed}_{name}", "labels.pt")
        if os.path.exists(img_p) and os.path.exists(lab_p):
            u8 = torch.load(img_p, map_location="cpu", weights_only=True)
            gen_feat = judge_features(feat_fn, u8.float() / 255.0 * 2.0 - 1.0, device)   # 快取影像 →[-1,1]
            lab = torch.load(lab_p, map_location="cpu", weights_only=True)
        elif generate_missing and cfg_model is not None:
            w = float(name.lstrip("w"))
            gen_feat, lab = stream_gen_judge(cfg_model, schedule, feat_fn, w, gseed_hash(seed, w),
                                             gp["per_class"], nc, device, batch=gp["batch"],
                                             steps=gp["steps"], eta=gp["eta"])
            regen.append(name)
        else:
            missing.append(name)
            continue
        prdc, _ = compute_prdc_per_class(real_feat.to(device), real_lab.to(device),
                                         gen_feat.to(device), lab.to(device),
                                         nearest_k=nearest_k, num_classes=nc)
        configs.append({"name": name, "precision": prdc["precision"], "coverage": prdc["coverage"],
                        "recall": prdc["recall"], "density": prdc["density"], "tstr": tstr_by_name.get(name)})
        print(f"  {ds} {name:>5}{' (regen)' if name in regen else ''}: prec={prdc['precision']:.4f}"
              f" cov={prdc['coverage']:.4f} recall={prdc['recall']:.4f} tstr={configs[-1]['tstr']}", flush=True)

    complete = not missing
    reports = {}
    have_tstr = all(c["tstr"] is not None for c in configs)
    if complete and have_tstr:
        for tag, signal in [("caf", "coverage"), ("caf_v2", "recall")]:
            rep = select_and_report(configs, real_ref_precision=ref_prec, tau_fraction=tau_fraction,
                                    utility_key="tstr", signal_key=signal)
            reports[tag] = rep
            oracle = rep["oracle_best"]
            print(f"    [{tag}/{signal}] selected {rep['selected']} (oracle {oracle}, "
                  f"regret {rep['regret_at_selected']}, tau {rep['tau']:.4f})")
    else:
        print(f"  {ds}: 未跑選擇（缺 {missing or 'TSTR'}）——需 regen 8 格影像後補（GPU，需授權）。")

    return {"dataset": ds, "seed": seed, "num_classes": nc, "feature_space": "judge penultimate 512-d",
            "signal_default": cfg["signal"], "real_ref_precision": ref_prec,
            "missing_cells": missing, "regenerated_cells": regen, "complete": complete,
            "configs": configs, "reports": reports}


def main():
    p = argparse.ArgumentParser(description="CIFAR judge 特徵堆疊版 CaF（A1b 反向交叉，T1c）。")
    p.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100"])
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--nearest-k", type=int, default=5)
    p.add_argument("--tau-fraction", type=float, default=0.9)
    p.add_argument("--generate-missing", action="store_true",
                   help="CIFAR-100 缺格以 streaming 生成補齊（低 RAM；決定性 hash gseed、同 confirmatory 分佈）")
    p.add_argument("--output", default="results/cifar_judgefeat_stack.json")
    args = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} seed={args.seed} datasets={args.datasets}")

    out = {"datasets": {}}
    for ds in args.datasets:
        print(f"\n########## {ds} (judge 特徵) ##########")
        out["datasets"][ds] = analyse(ds, args.seed, args.nearest_k, args.tau_fraction, device,
                                      generate_missing=args.generate_missing)

    print("\n" + "=" * 78)
    print("  T1c CIFAR judge 特徵堆疊 CaF（regret@selected，pp）")
    print("=" * 78)
    for ds in args.datasets:
        r = out["datasets"][ds]
        if r["complete"] and r["reports"]:
            caf = r["reports"]["caf"]
            print(f"  {ds}: CaF(coverage) 選 {caf['selected']}（oracle {caf['oracle_best']}、"
                  f"regret {caf['regret_at_selected']:.2f}）；判 coverage 是否選中 oracle → "
                  f"{'是' if caf['regret_at_selected'] == 0 else '否'}")
        else:
            print(f"  {ds}: 部分（缺 {r['missing_cells']}）——待 regen 補齊。")
    print("=" * 78)

    out["metadata"] = {
        "analysis": "cifar_judgefeat_stack", "status": "derived (judge features on cached images)",
        "feature_space": "cifar ResNet-18 judge penultimate 512-d ([-1,1] input)",
        "nearest_k": args.nearest_k, "tau_fraction": args.tau_fraction, "seed": args.seed,
        "note": "純衍生於 p1_assets 快取影像之 judge forward；CIFAR-100 缺 8 格影像需 regen（GPU、需授權）。",
        "start_timestamp": start_timestamp, "argv": " ".join(sys.argv),
        "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version()},
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
