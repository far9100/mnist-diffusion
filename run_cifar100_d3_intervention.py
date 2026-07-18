"""D3 介入臂：CIFAR-100 之 C3 coverage-matched pruning（機制複製的介入證據）。

預註冊 D3（`docs/prereg_cifar100.md`）：於 CIFAR-100 confirmatory 已生成集上跑一個 C3 型
coverage-matched pruning，定位為分支三論文之機制複製「介入」證據（觀察量已由 D3 driver 成立），
非分支路由輸入。剪枝規則沿 CIFAR-10 C3（`run_c2c3c5_intervention.py`，R-2026-07-06-13/-09-08）：
w2.5（高 coverage）剪至 w1（低 coverage）之 DINOv2 coverage 水準，移除離真實流形最近之樣本以降
coverage、二分搜尋匹配，重訓 TSTR；對照＝等計數隨機剪枝。

與 CIFAR-10 C3 的差異（另立新檔、不動凍結的 run_c2c3c5_intervention.py）：
  - num_classes=100、資產目錄 results/p1_assets_cifar100（regen_cifar100_cells.py 產）。
  - target_cov 改由**重生成的 w1 cell 現算**（cross-env 一致：base 與 target 在同一 DINOv2/PRDC
    環境算），而非讀凍結 JSON——故 w1 也須重生成（見 regen_cifar100_cells.py）。
  - 附 §5.2 metadata（start_timestamp/argv/env/nearest_k）。

判讀：coverage 匹配後 w2.5 之 TSTR 與等計數隨機剪枝之 TSTR 相比——若 coverage-matched 掉幅顯著大於
隨機剪枝，支持 coverage 本身承載效用（介入式）。N 死（每情境少重訓），為 exploratory 介入證據。

Usage:
    uv run python run_cifar100_d3_intervention.py                 # 需 GPU（Stage 3）
    uv run python run_cifar100_d3_intervention.py --n-retrain 5   # 提高 power（作者 Stage 3 閘定案）
"""
import argparse
import json
import os
import platform
import sys
import time

import torch

from cifar_classifier import run_tstr
from datasets.cifar import build_test_loader, load_real_per_class
from metrics_features import dinov2_features
from metrics_prdc import compute_prdc_per_class

ASSET = "results/p1_assets_cifar100"
CONF = "results/cifar100_cfg_confirmatory.json"
OUT = "results/cifar100_d3_intervention.json"
SEED = 10
TSTR_EPOCHS = 15              # 同 FROZEN_CIFAR100["tstr_epochs"]
NUM_CLASSES = 100
NEAREST_K = 5
RAND_SEED = 20260709         # 隨機剪枝對照之固定 seed（沿 CIFAR-10 C3）


def load_cell(name):
    """regen_cifar100_cells.py 落盤：影像 [-1,1]、labels、judge margins/preds、DINOv2 特徵。"""
    u8 = torch.load(f"{ASSET}/seed{SEED}_{name}/img_uint8.pt", map_location="cpu", weights_only=True)
    labels = torch.load(f"{ASSET}/seed{SEED}_{name}/labels.pt", map_location="cpu", weights_only=True)
    judge = torch.load(f"{ASSET}/seed{SEED}_{name}/judge_out.pt", map_location="cpu", weights_only=True)
    dino = torch.load(f"{ASSET}/seed{SEED}_{name}/dino_feat.pt", map_location="cpu", weights_only=True)
    imgs = u8.float() / 255.0 * 2.0 - 1.0
    return imgs, labels, judge["margins"], dino


def retrain_n(imgs, labels, test_loader, device, n):
    out = []
    for _ in range(n):
        overall, _ = run_tstr(imgs, labels, test_loader, device, num_classes=NUM_CLASSES, epochs=TSTR_EPOCHS)
        out.append(overall)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def frozen_tstr(d, name):
    sb = next(s for s in d["per_seed"] if s["seed"] == SEED)
    return next(c for c in sb["configs"] if c["name"] == name)["tstr"]


def coverage_of(gen_dino, gen_labels, real_dino, real_labels, device):
    prdc, _ = compute_prdc_per_class(real_dino.to(device), real_labels.to(device),
                                     gen_dino.to(device), gen_labels.to(device),
                                     nearest_k=NEAREST_K, num_classes=NUM_CLASSES)
    return prdc["coverage"]


def nearest_real_dist(gen_dino, gen_labels, real_dino, real_labels):
    """每 gen 樣本到同類最近真實點之距離（DINOv2）。用於「移除離真實流形最近者」以降 coverage。"""
    dist = torch.full((gen_dino.size(0),), float("inf"))
    for c in range(NUM_CLASSES):
        gm = (gen_labels == c).nonzero(as_tuple=True)[0]
        rm = (real_labels == c)
        if gm.numel() == 0 or rm.sum() == 0:
            continue
        dmat = torch.cdist(gen_dino[gm].float(), real_dino[rm].float())
        dist[gm] = dmat.min(dim=1).values
    return dist


def c3_coverage_matched(d, real_dino, real_labels, test_loader, device, n_retrain):
    imgs, labels, _, dino = load_cell("w2.5")
    # target_cov 由重生成的 w1 cell 現算（cross-env 一致），非讀凍結 JSON。
    w1_imgs, w1_labels, _, w1_dino = load_cell("w1")
    target_cov = coverage_of(w1_dino, w1_labels, real_dino, real_labels, device)
    base_cov = coverage_of(dino, labels, real_dino, real_labels, device)
    dist = nearest_real_dist(dino, labels, real_dino, real_labels)
    order = torch.argsort(dist)                                   # 近→遠；移除最近者降 coverage
    lo, hi = 0, imgs.size(0) - NUM_CLASSES * 6                    # 保每類 ≥ ~k+1
    best_k = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        keep = torch.ones(imgs.size(0), dtype=torch.bool); keep[order[:mid]] = False
        cov = coverage_of(dino[keep], labels[keep], real_dino, real_labels, device)
        if cov <= target_cov:
            best_k = mid; hi = mid - 1
        else:
            lo = mid + 1
    keep = torch.ones(imgs.size(0), dtype=torch.bool); keep[order[:best_k]] = False
    matched_cov = coverage_of(dino[keep], labels[keep], real_dino, real_labels, device)
    # 等計數隨機剪枝對照
    g = torch.Generator().manual_seed(RAND_SEED)
    perm = torch.randperm(imgs.size(0), generator=g)
    keep_rand = torch.ones(imgs.size(0), dtype=torch.bool); keep_rand[perm[:best_k]] = False
    res = {
        "w2.5_base_coverage": base_cov, "w1_target_coverage": target_cov,
        "target_cov_source": "regenerated w1 cell (cross-env)",
        "n_pruned_to_match": best_k, "matched_coverage": matched_cov,
        "w1_frozen_tstr": frozen_tstr(d, "w1"), "w2.5_frozen_tstr": frozen_tstr(d, "w2.5"),
        "tstr_w2.5_cov_matched": retrain_n(imgs[keep], labels[keep], test_loader, device, n_retrain),
        "tstr_w2.5_rand_pruned": retrain_n(imgs[keep_rand], labels[keep_rand], test_loader, device, n_retrain),
    }
    print(f"[C3] w2.5 cov {base_cov:.3f}→{matched_cov:.3f} (target w1 {target_cov:.3f}, pruned {best_k}); "
          f"matched={res['tstr_w2.5_cov_matched']} rand={res['tstr_w2.5_rand_pruned']} "
          f"w1={res['w1_frozen_tstr']} w2.5={res['w2.5_frozen_tstr']}", flush=True)
    return res


def main():
    p = argparse.ArgumentParser(description="CIFAR-100 D3 介入臂（C3 coverage-matched pruning）。")
    p.add_argument("--n-retrain", type=int, default=2,
                   help="每情境重訓次數；預設 2（沿 CIFAR-10 C3、N 死），作者可提高至 5（D4 power）")
    p.add_argument("--output", default=OUT,
                   help="輸出 JSON 路徑；預設覆寫原檔，提高 power 之 re-run 建議另存（保留原 N=2 結果）")
    args = p.parse_args()
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = json.load(open(CONF, encoding="utf-8"))
    test_loader = build_test_loader("cifar100")
    # 真實 DINOv2 特徵：優先載入 regen_cifar100_cells.py 已快取者（與此處 load_real_per_class(seed=0)
    # ＋dinov2_features 逐位同源，見 regen_cifar100_cells.py:123-127），免載 DINOv2 模型與 50k 前向、
    # 大幅降顯存（低顯存機器可跑；否則 fallback 重算）。
    real_feat_path, real_label_path = f"{ASSET}/real_dino_feat.pt", f"{ASSET}/real_labels.pt"
    if os.path.exists(real_feat_path) and os.path.exists(real_label_path):
        real_dino = torch.load(real_feat_path, map_location="cpu", weights_only=True)
        real_labels = torch.load(real_label_path, map_location="cpu", weights_only=True)
        print(f"loaded cached real DINOv2 feats {tuple(real_dino.shape)} from {ASSET}", flush=True)
    else:
        real_imgs, real_labels = load_real_per_class("cifar100", d["metadata"]["real_per_class"], seed=0)
        real_dino = dinov2_features((real_imgs + 1) / 2, device)
    print(f"device={device} seed={SEED} n_retrain={args.n_retrain}", flush=True)

    c3 = c3_coverage_matched(d, real_dino, real_labels, test_loader, device, args.n_retrain)
    out = {
        "c3_coverage_matched": c3,
        "metadata": {
            "analysis": "cifar100_d3_intervention_c3", "source": CONF, "asset": ASSET, "seed": SEED,
            "num_classes": NUM_CLASSES, "n_retrain": args.n_retrain, "tstr_epochs": TSTR_EPOCHS,
            "nearest_k": NEAREST_K, "rand_seed": RAND_SEED,
            "note": "分支三機制複製之介入證據（exploratory，禁因果）；剪枝規則沿 CIFAR-10 C3；"
                    "target_cov 由重生成 w1 cell cross-env 現算；N 死（少重訓）。",
            "start_timestamp": start_timestamp, "argv": " ".join(sys.argv),
            "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
                    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()},
        },
    }
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
