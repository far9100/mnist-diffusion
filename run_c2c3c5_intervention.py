"""C2/C3/C5：near-boundary 與 coverage 之介入式證據（剪枝／過濾＋重訓）。依凍結規則段
records/2026-07-06-12（C2）、-06-13（C3）、-06-14（C5）與 -09-04 §2.3。

三者皆取自 P1 落盤影像／DINOv2 特徵／judge 輸出（results/p1_assets，seed 10），禁二次重生成。全標
exploratory（C0）：規則先於計算、不因結果回改；因果措辭禁（H-C2a 顯著亦然，-06-05 §2 格殺）。

- C2 boundary-targeted pruning（-06-12）：固定組態 w1.5、w2.5，移除 margin<0.9525 之 near-boundary 樣本
  重訓 TSTR；對照＝等數量隨機移除（同計數）；每情境 2 次重訓（N 死）。判定：near-boundary 移除掉幅
  顯著大於隨機移除掉幅 → 支持 near-boundary 因果角色（介入式）；否則不支持。
- C3 coverage-matched pruning（-06-13）：w2.5（高 coverage）剪至 w1（低 coverage）之 DINOv2 coverage 水準
  （移除離真實流形最近之樣本以降 coverage），重訓；對照＝等計數隨機剪枝；每情境 2 次重訓。橋接：coverage
  匹配後 w2.5→w1 之 TSTR 差是否顯著縮小 → coverage 本身承載效用；否則差來自 coverage 以外之 w-共變項。
- C5 near-boundary 純度過濾（-06-14）：w1.5，按 margin 排序構造純度階梯 {原始, 高純度前 50%（低 margin）,
  低純度後 50%（高 margin）}，各重訓；每水準 2 次重訓。判定：TSTR 隨 near-boundary 純度單調上升 → 支持。

重訓沿 confirmatory TSTR 協定（run_tstr，15 epoch、未種子化 shuffle、augment on）。

Usage:
    uv run python run_c2c3c5_intervention.py
"""
import argparse
import json

import torch

from cifar_classifier import run_tstr
from datasets.cifar import build_test_loader, load_real_per_class
from metrics_features import dinov2_features
from metrics_prdc import compute_prdc_per_class

ASSET = "results/p1_assets"
CONF = "results/cifar10_cfg_confirmatory.json"
OUT = "results/cifar10_c2c3c5_intervention.json"
SEED = 10
THRESHOLD = 0.9525            # near-boundary margin 門檻（凍結）
N_RETRAIN = 2                 # 每情境 2 次（N 死）
TSTR_EPOCHS = 15
NUM_CLASSES = 10
RAND_SEED = 20260709         # 隨機移除／剪枝之固定 seed（事前定，對照可重現）


def load_cell(name):
    """P1 落盤：影像 [-1,1]、labels、judge margins/preds、DINOv2 特徵。"""
    u8 = torch.load(f"{ASSET}/seed{SEED}_{name}/img_uint8.pt", map_location="cpu", weights_only=True)
    labels = torch.load(f"{ASSET}/seed{SEED}_{name}/labels.pt", map_location="cpu", weights_only=True)
    judge = torch.load(f"{ASSET}/seed{SEED}_{name}/judge_out.pt", map_location="cpu", weights_only=True)
    dino = torch.load(f"{ASSET}/seed{SEED}_{name}/dino_feat.pt", map_location="cpu", weights_only=True)
    imgs = u8.float() / 255.0 * 2.0 - 1.0
    return imgs, labels, judge["margins"], dino


def retrain_n(imgs, labels, test_loader, device, n=N_RETRAIN):
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
                                     nearest_k=5, num_classes=NUM_CLASSES)
    return prdc["coverage"]


def nearest_real_dist(gen_dino, gen_labels, real_dino, real_labels):
    """每 gen 樣本到同類最近真實點之距離（DINOv2 空間）。用於 C3 之「移除離真實流形最近者」。"""
    dist = torch.full((gen_dino.size(0),), float("inf"))
    for c in range(NUM_CLASSES):
        gm = (gen_labels == c).nonzero(as_tuple=True)[0]
        rm = (real_labels == c)
        if gm.numel() == 0 or rm.sum() == 0:
            continue
        dmat = torch.cdist(gen_dino[gm].float(), real_dino[rm].float())  # (n_gen_c, n_real_c)
        dist[gm] = dmat.min(dim=1).values
    return dist


def c2_boundary_pruning(d, test_loader, device):
    res = {}
    for name in ["w1.5", "w2.5"]:
        imgs, labels, margins, _ = load_cell(name)
        nb_mask = margins < THRESHOLD
        n_rm = int(nb_mask.sum())
        keep_nb = ~nb_mask
        g = torch.Generator().manual_seed(RAND_SEED)
        perm = torch.randperm(imgs.size(0), generator=g)
        rand_rm = torch.zeros(imgs.size(0), dtype=torch.bool); rand_rm[perm[:n_rm]] = True
        keep_rand = ~rand_rm
        res[name] = {
            "n_total": imgs.size(0), "n_near_boundary_removed": n_rm,
            "frozen_full_tstr": frozen_tstr(d, name),
            "tstr_nb_removed": retrain_n(imgs[keep_nb], labels[keep_nb], test_loader, device),
            "tstr_rand_removed": retrain_n(imgs[keep_rand], labels[keep_rand], test_loader, device),
        }
        print(f"[C2] {name}: removed {n_rm} near-bnd; nb={res[name]['tstr_nb_removed']} "
              f"rand={res[name]['tstr_rand_removed']} full={res[name]['frozen_full_tstr']}", flush=True)
    return res


def c3_coverage_matched(d, real_dino, real_labels, test_loader, device):
    imgs, labels, _, dino = load_cell("w2.5")
    target_cov = next(c for c in next(s for s in d["per_seed"] if s["seed"] == SEED)["configs"]
                      if c["name"] == "w1")["coverage"]              # w1 之 coverage（凍結）
    base_cov = coverage_of(dino, labels, real_dino, real_labels, device)
    dist = nearest_real_dist(dino, labels, real_dino, real_labels)
    order = torch.argsort(dist)                                       # 近→遠；移除最近者降 coverage
    # 二分搜尋移除數，使 coverage ≤ target
    lo, hi = 0, imgs.size(0) - NUM_CLASSES * 6                        # 保每類 ≥ ~k+1
    best_k = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        rm = order[:mid]
        keep = torch.ones(imgs.size(0), dtype=torch.bool); keep[rm] = False
        cov = coverage_of(dino[keep], labels[keep], real_dino, real_labels, device)
        if cov <= target_cov:
            best_k = mid; hi = mid - 1
        else:
            lo = mid + 1
    rm = order[:best_k]
    keep = torch.ones(imgs.size(0), dtype=torch.bool); keep[rm] = False
    matched_cov = coverage_of(dino[keep], labels[keep], real_dino, real_labels, device)
    # 隨機剪枝對照（同計數）
    g = torch.Generator().manual_seed(RAND_SEED)
    perm = torch.randperm(imgs.size(0), generator=g)
    keep_rand = torch.ones(imgs.size(0), dtype=torch.bool); keep_rand[perm[:best_k]] = False
    res = {
        "w2.5_base_coverage": base_cov, "w1_target_coverage": target_cov,
        "n_pruned_to_match": best_k, "matched_coverage": matched_cov,
        "w1_frozen_tstr": frozen_tstr(d, "w1"), "w2.5_frozen_tstr": frozen_tstr(d, "w2.5"),
        "tstr_w2.5_cov_matched": retrain_n(imgs[keep], labels[keep], test_loader, device),
        "tstr_w2.5_rand_pruned": retrain_n(imgs[keep_rand], labels[keep_rand], test_loader, device),
    }
    print(f"[C3] w2.5 cov {base_cov:.3f}→{matched_cov:.3f} (target w1 {target_cov:.3f}, pruned {best_k}); "
          f"matched={res['tstr_w2.5_cov_matched']} rand={res['tstr_w2.5_rand_pruned']} "
          f"w1={res['w1_frozen_tstr']} w2.5={res['w2.5_frozen_tstr']}", flush=True)
    return res


def c5_purity_filter(test_loader, device):
    imgs, labels, margins, _ = load_cell("w1.5")
    order = torch.argsort(margins)                                   # 低 margin（高 near-bnd 純度）→ 高
    half = imgs.size(0) // 2
    high_purity = order[:half]                                       # 低 margin 前 50%
    low_purity = order[half:]                                        # 高 margin 後 50%
    res = {
        "n_total": imgs.size(0),
        "tstr_original": retrain_n(imgs, labels, test_loader, device),
        "tstr_high_purity_low_margin": retrain_n(imgs[high_purity], labels[high_purity], test_loader, device),
        "tstr_low_purity_high_margin": retrain_n(imgs[low_purity], labels[low_purity], test_loader, device),
    }
    print(f"[C5] w1.5: orig={res['tstr_original']} high_purity={res['tstr_high_purity_low_margin']} "
          f"low_purity={res['tstr_low_purity_high_margin']}", flush=True)
    return res


def main():
    argparse.ArgumentParser(description="C2/C3/C5 介入式證據。").parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = json.load(open(CONF, encoding="utf-8"))
    test_loader = build_test_loader("cifar10")
    real_imgs, real_labels = load_real_per_class("cifar10", d["metadata"]["real_per_class"], seed=0)
    real_dino = dinov2_features((real_imgs + 1) / 2, device)
    print(f"device={device} seed={SEED}", flush=True)

    out = {"c2_boundary_pruning": c2_boundary_pruning(d, test_loader, device),
           "c3_coverage_matched": c3_coverage_matched(d, real_dino, real_labels, test_loader, device),
           "c5_purity_filter": c5_purity_filter(test_loader, device),
           "note": "exploratory（C0）；gen/特徵/margin 取自 P1 落盤禁重生成；每情境 2 重訓 N 死；"
                   "因果措辭禁；判定於各 C record 對號入座事前判定。"}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
