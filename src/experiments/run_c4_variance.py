"""C4：變異分解（σ_cls 對 σ_gen），交付 D4 功效規劃。依凍結規則 R-2026-07-06-05 §6 C4、-09-04 §2.2。

目的：分離「分類器訓練變異」σ_cls（同一合成集、不同訓練 RNG）與「生成變異」σ_gen（不同 gen seed），
供 D4 決策單之功效規劃——若 CIFAR-100 σ_gen 同量級，3 seeds 定不住峰位 ±1 格，登記主張須為峰位噪聲
穩健形式（高原＋懸崖，不押點峰）。exploratory（C0）：規則先於計算、不因結果回改。

凍結設計（N 死）：
- cells：w ∈ {1, 1.5, 2, 2.5} × seed ∈ {10, 11, 12} = 12 cells（低中段，涵蓋上升肢與峰）。
- 每 cell 固定 +2 新 TSTR 重訓（fresh ResNet18、未種子化 shuffle＝σ_cls 來源，協定 dossier 甲-8）。
- 混池規則：P1 全逐位（R-2026-07-09-01）→ 凍結 confirmatory TSTR 計為第 3 replicate（同一逐位重現之
  合成集之一次抽樣），故 3 replicate/cell、within df = 12×2 = 24。（若 P1 非逐位則改 2/cell、12df，取嚴——
  本例 P1 逐位，用 3/cell。）
- 預期結論事前寫死：上升肢（w1→w1.5，+0.80pp、SE 1.88）維持 unresolved；record 不得因結果回改。

變異分解（2 層 nested random effects，平衡設計）：
  TSTR[w,seed,rep] = μ_w + gen[w,seed] + cls[w,seed,rep]，gen~N(0,σ_gen²)、cls~N(0,σ_cls²)。
  σ_cls² = MS_within（rep 內），df=24；σ_gen² = (MS_seed − MS_within)/n_rep，n_rep=3。

gen 用 P1 落盤影像（results/p1_assets），禁二次重生成。

Usage:
    uv run python run_c4_variance.py
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py
import argparse
import json
import os

import torch

from cifar_classifier import run_tstr
from datasets.cifar import build_test_loader

ASSET = "results/p1_assets"
CONF = "results/cifar10_cfg_confirmatory.json"
OUT = "results/cifar10_c4_variance.json"
WS = ["w1", "w1.5", "w2", "w2.5"]      # 低中段（凍結）
SEEDS = [10, 11, 12]
N_NEW_RETRAIN = 2                        # 每 cell +2（N 死）
TSTR_EPOCHS = 15                         # 同 confirmatory
NUM_CLASSES = 10


def load_gen(seed, name):
    """P1 落盤影像 uint8[0,255] → [-1,1]（run_tstr 期望值域）＋ labels。"""
    u8 = torch.load(f"{ASSET}/seed{seed}_{name}/img_uint8.pt", map_location="cpu", weights_only=True)
    labels = torch.load(f"{ASSET}/seed{seed}_{name}/labels.pt", map_location="cpu", weights_only=True)
    imgs = u8.float() / 255.0 * 2.0 - 1.0
    return imgs, labels


def variance_components(cells):
    """cells: {(w,seed): [rep TSTR...]}；回傳 σ_cls、σ_gen 與 ANOVA 中間量。平衡設計。"""
    ws = WS; seeds = SEEDS
    n_rep = len(next(iter(cells.values())))
    # within（rep 內）：SS_within、df=Σ(n_rep-1)
    ss_within = 0.0
    df_within = 0
    cell_means = {}
    for key, vals in cells.items():
        m = sum(vals) / len(vals)
        cell_means[key] = m
        ss_within += sum((v - m) ** 2 for v in vals)
        df_within += len(vals) - 1
    ms_within = ss_within / df_within                      # = σ_cls²

    # between-seed within w：SS_seed、df=Σ_w(n_seed-1)
    ss_seed = 0.0
    df_seed = 0
    for w in ws:
        seed_means = [cell_means[(w, s)] for s in seeds]
        mw = sum(seed_means) / len(seed_means)
        ss_seed += n_rep * sum((sm - mw) ** 2 for sm in seed_means)
        df_seed += len(seeds) - 1
    ms_seed = ss_seed / df_seed
    sigma_gen_sq = max(0.0, (ms_seed - ms_within) / n_rep)

    return {
        "n_rep_per_cell": n_rep, "df_within": df_within, "df_between_seed": df_seed,
        "ms_within": ms_within, "ms_between_seed": ms_seed,
        "sigma_cls": ms_within ** 0.5, "sigma_gen": sigma_gen_sq ** 0.5,
        "sigma_cls_sq": ms_within, "sigma_gen_sq": sigma_gen_sq,
    }


def main():
    argparse.ArgumentParser(description="C4 變異分解 σ_cls/σ_gen。").parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} cells={len(WS)}×{len(SEEDS)}, +{N_NEW_RETRAIN} 重訓/cell", flush=True)

    d = json.load(open(CONF, encoding="utf-8"))
    test_loader = build_test_loader("cifar10")

    # 凍結 confirmatory TSTR（第 3 replicate）
    frozen = {}
    for seed in SEEDS:
        sb = next(s for s in d["per_seed"] if s["seed"] == seed)
        for name in WS:
            frozen[(name, seed)] = next(c for c in sb["configs"] if c["name"] == name)["tstr"]

    cells = {}
    detail = []
    for name in WS:
        for seed in SEEDS:
            imgs, labels = load_gen(seed, name)
            reps = [frozen[(name, seed)]]                  # 第 3 replicate（凍結）
            for r in range(N_NEW_RETRAIN):
                overall, _ = run_tstr(imgs, labels, test_loader, device,
                                      num_classes=NUM_CLASSES, epochs=TSTR_EPOCHS)
                reps.append(overall)
                print(f"  {name} seed{seed} new-retrain {r+1}: TSTR={overall:.2f}", flush=True)
            cells[(name, seed)] = reps
            detail.append({"w": name, "seed": seed, "reps": reps,
                           "frozen_is_rep0": True, "cell_mean": sum(reps) / len(reps)})
            if device.type == "cuda":
                torch.cuda.empty_cache()

    vc = variance_components(cells)
    # per-w 之 seed 級離散（供上升肢 unresolved 佐證）
    per_w = {}
    for name in WS:
        seed_means = [sum(cells[(name, s)]) / len(cells[(name, s)]) for s in SEEDS]
        mu = sum(seed_means) / len(seed_means)
        per_w[name] = {"seed_cell_means": seed_means, "w_mean": mu}

    out = {"cells": [{"w": k[0], "seed": k[1], "reps": v} for k, v in cells.items()],
           "detail": detail, "per_w": per_w, "variance": vc,
           "note": "exploratory（C0）；gen 用 P1 落盤影像禁重生成；凍結 TSTR 為第 3 replicate（P1 逐位）；"
                   "上升肢維持 unresolved 為事前寫死之預期，record 不因結果回改；σ_cls/σ_gen 餵 D4 決策單。"}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nσ_cls={vc['sigma_cls']:.3f}  σ_gen={vc['sigma_gen']:.3f}  "
          f"(ms_within={vc['ms_within']:.3f}, ms_seed={vc['ms_between_seed']:.3f})", flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
