"""C7：small-probe FID 排序穩定性（依凍結規則段 records/2026-07-06-15）。

測 clean-fid 對 guidance 組態之排序在小 probe（少量真實參考）下是否穩定，餵 D5 之 matched-probe FID-min
baseline。exploratory（C0）：規則先於計算、不因結果回改。

凍結規則（records/2026-07-06-15）：
- 真實參考子抽樣至 {100, 250, 500}/class，每尺寸 5 次重抽（seed 事前定 = draw 0..4），重算各組態 clean-fid、得排序。
- 穩定性度量：小 probe 排序對全參考排序之 Kendall τ；並報 FID-argmin 是否隨 probe 尺寸改變。
- 判定（事前）：τ 高且 FID-argmin 穩定 → FID-min baseline 於小 probe 可靠（餵 D5）；否則 → D5 之 matched-probe
  baseline 須報其 probe 敏感度。

實作：gen 用 P1 落盤影像（results/p1_assets/seed{S}_{name}/img_uint8.pt，seed 10，10 組態），禁二次重生成。
clean-fid 走 cleanfid 特徵層 API（build_feature_extractor + get_folder_features + fid_from_feats）：gen 特徵每
組態抽一次快取、probe 每 draw 抽小集，兩者皆走 clean 模式（與 char_clean_fid 同表徵）。全參考排序 = 凍結
JSON 之 char_clean_fid（gen vs 全 CIFAR train）。probe 抽樣 = load_real_per_class(size, seed=draw, train=True)。

Usage:
    uv run python run_c7_probe_fid_stability.py
"""
import argparse
import json
import os
import tempfile

import torch
from torchvision.utils import save_image

from datasets.cifar import load_real_per_class
from fid_clean import _patch_scipy_sqrtm

ASSET = "results/p1_assets"
CONF = "results/cifar10_cfg_confirmatory.json"
OUT = "results/cifar10_c7_probe_fid_stability.json"
SEED = 10                      # 用 seed 10 之落盤影像作 10 組態排序（單 seed 排序穩定性；D5 餵 matched-probe）
PROBE_SIZES = [100, 250, 500]  # /class（凍結）
DRAWS = [0, 1, 2, 3, 4]        # 每尺寸 5 次、seed 事前定（凍結）


def kendall_tau(a, b):
    """Kendall τ-a（含平手以 sign 計）：兩排序向量之一致對比例。"""
    n = len(a)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    tot = n * (n - 1) / 2
    return (conc - disc) / tot if tot else float("nan")


def dump_and_features(imgs01, model, tmp_root):
    """imgs01 (N,3,32,32) in [0,1] → PNG → cleanfid clean 特徵。回傳 np 特徵陣列。"""
    from cleanfid import fid as cfid
    d = tempfile.mkdtemp(prefix="c7_", dir=tmp_root)
    try:
        for i in range(imgs01.size(0)):
            save_image(imgs01[i], os.path.join(d, f"{i:06d}.png"))
        return cfid.get_folder_features(d, model=model, mode="clean", num_workers=0,
                                        batch_size=256, verbose=False)
    finally:
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
        os.rmdir(d)


def main():
    p = argparse.ArgumentParser(description="C7 small-probe FID 排序穩定性。")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    from cleanfid import fid as cfid
    _patch_scipy_sqrtm()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cfid.build_feature_extractor("clean", device)
    tmp_root = tempfile.mkdtemp(prefix="c7root_")

    d = json.load(open(CONF, encoding="utf-8"))
    seed_block = next(s for s in d["per_seed"] if s["seed"] == args.seed)
    configs = seed_block["configs"]                 # 存檔順序 = grid 順序
    names = [c["name"] for c in configs]
    ws = [c["guidance"] for c in configs]
    full_fid = [c["char_clean_fid"] for c in configs]   # 全參考 clean-fid（gen vs 全 train），凍結 JSON
    full_rank_argmin_w = ws[min(range(len(ws)), key=lambda i: full_fid[i])]
    print(f"device={device} seed={args.seed} full-ref char_clean_fid argmin at w={full_rank_argmin_w}", flush=True)

    # gen 特徵：每組態抽一次（P1 落盤影像 uint8 → [0,1]）
    gen_feats = []
    for name in names:
        u8 = torch.load(f"{ASSET}/seed{args.seed}_{name}/img_uint8.pt", map_location="cpu", weights_only=True)
        feats = dump_and_features(u8.float() / 255.0, model, tmp_root)
        gen_feats.append(feats)
        print(f"  gen feat {name}: {feats.shape}", flush=True)

    # 逐 probe 尺寸 × draw：抽小 probe → 特徵 → 各組態 FID → 排序 → Kendall τ vs 全參考
    results = []
    for s in PROBE_SIZES:
        for draw in DRAWS:
            real_imgs, _ = load_real_per_class("cifar10", s, seed=draw, train=True)  # [-1,1]
            probe_feats = dump_and_features((real_imgs + 1) / 2, model, tmp_root)
            fids = [float(cfid.fid_from_feats(gf, probe_feats)) for gf in gen_feats]
            argmin_w = ws[min(range(len(fids)), key=lambda i: fids[i])]
            tau = kendall_tau([-x for x in full_fid], [-x for x in fids])  # 兩者同向（越小越好）故取負後同號
            results.append({"probe_per_class": s, "draw": draw, "fids": fids,
                            "argmin_w": argmin_w, "kendall_tau_vs_full": tau,
                            "argmin_matches_full": argmin_w == full_rank_argmin_w})
            print(f"  probe {s}/class draw {draw}: argmin w={argmin_w} (full={full_rank_argmin_w}) "
                  f"kendall_tau={tau:.3f}", flush=True)

    # 彙總（描述性，不下 D5 判定——判定於 δ 決策單）
    by_size = {}
    for s in PROBE_SIZES:
        rs = [r for r in results if r["probe_per_class"] == s]
        taus = [r["kendall_tau_vs_full"] for r in rs]
        argmin_stable = sum(r["argmin_matches_full"] for r in rs)
        by_size[str(s)] = {"mean_kendall_tau": sum(taus) / len(taus),
                           "min_kendall_tau": min(taus),
                           "argmin_matches_full_count": f"{argmin_stable}/{len(rs)}",
                           "argmin_ws": [r["argmin_w"] for r in rs]}

    out = {"seed": args.seed, "grid": ws, "full_ref_char_clean_fid": full_fid,
           "full_ref_argmin_w": full_rank_argmin_w,
           "probe_sizes": PROBE_SIZES, "draws": DRAWS,
           "per_probe": results, "summary_by_size": by_size,
           "note": "exploratory（C0）；clean-fid 特徵層（cleanfid clean 模式）；gen 用 P1 落盤影像禁重生成；"
                   "D5 判定（τ 高且 argmin 穩定 → FID-min 小 probe 可靠）於 δ 決策單，本檔僅產出數字。"}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    for f in os.listdir(tmp_root):
        os.remove(os.path.join(tmp_root, f)) if os.path.isfile(os.path.join(tmp_root, f)) else None
    os.rmdir(tmp_root) if os.path.isdir(tmp_root) and not os.listdir(tmp_root) else None
    print(f"Wrote {OUT}", flush=True)
    print(json.dumps(by_size, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
