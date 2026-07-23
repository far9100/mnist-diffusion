"""T5b（MNIST 側）：w<1 網格延伸 scout（對應審查 A8）。

問題設定（給第一次讀的研究生）：MNIST sandbox 的 guidance 網格為 {1,2,3,5,7,10}，下游效用（TSTR）的
oracle 落在網格下界 g1。這引出一個和 CIFAR-100 完全相同的疑慮：g1 之所以是 oracle，會不會只是「網格
恰好從 1 開始」的邊界假影？真正的 TSTR 峰是否其實在 w<1？本檔把 MNIST 的網格往下延伸到 g∈{0.5,0.75}，
量這兩個 sub-unity 組態的 TSTR，與凍結的 g1（取自 `selector_signal_multiseed.json`）並列判讀。

判讀目標（與 CIFAR-100 subunity scout 同構，見 `run_cifar_cfg_multiseed.py --off-protocol`）：
  - 若 TSTR(g0.5) < TSTR(g0.75) < TSTR(g1)：g1 為真實內部峰、非邊界效應。
  - 若某個 w<1 的 TSTR 反而 >= g1：峰其實在網格外，g1 只是邊界點。

為何 w<1 能跑：`ddpm.predict_eps` 的 CFG 判斷已由 `guidance_scale > 1.0` 擴充為 `!= 1.0`
（見 CHANGELOG 2026-07-21-08），故 w<1 會走 CFG 分支、往無條件方向內插（降低類別條件銳化）。
w==1 仍走純條件分支、與凍結資料逐位不變，本 scout 不重算 g1。

流程（逐 seed，沿 `run_selector_signal.py` 的 MNIST 協定：DDIM steps=50 eta=0、per_digit 1000、
judge-CNN 特徵 PRDC、TSTR epochs=20）：
  1. 由 `run_mnist_fid_arm.load_or_generate` 生成／載入 g∈{0.5,0.75} 的影像（決定性子種子
     `seed*1000+int(round(g*10))`，快取於 `results/mnist_gen_cache/`，與 T1a/T1b 共用同一批影像）。
  2. judge-CNN penultimate 特徵算 per-class PRDC（precision/coverage/recall，k=5、10 類）。
  3. 在生成影像上訓練全新 CNN、在真實 MNIST 測試集評估 → TSTR。
  4. 附上凍結 g1 的 per-seed TSTR 作參考峰。

本 scout 為 exploratory、off_protocol；不改任何凍結結果，寫新檔 `results/mnist_subunity_scout.json`。

Usage:
    uv run python src/experiments/run_mnist_subunity_scout.py                 # seeds 0 1 2, per_digit 1000
    uv run python src/experiments/run_mnist_subunity_scout.py --quick         # 冒煙：小規模
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py

import argparse
import json
import sys
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from ddpm import UNet, DiffusionSchedule
from fid import load_cnn
from analyze_distribution import extract_features, load_real_per_class
from metrics_prdc import compute_prdc_per_class
from evaluate import MNISTClassifier, train_classifier, evaluate, build_dataloaders
# 共用 T1a 的生成／快取，確保 subunity 影像與 FID-min／DINOv2 臂用同一套生成路徑
from run_mnist_fid_arm import load_or_generate, cfg_subseed, set_seed

SUBUNITY_GRID = [0.5, 0.75]        # 網格往下延伸的 sub-unity 點
REFERENCE = "g1"                    # 凍結參考峰（網格下界）
SAMPLER, STEPS, ETA = "ddim", 50, 0.0
CACHE_DIR = "results/mnist_gen_cache"


def summarize(values):
    """對一串純量計算 mean/std/min/max（忽略 None）；與 run_selector_signal 同型。"""
    vals = [float(v) for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return None
    m = sum(vals) / n
    std = (sum((v - m) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return {"mean": m, "std": std, "min": min(vals), "max": max(vals), "n": n, "values": vals}


def run_tstr(images, labels, real_test_loader, device, epochs, lr, batch_size):
    """純粹在 `images` 上訓練一個全新 CNN、在真實 MNIST 測試集評估（同 run_selector_signal 協定）。"""
    clf = MNISTClassifier().to(device)
    loader = DataLoader(TensorDataset(images, labels), batch_size=batch_size,
                        shuffle=True, num_workers=0)
    train_classifier(clf, loader, device, epochs, lr)
    acc, _, _ = evaluate(clf, real_test_loader, device)
    return acc


def analyse_seed(seed, ref_tstr, args, device, model, schedule, judge, real_test_loader):
    """單一 seed：對 g∈{0.5,0.75} 算 PRDC + TSTR，並附凍結 g1 參考峰。"""
    set_seed(seed)
    real_images, real_labels = load_real_per_class(args.data_dir, args.fid_per_class, seed)
    real_feats = extract_features(judge, real_images, device).to(device)
    real_labels_d = real_labels.to(device)

    configs = []
    for g in SUBUNITY_GRID:
        name = f"g{g:g}"
        gen_imgs, gen_labels = load_or_generate(seed, g, args.per_digit, args.batch_size,
                                                device, model, schedule, CACHE_DIR)
        gen_feats = extract_features(judge, gen_imgs, device).to(device)
        prdc, _ = compute_prdc_per_class(real_feats, real_labels_d,
                                         gen_feats, gen_labels.to(device), nearest_k=args.nearest_k)
        # TSTR：以決定性子種子播種，使 scout 可重現（cuDNN 殘留非決定性除外）。
        set_seed(cfg_subseed(seed, g))
        tstr = run_tstr(gen_imgs, gen_labels, real_test_loader, device,
                        args.tstr_epochs, args.tstr_lr, args.tstr_batch_size)
        configs.append({"name": name, "guidance": g,
                        "precision": prdc["precision"], "coverage": prdc["coverage"],
                        "recall": prdc["recall"], "density": prdc["density"], "tstr": tstr})
        print(f"  seed {seed} {name:>5}: prec={prdc['precision']:.4f} cov={prdc['coverage']:.4f}"
              f" recall={prdc['recall']:.4f} tstr={tstr:.2f}")
    # 凍結參考峰 g1（不重算）
    configs.append({"name": REFERENCE, "guidance": 1.0, "tstr": ref_tstr, "frozen_reference": True})
    print(f"  seed {seed} {REFERENCE:>5}: tstr={ref_tstr} (凍結參考，取自 selector_signal_multiseed)")
    return {"seed": seed, "configs": configs}


def main():
    p = argparse.ArgumentParser(description="MNIST w<1 網格延伸 scout（T5b MNIST 側，A8）。")
    p.add_argument("--checkpoint", default="ddpm_mnist.pt")
    p.add_argument("--cnn", default="mnist_cnn.pt")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--per-digit", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--fid-per-class", type=int, default=1000)
    p.add_argument("--nearest-k", type=int, default=5)
    p.add_argument("--tstr-epochs", type=int, default=20)
    p.add_argument("--tstr-lr", type=float, default=1e-3)
    p.add_argument("--tstr-batch-size", type=int, default=64)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--selector-signal", default="results/selector_signal_multiseed.json",
                   help="凍結 g1 TSTR 參考峰來源")
    p.add_argument("--output", default="results/mnist_subunity_scout.json")
    p.add_argument("--quick", action="store_true", help="冒煙：per_digit 60、seeds 0、epochs 2")
    args = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if args.quick:
        args.per_digit, args.seeds, args.fid_per_class, args.tstr_epochs = 60, [0], 200, 2
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} seeds={args.seeds} subunity_grid={SUBUNITY_GRID} "
          f"per_digit={args.per_digit} tstr_epochs={args.tstr_epochs}")

    model = UNet(in_channels=1, base_channels=64, num_classes=10).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
    model.eval()
    schedule = DiffusionSchedule(timesteps=1000, device=device).to(device)
    judge = load_cnn(args.cnn, device)
    _, real_test_loader = build_dataloaders(args.data_dir, args.tstr_batch_size, 2)

    # 凍結 g1 的 per-seed TSTR 參考峰
    with open(args.selector_signal, encoding="utf-8") as f:
        sig = json.load(f)
    g1_by_seed = {b["seed"]: next((c.get("tstr") for c in b["configs"] if c["name"] == REFERENCE), None)
                  for b in sig["per_seed"]}

    per_seed = [analyse_seed(seed, g1_by_seed.get(seed), args, device, model, schedule,
                             judge, real_test_loader)
                for seed in args.seeds]

    # 彙總：per_config TSTR，並判「峰是否在 w1（w<1 是否皆低於 g1）」
    names = [f"g{g:g}" for g in SUBUNITY_GRID] + [REFERENCE]
    per_config = []
    for name in names:
        tstr = [next((c["tstr"] for c in sr["configs"] if c["name"] == name), None) for sr in per_seed]
        prec = [next((c.get("precision") for c in sr["configs"] if c["name"] == name), None) for sr in per_seed]
        cov = [next((c.get("coverage") for c in sr["configs"] if c["name"] == name), None) for sr in per_seed]
        per_config.append({"name": name, "tstr": summarize(tstr),
                           "precision": summarize(prec), "coverage": summarize(cov)})
    tstr_by_name = {pc["name"]: (pc["tstr"]["mean"] if pc["tstr"] else None) for pc in per_config}
    # 主判準（穩健）：每個 w<1 組態逐 seed 皆低於同 seed 的 g1 → 真實峰不在網格外、g1 非邊界假影。
    all_subunity_below_w1 = all(
        (lambda cs: cs.get("g1") is not None
         and all(cs.get(f"g{g:g}") is not None and cs[f"g{g:g}"] < cs["g1"] for g in SUBUNITY_GRID))(
            {c["name"]: c["tstr"] for c in sr["configs"]})
        for sr in per_seed)
    # 次判準：單調上升 g0.5 < g0.75 < g1（與 CIFAR-100 同型；較嚴、對雜訊敏感）。
    monotone_up_to_w1 = all(
        (lambda cs: cs.get("g0.5") is not None and cs.get("g0.75") is not None and cs.get("g1") is not None
         and cs["g0.5"] < cs["g0.75"] < cs["g1"])(
            {c["name"]: c["tstr"] for c in sr["configs"]})
        for sr in per_seed)
    aggregate = {"n_seeds": len(per_seed), "per_config": per_config,
                 "tstr_mean_by_name": tstr_by_name,
                 "all_subunity_below_w1_all_seeds": all_subunity_below_w1,
                 "monotone_up_to_w1_all_seeds": monotone_up_to_w1}

    print("\n" + "=" * 78)
    print("  T5b MNIST w<1 scout：TSTR 是否在 w<1 反升？（判 g1 是否為真實內部峰）")
    print("=" * 78)
    for pc in per_config:
        if pc["tstr"]:
            t = pc["tstr"]
            print(f"  {pc['name']:>5}: TSTR 平均={t['mean']:.2f} per_seed={[round(v,2) for v in t['values']]}")
    print(f"  w<1 逐 seed 皆低於 g1（主判準）: {all_subunity_below_w1}"
          f" | 單調 g0.5<g0.75<g1（次判準）: {monotone_up_to_w1}")
    print("=" * 78)

    out = {"metadata": {
        "analysis": "mnist_subunity_scout", "status": "exploratory", "off_protocol": True,
        "dataset": "mnist", "axis": "CFG guidance (self-trained DDPM)",
        "subunity_grid": SUBUNITY_GRID, "reference_config": REFERENCE,
        "sampler": SAMPLER, "steps": STEPS, "eta": ETA, "seeds": args.seeds,
        "per_digit": args.per_digit, "fid_per_class": args.fid_per_class,
        "nearest_k": args.nearest_k, "tstr_epochs": args.tstr_epochs, "tstr_lr": args.tstr_lr,
        "tstr_batch_size": args.tstr_batch_size,
        "prdc_feature_space": "mnist_cnn.pt penultimate (judge CNN)",
        "cfg_condition": "ddpm.predict_eps guidance_scale != 1.0 (w<1 走 CFG 分支，見 CHANGELOG 2026-07-21-08)",
        "reference_source": args.selector_signal, "gen_cache": CACHE_DIR,
        "gen_subseed_formula": "seed*1000+int(round(g*10))",
        "tstr_seed_formula": "same subseed before each TSTR (scout 可重現)",
        "start_timestamp": start_timestamp, "end_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": " ".join(sys.argv),
        "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version()}},
        "aggregate": aggregate, "per_seed": per_seed}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
