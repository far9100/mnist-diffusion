"""自訓 CFG CIFAR-10 / CIFAR-100 的 1-seed 寬 grid scout（Stage 3 / D10 第三閘）。

固定量測 sampler (steps=50, eta=0)，guidance 掃寬（w∈{1,1.5,2,3,5,8}），每個 w 量四項：
  - precision + coverage（metrics_prdc，DINOv2 為主特徵、Inception 交叉檢查；務必保留 precision）
  - TSTR（cifar_classifier.run_tstr，from-scratch ResNet）
  - 標籤噪音 + near-boundary（該資料集的真實 judge + 校準 threshold）

目的：(a) 定位 coverage 真正開始崩的 w、找甜蜜點；(b) 驗 judge/threshold。scout 後據此一次定死
confirmatory grid、回填網格凍結 amendment。CIFAR-100 依 D10：scout 僅定網格，讀數不回饋判準。

Usage:
    uv run python run_cifar_cfg_scout.py --dataset cifar10 --guidance 1 1.5 2 3 5 8 --per-class 500
    uv run python run_cifar_cfg_scout.py --dataset cifar100 --per-class 200
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import torch

from cifar_cfg_sample import load_cfg_model, generate_balanced
from cifar_classifier import ResNet18, run_tstr
from mechanism import compute_margins, near_boundary_fraction
from metrics_prdc import compute_prdc_per_class
from metrics_features import dinov2_features
from datasets.cifar import load_real_per_class, build_test_loader, NUM_CLASSES


def load_inception_detector(device):
    """載入一次 Inception FID detector（供 coverage 的第二表徵交叉檢查）；失敗回傳 None。"""
    try:
        import pickle
        from phase1_edm_repro import DETECTOR_URL
        import dnnlib
        with dnnlib.util.open_url(DETECTOR_URL, verbose=False) as f:
            return pickle.load(f).to(device)
    except Exception as e:
        print(f"  [warn] 無法載入 Inception detector，交叉檢查略過：{e}", flush=True)
        return None


def inception_crosscheck(detector, real_imgs, real_labels, gen_imgs, gen_labels, nearest_k,
                         device, num_classes):
    """以已載入的 Inception detector 重算 per-class precision/coverage 作交叉檢查。"""
    from run_cifar_selector import inception_feats
    rf = inception_feats(real_imgs, detector, device)
    gf = inception_feats(gen_imgs, detector, device)
    prdc, _ = compute_prdc_per_class(rf.to(device), real_labels.to(device),
                                     gf.to(device), gen_labels.to(device),
                                     nearest_k=nearest_k, num_classes=num_classes)
    return prdc["coverage"], prdc["precision"]


def verdict(rows):
    """甜蜜點＝TSTR 峰位；崩點＝coverage 峰位之後首個跌破 90% 峰值的 w。

    崩點取「峰位後」是修正舊版對非單調曲線的誤判（舊版取第一個低於峰值 90% 者，會落在峰前，
    見 R-2026-07-05-06 更正）。
    """
    gs = [r["guidance"] for r in rows]
    cov = [r["coverage_dino"] for r in rows]
    tstr = [r["tstr"] for r in rows]
    sweet = gs[max(range(len(tstr)), key=lambda i: tstr[i])]
    peak_i = max(range(len(cov)), key=lambda i: cov[i])
    cov_max = cov[peak_i]
    collapse = next((gs[i] for i in range(peak_i + 1, len(cov)) if cov[i] < 0.9 * cov_max), None)
    return {"sweet_spot_guidance": sweet, "coverage_peak": cov_max,
            "coverage_peak_guidance": gs[peak_i],
            "coverage_collapse_guidance": collapse}


def resolve_threshold(dataset, judge_json):
    """未指定 --threshold 時，自 results/<dataset>_judge.json 讀該資料集校準好的 near-boundary threshold。"""
    if not os.path.exists(judge_json):
        raise FileNotFoundError(f"找不到 judge 校準檔 {judge_json}；請先跑 cifar_judge.py 或明確給 --threshold")
    with open(judge_json, encoding="utf-8") as f:
        return float(json.load(f)["near_boundary_threshold"])


def main():
    p = argparse.ArgumentParser(description="Self-trained CFG CIFAR-10/100 wide-grid scout (1 seed).")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--ckpt", default=None, help="預設 checkpoints/<dataset>_cfg.pt")
    p.add_argument("--judge", default=None, help="預設 checkpoints/<dataset>_judge.pt")
    p.add_argument("--guidance", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0, 5.0, 8.0])
    p.add_argument("--per-class", type=int, default=500)
    p.add_argument("--real-per-class", type=int, default=500)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--batch", type=int, default=250)
    p.add_argument("--nearest-k", type=int, default=5)
    p.add_argument("--threshold", type=float, default=None,
                   help="near-boundary threshold；未給則自 <dataset>_judge.json 讀")
    p.add_argument("--tstr-epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-inception", action="store_true", help="略過 Inception 交叉檢查")
    p.add_argument("--output", default=None, help="預設 results/<dataset>_cfg_scout.json")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    dataset = args.dataset
    num_classes = NUM_CLASSES[dataset]
    ckpt = args.ckpt or f"checkpoints/{dataset}_cfg.pt"
    judge_ckpt = args.judge or f"checkpoints/{dataset}_judge.pt"
    output = args.output or f"results/{dataset}_cfg_scout.json"
    if args.threshold is None:
        args.threshold = resolve_threshold(dataset, f"results/{dataset}_judge.json")

    if args.quick:
        args.guidance = [1.0, 8.0]
        args.per_class = 32
        args.real_per_class = 64
        args.tstr_epochs = 2

    start_ts = datetime.now(timezone.utc).isoformat()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  dataset={dataset} num_classes={num_classes}", flush=True)
    print(f"guidance={args.guidance} per_class={args.per_class} steps={args.steps} eta={args.eta} "
          f"tstr_epochs={args.tstr_epochs} threshold={args.threshold:.4f}", flush=True)
    os.makedirs("results", exist_ok=True)

    model, schedule, hp = load_cfg_model(ckpt, device)
    judge = ResNet18(num_classes=num_classes).to(device)
    judge.load_state_dict(torch.load(judge_ckpt, map_location=device, weights_only=True))
    judge.eval()
    print(f"Loaded CFG model (epoch {hp.get('epoch')}) and judge", flush=True)

    real_imgs, real_labels = load_real_per_class(dataset, args.real_per_class, seed=0)
    real_dino = dinov2_features((real_imgs + 1) / 2, device)
    # num_workers=0：Windows 下多 worker 在記憶體壓力時會崩，且 TensorDataset 無需多 worker。
    test_loader = build_test_loader(dataset, batch_size=256, num_workers=0)
    detector = None if args.no_inception else load_inception_detector(device)
    print(f"Real ref: {real_imgs.size(0)} imgs; inception 交叉檢查={'關' if detector is None else '開'}",
          flush=True)

    rows = []
    for w in args.guidance:
        gen, gen_labels = generate_balanced(model, schedule, args.per_class, device,
                                            args.steps, args.eta, guidance=w,
                                            num_classes=num_classes, batch=args.batch, seed=args.seed)
        gen_dino = dinov2_features((gen + 1) / 2, device)
        dino_prdc, _ = compute_prdc_per_class(real_dino.to(device), real_labels.to(device),
                                              gen_dino.to(device), gen_labels.to(device),
                                              nearest_k=args.nearest_k, num_classes=num_classes)
        tstr_overall, _ = run_tstr(gen, gen_labels, test_loader, device, num_classes=num_classes,
                                   epochs=args.tstr_epochs)
        margins, preds = compute_margins(judge, gen, device)
        nb = near_boundary_fraction(margins, args.threshold)
        label_noise = (preds != gen_labels).float().mean().item()

        incep_cov, incep_prec = (None, None)
        if detector is not None:
            incep_cov, incep_prec = inception_crosscheck(detector, real_imgs, real_labels,
                                                         gen, gen_labels, args.nearest_k, device,
                                                         num_classes)

        row = {"guidance": w, "precision_dino": dino_prdc["precision"],
               "coverage_dino": dino_prdc["coverage"], "coverage_inception": incep_cov,
               "precision_inception": incep_prec, "tstr": tstr_overall,
               "label_noise_frac": label_noise, "near_boundary_frac": nb}
        rows.append(row)
        print(f"  w={w:<4g} prec={dino_prdc['precision']:.3f} cov={dino_prdc['coverage']:.3f} "
              f"tstr={tstr_overall:.2f} label_noise={label_noise:.3f} near_boundary={nb:.3f}", flush=True)
        del gen, gen_dino, margins, preds
        if device.type == "cuda":
            torch.cuda.empty_cache()

    v = verdict(rows)
    print("\n" + "=" * 84)
    print(f"  自訓 CFG {dataset} 寬 grid scout（DINOv2 coverage）")
    print("=" * 84)
    print(f"  {'w':>4} {'precision':>10} {'coverage':>9} {'TSTR%':>8} {'label_noise':>12} {'near_bnd':>9}")
    for r in rows:
        print(f"  {r['guidance']:>4g} {r['precision_dino']:>10.3f} {r['coverage_dino']:>9.3f} "
              f"{r['tstr']:>8.2f} {r['label_noise_frac']:>12.3f} {r['near_boundary_frac']:>9.3f}")
    print("  " + "-" * 80)
    print(f"  甜蜜點 guidance = {v['sweet_spot_guidance']:g}")
    print(f"  coverage 峰值 = {v['coverage_peak']:.3f}（w={v['coverage_peak_guidance']:g}），"
          f"崩點 guidance = {v['coverage_collapse_guidance']}")
    print("=" * 84)

    out = {"metadata": {"dataset": dataset, "num_classes": num_classes,
                        "axis": "CFG guidance (self-trained model)",
                        "steps": args.steps, "eta": args.eta, "guidance_grid": args.guidance,
                        "per_class": args.per_class, "tstr_epochs": args.tstr_epochs,
                        "near_boundary_threshold": args.threshold,
                        "coverage_feature": "DINOv2 primary + Inception cross-check",
                        "start_timestamp": start_ts,
                        "end_timestamp": datetime.now(timezone.utc).isoformat(),
                        "argv": sys.argv,
                        "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                                "cudnn": torch.backends.cudnn.version()}},
           "rows": rows, "verdict": v, "args": {k: val for k, val in vars(args).items()}}
    with open(output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {output}", flush=True)


if __name__ == "__main__":
    main()
