"""Stage 3：自訓 CFG CIFAR-10 的 1-seed 寬 grid scout。

固定量測 sampler (steps=50, eta=0)，guidance 掃寬（w∈{1,1.5,2,3,5,8}），每個 w 量四項：
  - precision + coverage（metrics_prdc，DINOv2 為主特徵、Inception 交叉檢查；本次務必保留 precision）
  - TSTR（cifar_classifier.run_tstr，from-scratch ResNet）
  - 標籤噪音 + near-boundary（Stage 2 的真實 CIFAR judge + 校準 threshold 0.9525）

目的：(a) 確認 EDM proxy 上的 w≈1.5 甜蜜點是否遷移到自訓模型；(b) 定位 coverage 真正開始崩的 w；
(c) 驗 judge/threshold。scout 後據此一次定死最終 grid、回填協定增修（2026-07-05-03）。

Usage:
    uv run python run_cifar_cfg_scout.py --quick
    uv run python run_cifar_cfg_scout.py --guidance 1 1.5 2 3 5 8 --per-class 500
"""

import argparse
import json
import os

import torch

from cifar_cfg_sample import load_cfg_model, generate_balanced
from cifar_classifier import ResNet18, run_tstr
from mechanism import compute_margins, near_boundary_fraction
from metrics_prdc import compute_prdc_per_class
from metrics_features import dinov2_features
from datasets.cifar import load_real_per_class, build_test_loader


def _inception_crosscheck(real_imgs, real_labels, gen_imgs, gen_labels, nearest_k, device):
    """以 Inception 特徵重算 per-class coverage 作交叉檢查（失敗則回傳 None，不中斷 scout）。"""
    try:
        import pickle
        from run_cifar_selector import inception_feats
        from phase1_edm_repro import DETECTOR_URL
        import dnnlib
        with dnnlib.util.open_url(DETECTOR_URL, verbose=False) as f:
            detector = pickle.load(f).to(device)
        rf = inception_feats(real_imgs, detector, device)
        gf = inception_feats(gen_imgs, detector, device)
        prdc, _ = compute_prdc_per_class(rf.to(device), real_labels.to(device),
                                         gf.to(device), gen_labels.to(device),
                                         nearest_k=nearest_k, num_classes=10)
        return prdc["coverage"], prdc["precision"]
    except Exception as e:
        print(f"  [warn] Inception 交叉檢查略過：{e}", flush=True)
        return None, None


def verdict(rows):
    gs = [r["guidance"] for r in rows]
    cov = [r["coverage_dino"] for r in rows]
    tstr = [r["tstr"] for r in rows]
    sweet = gs[max(range(len(tstr)), key=lambda i: tstr[i])]
    cov_max = max(cov)
    # coverage 崩點：第一個 coverage 掉到峰值 90% 以下的 guidance
    collapse = next((gs[i] for i in range(len(cov)) if cov[i] < 0.9 * cov_max), None)
    return {"sweet_spot_guidance": sweet, "coverage_peak": cov_max,
            "coverage_collapse_guidance": collapse,
            "sweet_spot_matches_proxy_1p5": abs(sweet - 1.5) < 1e-6}


def main():
    p = argparse.ArgumentParser(description="Self-trained CFG CIFAR-10 wide-grid scout (1 seed).")
    p.add_argument("--ckpt", default="checkpoints/cifar10_cfg.pt")
    p.add_argument("--judge", default="checkpoints/cifar10_judge.pt")
    p.add_argument("--guidance", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0, 5.0, 8.0])
    p.add_argument("--per-class", type=int, default=500)
    p.add_argument("--real-per-class", type=int, default=500)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--batch", type=int, default=250)
    p.add_argument("--nearest-k", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.9525, help="CIFAR near-boundary threshold（Stage 2 校準）")
    p.add_argument("--tstr-epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-inception", action="store_true", help="略過 Inception 交叉檢查")
    p.add_argument("--output", default="results/cifar10_cfg_scout.json")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.guidance = [1.0, 8.0]
        args.per_class = 32
        args.real_per_class = 64
        args.tstr_epochs = 2
        args.no_inception = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(f"guidance={args.guidance} per_class={args.per_class} steps={args.steps} eta={args.eta} "
          f"tstr_epochs={args.tstr_epochs} threshold={args.threshold}", flush=True)
    os.makedirs("results", exist_ok=True)

    model, schedule, hp = load_cfg_model(args.ckpt, device)
    judge = ResNet18(num_classes=10).to(device)
    judge.load_state_dict(torch.load(args.judge, map_location=device, weights_only=True))
    judge.eval()
    print(f"Loaded CFG model (epoch {hp.get('epoch')}) and judge", flush=True)

    real_imgs, real_labels = load_real_per_class("cifar10", args.real_per_class, seed=0)
    real_dino = dinov2_features((real_imgs + 1) / 2, device)
    test_loader = build_test_loader("cifar10", batch_size=256)
    print(f"Real ref: {real_imgs.size(0)} imgs", flush=True)

    rows = []
    for w in args.guidance:
        gen, gen_labels = generate_balanced(model, schedule, args.per_class, device,
                                            args.steps, args.eta, guidance=w,
                                            num_classes=10, batch=args.batch, seed=args.seed)
        gen_dino = dinov2_features((gen + 1) / 2, device)
        dino_prdc, _ = compute_prdc_per_class(real_dino.to(device), real_labels.to(device),
                                              gen_dino.to(device), gen_labels.to(device),
                                              nearest_k=args.nearest_k, num_classes=10)
        tstr_overall, _ = run_tstr(gen, gen_labels, test_loader, device, num_classes=10,
                                   epochs=args.tstr_epochs)
        margins, preds = compute_margins(judge, gen, device)
        nb = near_boundary_fraction(margins, args.threshold)
        label_noise = (preds != gen_labels).float().mean().item()

        incep_cov, incep_prec = (None, None)
        if not args.no_inception:
            incep_cov, incep_prec = _inception_crosscheck(real_imgs, real_labels, gen, gen_labels,
                                                          args.nearest_k, device)

        row = {"guidance": w, "precision_dino": dino_prdc["precision"],
               "coverage_dino": dino_prdc["coverage"], "coverage_inception": incep_cov,
               "precision_inception": incep_prec, "tstr": tstr_overall,
               "label_noise_frac": label_noise, "near_boundary_frac": nb}
        rows.append(row)
        print(f"  w={w:<4g} prec={dino_prdc['precision']:.3f} cov={dino_prdc['coverage']:.3f} "
              f"tstr={tstr_overall:.2f} label_noise={label_noise:.3f} near_boundary={nb:.3f}", flush=True)

    v = verdict(rows)
    print("\n" + "=" * 84)
    print("  自訓 CFG CIFAR-10 寬 grid scout（DINOv2 coverage）")
    print("=" * 84)
    print(f"  {'w':>4} {'precision':>10} {'coverage':>9} {'TSTR%':>8} {'label_noise':>12} {'near_bnd':>9}")
    for r in rows:
        print(f"  {r['guidance']:>4g} {r['precision_dino']:>10.3f} {r['coverage_dino']:>9.3f} "
              f"{r['tstr']:>8.2f} {r['label_noise_frac']:>12.3f} {r['near_boundary_frac']:>9.3f}")
    print("  " + "-" * 80)
    print(f"  甜蜜點 guidance = {v['sweet_spot_guidance']:g}（與 proxy 的 1.5 相符：{v['sweet_spot_matches_proxy_1p5']}）")
    print(f"  coverage 峰值 = {v['coverage_peak']:.3f}，崩點 guidance = {v['coverage_collapse_guidance']}")
    print("=" * 84)

    out = {"metadata": {"dataset": "cifar10", "axis": "CFG guidance (self-trained model)",
                        "steps": args.steps, "eta": args.eta, "guidance_grid": args.guidance,
                        "per_class": args.per_class, "tstr_epochs": args.tstr_epochs,
                        "near_boundary_threshold": args.threshold,
                        "coverage_feature": "DINOv2 primary + Inception cross-check"},
           "rows": rows, "verdict": v, "args": {k: val for k, val in vars(args).items()}}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
