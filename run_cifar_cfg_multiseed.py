"""Stage 4：自訓 CFG CIFAR-10 多 seed 全量（confirmatory 主結果）。

在 Stage 3 定死的 grid（w∈{1,2,3,4,5,8}，固定 steps=50 eta=0）上跑 ≥3 seed，每個 (seed, w) 量
precision + coverage（DINOv2 主、Inception 交叉）+ TSTR + 標籤噪音/near-boundary。跨 seed 彙總帶
信賴區間；並以 CaF（argmax coverage s.t. precision ≥ tau，tau 自 real-vs-real 參考自動決定）在完整
網格（不剪枝）計 regret@selected / rank / top-k，作為 go/no-go 主指標（協定 §6、§8）。

Usage:
    uv run python run_cifar_cfg_multiseed.py --quick
    uv run python run_cifar_cfg_multiseed.py --guidance 1 2 3 4 5 8 --seeds 0 1 2 --per-class 500
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
from run_cifar_cfg_scout import load_inception_detector, inception_crosscheck
from run_cifar_selector import summarize
from selector import select_and_report


def real_ref_precision(real_dino, real_labels, nearest_k, device, seed):
    """real-vs-real 參考 precision（DINOv2 空間），供 CaF 的免 TSTR tau。"""
    n = real_dino.size(0)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    a, b = perm[:n // 2], perm[n // 2:]
    ref, _ = compute_prdc_per_class(real_dino[b].to(device), real_labels[b].to(device),
                                    real_dino[a].to(device), real_labels[a].to(device),
                                    nearest_k=nearest_k, num_classes=10)
    return ref["precision"]


def measure(model, schedule, judge, real_imgs, real_dino, real_labels, test_loader,
            detector, w, args, device, seed):
    gseed = seed * 10_000_000 + int(w * 1000) * 10_000
    gen, gen_labels = generate_balanced(model, schedule, args.per_class, device, args.steps,
                                        args.eta, guidance=w, num_classes=10, batch=args.batch, seed=gseed)
    gen_dino = dinov2_features((gen + 1) / 2, device)
    dino_prdc, _ = compute_prdc_per_class(real_dino.to(device), real_labels.to(device),
                                          gen_dino.to(device), gen_labels.to(device),
                                          nearest_k=args.nearest_k, num_classes=10)
    tstr, _ = run_tstr(gen, gen_labels, test_loader, device, num_classes=10, epochs=args.tstr_epochs)
    margins, preds = compute_margins(judge, gen, device)
    nb = near_boundary_fraction(margins, args.threshold)
    label_noise = (preds != gen_labels).float().mean().item()
    incep_cov, incep_prec = (None, None)
    if detector is not None:
        incep_cov, incep_prec = inception_crosscheck(detector, real_imgs, real_labels,
                                                     gen, gen_labels, args.nearest_k, device)
    row = {"name": f"w{w:g}", "guidance": w,
           "precision": dino_prdc["precision"], "coverage": dino_prdc["coverage"],
           "coverage_inception": incep_cov, "precision_inception": incep_prec,
           "tstr": tstr, "label_noise_frac": label_noise, "near_boundary_frac": nb}
    del gen, gen_dino, margins, preds
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main():
    p = argparse.ArgumentParser(description="Self-trained CFG CIFAR-10 multi-seed full run.")
    p.add_argument("--ckpt", default="checkpoints/cifar10_cfg.pt")
    p.add_argument("--judge", default="checkpoints/cifar10_judge.pt")
    p.add_argument("--guidance", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0, 5.0, 8.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--per-class", type=int, default=500)
    p.add_argument("--real-per-class", type=int, default=500)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--batch", type=int, default=250)
    p.add_argument("--nearest-k", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.9525)
    p.add_argument("--tau-fraction", type=float, default=0.9)
    p.add_argument("--tstr-epochs", type=int, default=15)
    p.add_argument("--no-inception", action="store_true")
    p.add_argument("--output", default="results/cifar10_cfg_multiseed.json")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.guidance = [1.0, 3.0]
        args.seeds = [0, 1]
        args.per_class = 32
        args.real_per_class = 64
        args.tstr_epochs = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(f"guidance={args.guidance} seeds={args.seeds} per_class={args.per_class} "
          f"tstr_epochs={args.tstr_epochs}", flush=True)
    os.makedirs("results", exist_ok=True)

    model, schedule, hp = load_cfg_model(args.ckpt, device)
    judge = ResNet18(num_classes=10).to(device)
    judge.load_state_dict(torch.load(args.judge, map_location=device, weights_only=True))
    judge.eval()

    real_imgs, real_labels = load_real_per_class("cifar10", args.real_per_class, seed=0)
    real_dino = dinov2_features((real_imgs + 1) / 2, device)
    test_loader = build_test_loader("cifar10", batch_size=256, num_workers=0)
    detector = None if args.no_inception else load_inception_detector(device)
    print(f"Real ref {real_imgs.size(0)} imgs; inception={'關' if detector is None else '開'}", flush=True)

    seed_results = []
    for seed in args.seeds:
        print(f"\n########## seed {seed} ##########", flush=True)
        ref_prec = real_ref_precision(real_dino, real_labels, args.nearest_k, device, seed)
        configs = []
        for w in args.guidance:
            row = measure(model, schedule, judge, real_imgs, real_dino, real_labels,
                          test_loader, detector, w, args, device, seed)
            configs.append(row)
            print(f"  w={w:<4g} prec={row['precision']:.3f} cov={row['coverage']:.3f} "
                  f"tstr={row['tstr']:.2f} label_noise={row['label_noise_frac']:.3f} "
                  f"near_bnd={row['near_boundary_frac']:.3f}", flush=True)
        report = select_and_report(configs, real_ref_precision=ref_prec,
                                   tau_fraction=args.tau_fraction, utility_key="tstr")
        print(f"  -> CaF 選 {report['selected']}（oracle {report['oracle_best']}, "
              f"regret {report['regret_at_selected']}, rank {report['rank']}/{report['n_configs']}）", flush=True)
        seed_results.append({"seed": seed, "ref_precision": ref_prec,
                             "configs": configs, "report": report})

    # 跨 seed 彙總
    names = [f"w{w:g}" for w in args.guidance]
    per_config = []
    for nm in names:
        keys = ["precision", "coverage", "tstr", "label_noise_frac", "near_boundary_frac"]
        agg = {"name": nm}
        for k in keys:
            agg[k] = summarize([next(c for c in sr["configs"] if c["name"] == nm)[k]
                                for sr in seed_results])
        per_config.append(agg)
    selected = [sr["report"]["selected"] for sr in seed_results]
    counts = {nm: selected.count(nm) for nm in sorted(set(selected))}
    modal = max(counts, key=counts.get)
    agg = {"n_seeds": len(seed_results), "per_config": per_config,
           "selection": {"per_seed": selected, "modal": modal,
                         "modal_fraction": counts[modal] / len(selected)},
           "regret_at_selected": summarize([sr["report"]["regret_at_selected"] for sr in seed_results]),
           "rank_per_seed": [sr["report"]["rank"] for sr in seed_results],
           "topk_hit_rate": sum(bool(sr["report"]["topk_hit"]) for sr in seed_results) / len(seed_results),
           "oracle_best_per_seed": [sr["report"]["oracle_best"] for sr in seed_results]}

    print("\n" + "=" * 92)
    print(f"  自訓 CFG CIFAR-10 多 seed（{agg['n_seeds']} seeds，grid {args.guidance}）")
    print("=" * 92)
    print(f"  {'w':>4} {'precision':>16} {'coverage':>16} {'TSTR%':>16} {'label_noise':>14} {'near_bnd':>12}")
    for pc in per_config:
        def f(s, d=3): return f"{s['mean']:.{d}f}+/-{s['std']:.{d}f}" if s else "n/a"
        print(f"  {pc['name'][1:]:>4} {f(pc['precision']):>16} {f(pc['coverage']):>16} "
              f"{f(pc['tstr'], 2):>16} {f(pc['label_noise_frac']):>14} {f(pc['near_boundary_frac']):>12}")
    print("  " + "-" * 88)
    print(f"  CaF 選擇/seed : {agg['selection']['per_seed']}（modal {agg['selection']['modal']}, "
          f"{agg['selection']['modal_fraction']*100:.0f}%）")
    print(f"  oracle TSTR-best/seed : {agg['oracle_best_per_seed']}")
    if agg["regret_at_selected"]:
        r = agg["regret_at_selected"]
        print(f"  regret@selected : {r['mean']:.3f} +/- {r['std']:.3f} pp（max {r['max']:.3f}）")
    print(f"  rank/seed : {agg['rank_per_seed']}   top-3 命中率 : {agg['topk_hit_rate']*100:.0f}%")
    print("=" * 92)

    out = {"metadata": {"dataset": "cifar10", "axis": "CFG guidance (self-trained)",
                        "steps": args.steps, "eta": args.eta, "guidance_grid": args.guidance,
                        "seeds": args.seeds, "per_class": args.per_class,
                        "near_boundary_threshold": args.threshold,
                        "coverage_feature": "DINOv2 primary + Inception cross-check"},
           "aggregate": agg, "per_seed": seed_results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
