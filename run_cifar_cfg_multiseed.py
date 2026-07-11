"""自訓 CFG CIFAR-10 / CIFAR-100 多 seed 全量（confirmatory 主結果）。

CIFAR-100 一般化（D 包 `records/2026-07-09-13`）：--dataset cifar100 時類數 100、生成種子改用無碰撞
hash 公式（datasets/cifar100_gseed）、selector 改用 CaF-v2（argmax recall s.t. precision≥τ，D8）、
char clean-fid 用 base gate 自建的 cifar100_train_clean32 參考、每 (seed,w) 做 --reps 次 TSTR 重訓。

在封頂 amendment（records/2026-07-05-11）凍結的 grid（w∈{1,1.5,2,2.5,3,4,5,6,7,8}，固定 steps=50
eta=0）上跑 fresh seeds {10,11,12}，每個 (seed, w) 量：precision + coverage（DINOv2 主、Inception 交叉）
+ TSTR + near-boundary + 標籤噪音。跨 seed 彙總帶信賴區間；並以 CaF（argmax coverage s.t. precision
≥ tau，tau 自 real-vs-real 參考自動決定）在完整網格（不剪枝）計 regret@selected / rank / top-k，作為
go/no-go 主指標（協定 §6、§8）。

凍結規格的儀器（confirmatory）：
  - per-class 超額 label-noise（增修 records/2026-07-05-08 規格2）：逐類合成 label-noise 減真實 judge
    floor（floor 自 cifar10_judge.json 的 per_class_accuracy），跨 seed 帶 CI。難類 floor 本高，逐類相減
    才能分離「難類本難判」與「guidance 製造污染」。
  - per-config characterization clean-fid（規格1）：重用生成集算 clean-fid，只報告不 gate，呈現 FID 隨
    guidance 的軌跡。
  - per_class 依協定 2026-07-05-02 §3 用 1000（消 coverage 樣本數假影），非 pilot 的 500。
  - C2 裁決（全網格偏相關）由生成後腳本 run_c2_partial.py 讀本檔輸出另算（records/2026-07-05-13）。

Usage:
    uv run python run_cifar_cfg_multiseed.py --quick
    uv run python run_cifar_cfg_multiseed.py --guidance 1 1.5 2 2.5 3 4 5 6 7 8 \
        --seeds 10 11 12 --per-class 1000 --real-per-class 1000 \
        --output results/cifar10_cfg_confirmatory.json
"""

import argparse
import json
import os
import sys
import time

import torch

from cifar_cfg_sample import load_cfg_model, generate_balanced
from cifar_classifier import ResNet18, run_tstr
from mechanism import compute_margins, near_boundary_fraction
from metrics_prdc import compute_prdc_per_class
from metrics_features import dinov2_features
from datasets.cifar import load_real_per_class, build_test_loader, NUM_CLASSES
from datasets.cifar100_gseed import gseed as gseed_hash
from run_cifar_cfg_scout import load_inception_detector, inception_crosscheck
from run_cifar_selector import summarize
from selector import select_and_report
from fid_clean import clean_fid_vs_dataset
from cifar100_base_gate import clean_fid_gen_vs_ref


def gen_seed(dataset, seed, w):
    """(seed, w) → 生成種子。CIFAR-100 用無碰撞 hash 公式（D9 datasets/cifar100_gseed.py）；
    CIFAR-10 沿用原 (seed+w)×1e7 公式以保凍結資料之逐位重現（該公式已知在 CIFAR-10 網格有退化
    碰撞，見 records/2026-07-06-05 §1.12，但 CIFAR-10 confirmatory 已用它跑完並定稿，不改）。"""
    if dataset == "cifar100":
        return gseed_hash(seed, w)
    return seed * 10_000_000 + int(w * 1000) * 10_000


def load_judge_floor(path="results/cifar10_judge.json", num_classes=10):
    """逐類真實 judge 誤判率（floor），依增修 2026-07-05-08 規格2 作固定儀器。

    floor_c = 1 - per_class_accuracy[c] / 100。判定 super 額外污染時逐類相減此 floor。
    """
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    acc = j["per_class_accuracy"]  # {"0": 94.5, ...}，真實測試集上 judge 逐類準確率
    return [1.0 - float(acc[str(c)]) / 100.0 for c in range(num_classes)]


def real_ref_precision(real_dino, real_labels, nearest_k, device, seed, num_classes=10):
    """real-vs-real 參考 precision（DINOv2 空間），供 CaF 的免 TSTR tau。"""
    n = real_dino.size(0)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    a, b = perm[:n // 2], perm[n // 2:]
    ref, _ = compute_prdc_per_class(real_dino[b].to(device), real_labels[b].to(device),
                                    real_dino[a].to(device), real_labels[a].to(device),
                                    nearest_k=nearest_k, num_classes=num_classes)
    return ref["precision"]


def measure(model, schedule, judge, real_imgs, real_dino, real_labels, test_loader,
            detector, judge_floor, w, args, device, seed, do_fid=True):
    nc = args.num_classes
    g = gen_seed(args.dataset, seed, w)
    gen, gen_labels = generate_balanced(model, schedule, args.per_class, device, args.steps,
                                        args.eta, guidance=w, num_classes=nc, batch=args.batch, seed=g)
    gen_dino = dinov2_features((gen + 1) / 2, device)
    dino_prdc, _ = compute_prdc_per_class(real_dino.to(device), real_labels.to(device),
                                          gen_dino.to(device), gen_labels.to(device),
                                          nearest_k=args.nearest_k, num_classes=nc)
    # TSTR reps（D4：n_rep 個獨立 from-scratch 分類器，消 σ_cls 分類器訓練變異；均值餵 selector）。
    tstr_reps = []
    for _ in range(args.reps):
        t, _ = run_tstr(gen, gen_labels, test_loader, device, num_classes=nc, epochs=args.tstr_epochs)
        tstr_reps.append(t)
    tstr = sum(tstr_reps) / len(tstr_reps)
    margins, preds = compute_margins(judge, gen, device)
    nb = near_boundary_fraction(margins, args.threshold)
    label_noise = (preds != gen_labels).float().mean().item()

    # per-class 超額 label-noise：逐類合成 label-noise 減真實 judge floor（增修 2026-07-05-08 規格2）。
    # 全域相減會混淆「難類本難判」（floor 高）與「guidance 製造污染」，故逐類算、逐類相減。
    per_class_ln, per_class_excess = [], []
    for c in range(nc):
        m = (gen_labels == c)
        ln_c = (preds[m] != c).float().mean().item() if bool(m.any()) else float("nan")
        per_class_ln.append(ln_c)
        per_class_excess.append(ln_c - judge_floor[c])
    excess_mean = sum(per_class_excess) / len(per_class_excess)

    # per-config characterization clean-fid：重用生成集，只報告不 gate（規格1）。
    # 每 config 生成張數固定（per_class×類數），故小樣本正偏誤在各 config 間一致，軌跡可比。
    # CIFAR-100 無內建 clean-fid stats，用 base gate 建的自訂參考 cifar100_train_clean32。
    char_fid = None
    if do_fid:
        if args.dataset == "cifar100":
            char_fid = clean_fid_gen_vs_ref((gen + 1) / 2, "cifar100_train_clean32")
        else:
            char_fid = float(clean_fid_vs_dataset((gen + 1) / 2, dataset_name="cifar10",
                                                  dataset_split="train", dataset_res=32))

    incep_cov, incep_prec = (None, None)
    if detector is not None:
        incep_cov, incep_prec = inception_crosscheck(detector, real_imgs, real_labels,
                                                     gen, gen_labels, args.nearest_k, device, nc)
    row = {"name": f"w{w:g}", "guidance": w,
           "precision": dino_prdc["precision"], "coverage": dino_prdc["coverage"],
           "recall": dino_prdc["recall"], "density": dino_prdc["density"],
           "coverage_inception": incep_cov, "precision_inception": incep_prec,
           "tstr": tstr, "tstr_reps": tstr_reps, "label_noise_frac": label_noise,
           "label_noise_per_class": per_class_ln,
           "label_noise_excess_per_class": per_class_excess,
           "label_noise_excess_mean": excess_mean,
           "char_clean_fid": char_fid, "near_boundary_frac": nb}
    del gen, gen_dino, margins, preds
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main():
    p = argparse.ArgumentParser(description="Self-trained CFG CIFAR-10/100 multi-seed full run.")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--ckpt", default=None, help="預設 checkpoints/<dataset>_cfg.pt")
    p.add_argument("--judge", default=None, help="預設 checkpoints/<dataset>_judge.pt")
    p.add_argument("--guidance", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0, 5.0, 8.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--reps", type=int, default=1, help="每 (seed,w) 之 TSTR 重訓次數（D4 CIFAR-100=5）")
    p.add_argument("--per-class", type=int, default=500)
    p.add_argument("--real-per-class", type=int, default=500)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--batch", type=int, default=250)
    p.add_argument("--nearest-k", type=int, default=5)
    p.add_argument("--threshold", type=float, default=None,
                   help="near-boundary threshold；未給則自 <dataset>_judge.json 讀")
    p.add_argument("--tau-fraction", type=float, default=0.9)
    p.add_argument("--tstr-epochs", type=int, default=15)
    p.add_argument("--judge-json", default=None, help="預設 results/<dataset>_judge.json")
    p.add_argument("--no-inception", action="store_true")
    p.add_argument("--no-fid", action="store_true",
                   help="略過 per-config characterization clean-fid（僅 smoke 用；confirmatory 須開）")
    p.add_argument("--output", default=None, help="預設 results/<dataset>_cfg_multiseed.json")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # F1（流程債）：記錄開跑時刻，供事後時序對賬

    # dataset 相依預設。CaF-v2（CIFAR-100）用 recall 訊號，CaF（CIFAR-10）用 coverage（D8）。
    args.num_classes = NUM_CLASSES[args.dataset]
    ckpt = args.ckpt or f"checkpoints/{args.dataset}_cfg.pt"
    judge_ckpt = args.judge or f"checkpoints/{args.dataset}_judge.pt"
    judge_json = args.judge_json or f"results/{args.dataset}_judge.json"
    output = args.output or f"results/{args.dataset}_cfg_multiseed.json"
    signal_key = "recall" if args.dataset == "cifar100" else "coverage"
    if args.threshold is None:
        with open(judge_json, encoding="utf-8") as f:
            args.threshold = float(json.load(f)["near_boundary_threshold"])

    if args.quick:
        args.guidance = [1.0, 3.0]
        args.seeds = [0, 1]
        args.reps = 2
        args.per_class = 32
        args.real_per_class = 64
        args.tstr_epochs = 2
        args.no_fid = True  # 32/class 太少，cleanfid 不穩；smoke 不量 FID

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  dataset={args.dataset} num_classes={args.num_classes} "
          f"selector_signal={signal_key}", flush=True)
    print(f"guidance={args.guidance} seeds={args.seeds} reps={args.reps} per_class={args.per_class} "
          f"tstr_epochs={args.tstr_epochs} threshold={args.threshold:.4f}", flush=True)
    os.makedirs("results", exist_ok=True)

    model, schedule, hp = load_cfg_model(ckpt, device)
    judge = ResNet18(num_classes=args.num_classes).to(device)
    judge.load_state_dict(torch.load(judge_ckpt, map_location=device, weights_only=True))
    judge.eval()
    judge_floor = load_judge_floor(judge_json, args.num_classes)
    print(f"judge floor（逐類誤判率）前5類: {[round(f, 3) for f in judge_floor[:5]]} ...", flush=True)
    print(f"per-config characterization FID: {'關（smoke）' if args.no_fid else '開'}", flush=True)

    real_imgs, real_labels = load_real_per_class(args.dataset, args.real_per_class, seed=0)
    real_dino = dinov2_features((real_imgs + 1) / 2, device)
    test_loader = build_test_loader(args.dataset, batch_size=256, num_workers=0)
    detector = None if args.no_inception else load_inception_detector(device)
    print(f"Real ref {real_imgs.size(0)} imgs; inception={'關' if detector is None else '開'}", flush=True)

    seed_results = []
    for seed in args.seeds:
        print(f"\n########## seed {seed} ##########", flush=True)
        ref_prec = real_ref_precision(real_dino, real_labels, args.nearest_k, device, seed,
                                      args.num_classes)
        configs = []
        for w in args.guidance:
            row = measure(model, schedule, judge, real_imgs, real_dino, real_labels,
                          test_loader, detector, judge_floor, w, args, device, seed,
                          do_fid=not args.no_fid)
            configs.append(row)
            fid_s = f"{row['char_clean_fid']:.2f}" if row["char_clean_fid"] is not None else "n/a"
            print(f"  w={w:<4g} prec={row['precision']:.3f} cov={row['coverage']:.3f} "
                  f"tstr={row['tstr']:.2f} excess_ln={row['label_noise_excess_mean']:+.3f} "
                  f"char_fid={fid_s} near_bnd={row['near_boundary_frac']:.3f}", flush=True)
        report = select_and_report(configs, real_ref_precision=ref_prec,
                                   tau_fraction=args.tau_fraction, utility_key="tstr",
                                   signal_key=signal_key)
        print(f"  -> CaF 選 {report['selected']}（oracle {report['oracle_best']}, "
              f"regret {report['regret_at_selected']}, rank {report['rank']}/{report['n_configs']}）", flush=True)
        seed_results.append({"seed": seed, "ref_precision": ref_prec,
                             "configs": configs, "report": report})

    # 跨 seed 彙總
    names = [f"w{w:g}" for w in args.guidance]
    per_config = []
    for nm in names:
        def rows_for(nm=nm):
            return [next(c for c in sr["configs"] if c["name"] == nm) for sr in seed_results]
        keys = ["precision", "coverage", "recall", "density",
                "coverage_inception", "precision_inception",
                "tstr", "label_noise_frac", "label_noise_excess_mean",
                "char_clean_fid", "near_boundary_frac"]
        agg = {"name": nm}
        for k in keys:
            agg[k] = summarize([r[k] for r in rows_for()])
        # per-class 超額 label-noise：逐類跨 seed 彙總帶 CI（規格2 要求逐類、帶 CI）
        agg["label_noise_excess_per_class"] = [
            summarize([r["label_noise_excess_per_class"][c] for r in rows_for()])
            for c in range(args.num_classes)]
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
    print(f"  自訓 CFG {args.dataset} 多 seed（{agg['n_seeds']} seeds × {args.reps} reps，"
          f"grid {args.guidance}，selector {signal_key}）")
    print("=" * 92)
    print(f"  {'w':>4} {'precision':>14} {'coverage':>14} {'TSTR%':>14} "
          f"{'excess_ln':>14} {'char_fid':>10}")
    for pc in per_config:
        def f(s, d=3): return f"{s['mean']:.{d}f}+/-{s['std']:.{d}f}" if s else "n/a"
        print(f"  {pc['name'][1:]:>4} {f(pc['precision']):>14} {f(pc['coverage']):>14} "
              f"{f(pc['tstr'], 2):>14} {f(pc['label_noise_excess_mean']):>14} "
              f"{f(pc['char_clean_fid'], 2):>10}")
    print("  " + "-" * 88)
    print(f"  CaF 選擇/seed : {agg['selection']['per_seed']}（modal {agg['selection']['modal']}, "
          f"{agg['selection']['modal_fraction']*100:.0f}%）")
    print(f"  oracle TSTR-best/seed : {agg['oracle_best_per_seed']}")
    if agg["regret_at_selected"]:
        r = agg["regret_at_selected"]
        print(f"  regret@selected : {r['mean']:.3f} +/- {r['std']:.3f} pp（max {r['max']:.3f}）")
    print(f"  rank/seed : {agg['rank_per_seed']}   top-3 命中率 : {agg['topk_hit_rate']*100:.0f}%")
    print("=" * 92)

    out = {"metadata": {"dataset": args.dataset, "num_classes": args.num_classes,
                        "axis": "CFG guidance (self-trained)",
                        "selector": "CaF-v2 (recall)" if signal_key == "recall" else "CaF (coverage)",
                        "selector_signal_key": signal_key,
                        "reps": args.reps,
                        "gen_seed_formula": ("cifar100_gseed sha256[:15]" if args.dataset == "cifar100"
                                             else "(seed*1e7 + int(w*1e3)*1e4)"),
                        "steps": args.steps, "eta": args.eta, "guidance_grid": args.guidance,
                        "seeds": args.seeds, "per_class": args.per_class,
                        "real_per_class": args.real_per_class,
                        "near_boundary_threshold": args.threshold,
                        "judge_floor_per_class": judge_floor,
                        "char_clean_fid_enabled": not args.no_fid,
                        "coverage_feature": "DINOv2 primary + Inception cross-check",
                        "prereg": {"grid_cap": "2026-07-05-11", "repositioning": "2026-07-05-12",
                                   "c2_stats": "2026-07-05-13", "label_noise_spec": "2026-07-05-08 規格2",
                                   "cifar100_d_package": "2026-07-09-13",
                                   "cifar100_grid_freeze": "2026-07-11-05"},
                        "start_timestamp": start_timestamp,      # F1：開跑時刻
                        "argv": " ".join(sys.argv),              # F1：完整命令留痕（含 --nearest-k / --batch 實傳值）
                        "nearest_k": args.nearest_k, "batch": args.batch,
                        "tau_fraction": args.tau_fraction,
                        "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                                "cudnn": torch.backends.cudnn.version()}},
           "aggregate": agg, "per_seed": seed_results}
    with open(output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {output}", flush=True)


if __name__ == "__main__":
    main()
