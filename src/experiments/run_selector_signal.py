"""MNIST sandbox 上 CaF 選擇器的 Gate A go/no-go 訊號（多 seed）。

問題（Phase 0 gate，準則 b）：CaF——在 precision >= tau 的限制下取 argmax
coverage，由一組真實 probe set 免訓練地算出——能否穩健地選出 downstream TSTR
會挑的 sampler 組態，而不需為每個組態各訓練一個分類器，且所選結果不是單一 seed
雜訊造成的假象？

為何要多 seed（修訂版計畫，brief 第 3 節）：整個論點建立在「utility 最佳的組態
!= FID 最佳的組態」上。在一個帶雜訊的網格上做單一 seed 的 argmax，本身可能就是
雜訊（單一 seed 的執行中，g1 與 g2 的 coverage 只差約 0.02）。因此我們把整條
pipeline 在 >=3 個 seed 上重複，並回報選擇器決策的*分布*與信賴區間，而非單一
僥倖的選擇。

相較於舊的單一 seed 腳本的方法論升級：PRDC（選擇器輸入）與 TSTR（oracle
utility）在每個 seed 下由相同的生成樣本計算，因此選擇器是對照在相同資料上量得的
utility 來評分，而不是對照一份陳舊、另外生成的 guidance_study.json。

對每個 seed s 與 guidance g 我們：
  1. 用 s 對所有 RNG 設定 seed（可重現，且各 seed 抽樣互異），
  2. 在 (DDIM, eta=0, steps=50, guidance=g) 下生成 per_digit*10 張影像，
  3. 在 judge-CNN 特徵空間計算各類別 PRDC（precision/coverage），
  4. 在這些影像上訓練一個全新的 CNN，並在真實 MNIST 上評估 -> TSTR，
  5. 執行 CaF（由真對真的參考 precision 得出 auto-tau）。
接著跨 seed 彙整：選擇分布、regret@selected（平均/CI）、rank、top-k 命中率與
tau-robustness。

Usage:
    uv run python run_selector_signal.py                       # seeds 0 1 2, full
    uv run python run_selector_signal.py --seeds 0 1 2 3 4     # 5 seeds
    uv run python run_selector_signal.py --quick               # fast smoke test
    uv run python run_selector_signal.py --no-compute-tstr     # read oracle from json
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from ddpm import UNet, DiffusionSchedule
from inference import generate as gen_batches
from fid import load_cnn
from analyze_distribution import extract_features, load_real_per_class
from metrics_prdc import compute_prdc_per_class
from evaluate import (MNISTClassifier, train_classifier, evaluate,
                      build_dataloaders, get_git_commit)
from mechanism import analyze_dataset
from selector import select_and_report

NEAR_BOUNDARY_THRESHOLD = 0.5

FULL_GUIDANCE = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
QUICK_GUIDANCE = [1.0, 3.0, 7.0]
SAMPLER, STEPS, ETA = "ddim", 50, 0.0


def set_seed(seed):
    """對所有 RNG 設定 seed，使一次執行在同一 seed 下可重現，且各 seed 之間互異。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summarize(values):
    """對一串純量計算 mean/std/sem/min/max（忽略 None）。"""
    vals = [float(v) for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return None
    mean = sum(vals) / n
    std = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    sem = std / (n ** 0.5) if n > 0 else 0.0
    return {"mean": mean, "std": std, "sem": sem, "ci95": 1.96 * sem,
            "min": min(vals), "max": max(vals), "n": n, "values": vals}


def generate_config(model, schedule, guidance, per_digit, batch_size, device):
    imgs, labels = [], []
    for digit in range(10):
        for batch in gen_batches(model, schedule, digit, per_digit, batch_size,
                                 guidance, device, sampler=SAMPLER, steps=STEPS, eta=ETA):
            imgs.append(batch.cpu())
            labels.append(torch.full((batch.size(0),), digit, dtype=torch.long))
    return torch.cat(imgs), torch.cat(labels)


def run_tstr(images, labels, real_test_loader, device, epochs, lr, batch_size):
    """純粹在 `images` 上訓練一個全新的 CNN，並在真實 MNIST 測試集上評估。"""
    clf = MNISTClassifier().to(device)
    loader = DataLoader(TensorDataset(images, labels), batch_size=batch_size,
                        shuffle=True, num_workers=0)
    train_classifier(clf, loader, device, epochs, lr)
    acc, _, _ = evaluate(clf, real_test_loader, device)
    return acc


def load_tstr_map(path):
    """從先前的 guidance_study.json 建立 guidance value -> tstr_acc 的對照（fallback）。"""
    if not path or not os.path.exists(path):
        return {}
    data = json.load(open(path, encoding="utf-8"))
    return {float(r["guidance"]): float(r["tstr_acc"]) for r in data.get("rows", [])}


def run_one_seed(seed, model, schedule, judge, guidance_list, real_test_loader,
                 tstr_map, args, device):
    """單一 seed 下完整的 PRDC + TSTR + CaF pipeline。回傳一個結果 dict。"""
    set_seed(seed)
    print(f"\n########## seed {seed} ##########")

    # 由真對真的切分得出真實 probe set 與參考 precision（不依賴 TSTR 的 tau）。
    real_images, real_labels = load_real_per_class(args.data_dir, args.fid_per_class, seed)
    real_feats = extract_features(judge, real_images, device).to(device)
    real_labels = real_labels.to(device)
    n = real_feats.size(0)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    half = n // 2
    a_idx, b_idx = perm[:half], perm[half:]
    ref_mean, _ = compute_prdc_per_class(
        real_feats[b_idx], real_labels[b_idx],   # manifold = 切分 B
        real_feats[a_idx], real_labels[a_idx],   # 「fake」= 切分 A
        nearest_k=args.nearest_k)
    real_ref_precision = ref_mean["precision"]
    print(f"  real-vs-real reference precision (per-class): {real_ref_precision:.4f}")

    # 真實資料的參考 margin 分布（C2 mechanism 基線）。
    real_margin, _, _ = analyze_dataset(judge, real_images, real_labels, device,
                                        NEAR_BOUNDARY_THRESHOLD)
    real_near_boundary = real_margin["near_boundary_frac"][str(NEAR_BOUNDARY_THRESHOLD)]
    print(f"  real near-boundary frac (<{NEAR_BOUNDARY_THRESHOLD}): {real_near_boundary:.4f}")

    configs = []
    for g in guidance_list:
        gen_imgs, gen_labels = generate_config(model, schedule, g, args.per_digit,
                                               args.batch_size, device)
        gen_feats = extract_features(judge, gen_imgs, device).to(device)
        gen_labels_d = gen_labels.to(device)
        prdc, _ = compute_prdc_per_class(real_feats, real_labels,
                                         gen_feats, gen_labels_d, nearest_k=args.nearest_k)
        # C2 mechanism：near-boundary 耗竭 + label-noise 競爭控制，
        # 在與 PRDC/TSTR 相同的樣本上量測。
        msum, _, _ = analyze_dataset(judge, gen_imgs, gen_labels, device,
                                     NEAR_BOUNDARY_THRESHOLD)
        cfg = {"name": f"g{g:g}", "guidance": g,
               "precision": prdc["precision"], "coverage": prdc["coverage"],
               "recall": prdc["recall"], "density": prdc["density"],
               "mean_margin": msum["mean_margin"],
               "near_boundary_frac": msum["near_boundary_frac"][str(NEAR_BOUNDARY_THRESHOLD)],
               "label_noise_frac": msum["label_noise_frac"]}

        if args.compute_tstr:
            tstr = run_tstr(gen_imgs, gen_labels, real_test_loader, device,
                            args.tstr_epochs, args.tstr_lr, args.tstr_batch_size)
            cfg["tstr"] = tstr
        elif g in tstr_map:
            cfg["tstr"] = tstr_map[g]

        configs.append(cfg)
        tstr_str = f", tstr={cfg['tstr']:.2f}" if "tstr" in cfg else ""
        print(f"  g={g:<4g} precision={prdc['precision']:.4f} coverage={prdc['coverage']:.4f}"
              f" recall={prdc['recall']:.4f} nb_frac={cfg['near_boundary_frac']:.4f}"
              f" lbl_noise={cfg['label_noise_frac']:.4f}{tstr_str}")

    have_tstr = all("tstr" in c for c in configs)
    report = select_and_report(configs, real_ref_precision=real_ref_precision,
                               tau_fraction=args.tau_fraction,
                               utility_key="tstr" if have_tstr else "__none__")
    print(f"  -> CaF selected {report['selected']} "
          f"(oracle {report['oracle_best']}, regret {report['regret_at_selected']}, "
          f"rank {report['rank']}/{report['n_configs']})")
    return {"seed": seed, "real_ref_precision": real_ref_precision,
            "real_near_boundary_frac": real_near_boundary,
            "configs": configs, "report": report}


def aggregate(seed_results, guidance_list, topk):
    """彙整各 seed 的結果：各組態的 CI 加上選擇器決策分布。"""
    names = [f"g{g:g}" for g in guidance_list]

    per_config = []
    for name in names:
        prec, cov, tstr, nb, mm, ln = [], [], [], [], [], []
        for sr in seed_results:
            c = next(c for c in sr["configs"] if c["name"] == name)
            prec.append(c["precision"]); cov.append(c["coverage"])
            tstr.append(c.get("tstr"))
            nb.append(c["near_boundary_frac"]); mm.append(c["mean_margin"])
            ln.append(c["label_noise_frac"])
        per_config.append({"name": name,
                           "guidance": next(g for g in guidance_list if f"g{g:g}" == name),
                           "precision": summarize(prec),
                           "coverage": summarize(cov),
                           "tstr": summarize(tstr),
                           "near_boundary_frac": summarize(nb),
                           "mean_margin": summarize(mm),
                           "label_noise_frac": summarize(ln)})

    # C2 mechanism 的方向性：near-boundary 比例應隨 guidance 上升而下降
    # （prototype 銳化耗竭了邊界支持），同時 label-noise 維持在低點。
    by_g = sorted(per_config, key=lambda pc: pc["guidance"])
    nb_by_g = [pc["near_boundary_frac"]["mean"] for pc in by_g]
    tstr_by_g = [pc["tstr"]["mean"] if pc["tstr"] else None for pc in by_g]
    mechanism = {
        "near_boundary_frac_by_guidance": {pc["name"]: pc["near_boundary_frac"]["mean"] for pc in by_g},
        "monotone_depletion": all(nb_by_g[i] >= nb_by_g[i + 1] for i in range(len(nb_by_g) - 1)),
        "nb_frac_drop_low_to_high": nb_by_g[0] - nb_by_g[-1],
        "label_noise_by_guidance": {pc["name"]: pc["label_noise_frac"]["mean"] for pc in by_g},
        "real_near_boundary_frac": summarize([sr["real_near_boundary_frac"] for sr in seed_results]),
        "nb_vs_tstr_aligned": (nb_by_g[0] >= nb_by_g[-1] and tstr_by_g[0] is not None
                               and tstr_by_g[0] >= tstr_by_g[-1]),
    }

    selected = [sr["report"]["selected"] for sr in seed_results]
    counts = {nm: selected.count(nm) for nm in sorted(set(selected))}
    modal = max(counts, key=counts.get)
    regrets = [sr["report"]["regret_at_selected"] for sr in seed_results]
    ranks = [sr["report"]["rank"] for sr in seed_results]
    topk_hits = [sr["report"]["topk_hit"] for sr in seed_results]
    oracle_best = [sr["report"]["oracle_best"] for sr in seed_results]
    taus = [sr["report"]["tau"] for sr in seed_results]
    tau_stability = [sr["report"]["tau_robustness"]["stability"] for sr in seed_results]

    return {
        "n_seeds": len(seed_results),
        "per_config": per_config,
        "selection": {"per_seed": selected, "counts": counts,
                      "modal": modal, "modal_fraction": counts[modal] / len(selected)},
        "regret_at_selected": summarize(regrets),
        "rank": summarize([r for r in ranks if r is not None]),
        "rank_per_seed": ranks,
        "topk": topk,
        "topk_hit_rate": sum(bool(h) for h in topk_hits) / len(topk_hits),
        "oracle_best_per_seed": oracle_best,
        "tau": summarize(taus),
        "tau_stability": summarize(tau_stability),
        "mechanism": mechanism,
    }


def print_summary(agg):
    print("\n" + "=" * 78)
    print(f"  Gate A — CaF selector signal over {agg['n_seeds']} seeds")
    print("=" * 78)
    hdr = (f"  {'config':>7} {'precision':>16} {'coverage':>16} {'nb_frac':>16} "
           f"{'TSTR%':>16}")
    print(hdr)
    print("  " + "-" * 88)
    for pc in agg["per_config"]:
        def fmt(s):
            return f"{s['mean']:.4f}+/-{s['std']:.4f}" if s else "   n/a"
        def fmt2(s):
            return f"{s['mean']:.2f}+/-{s['std']:.2f}" if s else "   n/a"
        print(f"  {pc['name']:>7} {fmt(pc['precision']):>16} {fmt(pc['coverage']):>16} "
              f"{fmt(pc['near_boundary_frac']):>16} {fmt2(pc['tstr']):>16}")
    print("  " + "-" * 88)
    sel = agg["selection"]
    print(f"  CaF selection per seed : {sel['per_seed']}")
    print(f"  selection counts       : {sel['counts']}  (modal {sel['modal']}, "
          f"{sel['modal_fraction']*100:.0f}% of seeds)")
    print(f"  oracle TSTR-best/seed  : {agg['oracle_best_per_seed']}")
    if agg["regret_at_selected"]:
        r = agg["regret_at_selected"]
        print(f"  regret@selected        : {r['mean']:.3f} +/- {r['std']:.3f} pp "
              f"(max {r['max']:.3f}, 95%CI +/-{r['ci95']:.3f})")
    if agg["rank"]:
        print(f"  rank of selected       : mean {agg['rank']['mean']:.2f}, "
              f"per-seed {agg['rank_per_seed']}")
    print(f"  top-{agg['topk']} hit rate         : {agg['topk_hit_rate']*100:.0f}% of seeds")
    if agg["tau"]:
        print(f"  auto-tau               : {agg['tau']['mean']:.4f} +/- {agg['tau']['std']:.4f}")
    mech = agg["mechanism"]
    print("  " + "-" * 74)
    print(f"  [C2 mechanism] near-boundary frac by guidance: "
          f"{ {k: round(v,3) for k,v in mech['near_boundary_frac_by_guidance'].items()} }")
    print(f"  monotone depletion (nb falls as g rises): {mech['monotone_depletion']} "
          f"(drop low->high = {mech['nb_frac_drop_low_to_high']:.3f})")
    print(f"  label-noise by guidance: "
          f"{ {k: round(v,3) for k,v in mech['label_noise_by_guidance'].items()} }")
    print(f"  nb-depletion aligned with TSTR drop: {mech['nb_vs_tstr_aligned']}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Multi-seed CaF selector signal (Gate A).")
    parser.add_argument("--checkpoint", default="ddpm_mnist.pt")
    parser.add_argument("--cnn", default="mnist_cnn.pt")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--per-digit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fid-per-class", type=int, default=1000)
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--tau-fraction", type=float, default=0.9)
    parser.add_argument("--tstr-epochs", type=int, default=20)
    parser.add_argument("--tstr-lr", type=float, default=1e-3)
    parser.add_argument("--tstr-batch-size", type=int, default=64)
    parser.add_argument("--compute-tstr", action=argparse.BooleanOptionalAction, default=True,
                        help="Compute TSTR from the same samples (default). "
                             "--no-compute-tstr reads oracle from --tstr-json instead.")
    parser.add_argument("--tstr-json", default="results/guidance_study.json",
                        help="Fallback oracle TSTR source when --no-compute-tstr")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--output", default="results/selector_signal_multiseed.json")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    guidance_list = QUICK_GUIDANCE if args.quick else FULL_GUIDANCE
    if args.quick:
        if args.per_digit == 1000:
            args.per_digit = 50
        if args.tstr_epochs == 20:
            args.tstr_epochs = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Seeds: {args.seeds} | guidance grid: {guidance_list} | "
          f"per_digit={args.per_digit} | compute_tstr={args.compute_tstr}")
    os.makedirs(args.output_dir, exist_ok=True)

    model = UNet(in_channels=1, base_channels=64, num_classes=10).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)
    model.eval()
    schedule = DiffusionSchedule(timesteps=1000, device=device).to(device)
    judge = load_cnn(args.cnn, device)
    print(f"Loaded checkpoint {args.checkpoint} and judge CNN {args.cnn}")

    real_test_loader = None
    if args.compute_tstr:
        _, real_test_loader = build_dataloaders(args.data_dir, args.tstr_batch_size, 2)
    tstr_map = {} if args.compute_tstr else load_tstr_map(args.tstr_json)
    if tstr_map:
        print(f"Loaded fallback oracle TSTR for guidances: {sorted(tstr_map)}")

    seed_results = []
    for seed in args.seeds:
        seed_results.append(run_one_seed(seed, model, schedule, judge, guidance_list,
                                         real_test_loader, tstr_map, args, device))

    agg = aggregate(seed_results, guidance_list, topk=3)
    print_summary(agg)

    out = {
        "metadata": {
            "checkpoint": args.checkpoint, "git_commit": get_git_commit(),
            "sampler": SAMPLER, "steps": STEPS, "eta": ETA,
            "seeds": args.seeds, "per_digit": args.per_digit,
            "compute_tstr": args.compute_tstr, "tstr_epochs": args.tstr_epochs,
            "guidance_grid": guidance_list, "tau_fraction": args.tau_fraction,
            "prdc_feature_space": "mnist_cnn.pt penultimate (judge CNN)",
        },
        "aggregate": agg,
        "per_seed": seed_results,
        "args": vars(args),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
