"""(steps × η × guidance) 聯合取樣掃描——C1 效用曲面與 C3a 選擇器輸入的框架。

在單一已訓練的 DDPM checkpoint 上，對每個 sampler config 量測：
  - 取樣的 wall-clock 時間與 images/sec，
  - TSTR 整體與各類別準確率（重用 evaluate.py），
  - MNIST-FID（重用 fid.py），
  - PRDC precision/recall/density/coverage，各類別平均（metrics_prdc.py）。
接著在整個 grid 上執行 CaF 選擇器並回報 regret@selected——
也就是用免訓練規則挑 config、而非 oracle-best 時，下游準確率損失了多少。

DDIM 與 DDPM 重用同一個已訓練模型（它是一種 sampler，而非另一個模型），
因此這是在聯合 grid 上、同一組權重的乾淨比較。

Usage:
    uv run python run_comparison.py --quick
    uv run python run_comparison.py --per-digit 1000 --tstr-epochs 20
"""

import argparse
import csv
import json
import os
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from ddpm import UNet, DiffusionSchedule
from inference import generate as gen_batches
from evaluate import (MNISTClassifier, train_classifier, evaluate,
                      build_dataloaders, determine_verdict, get_git_commit)
from fid import load_cnn, real_feature_stats, compute_fid
from analyze_distribution import extract_features, load_real_per_class
from metrics_prdc import compute_prdc_per_class
from selector import select_and_report

# 聯合 grid 元組：(sampler, steps, eta, guidance)。
FULL_GRID = (
    [("ddpm", 1000, 0.0, 3.0)]
    + [("ddim", s, 0.0, g) for s in (50, 20, 10) for g in (1.0, 2.0, 3.0, 5.0, 7.0)]
    + [("ddim", 50, 0.5, 3.0), ("ddim", 50, 1.0, 3.0)]
)
QUICK_GRID = [("ddim", s, 0.0, g) for s in (50, 10) for g in (1.0, 3.0, 7.0)]


def cell_name(sampler, steps, eta, guidance):
    if sampler == "ddpm":
        return f"ddpm_s1000_g{guidance:g}"
    return f"ddim_s{steps}_eta{eta:g}_g{guidance:g}"


def generate_dataset(model, schedule, sampler, steps, eta, guidance, per_digit,
                     batch_size, device):
    all_images, all_labels = [], []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.monotonic()
    for digit in range(10):
        for batch in gen_batches(model, schedule, digit, per_digit, batch_size,
                                 guidance, device, sampler=sampler, steps=steps, eta=eta):
            all_images.append(batch.cpu())
            all_labels.append(torch.full((batch.size(0),), digit, dtype=torch.long))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return torch.cat(all_images), torch.cat(all_labels), time.monotonic() - start


def run_tstr(images, labels, real_test_loader, device, epochs, lr, batch_size):
    """在產生的資料集上重新訓練一個 CNN；回傳 (overall, per_class_acc)。"""
    model = MNISTClassifier().to(device)
    loader = DataLoader(TensorDataset(images, labels), batch_size=batch_size,
                        shuffle=True, num_workers=0)
    train_classifier(model, loader, device, epochs, lr)
    acc, correct, total = evaluate(model, real_test_loader, device)
    per_class = {str(d): (100.0 * correct.get(d, 0) / total[d] if total.get(d) else None)
                 for d in range(10)}
    return acc, per_class


def format_table(rows):
    header = (f"  {'config':<22} {'imgs/s':>7} {'TSTR%':>7} {'FID':>8} "
              f"{'prec':>6} {'cov':>6} {'recall':>7}")
    lines = ["=" * 78, "  Joint (steps x eta x guidance) sweep", "=" * 78, header,
             "  " + "-" * 74]
    for r in rows:
        lines.append(f"  {r['config']:<22} {r['imgs_per_s']:>7.1f} {r['tstr_acc']:>7.2f} "
                     f"{r['fid']:>8.3f} {r['precision']:>6.3f} {r['coverage']:>6.3f} "
                     f"{r['recall']:>7.3f}")
    lines.append("=" * 78)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Joint sampler sweep + CaF selector.")
    parser.add_argument("--checkpoint", default="ddpm_mnist.pt")
    parser.add_argument("--cnn", default="mnist_cnn.pt")
    parser.add_argument("--per-digit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tstr-epochs", type=int, default=20)
    parser.add_argument("--tstr-lr", type=float, default=1e-3)
    parser.add_argument("--tstr-batch-size", type=int, default=64)
    parser.add_argument("--fid-per-class", type=int, default=1000)
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--tau-fraction", type=float, default=0.9)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        grid = QUICK_GRID
        if args.per_digit == 1000:
            args.per_digit = 50
        if args.tstr_epochs == 20:
            args.tstr_epochs = 3
    else:
        grid = FULL_GRID

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    model = UNet(in_channels=1, base_channels=64, num_classes=10).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)
    model.eval()
    schedule = DiffusionSchedule(timesteps=1000, device=device).to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    _, real_test_loader = build_dataloaders(args.data_dir, args.tstr_batch_size, 2)
    judge_cnn = load_cnn(args.cnn, device)
    real_images, real_labels = load_real_per_class(args.data_dir, args.fid_per_class, seed=0)
    real_stats = real_feature_stats(real_images, device=device, model=judge_cnn)
    real_feats = extract_features(judge_cnn, real_images, device).to(device)
    real_labels_d = real_labels.to(device)
    print(f"Cached FID + PRDC reference from {real_images.size(0)} real images.\n")

    rows = []
    for sampler, steps, eta, guidance in grid:
        name = cell_name(sampler, steps, eta, guidance)
        print(f"=== {name} ===")
        images, labels, secs = generate_dataset(
            model, schedule, sampler, steps, eta, guidance, args.per_digit,
            args.batch_size, device)
        n = images.size(0)
        imgs_per_s = n / secs if secs > 0 else float("nan")

        fid = compute_fid(images, device=device, model=judge_cnn, real_stats=real_stats)
        gen_feats = extract_features(judge_cnn, images, device).to(device)
        prdc, _ = compute_prdc_per_class(real_feats, real_labels_d, gen_feats,
                                         labels.to(device), nearest_k=args.nearest_k)
        tstr, per_class = run_tstr(images, labels, real_test_loader, device,
                                   args.tstr_epochs, args.tstr_lr, args.tstr_batch_size)
        verdict, _ = determine_verdict(tstr, 95.0, 90.0)
        print(f"  {imgs_per_s:.1f} imgs/s  FID={fid:.3f}  TSTR={tstr:.2f}%  "
              f"prec={prdc['precision']:.3f} cov={prdc['coverage']:.3f} recall={prdc['recall']:.3f}\n")

        rows.append({
            "config": name, "sampler": sampler, "steps": steps, "eta": eta,
            "guidance": guidance, "n_images": n, "wall_clock_s": round(secs, 3),
            "imgs_per_s": round(imgs_per_s, 2), "tstr_acc": round(tstr, 3),
            "tstr_per_class": per_class, "fid": round(fid, 4),
            "precision": round(prdc["precision"], 4), "recall": round(prdc["recall"], 4),
            "density": round(prdc["density"], 4), "coverage": round(prdc["coverage"], 4),
            "verdict": verdict,
        })

    # CaF tau 的參考 precision（real-vs-real 切分），以及在 grid 上執行選擇器。
    n_real = real_feats.size(0)
    perm = torch.randperm(n_real, generator=torch.Generator().manual_seed(0))
    h = n_real // 2
    ref_mean, _ = compute_prdc_per_class(
        real_feats[perm[h:]], real_labels_d[perm[h:]],
        real_feats[perm[:h]], real_labels_d[perm[:h]], nearest_k=args.nearest_k)
    sel_configs = [{"name": r["config"], "precision": r["precision"],
                    "coverage": r["coverage"], "recall": r["recall"],
                    "tstr": r["tstr_acc"]} for r in rows]
    selector_report = select_and_report(sel_configs, real_ref_precision=ref_mean["precision"],
                                         tau_fraction=args.tau_fraction, utility_key="tstr")

    csv_path = os.path.join(args.output_dir, "comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        cols = [k for k in rows[0].keys() if k != "tstr_per_class"]
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(args.output_dir, "comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {"checkpoint": args.checkpoint, "git_commit": get_git_commit(),
                         "device": str(device), "per_digit": args.per_digit,
                         "tstr_epochs": args.tstr_epochs,
                         "real_ref_precision": ref_mean["precision"],
                         "fid_note": "MNIST-FID (mnist_cnn.pt features), not Inception-FID"},
            "rows": rows, "selector_report": selector_report,
        }, f, indent=2, ensure_ascii=False)

    table = format_table(rows)
    with open(os.path.join(args.output_dir, "comparison.txt"), "w", encoding="utf-8") as f:
        f.write(table + "\n")

    print()
    print(table)
    print(f"\n  CaF selector: picked {selector_report['selected']} "
          f"(oracle {selector_report['oracle_best']}), "
          f"regret={selector_report['regret_at_selected']:.3f} pp, "
          f"rank {selector_report['rank']}/{selector_report['n_configs']}, "
          f"top-{selector_report['topk']} hit={selector_report['topk_hit']}")
    print(f"\nWrote {csv_path}, {json_path}, and comparison.txt")


if __name__ == "__main__":
    main()
