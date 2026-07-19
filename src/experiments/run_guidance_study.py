"""Guidance 強度研究：解耦視覺保真度（FID）與下游效用（TSTR），並以類內多樣性解釋落差。

sampler 固定不變（DDIM, eta=0, steps=50），使 guidance scale 成為
唯一的變數。對每個 guidance 值量測：
  - FID（視覺保真度；重用 fid.py），
  - TSTR 準確率（下游效用；重用 evaluate.py），
  - 於 pixel 與 CNN 特徵空間的類內多樣性、各類別 pixel std、
    平均 max-softmax confidence，以及 gen/real 多樣性比值（重用
    analyze_distribution.py）。

假設：提高 guidance 會銳化個別樣本（FID 在某個範圍內改善），
但會塌陷類內多樣性，因此 FID 最佳的 guidance 與
TSTR 最佳的 guidance 不同。結果 -> results/guidance_study.{csv,json,txt}。

Usage:
    uv run python run_guidance_study.py --quick
    uv run python run_guidance_study.py --per-digit 1000 --tstr-epochs 20
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py

import argparse
import csv
import json
import os
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from ddpm import UNet, DiffusionSchedule
from inference import generate as gen_batches
from evaluate import MNISTClassifier, train_classifier, evaluate, build_dataloaders, get_git_commit
from fid import load_cnn, real_feature_stats, compute_fid
from analyze_distribution import (extract_features, softmax_confidence,
                                  mean_pairwise_l2, load_real_per_class)

FULL_GUIDANCE = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
QUICK_GUIDANCE = [1.0, 3.0, 7.0]

SAMPLER, STEPS, ETA = "ddim", 50, 0.0  # 整個掃描過程固定不變


def generate_dataset(model, schedule, guidance, per_digit, batch_size, device):
    all_images, all_labels = [], []
    for digit in range(10):
        for batch in gen_batches(model, schedule, digit, per_digit, batch_size,
                                 guidance, device, sampler=SAMPLER, steps=STEPS, eta=ETA):
            all_images.append(batch.cpu())
            all_labels.append(torch.full((batch.size(0),), digit, dtype=torch.long))
    return torch.cat(all_images), torch.cat(all_labels)


def diversity_metrics(images, labels, judge_cnn, device):
    """對一個有標籤的資料集，計算各類別平均的多樣性 / confidence 指標。"""
    feats = extract_features(judge_cnn, images, device)
    conf = softmax_confidence(judge_cnn, images, device)
    pix_divs, feat_divs, pix_stds = [], [], []
    for d in range(10):
        m = labels == d
        if m.sum().item() < 2:
            continue
        pix = images[m].view(m.sum().item(), -1)
        pix_divs.append(mean_pairwise_l2(pix))
        feat_divs.append(mean_pairwise_l2(feats[m]))
        pix_stds.append(images[m].std(dim=0).mean().item())
    return {
        "diversity_pixel": sum(pix_divs) / len(pix_divs),
        "diversity_feat": sum(feat_divs) / len(feat_divs),
        "pixel_std": sum(pix_stds) / len(pix_stds),
        "mean_confidence": conf.mean().item(),
    }


def run_tstr(images, labels, real_test_loader, device, epochs, lr, batch_size):
    model = MNISTClassifier().to(device)
    loader = DataLoader(TensorDataset(images, labels), batch_size=batch_size,
                        shuffle=True, num_workers=0)
    train_classifier(model, loader, device, epochs, lr)
    acc, _, _ = evaluate(model, real_test_loader, device)
    return acc


def format_table(rows):
    header = (f"  {'guidance':>8} {'FID':>9} {'TSTR%':>7} {'div_pix':>8} "
              f"{'div_feat':>9} {'pix_std':>8} {'confid':>7} {'div_ratio':>9}")
    lines = ["=" * 82, "  Guidance study (DDIM, eta=0, steps=50)", "=" * 82, header,
             "  " + "-" * 78]
    for r in rows:
        lines.append(f"  {r['guidance']:>8g} {r['fid']:>9.3f} {r['tstr_acc']:>7.2f} "
                     f"{r['diversity_pixel']:>8.3f} {r['diversity_feat']:>9.3f} "
                     f"{r['pixel_std']:>8.4f} {r['mean_confidence']:>7.4f} "
                     f"{r['diversity_ratio_feat']:>9.3f}")
    lines.append("=" * 82)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Guidance vs diversity vs utility study.")
    parser.add_argument("--checkpoint", default="ddpm_mnist.pt")
    parser.add_argument("--cnn", default="mnist_cnn.pt")
    parser.add_argument("--per-digit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tstr-epochs", type=int, default=20)
    parser.add_argument("--tstr-lr", type=float, default=1e-3)
    parser.add_argument("--tstr-batch-size", type=int, default=64)
    parser.add_argument("--fid-per-class", type=int, default=1000)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    guidance_list = FULL_GUIDANCE
    if args.quick:
        guidance_list = QUICK_GUIDANCE
        if args.per_digit == 1000:
            args.per_digit = 50
        if args.tstr_epochs == 20:
            args.tstr_epochs = 3

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
    real_div = diversity_metrics(real_images, real_labels, judge_cnn, device)
    print(f"Real reference: div_feat={real_div['diversity_feat']:.3f}, "
          f"confidence={real_div['mean_confidence']:.4f}\n")

    rows = []
    for guidance in guidance_list:
        print(f"=== guidance={guidance:g} ===")
        t0 = time.monotonic()
        images, labels = generate_dataset(model, schedule, guidance,
                                          args.per_digit, args.batch_size, device)
        print(f"  generated {images.size(0)} images in {time.monotonic()-t0:.1f}s")

        fid = compute_fid(images, device=device, model=judge_cnn, real_stats=real_stats)
        div = diversity_metrics(images, labels, judge_cnn, device)
        tstr = run_tstr(images, labels, real_test_loader, device,
                        args.tstr_epochs, args.tstr_lr, args.tstr_batch_size)
        ratio_feat = (div["diversity_feat"] / real_div["diversity_feat"]
                      if real_div["diversity_feat"] > 0 else float("nan"))
        print(f"  FID={fid:.3f}  TSTR={tstr:.2f}%  div_feat={div['diversity_feat']:.3f} "
              f"(ratio {ratio_feat:.3f})  confidence={div['mean_confidence']:.4f}\n")

        rows.append({
            "guidance": guidance, "fid": round(fid, 4), "tstr_acc": round(tstr, 3),
            "diversity_pixel": round(div["diversity_pixel"], 4),
            "diversity_feat": round(div["diversity_feat"], 4),
            "pixel_std": round(div["pixel_std"], 5),
            "mean_confidence": round(div["mean_confidence"], 5),
            "diversity_ratio_feat": round(ratio_feat, 4),
        })

    # 核心分析：FID 最佳的 guidance 是否 != TSTR 最佳的 guidance？
    best_fid = min(rows, key=lambda r: r["fid"])
    best_tstr = max(rows, key=lambda r: r["tstr_acc"])
    decoupled = best_fid["guidance"] != best_tstr["guidance"]
    analysis = {
        "fid_optimal_guidance": best_fid["guidance"],
        "fid_optimal_value": best_fid["fid"],
        "tstr_optimal_guidance": best_tstr["guidance"],
        "tstr_optimal_value": best_tstr["tstr_acc"],
        "fid_and_tstr_optima_differ": decoupled,
    }

    csv_path = os.path.join(args.output_dir, "guidance_study.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(args.output_dir, "guidance_study.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "checkpoint": args.checkpoint, "git_commit": get_git_commit(),
                "sampler": SAMPLER, "steps": STEPS, "eta": ETA,
                "per_digit": args.per_digit, "tstr_epochs": args.tstr_epochs,
                "fid_note": "MNIST-FID (mnist_cnn.pt features), not Inception-FID",
            },
            "real_reference": real_div,
            "rows": rows,
            "analysis": analysis,
        }, f, indent=2, ensure_ascii=False)

    table = format_table(rows)
    txt_path = os.path.join(args.output_dir, "guidance_study.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")

    print()
    print(table)
    print()
    print(f"  FID-optimal guidance : {best_fid['guidance']:g} (FID={best_fid['fid']:.3f})")
    print(f"  TSTR-optimal guidance: {best_tstr['guidance']:g} (TSTR={best_tstr['tstr_acc']:.2f}%)")
    print(f"  => FID and TSTR optima {'DIFFER' if decoupled else 'coincide'}"
          f" (core hypothesis {'supported' if decoupled else 'not supported'} at this resolution)")
    print(f"\nWrote {csv_path}, {json_path}, {txt_path}")


if __name__ == "__main__":
    main()
