"""在真實 CIFAR-10 上訓練 ResNet-18 judge，並校準 CIFAR 專用的 near-boundary threshold。

judge 的用途：作為標籤噪音與 near-boundary 診斷（mechanism.py）的「裁判」——量合成樣本的
margin = p(top1) - p(top2) 與 top1 是否等於給定標籤。judge 必須夠準，這些診斷才有意義（judge
弱則分不清「judge 爛」與「合成品質差」）。

threshold 校準：mechanism 的 near-boundary threshold 預設 0.5 是 MNIST 調的（MNIST 近乎可分，真實
near-boundary 比例僅約 0.9%，訊號飽和）。CIFAR 難得多，需以真實資料的 margin 分布分位數定一個
CIFAR 專用 threshold，使「near-boundary」在真實資料上有非平凡的比例、讓儀器有解析度。此處取真實
測試 margin 的 20 百分位（約 20% 真實樣本落在邊界附近）為 threshold。

Usage:
    uv run python cifar_judge.py --epochs 25
"""

import argparse
import json
import os

import torch

from cifar_classifier import ResNet18, train_classifier, evaluate
from mechanism import compute_margins
from datasets.cifar import load_real_per_class, build_test_loader


def main():
    p = argparse.ArgumentParser(description="Train a real-CIFAR-10 judge and calibrate near-boundary threshold.")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--save", default="checkpoints/cifar10_judge.pt")
    p.add_argument("--output", default="results/cifar10_judge.json")
    p.add_argument("--quantile", type=float, default=0.20,
                   help="以真實測試 margin 的此分位數作 near-boundary threshold")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 真實 CIFAR-10 全訓練集（每類 5000）
    train_imgs, train_labels = load_real_per_class("cifar10", 5000, seed=0, train=True)
    print(f"Judge 訓練集：{train_imgs.size(0)} 張真實 CIFAR-10", flush=True)

    model = ResNet18(num_classes=10).to(device)
    train_classifier(model, train_imgs, train_labels, device, epochs=args.epochs,
                     lr=args.lr, batch_size=args.batch_size, augment=True, verbose=True)

    test_loader = build_test_loader("cifar10", batch_size=256)
    overall, per_class = evaluate(model, test_loader, device, num_classes=10)
    print(f"\nJudge 真實測試準確度：{overall:.2f}%", flush=True)

    torch.save(model.state_dict(), args.save)
    print(f"存 judge -> {args.save}", flush=True)

    # 校準 near-boundary threshold：真實測試 margin 分位數
    test_imgs, test_labels = load_real_per_class("cifar10", 1000, seed=1, train=False)
    margins, _ = compute_margins(model, test_imgs, device)
    qs = [0.05, 0.10, 0.20, 0.25, 0.50]
    qvals = torch.quantile(margins, torch.tensor(qs)).tolist()
    threshold = float(torch.quantile(margins, torch.tensor(args.quantile)).item())
    real_nb_frac = float((margins < threshold).float().mean().item())

    print("\n真實測試 margin 分位數：")
    for q, v in zip(qs, qvals):
        print(f"  p{int(q*100):02d} = {v:.4f}")
    print(f"\nCIFAR near-boundary threshold（p{int(args.quantile*100)}）= {threshold:.4f}"
          f"  -> 真實 near-boundary 比例 {real_nb_frac:.3f}")

    out = {"judge_ckpt": args.save, "epochs": args.epochs,
           "test_accuracy": overall, "per_class_accuracy": per_class,
           "margin_quantiles": {f"p{int(q*100):02d}": v for q, v in zip(qs, qvals)},
           "near_boundary_threshold": threshold, "threshold_quantile": args.quantile,
           "real_near_boundary_frac": real_nb_frac}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
