"""在真實 CIFAR-10 / CIFAR-100 上訓練 ResNet-18 judge，並校準 near-boundary threshold。

judge 的用途：作為標籤噪音與 near-boundary 診斷（mechanism.py）的「裁判」——量合成樣本的
margin = p(top1) - p(top2) 與 top1 是否等於給定標籤。judge 必須夠準，這些診斷才有意義（judge
弱則分不清「judge 爛」與「合成品質差」）。

threshold 校準：mechanism 的 near-boundary threshold 預設 0.5 是 MNIST 調的（MNIST 近乎可分，真實
near-boundary 比例僅約 0.9%，訊號飽和）。CIFAR 難得多，需以真實資料的 margin 分布分位數定一個
CIFAR 專用 threshold，使「near-boundary」在真實資料上有非平凡的比例、讓儀器有解析度。此處取真實
測試 margin 的 20 百分位（約 20% 真實樣本落在邊界附近）為 threshold。同一 p20 相對分位程序沿用到
CIFAR-100（D 包 D9「凍程序不凍數字」：程序固定，門檻數字隨資料）。

Usage:
    uv run python cifar_judge.py --dataset cifar10 --epochs 25
    uv run python cifar_judge.py --dataset cifar100 --epochs 25
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import torch

from cifar_classifier import ResNet18, train_classifier, evaluate
from mechanism import compute_margins
from datasets.cifar import load_real_per_class, build_test_loader, NUM_CLASSES


def main():
    p = argparse.ArgumentParser(description="Train a real-CIFAR judge and calibrate near-boundary threshold.")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--save", default=None, help="預設 checkpoints/<dataset>_judge.pt")
    p.add_argument("--output", default=None, help="預設 results/<dataset>_judge.json")
    p.add_argument("--quantile", type=float, default=0.20,
                   help="以真實測試 margin 的此分位數作 near-boundary threshold")
    args = p.parse_args()

    dataset = args.dataset
    num_classes = NUM_CLASSES[dataset]
    save = args.save or f"checkpoints/{dataset}_judge.pt"
    output = args.output or f"results/{dataset}_judge.json"

    start_ts = datetime.now(timezone.utc).isoformat()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  dataset={dataset} num_classes={num_classes}", flush=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 真實全訓練集（50000 張，平均分到各類：CIFAR-10 每類 5000、CIFAR-100 每類 500）
    train_per_class = 50000 // num_classes
    train_imgs, train_labels = load_real_per_class(dataset, train_per_class, seed=0, train=True)
    print(f"Judge 訓練集：{train_imgs.size(0)} 張真實 {dataset}", flush=True)

    model = ResNet18(num_classes=num_classes).to(device)
    train_classifier(model, train_imgs, train_labels, device, epochs=args.epochs,
                     lr=args.lr, batch_size=args.batch_size, augment=True, verbose=True)

    test_loader = build_test_loader(dataset, batch_size=256)
    overall, per_class = evaluate(model, test_loader, device, num_classes=num_classes)
    print(f"\nJudge 真實測試準確度：{overall:.2f}%", flush=True)

    torch.save(model.state_dict(), save)
    print(f"存 judge -> {save}", flush=True)

    # 校準 near-boundary threshold：真實測試 margin 分位數（全測試集，平均分到各類）
    cal_per_class = 10000 // num_classes
    test_imgs, test_labels = load_real_per_class(dataset, cal_per_class, seed=1, train=False)
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

    out = {"dataset": dataset, "num_classes": num_classes,
           "judge_ckpt": save, "epochs": args.epochs,
           "test_accuracy": overall, "per_class_accuracy": per_class,
           "margin_quantiles": {f"p{int(q*100):02d}": v for q, v in zip(qs, qvals)},
           "near_boundary_threshold": threshold, "threshold_quantile": args.quantile,
           "real_near_boundary_frac": real_nb_frac,
           "start_timestamp": start_ts, "end_timestamp": datetime.now(timezone.utc).isoformat(),
           "argv": sys.argv,
           "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                   "cudnn": torch.backends.cudnn.version()}}
    with open(output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
