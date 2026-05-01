"""
CNN classifier for validating MNIST diffusion model output.

Trains a CNN on real MNIST data, then evaluates it on generated images
to check whether the diffusion model produces realistic, classifiable digits.

Usage:
    uv run python evaluate.py
    uv run python evaluate.py --generated generated/dataset.pt
    uv run python evaluate.py --save-cnn mnist_cnn.pt
    uv run python evaluate.py --checkpoint mnist_cnn.pt
    uv run python evaluate.py --epochs 5 --batch-size 512 --confusion-matrix
    uv run python evaluate.py --report report.txt --report-json report.json
    uv run python evaluate.py --threshold 95 --threshold-warn 90 --strict
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

from train import download_mnist


class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_dataloaders(data_dir, batch_size, num_workers):
    download_mnist(data_dir)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_set = datasets.MNIST(data_dir, train=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def train_classifier(model, train_loader, device, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
        scheduler.step()
        acc = 100.0 * correct / total
        print(f"  Epoch {epoch}/{epochs}  loss={total_loss/total:.4f}  train_acc={acc:.2f}%")


def evaluate(model, loader, device):
    model.eval()
    per_class_correct = {}
    per_class_total = {}
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            for pred, label in zip(preds, labels):
                key = label.item()
                per_class_total[key] = per_class_total.get(key, 0) + 1
                if pred.item() == key:
                    per_class_correct[key] = per_class_correct.get(key, 0) + 1
    total_correct = sum(per_class_correct.values())
    total_images = sum(per_class_total.values())
    overall_acc = 100.0 * total_correct / total_images if total_images > 0 else 0.0
    return overall_acc, per_class_correct, per_class_total


def build_confusion_matrix(model, loader, device):
    model.eval()
    matrix = torch.zeros(10, 10, dtype=torch.int64)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu()
            for pred, label in zip(preds, labels):
                matrix[label.item()][pred.item()] += 1
    return matrix


def format_metrics_block(overall_acc, per_class_correct, per_class_total, title):
    total_correct = sum(per_class_correct.values())
    total_images = sum(per_class_total.values())
    lines = []
    lines.append("=" * 50)
    lines.append(f"  {title}")
    lines.append("=" * 50)
    lines.append(f"  {'Digit':<8} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    lines.append(f"  {'-'*38}")
    for digit in range(10):
        if digit in per_class_total:
            correct = per_class_correct.get(digit, 0)
            total = per_class_total[digit]
            acc = 100.0 * correct / total
            lines.append(f"  {digit:<8} {correct:>8} {total:>8} {acc:>9.2f}%")
        else:
            lines.append(f"  {digit:<8} {'N/A':>8} {'N/A':>8} {'N/A':>10}")
    lines.append(f"  {'-'*38}")
    lines.append(f"  {'TOTAL':<8} {total_correct:>8} {total_images:>8} {overall_acc:>9.2f}%")
    lines.append("=" * 50)
    return "\n".join(lines)


def format_confusion_matrix(matrix):
    lines = ["  Confusion Matrix (rows=actual, cols=predicted):"]
    lines.append("       " + "".join(f"{d:>6}" for d in range(10)))
    for i in range(10):
        row = f"  [{i}]  " + "".join(f"{matrix[i][j].item():>6}" for j in range(10))
        lines.append(row)
    return "\n".join(lines)


def print_report(overall_acc, per_class_correct, per_class_total, title):
    print()
    print(format_metrics_block(overall_acc, per_class_correct, per_class_total, title))


def print_confusion_matrix(matrix):
    print()
    print(format_confusion_matrix(matrix))


def get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def determine_verdict(gen_acc, threshold, threshold_warn):
    if gen_acc >= threshold:
        return f"PASS (high quality, accuracy >= {threshold:.2f}%)", 0
    if gen_acc >= threshold_warn:
        return (
            f"PASS (acceptable, {threshold_warn:.2f}% <= accuracy < {threshold:.2f}%)",
            0,
        )
    return f"FAIL (accuracy < {threshold_warn:.2f}%)", 1


def build_report_text(*, args, device, real_metrics, gen_metrics, gen_matrix,
                     source_path, elapsed_s, verdict):
    gen_acc, gen_correct, gen_total = gen_metrics
    real_acc = real_metrics[0] if real_metrics else None
    total_samples = sum(gen_total.values())

    lines = []
    lines.append("=" * 60)
    lines.append("  MNIST DIFFUSION QUALITY EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Metadata")
    lines.append("-" * 60)
    lines.append(f"  Generated at      : {datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"  Device            : {device}")
    if device.type == "cuda":
        lines.append(f"  GPU               : {torch.cuda.get_device_name(0)}")
    lines.append(f"  PyTorch           : {torch.__version__}")
    commit = get_git_commit()
    if commit:
        lines.append(f"  Git commit        : {commit}")
    lines.append(f"  Generated dataset : {source_path}")
    lines.append(f"  Total samples     : {total_samples}")
    if args.checkpoint:
        ckpt_path = args.checkpoint
        try:
            ckpt_mtime = datetime.datetime.fromtimestamp(
                os.path.getmtime(ckpt_path)
            ).isoformat(timespec="seconds")
        except OSError:
            ckpt_mtime = "n/a"
        lines.append(f"  CNN checkpoint    : {ckpt_path} (mtime {ckpt_mtime})")
    else:
        lines.append(
            f"  CNN trained       : {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}"
        )
    lines.append(f"  Elapsed           : {elapsed_s:.1f}s")
    lines.append("")

    if real_metrics is not None:
        real_acc, real_correct, real_total = real_metrics
        lines.append(format_metrics_block(
            real_acc, real_correct, real_total,
            "CNN Baseline on Real MNIST Test Set",
        ))
        lines.append("")

    lines.append(format_metrics_block(
        gen_acc, gen_correct, gen_total,
        "CNN Evaluation on Generated Images",
    ))
    lines.append(f"  Source: {source_path}")
    if real_acc is not None:
        gap = gen_acc - real_acc
        lines.append(f"  Gap vs baseline   : {gap:+.2f} pp")
    lines.append("")

    lines.append(format_confusion_matrix(gen_matrix))
    lines.append("")

    lines.append("Quality Verdict")
    lines.append("-" * 60)
    lines.append(f"  Threshold (pass)        : {args.threshold:.2f}%")
    lines.append(f"  Threshold (acceptable)  : {args.threshold_warn:.2f}%")
    lines.append(f"  Generated accuracy      : {gen_acc:.2f}%")
    lines.append(f"  Result                  : {verdict}")
    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def build_report_json(*, args, device, real_metrics, gen_metrics, gen_matrix,
                     source_path, elapsed_s, verdict, exit_code):
    gen_acc, gen_correct, gen_total = gen_metrics

    def _per_class_dict(correct, total):
        return {
            str(d): {
                "correct": correct.get(d, 0),
                "total": total[d],
                "accuracy": 100.0 * correct.get(d, 0) / total[d],
            }
            for d in sorted(total.keys())
        }

    real_block = None
    real_acc = None
    if real_metrics is not None:
        real_acc, real_correct, real_total = real_metrics
        real_block = {
            "overall_accuracy": real_acc,
            "per_class": _per_class_dict(real_correct, real_total),
        }

    ckpt_meta = None
    if args.checkpoint:
        try:
            ckpt_meta = {
                "path": args.checkpoint,
                "mtime": datetime.datetime.fromtimestamp(
                    os.path.getmtime(args.checkpoint)
                ).isoformat(timespec="seconds"),
            }
        except OSError:
            ckpt_meta = {"path": args.checkpoint, "mtime": None}

    return {
        "metadata": {
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "pytorch": torch.__version__,
            "git_commit": get_git_commit(),
            "generated_dataset": source_path,
            "total_samples": sum(gen_total.values()),
            "cnn_checkpoint": ckpt_meta,
            "cnn_train_config": (
                None if args.checkpoint
                else {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr}
            ),
            "elapsed_s": elapsed_s,
        },
        "real_baseline": real_block,
        "generated": {
            "overall_accuracy": gen_acc,
            "per_class": _per_class_dict(gen_correct, gen_total),
            "gap_vs_baseline_pp": (gen_acc - real_acc) if real_acc is not None else None,
        },
        "confusion_matrix": gen_matrix.tolist(),
        "thresholds": {
            "pass": args.threshold,
            "acceptable": args.threshold_warn,
        },
        "verdict": verdict,
        "exit_code": exit_code,
    }


def load_generated(path):
    if not os.path.exists(path):
        print(f"\nERROR: Generated dataset not found at '{path}'")
        if os.path.exists("ex/dataset.pt"):
            print("Tip: this repo ships a ready-to-use example you can evaluate immediately:")
            print("  uv run python evaluate.py --generated ex/dataset.pt")
        print("Or run inference to produce a fresh one:")
        print(f"  uv run python inference.py --per-digit 100 --output-dir generated")
        print("Then re-run evaluate.py.")
        sys.exit(1)

    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or "images" not in data or "labels" not in data:
        print(f"\nERROR: '{path}' does not have the expected format.")
        print("Expected a dict with keys 'images' and 'labels'.")
        print("Re-generate using: uv run python inference.py")
        sys.exit(1)

    images = data["images"]
    labels = data["labels"]
    print(f"Loaded generated dataset: {images.shape[0]} images, "
          f"labels {labels.min().item()}–{labels.max().item()}")
    return images, labels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a CNN on real MNIST and evaluate on diffusion-generated images."
    )
    parser.add_argument("--generated", default="generated/dataset.pt",
                        help="Path to generated .pt file (default: generated/dataset.pt)")
    parser.add_argument("--checkpoint", default=None,
                        help="Pre-trained CNN checkpoint path; skips training if provided")
    parser.add_argument("--save-cnn", default=None,
                        help="Save trained CNN weights to this path")
    parser.add_argument("--epochs", type=int, default=10,
                        help="CNN training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for training and evaluation (default: 256)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--data-dir", default="./data",
                        help="MNIST data directory (default: ./data)")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader workers for MNIST (default: 2)")
    parser.add_argument("--confusion-matrix", action="store_true",
                        help="Print confusion matrix for generated images")
    parser.add_argument("--report", default=None,
                        help="Write a human-readable text report to this path")
    parser.add_argument("--report-json", default=None,
                        help="Write a machine-readable JSON report to this path")
    parser.add_argument("--threshold", type=float, default=95.0,
                        help="Pass threshold for overall accuracy in %% (default: 95)")
    parser.add_argument("--threshold-warn", type=float, default=90.0,
                        help="Acceptable threshold for overall accuracy in %% (default: 90)")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with non-zero code if accuracy is below the warn threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MNISTClassifier().to(device)

    real_metrics = None

    if args.checkpoint:
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=device, weights_only=True)
        )
        print(f"Loaded CNN checkpoint: {args.checkpoint}")
    else:
        print("\nTraining CNN on real MNIST...")
        train_loader, test_loader = build_dataloaders(
            args.data_dir, args.batch_size, args.num_workers
        )
        train_classifier(model, train_loader, device, args.epochs, args.lr)

        overall_acc, per_class_correct, per_class_total = evaluate(model, test_loader, device)
        real_metrics = (overall_acc, per_class_correct, per_class_total)
        print_report(overall_acc, per_class_correct, per_class_total,
                     "CNN Accuracy on Real MNIST Test Set")

        if args.save_cnn:
            torch.save(model.state_dict(), args.save_cnn)
            print(f"\nCNN saved to {args.save_cnn}")

    images, labels = load_generated(args.generated)
    gen_dataset = TensorDataset(images, labels)
    gen_loader = DataLoader(gen_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    gen_overall, gen_correct, gen_total = evaluate(model, gen_loader, device)
    gen_metrics = (gen_overall, gen_correct, gen_total)
    print_report(gen_overall, gen_correct, gen_total,
                 "CNN Evaluation on Generated Images")
    print(f"  Source: {args.generated}")

    need_matrix = args.confusion_matrix or args.report or args.report_json
    gen_matrix = None
    if need_matrix:
        matrix_loader = DataLoader(gen_dataset, batch_size=args.batch_size, shuffle=False,
                                   num_workers=0)
        gen_matrix = build_confusion_matrix(model, matrix_loader, device)
        if args.confusion_matrix:
            print_confusion_matrix(gen_matrix)

    verdict, exit_code = determine_verdict(
        gen_overall, args.threshold, args.threshold_warn
    )
    elapsed_s = time.monotonic() - start_time

    print()
    print("=" * 50)
    print("  Quality Verdict")
    print("=" * 50)
    print(f"  Threshold (pass)        : {args.threshold:.2f}%")
    print(f"  Threshold (acceptable)  : {args.threshold_warn:.2f}%")
    print(f"  Generated accuracy      : {gen_overall:.2f}%")
    print(f"  Result                  : {verdict}")
    print("=" * 50)

    if args.report:
        text = build_report_text(
            args=args, device=device,
            real_metrics=real_metrics, gen_metrics=gen_metrics,
            gen_matrix=gen_matrix,
            source_path=args.generated, elapsed_s=elapsed_s,
            verdict=verdict,
        )
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\nText report written to {args.report}")

    if args.report_json:
        data = build_report_json(
            args=args, device=device,
            real_metrics=real_metrics, gen_metrics=gen_metrics,
            gen_matrix=gen_matrix,
            source_path=args.generated, elapsed_s=elapsed_s,
            verdict=verdict, exit_code=exit_code,
        )
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"JSON report written to {args.report_json}")

    if args.strict and exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
