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
"""

import argparse
import os
import sys

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


def print_report(overall_acc, per_class_correct, per_class_total, title):
    total_correct = sum(per_class_correct.values())
    total_images = sum(per_class_total.values())
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  {'Digit':<8} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-'*38}")
    for digit in range(10):
        if digit in per_class_total:
            correct = per_class_correct.get(digit, 0)
            total = per_class_total[digit]
            acc = 100.0 * correct / total
            print(f"  {digit:<8} {correct:>8} {total:>8} {acc:>9.2f}%")
        else:
            print(f"  {digit:<8} {'N/A':>8} {'N/A':>8} {'N/A':>10}")
    print(f"  {'-'*38}")
    print(f"  {'TOTAL':<8} {total_correct:>8} {total_images:>8} {overall_acc:>9.2f}%")
    print(f"{'='*50}")


def print_confusion_matrix(matrix):
    print("\n  Confusion Matrix (rows=actual, cols=predicted):")
    header = "       " + "".join(f"{d:>6}" for d in range(10))
    print(header)
    for i in range(10):
        row = f"  [{i}]  " + "".join(f"{matrix[i][j].item():>6}" for j in range(10))
        print(row)


def load_generated(path):
    if not os.path.exists(path):
        print(f"\nERROR: Generated dataset not found at '{path}'")
        print("Run inference first to create it:")
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
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MNISTClassifier().to(device)

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
        print_report(overall_acc, per_class_correct, per_class_total,
                     "CNN Accuracy on Real MNIST Test Set")

        if args.save_cnn:
            torch.save(model.state_dict(), args.save_cnn)
            print(f"\nCNN saved to {args.save_cnn}")

    images, labels = load_generated(args.generated)
    gen_dataset = TensorDataset(images, labels)
    gen_loader = DataLoader(gen_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    overall_acc, per_class_correct, per_class_total = evaluate(model, gen_loader, device)
    print_report(overall_acc, per_class_correct, per_class_total,
                 f"CNN Evaluation on Generated Images")
    print(f"  Source: {args.generated}")

    if args.confusion_matrix:
        gen_loader2 = DataLoader(gen_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=0)
        matrix = build_confusion_matrix(model, gen_loader2, device)
        print_confusion_matrix(matrix)


if __name__ == "__main__":
    main()
