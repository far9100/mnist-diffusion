"""
Verify that the CNN evaluator (evaluate.MNISTClassifier) is competent enough
to be trusted as the "judge" of generated digit quality.

Loads a trained CNN checkpoint, runs it on the real MNIST test set, and checks
that overall accuracy and every per-class accuracy clear the configured thresholds.

Usage:
    uv run python test_classifier.py
    uv run python test_classifier.py --checkpoint mnist_cnn.pt
    uv run python test_classifier.py --threshold-overall 99.0 --threshold-per-class 97.0
"""

import argparse
import os
import sys

import torch

from evaluate import (
    MNISTClassifier,
    build_dataloaders,
    evaluate,
    format_metrics_block,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sanity-check the CNN evaluator on the real MNIST test set."
    )
    parser.add_argument("--checkpoint", default="mnist_cnn.pt",
                        help="CNN checkpoint to verify (default: mnist_cnn.pt)")
    parser.add_argument("--data-dir", default="./data",
                        help="MNIST data directory (default: ./data)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Evaluation batch size (default: 256)")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader workers (default: 2)")
    parser.add_argument("--threshold-overall", type=float, default=99.0,
                        help="Required overall accuracy in %% (default: 99.0)")
    parser.add_argument("--threshold-per-class", type=float, default=97.0,
                        help="Required per-class accuracy in %% (default: 97.0)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"\nERROR: CNN checkpoint not found at '{args.checkpoint}'")
        print("Train and save one first, e.g.:")
        print("  uv run python evaluate.py --save-cnn mnist_cnn.pt")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MNISTClassifier().to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    print(f"Loaded CNN checkpoint: {args.checkpoint}")

    _, test_loader = build_dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )

    overall_acc, per_class_correct, per_class_total = evaluate(model, test_loader, device)

    print()
    print(format_metrics_block(
        overall_acc, per_class_correct, per_class_total,
        "CNN Capability Check on Real MNIST Test Set",
    ))

    per_class_acc = {
        d: 100.0 * per_class_correct.get(d, 0) / per_class_total[d]
        for d in sorted(per_class_total.keys())
    }
    min_digit, min_acc = min(per_class_acc.items(), key=lambda kv: kv[1])

    failures = []
    if overall_acc < args.threshold_overall:
        failures.append(
            f"overall accuracy {overall_acc:.2f}% < {args.threshold_overall:.2f}%"
        )
    below_per_class = [
        (d, acc) for d, acc in per_class_acc.items()
        if acc < args.threshold_per_class
    ]
    for d, acc in below_per_class:
        failures.append(
            f"digit {d} accuracy {acc:.2f}% < {args.threshold_per_class:.2f}%"
        )

    verdict = "PASS" if not failures else "FAIL"

    print()
    print("=" * 50)
    print("  Classifier Capability Verdict")
    print("=" * 50)
    print(f"  Overall threshold       : {args.threshold_overall:.2f}%")
    print(f"  Per-class threshold     : {args.threshold_per_class:.2f}%")
    print(f"  Overall accuracy        : {overall_acc:.2f}%")
    print(f"  Min per-class accuracy  : {min_acc:.2f}% (digit {min_digit})")
    print(f"  Result                  : {verdict}")
    if failures:
        print("  Failures:")
        for f in failures:
            print(f"    - {f}")
    print("=" * 50)

    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
