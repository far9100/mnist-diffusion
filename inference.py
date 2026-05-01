"""
Inference script for generating MNIST digits with a trained DDPM.

Outputs only the .pt tensor dataset (consumed by evaluate.py for the TSTR
flow). Visualizations of the model's generation quality over time are
produced during training by train.py and live in samples/.

Usage:
    # Generate 100 images per digit (0-9), saved as a dataset
    uv run python inference.py

    # Only specific digits
    uv run python inference.py --digits 3 7

    # More images per digit (recommended for the canonical TSTR run)
    uv run python inference.py --per-digit 1000
"""

import argparse
import os

import torch
from tqdm import tqdm

from ddpm import UNet, DiffusionSchedule


def generate(model, schedule, digit, count, batch_size, guidance_scale, device):
    """Generate `count` images of a single digit, yielding in batches."""
    generated = 0
    while generated < count:
        n = min(batch_size, count - generated)
        labels = torch.full((n,), digit, device=device, dtype=torch.long)
        images = schedule.p_sample_loop(
            model, shape=(n, 1, 28, 28),
            class_labels=labels, guidance_scale=guidance_scale,
        )
        images = images.clamp(-1, 1)
        yield images
        generated += n


def main():
    parser = argparse.ArgumentParser(description="Generate MNIST digits with trained DDPM")
    parser.add_argument("--checkpoint", type=str, default="ddpm_mnist.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--digits", type=int, nargs="+", default=list(range(10)),
                        help="Which digits to generate (default: 0-9)")
    parser.add_argument("--per-digit", type=int, default=100,
                        help="Number of images per digit (default: 100)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for generation (default: 64)")
    parser.add_argument("--guidance-scale", type=float, default=3.0,
                        help="Classifier-free guidance scale (default: 3.0)")
    parser.add_argument("--output-dir", type=str, default="generated",
                        help="Output directory (default: generated/)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet(in_channels=1, base_channels=64, num_classes=10).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    schedule = DiffusionSchedule(timesteps=1000, device=device)
    os.makedirs(args.output_dir, exist_ok=True)

    all_images = []
    all_labels = []

    for digit in args.digits:
        print(f"\nGenerating digit {digit} ({args.per_digit} images)...")
        digit_images = []

        for batch in tqdm(generate(model, schedule, digit, args.per_digit,
                                   args.batch_size, args.guidance_scale, device),
                          total=(args.per_digit + args.batch_size - 1) // args.batch_size):
            digit_images.append(batch.cpu())

        digit_images = torch.cat(digit_images, dim=0)
        digit_labels = torch.full((digit_images.shape[0],), digit, dtype=torch.long)

        all_images.append(digit_images)
        all_labels.append(digit_labels)

    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    dataset_path = os.path.join(args.output_dir, "dataset.pt")
    torch.save({"images": all_images, "labels": all_labels}, dataset_path)
    print(f"\nDataset saved to {dataset_path}")
    print(f"  images: {all_images.shape}  (range [-1, 1], float32)")
    print(f"  labels: {all_labels.shape}  (int64)")

    total = all_images.shape[0]
    print(f"\nDone! Generated {total} images total.")
    print(f"\nUsage in downstream training:")
    print(f'  data = torch.load("{dataset_path}")')
    print(f'  images, labels = data["images"], data["labels"]')


if __name__ == "__main__":
    main()
