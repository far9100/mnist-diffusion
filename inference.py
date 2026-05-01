"""
Inference script for generating MNIST digits with a trained DDPM.

Usage:
    # Generate 100 images per digit (0-9), saved as a dataset
    uv run python inference.py

    # Generate only specific digits
    uv run python inference.py --digits 3 7

    # Generate more images per digit
    uv run python inference.py --per-digit 500

    # Save a preview grid
    uv run python inference.py --digits 5 --per-digit 64 --save-grid
"""

import argparse
import os

import torch
from torchvision.utils import save_image, make_grid
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
    parser.add_argument("--save-grid", action="store_true",
                        help="Also save a preview grid image per digit (one row of 10 per 10 samples)")
    parser.add_argument("--save-images", action="store_true",
                        help="Save each generated image as an individual PNG under "
                             "<output-dir>/images/digit_<d>_sample_<NNN>.png")
    parser.add_argument("--save-pt", action="store_true", default=True,
                        help="Save as .pt tensor dataset (default: True)")
    parser.add_argument("--save-denoising", action="store_true",
                        help="Save a denoising process visualization grid")
    parser.add_argument("--denoising-steps", type=int, default=9,
                        help="Number of intermediate snapshots in denoising grid (default: 9)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
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

        # Save preview grid (full set, square layout)
        if args.save_grid:
            n_total = digit_images.shape[0]
            nrow = max(1, int(round(n_total ** 0.5)))
            grid = make_grid(digit_images * 0.5 + 0.5, nrow=nrow)
            grid_path = os.path.join(args.output_dir, f"grid_digit_{digit}.png")
            save_image(grid, grid_path)
            print(f"  Grid saved to {grid_path}  ({n_total} images, nrow={nrow})")

        # Save each image as an individual PNG (flat layout, all digits in one folder)
        if args.save_images:
            img_dir = os.path.join(args.output_dir, "images")
            os.makedirs(img_dir, exist_ok=True)
            for idx in range(digit_images.shape[0]):
                save_image(
                    digit_images[idx] * 0.5 + 0.5,
                    os.path.join(img_dir, f"digit_{digit}_sample_{idx:03d}.png"),
                )
            print(f"  Saved {digit_images.shape[0]} individual PNGs to {img_dir}/digit_{digit}_*.png")

    # Combine and save dataset
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if args.save_pt:
        dataset_path = os.path.join(args.output_dir, "dataset.pt")
        torch.save({"images": all_images, "labels": all_labels}, dataset_path)
        print(f"\nDataset saved to {dataset_path}")
        print(f"  images: {all_images.shape}  (range [-1, 1], float32)")
        print(f"  labels: {all_labels.shape}  (int64)")

    # Denoising process visualization
    if args.save_denoising:
        print("\nGenerating denoising process visualization...")
        denoise_digits = args.digits
        denoise_labels = torch.tensor(denoise_digits, device=device, dtype=torch.long)
        n_snapshots = args.denoising_steps
        _, snapshots = schedule.p_sample_loop(
            model, shape=(len(denoise_digits), 1, 28, 28),
            class_labels=denoise_labels, guidance_scale=args.guidance_scale,
            snapshot_steps=n_snapshots,
        )
        # rows = digits, columns = denoising steps (noise -> clean)
        rows = torch.cat([s.cpu() for s in snapshots], dim=0)
        rows = rows.clamp(-1, 1) * 0.5 + 0.5
        denoise_grid = make_grid(rows, nrow=len(snapshots))
        denoise_path = os.path.join(args.output_dir, "denoising_process.png")
        save_image(denoise_grid, denoise_path)
        print(f"Saved denoising process to {denoise_path}")

    total = all_images.shape[0]
    print(f"\nDone! Generated {total} images total.")
    print(f"\nUsage in downstream training:")
    print(f'  data = torch.load("{dataset_path}")')
    print(f'  images, labels = data["images"], data["labels"]')


if __name__ == "__main__":
    main()
