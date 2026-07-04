"""
以訓練好的 DDPM 生成 MNIST 數字的推論腳本。

只輸出 .pt tensor 資料集（供 evaluate.py 在 TSTR 流程中使用）。模型隨時間
變化的生成品質視覺化由 train.py 在訓練期間產生，並存放在 samples/ 中。

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


def generate(model, schedule, digit, count, batch_size, guidance_scale, device,
             sampler="ddpm", steps=None, eta=0.0):
    """生成單一數字的 `count` 張影像，以 batch 為單位逐次 yield。

    sampler："ddpm"（ancestral，完整 T 個 steps）或 "ddim"（子序列 sampler，
    重用同一個訓練好的模型）。steps：DDIM 子序列長度。eta：DDIM 的隨機性
    （0=deterministic，配合完整 steps 時 1 ~ DDPM）。
    """
    generated = 0
    while generated < count:
        n = min(batch_size, count - generated)
        labels = torch.full((n,), digit, device=device, dtype=torch.long)
        if sampler == "ddim":
            images = schedule.ddim_sample_loop(
                model, shape=(n, 1, 28, 28), num_steps=steps, eta=eta,
                class_labels=labels, guidance_scale=guidance_scale,
            )
        else:
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
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm",
                        help="Sampler: ddpm (ancestral, full T) or ddim (default: ddpm)")
    parser.add_argument("--steps", type=int, default=50,
                        help="DDIM sub-sequence length (only used for --sampler ddim, default: 50)")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM stochasticity: 0=deterministic, 1 with full steps ~ DDPM (default: 0.0)")
    parser.add_argument("--output-dir", type=str, default="generated",
                        help="Output directory (default: generated/)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet(in_channels=1, base_channels=64, num_classes=10).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    schedule = DiffusionSchedule(timesteps=1000, device=device)
    if args.sampler == "ddim":
        print(f"Sampler: DDIM (steps={args.steps}, eta={args.eta})")
    else:
        print("Sampler: DDPM ancestral (1000 steps)")
    os.makedirs(args.output_dir, exist_ok=True)

    all_images = []
    all_labels = []

    for digit in args.digits:
        print(f"\nGenerating digit {digit} ({args.per_digit} images)...")
        digit_images = []

        for batch in tqdm(generate(model, schedule, digit, args.per_digit,
                                   args.batch_size, args.guidance_scale, device,
                                   sampler=args.sampler, steps=args.steps, eta=args.eta),
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
