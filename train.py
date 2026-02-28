"""
Training loop and sampling for DDPM on MNIST.

Usage:
    uv run python train.py
    uv run python train.py --epochs 50 --lr 1e-4 --batch-size 256
    uv run python train.py --resume ddpm_mnist.pt --epochs 10
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from ddpm import UNet, DiffusionSchedule


def download_mnist(data_dir="./data"):
    """Download MNIST with fallback mirrors if the default source fails."""
    raw_dir = os.path.join(data_dir, "MNIST", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    files = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
    ]

    mirrors = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
    ]

    import gzip
    import shutil
    import hashlib
    from urllib.request import urlretrieve

    for filename, expected_md5 in files:
        dest = os.path.join(raw_dir, filename)
        # Check if already downloaded and valid
        if os.path.exists(dest):
            md5 = hashlib.md5(open(dest, "rb").read()).hexdigest()
            if md5 == expected_md5:
                continue
            os.remove(dest)
        # Also remove any corrupt extracted file
        extracted_path = os.path.join(raw_dir, filename.replace(".gz", ""))
        if os.path.exists(extracted_path):
            os.remove(extracted_path)
        downloaded = False
        for mirror in mirrors:
            url = mirror + filename
            try:
                print(f"Downloading {url} ...")
                urlretrieve(url, dest)
                md5 = hashlib.md5(open(dest, "rb").read()).hexdigest()
                if md5 != expected_md5:
                    os.remove(dest)
                    continue
                downloaded = True
                break
            except Exception as e:
                print(f"  Mirror failed: {e}")
                if os.path.exists(dest):
                    os.remove(dest)
        if not downloaded:
            raise RuntimeError(f"Failed to download {filename} from all mirrors.")

    # Extract .gz files
    for filename, _ in files:
        gz_path = os.path.join(raw_dir, filename)
        extracted_path = os.path.join(raw_dir, filename.replace(".gz", ""))
        if os.path.exists(extracted_path) and os.path.getsize(extracted_path) > 0:
            continue
        with gzip.open(gz_path, "rb") as f_in, open(extracted_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DDPM on MNIST")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Training batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument("--timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps (default: 1000)")
    parser.add_argument("--base-channels", type=int, default=64,
                        help="Base channel count for UNet (default: 64)")
    parser.add_argument("--sample-interval", type=int, default=5,
                        help="Save samples every N epochs (default: 5)")
    parser.add_argument("--guidance-scale", type=float, default=3.0,
                        help="Classifier-free guidance scale for sampling (default: 3.0)")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader workers (default: 2)")
    parser.add_argument("--output-dir", type=str, default="samples",
                        help="Directory for sample images (default: samples/)")
    parser.add_argument("--save-path", type=str, default="ddpm_mnist.pt",
                        help="Path to save model weights (default: ddpm_mnist.pt)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from a checkpoint")
    return parser.parse_args()


def train():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data — normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Download MNIST with fallback mirrors, then load without re-downloading
    try:
        dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    except RuntimeError:
        print("Default MNIST download failed, trying fallback mirrors...")
        download_mnist("./data")
        dataset = datasets.MNIST("./data", train=True, download=False, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # Model
    model = UNet(in_channels=1, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    schedule = DiffusionSchedule(timesteps=args.timesteps, device=device)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
        print(f"Resumed from checkpoint: {args.resume}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            bs = images.shape[0]

            # Classifier-free guidance: 10% label dropout
            drop_mask = torch.rand(bs, device=device) < 0.1
            labels = torch.where(drop_mask, torch.full_like(labels, 10), labels)

            # Random timesteps
            t = torch.randint(0, args.timesteps, (bs,), device=device)

            # Forward diffusion
            noise = torch.randn_like(images)
            x_t = schedule.q_sample(images, t, noise)

            # Predict noise
            pred_noise = model(x_t, t, labels)
            loss = torch.nn.functional.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} — avg loss: {avg_loss:.4f}")

        # Generate samples periodically
        if epoch % args.sample_interval == 0 or epoch == 1:
            generate_samples(model, schedule, epoch, device,
                             args.output_dir, args.guidance_scale)

    # Final samples
    generate_samples(model, schedule, "final", device,
                     args.output_dir, args.guidance_scale)
    torch.save(model.state_dict(), args.save_path)
    print(f"Training complete. Model saved to {args.save_path}")


@torch.no_grad()
def generate_samples(model, schedule, label, device,
                     output_dir="samples", guidance_scale=3.0):
    """Generate a 10x8 grid of samples (8 per digit) and save to disk."""
    model.eval()
    # 8 samples per digit, 10 digits
    class_labels = torch.arange(10, device=device).repeat_interleave(8)
    images = schedule.p_sample_loop(model, shape=(80, 1, 28, 28),
                                    class_labels=class_labels,
                                    guidance_scale=guidance_scale)
    images = images.clamp(-1, 1) * 0.5 + 0.5
    grid = make_grid(images, nrow=8)
    path = os.path.join(output_dir, f"epoch_{label}.png")
    save_image(grid, path)
    print(f"Saved samples to {path}")

    # Denoising process visualization: one sample per digit, 10 snapshots
    denoise_labels = torch.arange(10, device=device)
    _, snapshots = schedule.p_sample_loop(
        model, shape=(10, 1, 28, 28),
        class_labels=denoise_labels, guidance_scale=guidance_scale,
        snapshot_steps=9,
    )
    # snapshots: list of 10 tensors (9 intermediate + 1 final), each (10, 1, 28, 28)
    # Build grid: rows = digits (0-9), columns = denoising steps (noise -> clean)
    rows = torch.cat(snapshots, dim=0)  # (10*10, 1, 28, 28)
    rows = rows.clamp(-1, 1) * 0.5 + 0.5
    denoise_grid = make_grid(rows, nrow=len(snapshots))
    denoise_path = os.path.join(output_dir, f"denoise_epoch_{label}.png")
    save_image(denoise_grid, denoise_path)
    print(f"Saved denoising process to {denoise_path}")

    model.train()


if __name__ == "__main__":
    train()
