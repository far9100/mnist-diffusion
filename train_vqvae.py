"""
Stage 1 training: multi-scale residual VQ-VAE for VAR-mini.

Trains the tokenizer on MNIST. Outputs a checkpoint that Stage 2 freezes and
uses to build the token sequences fed to the scale-wise transformer.

Usage:
    uv run python train_vqvae.py
    uv run python train_vqvae.py --epochs 50 --batch-size 256
    uv run python train_vqvae.py --resume var_vqvae.pt --epochs 10
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from train import download_mnist
from var import VARVQVAE


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAR-mini VQ-VAE on MNIST")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4, 7])
    parser.add_argument("--commitment-cost", type=float, default=0.25)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--reinit-interval", type=int, default=500,
                        help="Re-init dead codebook entries every N steps (0 to disable)")
    parser.add_argument("--reinit-threshold", type=float, default=1.0,
                        help="EMA cluster size below which a code is dead")
    parser.add_argument("--sample-interval", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="samples_vqvae")
    parser.add_argument("--save-path", type=str, default="var_vqvae.pt")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def train():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    try:
        dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    except RuntimeError:
        print("Default MNIST download failed, trying fallback mirrors...")
        download_mnist("./data")
        dataset = datasets.MNIST("./data", train=True, download=False, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)

    model = VARVQVAE(
        in_channels=1,
        embedding_dim=args.embedding_dim,
        codebook_size=args.codebook_size,
        scales=tuple(args.scales),
        commitment_cost=args.commitment_cost,
        decay=args.ema_decay,
    ).to(device)

    if args.resume:
        sd = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(sd)
        print(f"Resumed from {args.resume}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Scales: {args.scales}  -> sequence length (excl. SOS): "
          f"{sum(s * s for s in args.scales)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Hold a fixed batch of validation images for visualisation across epochs
    vis_batch, _ = next(iter(dataloader))
    vis_batch = vis_batch[:16].to(device)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_recon = total_commit = 0.0
        running_perp = torch.zeros(len(args.scales), device=device)
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, _ in pbar:
            images = images.to(device, non_blocking=True)

            x_hat, _, commit_loss, perplexity = model(images)
            recon_loss = F.l1_loss(x_hat, images)
            loss = recon_loss + commit_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon += recon_loss.item()
            total_commit += commit_loss.item()
            running_perp += perplexity.detach()
            n_batches += 1
            global_step += 1

            if args.reinit_interval > 0 and global_step % args.reinit_interval == 0:
                n_dead = model.rvq.quantizer.reinit_dead_codes(
                    threshold=args.reinit_threshold)
                if n_dead > 0:
                    pbar.write(f"  [step {global_step}] reinitialised {n_dead} dead codes")

            pbar.set_postfix(
                recon=f"{recon_loss.item():.4f}",
                commit=f"{commit_loss.item():.4f}",
                ppl_max=f"{perplexity.max().item():.1f}",
            )

        avg_perp = (running_perp / n_batches).tolist()
        print(f"Epoch {epoch} - recon: {total_recon / n_batches:.4f}  "
              f"commit: {total_commit / n_batches:.4f}  "
              f"perplexity: {[f'{p:.1f}' for p in avg_perp]}")

        if epoch % args.sample_interval == 0 or epoch == 1 or epoch == args.epochs:
            save_reconstructions(model, vis_batch, epoch, args.output_dir)

    torch.save(model.state_dict(), args.save_path)
    print(f"Training complete. Model saved to {args.save_path}")


@torch.no_grad()
def save_reconstructions(model, batch, epoch, output_dir):
    """Save side-by-side input vs reconstruction grid."""
    model.eval()
    x_hat, _, _, _ = model(batch)
    # Stack: row 0 = inputs, row 1 = reconstructions
    pairs = torch.stack([batch, x_hat.clamp(-1, 1)], dim=1).flatten(0, 1)  # (32, 1, 28, 28)
    pairs = pairs * 0.5 + 0.5
    grid = make_grid(pairs, nrow=batch.shape[0])
    path = os.path.join(output_dir, f"recon_epoch_{epoch}.png")
    save_image(grid, path)
    print(f"  saved reconstruction to {path}")
    model.train()


if __name__ == "__main__":
    train()
