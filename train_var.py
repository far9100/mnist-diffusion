"""
Stage 2 訓練：VAR-mini 的 scale-wise transformer。

載入凍結的 VQ-VAE checkpoint，先把 MNIST 訓練集預先編碼成多尺度的
token 網格（快取到磁碟），接著以 10% 的類別標籤 dropout 進行
classifier-free guidance 來訓練 transformer。

Usage:
    uv run python train_var.py
    uv run python train_var.py --epochs 100 --batch-size 256
    uv run python train_var.py --resume var_transformer.pt --epochs 10
"""

import argparse
import math
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from train import download_mnist
from var import VARVQVAE, VARTransformer, generate


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAR-mini transformer on MNIST")
    parser.add_argument("--vqvae", type=str, default="var_vqvae.pt",
                        help="Path to a trained VQ-VAE checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-dropout", type=float, default=0.1,
                        help="Probability of replacing label with null class for CFG")
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cfg-scale", type=float, default=2.0,
                        help="CFG scale for periodic sampling visualisations")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--sample-interval", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--token-cache", type=str, default="data/var_tokens.pt")
    parser.add_argument("--output-dir", type=str, default="samples_var")
    parser.add_argument("--save-path", type=str, default="var_transformer.pt")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def build_token_cache(vqvae, mnist_dataset, cache_path, batch_size, num_workers, device):
    """把 `mnist_dataset` 中的每張影像編碼成多尺度的 token 網格，
    並將結果快取到磁碟。之後的訓練會重用這份快取。"""
    if os.path.exists(cache_path):
        print(f"Loading cached tokens from {cache_path}")
        return torch.load(cache_path, weights_only=True)

    print(f"Building token cache (one-time, ~30s)...")
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    vqvae.eval()
    loader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    K = len(vqvae.scales)
    scale_chunks = [[] for _ in range(K)]
    label_chunks = []
    for x, y in tqdm(loader, desc="encoding"):
        x = x.to(device, non_blocking=True)
        idx_list = vqvae.encode_to_indices(x)
        for k in range(K):
            scale_chunks[k].append(idx_list[k].cpu())
        label_chunks.append(y)

    cache = {
        "scales": list(vqvae.scales),
        "labels": torch.cat(label_chunks),
    }
    for k in range(K):
        cache[f"scale_{k}"] = torch.cat(scale_chunks[k])
    torch.save(cache, cache_path)
    n = cache["labels"].shape[0]
    size_mb = sum(cache[f"scale_{k}"].numel() * 8 for k in range(K)) / 1e6
    print(f"  cached {n} samples, scale tensors total ~{size_mb:.1f} MB")
    return cache


class TokenDataset(Dataset):
    """常駐記憶體的多尺度 token 資料集。"""

    def __init__(self, cache):
        self.scales = cache["scales"]
        self.labels = cache["labels"]
        self.scale_tensors = [cache[f"scale_{k}"] for k in range(len(self.scales))]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, i):
        # tuple：(label, scale0_idx, scale1_idx, ..., scaleK-1_idx)
        return (self.labels[i].clone(), *(t[i].clone() for t in self.scale_tensors))


def collate_tokens(batch):
    """模組層級的 collate（Windows 的 multiprocessing 不能用 closure）。

    每個 batch 元素是 (label, idx_0, idx_1, ..., idx_{K-1})；我們分別把每個
    scale 與 labels 各自堆疊起來。"""
    K = len(batch[0]) - 1
    labels = torch.stack([b[0] for b in batch])
    scales = [torch.stack([b[k + 1] for b in batch]) for k in range(K)]
    return labels, scales


def get_lr(step, warmup, total_steps, lr_max, lr_min):
    if step < warmup:
        return lr_max * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total_steps - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def train():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- 載入凍結的 VQ-VAE -----------------------------------------------
    vqvae = VARVQVAE().to(device)
    sd = torch.load(args.vqvae, map_location=device, weights_only=True)
    vqvae.load_state_dict(sd)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    print(f"Loaded VQ-VAE from {args.vqvae}")

    # ---- 建立／載入 token 快取 -----------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    try:
        mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transform)
    except RuntimeError:
        download_mnist("./data")
        mnist_train = datasets.MNIST("./data", train=True, download=False, transform=transform)

    cache = build_token_cache(vqvae, mnist_train, args.token_cache,
                              batch_size=512, num_workers=args.num_workers, device=device)
    dataset = TokenDataset(cache)
    K = len(dataset.scales)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_tokens, drop_last=True,
    )

    # ---- 建立 transformer ------------------------------------------------
    transformer = VARTransformer(
        scales=tuple(dataset.scales),
        embedding_dim=vqvae.embedding_dim,
        num_classes=10,
        codebook_size=vqvae.codebook_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    ).to(device)
    if args.resume:
        transformer.load_state_dict(
            torch.load(args.resume, map_location=device, weights_only=True))
        print(f"Resumed transformer from {args.resume}")

    optimizer = torch.optim.AdamW(transformer.parameters(),
                                  lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=args.weight_decay)

    n_params = sum(p.numel() for p in transformer.parameters())
    total_steps = args.epochs * len(dataloader)
    print(f"Transformer parameters: {n_params:,}")
    print(f"Sequence length: {transformer.total_tokens}  "
          f"Total training steps: {total_steps}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 訓練迴圈 ----------------------------------------------------
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        transformer.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for labels, scale_indices in pbar:
            labels = labels.to(device, non_blocking=True)
            scale_indices = [s.to(device, non_blocking=True) for s in scale_indices]

            # Classifier-free guidance：以 10% 的機率把標籤 dropout 成 null class。
            drop = torch.rand(labels.shape[0], device=device) < args.label_dropout
            labels = torch.where(drop, torch.full_like(labels, transformer.num_classes), labels)

            with torch.no_grad():
                cumulative = vqvae.rvq.cumulative_f_hat_per_scale(scale_indices)

            _, loss = transformer(labels, cumulative, target_indices=scale_indices)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), args.grad_clip)
            lr_now = get_lr(global_step, args.warmup_steps, total_steps,
                            args.lr, args.lr_min)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_now:.2e}")

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch} - avg loss: {avg_loss:.4f}  perplexity: {math.exp(avg_loss):.2f}")

        if epoch % args.sample_interval == 0 or epoch == 1 or epoch == args.epochs:
            save_samples(vqvae, transformer, epoch, device,
                         args.output_dir, args.cfg_scale, args.top_k, args.top_p)

    torch.save(transformer.state_dict(), args.save_path)
    print(f"Training complete. Transformer saved to {args.save_path}")


@torch.no_grad()
def save_samples(vqvae, transformer, label, device, output_dir,
                 cfg_scale, top_k, top_p, per_class=8):
    """生成一張 per_class×10 的網格（每一列 = 一個數字）並存檔。"""
    class_labels = torch.arange(10, device=device).repeat_interleave(per_class)
    images, _ = generate(transformer, vqvae, class_labels,
                         cfg_scale=cfg_scale, top_k=top_k, top_p=top_p)
    images = images * 0.5 + 0.5  # 轉到 [0,1]
    grid = make_grid(images, nrow=per_class)
    path = os.path.join(output_dir, f"epoch_{label}.png")
    save_image(grid, path)
    print(f"  saved samples to {path}")


if __name__ == "__main__":
    train()
