"""在 CIFAR-10 / CIFAR-100 上訓練類別條件、支援 CFG 的擴散模型。

Phase 1 主軸 backbone。原封不動重用 sandbox 已驗證的元件：
UNet + DiffusionSchedule（皆與解析度/通道數無關）+ classifier-free
guidance（把 label dropout 到 null class）+ DDIM(eta) sampler。只有資料
（3 通道 32x32）、輕量增強與 EMA 是新的——EMA 對 CIFAR FID 影響很大。

與計畫對應的設計選擇：
  - 訓練全程週期性存 checkpoint（--ckpt-interval）。早期 checkpoint 同時
    充當 autoguidance（Karras 2024）所需的「模型的弱版本」＝第二種 guidance
    方法（C3b）。
  - Label dropout p（預設 0.1）會訓練無條件路徑，使在 MNIST 上驗證過的同一個
    classifier-free guidance 旋鈕能遷移到 CIFAR。

Usage:
    uv run python train_cifar.py --dataset cifar10 --epochs 500
    uv run python train_cifar.py --dataset cifar100 --num-classes 100 --epochs 600
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py

import argparse
import copy
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from ddpm import UNet, DiffusionSchedule


class EMA:
    """模型參數的指數移動平均（EMA），是擴散模型 FID 的標準作法。"""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


def parse_args():
    p = argparse.ArgumentParser(description="Train CFG-capable diffusion on CIFAR.")
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--base-channels", type=int, default=128)
    p.add_argument("--channel-mults", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--label-dropout", type=float, default=0.1)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--guidance-scale", type=float, default=2.0)
    p.add_argument("--sample-interval", type=int, default=10)
    p.add_argument("--ckpt-interval", type=int, default=25,
                   help="Save a (numbered) checkpoint every N epochs; early ones "
                        "serve as the weak model for autoguidance.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", type=str, default="samples_cifar")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--save-path", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def build_loader(args):
    # 正規化到 [-1, 1]（3 通道），輕量水平翻轉增強（CIFAR 的標準作法）。
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    cls = datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100
    ds = cls("./data", train=True, download=True, transform=transform)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                      num_workers=args.num_workers, pin_memory=True, drop_last=True)


@torch.no_grad()
def sample_grid(model, schedule, path, num_classes, device, guidance_scale):
    model.eval()
    per = 8 if num_classes <= 10 else 1
    shown = min(num_classes, 10)
    labels = torch.arange(shown, device=device).repeat_interleave(per)
    imgs = schedule.ddim_sample_loop(model, shape=(labels.size(0), 3, 32, 32),
                                     num_steps=50, eta=0.0, class_labels=labels,
                                     guidance_scale=guidance_scale)
    imgs = imgs.clamp(-1, 1) * 0.5 + 0.5
    save_image(make_grid(imgs, nrow=per), path)


def save_ckpt(path, model, ema, optimizer, epoch, args):
    torch.save({"model_state_dict": model.state_dict(),
                "ema_state_dict": ema.shadow.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch, "hyperparams": vars(args)}, path)


def train():
    args = parse_args()
    if args.dataset == "cifar100":
        args.num_classes = 100
    if args.save_path is None:
        args.save_path = f"checkpoints/{args.dataset}_cfg.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | dataset={args.dataset} classes={args.num_classes}")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    loader = build_loader(args)
    model = UNet(in_channels=3, base_channels=args.base_channels,
                 channel_mults=tuple(args.channel_mults),
                 num_classes=args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    schedule = DiffusionSchedule(timesteps=args.timesteps, device=device)
    ema = EMA(model, decay=args.ema_decay)

    start_epoch = 1
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        if "ema_state_dict" in ck:
            ema.shadow.load_state_dict(ck["ema_state_dict"])
        if "optimizer_state_dict" in ck:
            optimizer.load_state_dict(ck["optimizer_state_dict"])
        start_epoch = ck.get("epoch", 0) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    null_index = args.num_classes  # CFG 的 null class

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            bs = images.size(0)
            drop = torch.rand(bs, device=device) < args.label_dropout
            labels = torch.where(drop, torch.full_like(labels, null_index), labels)
            t = torch.randint(0, args.timesteps, (bs,), device=device)
            noise = torch.randn_like(images)
            x_t = schedule.q_sample(images, t, noise)
            loss = torch.nn.functional.mse_loss(model(x_t, t, labels), noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)
            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"Epoch {epoch} — avg loss: {total/len(loader):.4f}")

        if epoch % args.sample_interval == 0 or epoch == 1:
            sample_grid(ema.shadow, schedule,
                        os.path.join(args.output_dir, f"epoch_{epoch}.png"),
                        args.num_classes, device, args.guidance_scale)
        if epoch % args.ckpt_interval == 0:
            save_ckpt(os.path.join(args.ckpt_dir, f"{args.dataset}_cfg_ep{epoch}.pt"),
                      model, ema, optimizer, epoch, args)
        save_ckpt(args.save_path, model, ema, optimizer, epoch, args)

    print(f"Training complete. Saved to {args.save_path}")


if __name__ == "__main__":
    train()
