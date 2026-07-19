"""
VAR-mini 的推論腳本。生成 MNIST 數字，並輸出與現有 TSTR pipeline（evaluate.py）相同格式的資料集檔案：

    {"images": (N, 1, 28, 28) float32 in [-1, 1], "labels": (N,) int64}

Usage:
    # 預設：每個數字（0-9）生成 100 張影像
    uv run python inference_var.py

    # 標準的 TSTR 執行（每個數字 1000 張 -> 共 10K 張）
    uv run python inference_var.py --per-digit 1000

    # 覆寫 checkpoint／取樣控制參數
    uv run python inference_var.py --vqvae var_vqvae.pt --transformer var_transformer.pt \
        --cfg-scale 2.0 --top-k 50 --top-p 0.95
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py

import argparse
import os

import torch
from tqdm import tqdm

from var import VARVQVAE, VARTransformer, generate


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MNIST digits with trained VAR-mini")
    parser.add_argument("--vqvae", type=str, default="var_vqvae.pt",
                        help="Path to trained VQ-VAE checkpoint")
    parser.add_argument("--transformer", type=str, default="var_transformer.pt",
                        help="Path to trained transformer checkpoint")
    parser.add_argument("--digits", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--per-digit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default="generated")
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    return parser.parse_args()


def generate_for_digit(transformer, vqvae, digit, count, batch_size, args, device):
    """為單一數字持續產生一批批影像，直到生成 `count` 張為止。"""
    generated = 0
    while generated < count:
        n = min(batch_size, count - generated)
        labels = torch.full((n,), digit, device=device, dtype=torch.long)
        images, _ = generate(
            transformer, vqvae, labels,
            cfg_scale=args.cfg_scale,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
        )
        yield images.clamp(-1, 1)
        generated += n


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vqvae = VARVQVAE().to(device)
    vqvae.load_state_dict(torch.load(args.vqvae, map_location=device, weights_only=True))
    vqvae.eval()
    print(f"Loaded VQ-VAE from {args.vqvae}")

    transformer = VARTransformer(
        scales=tuple(vqvae.scales),
        embedding_dim=vqvae.embedding_dim,
        num_classes=10,
        codebook_size=vqvae.codebook_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
    ).to(device)
    transformer.load_state_dict(
        torch.load(args.transformer, map_location=device, weights_only=True))
    transformer.eval()
    print(f"Loaded transformer from {args.transformer}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_images = []
    all_labels = []
    for digit in args.digits:
        print(f"\nGenerating digit {digit} ({args.per_digit} images)...")
        digit_imgs = []
        n_batches = (args.per_digit + args.batch_size - 1) // args.batch_size
        for batch in tqdm(generate_for_digit(transformer, vqvae, digit,
                                             args.per_digit, args.batch_size,
                                             args, device),
                          total=n_batches):
            digit_imgs.append(batch.cpu())
        digit_imgs = torch.cat(digit_imgs, dim=0)
        digit_lbls = torch.full((digit_imgs.shape[0],), digit, dtype=torch.long)
        all_images.append(digit_imgs)
        all_labels.append(digit_lbls)

    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    dataset_path = os.path.join(args.output_dir, "dataset.pt")
    torch.save({"images": all_images, "labels": all_labels}, dataset_path)
    print(f"\nDataset saved to {dataset_path}")
    print(f"  images: {all_images.shape}  (range [-1, 1], float32)")
    print(f"  labels: {all_labels.shape}  (int64)")

    print(f"\nDone! Generated {all_images.shape[0]} images total.")
    print(f"\nUsage in downstream training:")
    print(f'  data = torch.load("{dataset_path}")')
    print(f'  images, labels = data["images"], data["labels"]')


if __name__ == "__main__":
    main()
