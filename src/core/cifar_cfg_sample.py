"""自訓 CFG CIFAR 擴散模型的平衡生成，與 FID gate（clean-fid + FD-DINOv2）。

載入 train_cifar.py 產出的 checkpoint（用 EMA 權重，CIFAR FID 的標準作法），以
ddpm.DiffusionSchedule 的 DDIM(eta) 取樣器做各類別平衡生成。此模組同時作為自訓主軸的
生成入口，供 Stage 1 的 FID gate 與 Stage 3 的寬 grid scout 共用（取代早期預警用的 EDM 代理）。

FID gate（Stage 1）：以固定 (steps, eta) 在代表性設定（預設 guidance=1，純條件）量 clean-fid
（Inception 錨點）與 FD-DINOv2，確認自訓模型落在 repo 內自證的堪用帶（非 EDM 的 1.79）。

Usage:
    uv run python cifar_cfg_sample.py --num 5000 --steps 50 --eta 0 --guidance 1.0
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py

import argparse
import json
import os
import sys
import time

import torch

from ddpm import UNet, DiffusionSchedule


def load_cfg_model(ckpt, device):
    """載入自訓 CFG 模型（EMA 權重），依 checkpoint 的 hyperparams 建 UNet 與 schedule。"""
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    hp = ck.get("hyperparams", {})
    model = UNet(in_channels=3,
                 base_channels=hp.get("base_channels", 128),
                 channel_mults=tuple(hp.get("channel_mults", [1, 2, 4])),
                 num_classes=hp.get("num_classes", 10)).to(device)
    state = ck.get("ema_state_dict") or ck.get("model_state_dict")
    model.load_state_dict(state)
    model.eval()
    schedule = DiffusionSchedule(timesteps=hp.get("timesteps", 1000), device=device)
    return model, schedule, hp


@torch.no_grad()
def generate_balanced(model, schedule, per_class, device, steps=50, eta=0.0,
                      guidance=1.0, num_classes=10, batch=250, seed=0):
    """每類生成 per_class 張，回傳 [-1,1] 影像與標籤（DDIM(eta) 取樣、CFG 強度 guidance）。"""
    gen = torch.Generator(device=device).manual_seed(seed)
    imgs, labels = [], []
    for c in range(num_classes):
        remaining = per_class
        while remaining > 0:
            bs = min(batch, remaining)
            lab = torch.full((bs,), c, device=device, dtype=torch.long)
            x = schedule.ddim_sample_loop(model, shape=(bs, 3, 32, 32), num_steps=steps,
                                          eta=eta, class_labels=lab, guidance_scale=guidance,
                                          generator=gen)
            imgs.append(x.clamp(-1, 1).cpu())
            labels.append(torch.full((bs,), c, dtype=torch.long))
            remaining -= bs
    return torch.cat(imgs), torch.cat(labels)


def main():
    p = argparse.ArgumentParser(description="Self-trained CFG CIFAR FID gate (clean-fid + FD-DINOv2).")
    p.add_argument("--ckpt", default="checkpoints/cifar10_cfg.pt")
    p.add_argument("--num", type=int, default=5000, help="生成張數（會平均分到 10 類）")
    p.add_argument("--steps", type=int, default=50, help="DDIM 步數（量測用，固定）")
    p.add_argument("--eta", type=float, default=0.0, help="DDIM 隨機性（量測用，固定）")
    p.add_argument("--guidance", type=float, default=1.0, help="FID gate 用的代表性 CFG 強度")
    p.add_argument("--batch", type=int, default=250)
    p.add_argument("--real-per-class", type=int, default=500, help="FD-DINOv2 的真實參考張數/類")
    # 協定門檻：CIFAR-10 base model 50k clean-fid <= 10（增修 R-2026-07-05-08）。舊版此處硬編一個
    # 「堪用帶 5-15」的欄位，與協定不一致；改為明示的 gate 門檻，欄名與 cifar100_base_gate.py 對齊。
    p.add_argument("--gate-threshold", type=float, default=10.0,
                   help="clean-fid gate 門檻（CIFAR-10 協定為 <= 10）")
    p.add_argument("--output", default="results/cifar10_cfg_fid.json")
    args = p.parse_args()
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    os.makedirs("results", exist_ok=True)

    model, schedule, hp = load_cfg_model(args.ckpt, device)
    print(f"Loaded EMA model from {args.ckpt} (epoch {hp.get('epoch', '?')}, "
          f"base_channels={hp.get('base_channels')}, timesteps={hp.get('timesteps')})", flush=True)

    per_class = args.num // 10
    print(f"Generating {per_class*10} imgs (steps={args.steps} eta={args.eta} "
          f"guidance={args.guidance}) ...", flush=True)
    gen, _ = generate_balanced(model, schedule, per_class, device, args.steps, args.eta,
                               args.guidance, num_classes=10, batch=args.batch)
    gen01 = (gen + 1) / 2  # [-1,1] -> [0,1]

    # clean-fid（Inception 錨點）
    from fid_clean import clean_fid_vs_dataset
    print("Computing clean-fid vs CIFAR-10 train stats ...", flush=True)
    cfid = clean_fid_vs_dataset(gen01, dataset_name="cifar10", dataset_split="train", dataset_res=32)

    # FD-DINOv2
    from datasets.cifar import load_real_per_class
    from metrics_features import dinov2_features, fd_from_features
    print("Computing FD-DINOv2 ...", flush=True)
    real, _ = load_real_per_class("cifar10", args.real_per_class, seed=0)
    real01 = (real + 1) / 2
    rf = dinov2_features(real01, device)
    gf = dinov2_features(gen01, device)
    fd_dino = fd_from_features(rf, gf)

    cfid = float(cfid)
    fd_dino = float(fd_dino)
    passed = bool(cfid <= args.gate_threshold)
    print("\n" + "=" * 60)
    print(f"  自訓 CFG CIFAR-10 FID gate (steps={args.steps} eta={args.eta} w={args.guidance})")
    print("=" * 60)
    print(f"  clean-fid (Inception) : {cfid:.3f}   "
          f"(gate <= {args.gate_threshold:g}: {'通過' if passed else '未通過'})")
    print(f"  FD-DINOv2             : {fd_dino:.3f}")
    print("=" * 60)

    out = {"ckpt": args.ckpt, "epoch": hp.get("epoch"), "steps": args.steps, "eta": args.eta,
           "guidance": args.guidance, "num": per_class * 10, "batch": args.batch,
           "real_per_class": args.real_per_class,
           "clean_fid": cfid, "fd_dinov2": fd_dino,
           "gate_threshold": args.gate_threshold, "passed": passed,
           # claude.md §5.2：量測 driver 須留下可重現這些數字的全部參數與環境。
           "start_timestamp": start_timestamp,
           "argv": " ".join(sys.argv),
           "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                   "cudnn": torch.backends.cudnn.version()}}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
