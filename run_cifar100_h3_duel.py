"""H3 三臂 matched-budget 對決：FID-min／CaF-v2／Chamfer（分支三護城河對決）。

預註冊 D5/D7：在 matched-budget 下比 FID-min、CaF-v2、簡化 Chamfer 三臂之下游 TSTR。CIFAR-100 上
CaF-v2 與 FID-min 逐 seed 同選 w1.5（`2026-07-16-01`），故此兩臂＝vanilla w1.5；Chamfer 臂為新生成。

三臂（同 budget：各 per_class=500、100 類 = 5 萬張；真實參考 500/class）：
  - FID-min 臂 ＝ vanilla w1.5（char-fid 最小之選中組態）。
  - CaF-v2 臂 ＝ w1.5（CIFAR-100 上與 FID-min 同選）。→ 兩臂同一份，TSTR 取凍結 confirmatory w1.5
    seed-10 值（重生成已證逐位重現，`2026-07-17-05`，故不重生成，省 GPU）。
  - Chamfer 臂 ＝ 新生成：guidance_scale=1.0（純條件，不靠 CFG 銳化）＋ chamfer_weight（靠 exemplar
    覆蓋提多樣性，作者裁定 weight≈1.0）。**逐類生成、chamfer 導引對同類 exemplar**（避免跨類混淆）。

判讀：Chamfer 臂 TSTR 是否 tie/beat vanilla w1.5。契合 so-what——便宜選中的 vanilla 是否敵得過較
複雜的 Chamfer。禁因果措辭。

Usage:
    uv run python run_cifar100_h3_duel.py                       # 需 GPU
    uv run python run_cifar100_h3_duel.py --chamfer-weight 1.0 --reps 3
"""
import argparse
import json
import os
import platform
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F

from cifar_cfg_sample import load_cfg_model
from cifar_classifier import ResNet18, run_tstr
from chamfer import chamfer_guided_ddim_sample, cifar_penultimate_feature_fn
from datasets.cifar import build_test_loader, load_real_per_class
from metrics_features import dinov2_features, get_dinov2, IMAGENET_MEAN, IMAGENET_STD
from metrics_prdc import compute_prdc_per_class


def make_dinov2_feature_fn(device, image_size=224):
    """可微的 DINOv2 feature_fn，供 chamfer 導引用（任務無關特徵，對比 judge 的任務對齊特徵）。

    複製 metrics_features.dinov2_features 的預處理，但不包 @no_grad，讓 guidance 梯度能穿過 DINOv2
    回傳到影像。輸入 x 為 [-1,1] 的 (N,3,32,32)（擴散輸出口徑）。DINOv2 用的是量測 coverage 的同一
    backbone，故此臂會把生成推去在 DINOv2 空間覆蓋 exemplar——正是「任務無關 coverage 是否驅動效用」
    的公平測試。"""
    model = get_dinov2("dinov2_vitb14", device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def feature_fn(x):
        x01 = (x + 1.0) / 2.0
        x01 = F.interpolate(x01, size=image_size, mode="bicubic", align_corners=False)
        xn = (x01.clamp(0, 1) - mean) / std
        return model(xn)                       # 不包 no_grad -> 梯度可回傳

    return feature_fn

CONF = "results/cifar100_cfg_confirmatory.json"
OUT = "results/cifar100_h3_duel.json"
SEED = 10
NUM_CLASSES = 100
PER_CLASS = 500
STEPS = 50
ETA = 0.0
NEAREST_K = 5
TSTR_EPOCHS = 15


def frozen_w15_tstr(d):
    """凍結 confirmatory 之 vanilla w1.5 TSTR：seed-10 值與 8-seed 均值。"""
    sb = next(s for s in d["per_seed"] if s["seed"] == SEED)
    seed10 = next(c for c in sb["configs"] if c["name"] == "w1.5")["tstr"]
    per_config = next(pc for pc in d["aggregate"]["per_config"] if pc["name"] == "w1.5")
    return seed10, per_config["tstr"]["mean"]


def generate_chamfer_balanced(model, schedule, feature_fn, real_imgs, real_labels,
                              chamfer_weight, exemplars_per_class, batch, device):
    """逐類生成 PER_CLASS 張：每類 chamfer 導引對「同類」exemplar（class-conditional coverage）。

    對每個類別 c，取該類 exemplars_per_class 張真實影像之特徵當覆蓋目標，分批以
    chamfer_guided_ddim_sample（guidance_scale=1.0 純條件 + chamfer_weight）生成。逐類進行使
    chamfer 覆蓋項推向同類真實多樣性，而非跨類混淆。
    """
    all_imgs, all_labels = [], []
    for c in range(NUM_CLASSES):
        idx = (real_labels == c).nonzero(as_tuple=True)[0][:exemplars_per_class]
        with torch.no_grad():
            ex_feats = feature_fn(real_imgs[idx].to(device))
        made = 0
        while made < PER_CLASS:
            n = min(batch, PER_CLASS - made)
            labels_c = torch.full((n,), c, device=device, dtype=torch.long)
            gen = torch.Generator(device=device).manual_seed(SEED + c * 100_000 + made)
            imgs = chamfer_guided_ddim_sample(model, schedule, feature_fn, ex_feats,
                                              shape=(n, 3, 32, 32), num_steps=STEPS, eta=ETA,
                                              class_labels=labels_c, guidance_scale=1.0,
                                              chamfer_weight=chamfer_weight, term="coverage", generator=gen)
            all_imgs.append(imgs.clamp(-1, 1).cpu())
            all_labels.append(labels_c.cpu())
            made += n
        if (c + 1) % 20 == 0:
            print(f"  chamfer 生成：class {c + 1}/{NUM_CLASSES}", flush=True)
    return torch.cat(all_imgs), torch.cat(all_labels)


def main():
    p = argparse.ArgumentParser(description="H3 三臂 matched-budget 對決（FID-min/CaF-v2/Chamfer）。")
    p.add_argument("--chamfer-weight", type=float, default=1.0, help="Chamfer 導引強度（作者裁定≈1.0）")
    p.add_argument("--chamfer-features", choices=["dinov2", "judge"], default="dinov2",
                   help="chamfer 導引特徵空間：dinov2（任務無關、公平對 CaF，主結果）或 judge（任務對齊、對照）")
    p.add_argument("--exemplars-per-class", type=int, default=16, help="每類 chamfer exemplar 數")
    p.add_argument("--reps", type=int, default=3, help="每臂 TSTR 重訓次數")
    p.add_argument("--batch", type=int, default=250, help="生成 batch；DINOv2 特徵較重，建議 96")
    p.add_argument("--dinov2-image-size", type=int, default=112,
                   help="DINOv2 guidance 的輸入解析度（導引推力方向；coverage 量測仍用標準 224）")
    p.add_argument("--output", default=None, help="預設 results/cifar100_h3_duel_<features>.json")
    args = p.parse_args()
    out_path = args.output or f"results/cifar100_h3_duel_{args.chamfer_features}.json"
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = json.load(open(CONF, encoding="utf-8"))
    van_seed10, van_mean = frozen_w15_tstr(d)
    print(f"device={device} vanilla w1.5 TSTR: seed10={van_seed10} 8seed_mean={van_mean}", flush=True)

    model, schedule, hp = load_cfg_model("checkpoints/cifar100_cfg.pt", device)
    if args.chamfer_features == "judge":
        judge = ResNet18(num_classes=NUM_CLASSES).to(device)
        judge.load_state_dict(torch.load("checkpoints/cifar100_judge.pt", map_location=device, weights_only=True))
        judge.eval()
        feature_fn = cifar_penultimate_feature_fn(judge)   # 任務對齊（對照）
    else:
        feature_fn = make_dinov2_feature_fn(device, image_size=args.dinov2_image_size)  # 任務無關（公平對 CaF，主結果）
    print(f"chamfer feature space: {args.chamfer_features}", flush=True)

    real_imgs, real_labels = load_real_per_class("cifar100", 500, seed=0)
    real_dino = dinov2_features((real_imgs + 1) / 2, device)
    test_loader = build_test_loader("cifar100")

    # Chamfer 臂：逐類生成
    print(f"Chamfer 臂生成（weight={args.chamfer_weight}, exemplars/class={args.exemplars_per_class}）...", flush=True)
    t0 = time.time()
    ch_imgs, ch_labels = generate_chamfer_balanced(model, schedule, feature_fn, real_imgs, real_labels,
                                                   args.chamfer_weight, args.exemplars_per_class, args.batch, device)
    gen_seconds = time.time() - t0

    # Chamfer 臂特徵/coverage（DINOv2，與 confirmatory 同空間）
    ch_dino = dinov2_features((ch_imgs + 1) / 2, device)
    prdc, _ = compute_prdc_per_class(real_dino.to(device), real_labels.to(device),
                                     ch_dino.to(device), ch_labels.to(device),
                                     nearest_k=NEAREST_K, num_classes=NUM_CLASSES)
    ch_coverage, ch_precision = prdc["coverage"], prdc["precision"]

    # Chamfer 臂 TSTR（reps 次）
    ch_tstr = []
    for r in range(args.reps):
        overall, _ = run_tstr(ch_imgs, ch_labels, test_loader, device, num_classes=NUM_CLASSES, epochs=TSTR_EPOCHS)
        ch_tstr.append(overall)
        print(f"  chamfer TSTR rep{r + 1}/{args.reps}: {overall:.2f}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    ch_mean = sum(ch_tstr) / len(ch_tstr)

    out = {
        "arms": {
            "fidmin": {"config": "w1.5", "tstr_seed10": van_seed10, "tstr_8seed_mean": van_mean,
                       "note": "凍結 confirmatory（重生成已證逐位重現）"},
            "caf_v2": {"config": "w1.5", "tstr_seed10": van_seed10,
                       "note": "CIFAR-100 上與 FID-min 同選 w1.5，兩臂同一份"},
            "chamfer": {"guidance_scale": 1.0, "chamfer_weight": args.chamfer_weight,
                        "feature_space": args.chamfer_features,
                        "exemplars_per_class": args.exemplars_per_class,
                        "tstr_reps": ch_tstr, "tstr_mean": round(ch_mean, 2),
                        "coverage_dinov2": round(ch_coverage, 4), "precision_dinov2": round(ch_precision, 4),
                        "gen_seconds": round(gen_seconds, 1)},
        },
        "duel": {
            "chamfer_minus_vanilla_seed10": round(ch_mean - van_seed10, 2),
            "vanilla_ties_or_beats_chamfer": van_seed10 >= ch_mean,
        },
        "metadata": {
            "analysis": "cifar100_h3_duel", "source": CONF, "seed": SEED,
            "chamfer_features": args.chamfer_features,
            "per_class": PER_CLASS, "num_classes": NUM_CLASSES, "steps": STEPS, "eta": ETA,
            "nearest_k": NEAREST_K, "tstr_epochs": TSTR_EPOCHS, "reps": args.reps,
            "note": "matched-budget（各 5 萬張、真實參考 500/class）；Chamfer 臂逐類對同類 exemplar 導引；"
                    "vanilla 臂用凍結 confirmatory w1.5；exploratory、禁因果。dinov2 特徵＝任務無關公平對比，"
                    "judge 特徵＝任務對齊對照。",
            "start_timestamp": start_timestamp, "argv": " ".join(sys.argv),
            "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
                    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()},
        },
    }
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[H3:{args.chamfer_features}] Chamfer TSTR {ch_mean:.2f} (DINOv2 cov {ch_coverage:.3f}) vs "
          f"vanilla w1.5 {van_seed10} → diff {ch_mean - van_seed10:+.2f}pp；"
          f"vanilla tie/beat={out['duel']['vanilla_ties_or_beats_chamfer']}", flush=True)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
