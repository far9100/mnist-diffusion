"""T8：H3 公平化 v2（雙向 Chamfer、官方 G_freq 導引頻率、matched-budget FLOPs 記帳）。對應審查 A5。

問題設定（給第一次讀的研究生）：v1（`run_cifar100_h3_duel.py`）的 Chamfer 臂用**單向**覆蓋項
（exemplar→最近生成）、每步導引、單 weight，量得 coverage 反低、據以攻擊「便宜代理不追蹤效用」。
但官方 Chamfer Guidance 是**雙向**距離、導引頻率 G_freq=5、且報告 coverage **上升**（0.603→0.912）。
v1 的「coverage 反低」可能是簡化假影。本 v2 做公平化重跑以定論：

  1. 雙向項 term="chamfer"（主結果）＋單向 term="coverage"（ablation，對 v1）。
  2. `--guide-every 5`（官方 G_freq）而非每步。
  3. weight 掃描 {0.05, 0.1, 0.3, 1.0}（單 seed 10 scout）。
  4. Chamfer 臂與 vanilla 臂 rep 對稱（各 5）。
  5. coverage 量兩個空間：DINOv2-224（本文量測空間）＋導引所用空間（DINOv2-112 或 judge）——
     檢驗「增益不在 coverage」是否為量測空間錯位。
  6. FLOPs 記帳：生成 NFE ＋ 導引反傳次數 × 特徵網路成本；matched-budget 口徑（張數 vs FLOPs）明寫。

凍結不動：v1 driver 與其輸出 `cifar100_h3_duel_dinov2.json`／`_judge.json`。本 v2 寫新檔
`results/cifar100_h3_duel_v2_<features>.json`。GPU 中（導引生成較慢）；exploratory、禁因果。

Usage:
    uv run python src/experiments/run_cifar100_h3_duel_v2.py                    # dinov2、雙向掃 4 weight＋單向 ablation
    uv run python src/experiments/run_cifar100_h3_duel_v2.py --weights 1.0 --reps 3   # smoke
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py
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

CONF = "results/cifar100_cfg_confirmatory.json"
SEED = 10
NUM_CLASSES = 100
PER_CLASS = 500
STEPS = 50
ETA = 0.0
NEAREST_K = 5
TSTR_EPOCHS = 15
# 粗略每次 forward 之 GFLOPs（order-of-magnitude，僅供 matched-budget 口徑討論；backward≈2×forward）。
GFLOPS_DIFF_FWD = 6.0        # cifar100 CFG UNet @32x32（估）
GFLOPS_DINO_FWD = {224: 17.5, 112: 4.5}   # DINOv2 ViT-B/14（估，依 token 數縮放）


def make_dinov2_feature_fn(device, image_size=112):
    """可微 DINOv2 feature_fn（導引用；[-1,1] 輸入）。複製 dinov2_features 前處理但不包 no_grad。"""
    model = get_dinov2("dinov2_vitb14", device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def feature_fn(x):
        x01 = (x + 1.0) / 2.0
        x01 = F.interpolate(x01, size=image_size, mode="bicubic", align_corners=False)
        xn = (x01.clamp(0, 1) - mean) / std
        return model(xn)
    return feature_fn


def frozen_w15_tstr(d):
    """凍結 confirmatory 之 vanilla w1.5：seed-10 值、8-seed 均值、per-seed reps（供對稱 rep 對比）。"""
    sb = next(s for s in d["per_seed"] if s["seed"] == SEED)
    cfg = next(c for c in sb["configs"] if c["name"] == "w1.5")
    per_config = next(pc for pc in d["aggregate"]["per_config"] if pc["name"] == "w1.5")
    return cfg["tstr"], cfg.get("tstr_reps", []), per_config["tstr"]["mean"], cfg.get("coverage")


def generate_chamfer(model, schedule, feature_fn, real_imgs, real_labels, weight, term,
                     guide_every, exemplars_per_class, batch, device):
    """逐類生成 PER_CLASS 張：term/guide_every 可調。回傳影像、標籤與 NFE 計數。"""
    guided_steps = len([i for i in range(STEPS) if i % guide_every == 0])   # 每張導引步數
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
                                              chamfer_weight=weight, term=term,
                                              guide_every=guide_every, generator=gen)
            all_imgs.append(imgs.clamp(-1, 1).cpu())
            all_labels.append(labels_c.cpu())
            made += n
        if (c + 1) % 25 == 0:
            print(f"    [{term} w{weight} ge{guide_every}] class {c + 1}/{NUM_CLASSES}", flush=True)
    n_img = NUM_CLASSES * PER_CLASS
    nfe = {"images": n_img, "diffusion_fwd_per_img": STEPS, "guided_steps_per_img": guided_steps,
           "guidance_fwd_bwd_per_img": guided_steps * 3}   # 1 fwd + 1 bwd(≈2×) = 3× forward-equiv
    return torch.cat(all_imgs), torch.cat(all_labels), nfe


def coverage_in_space(gen_imgs, gen_labels, real_imgs, real_labels, device, image_size):
    """在指定 DINOv2 解析度空間量 per-class PRDC（[-1,1] →[0,1]）。"""
    real_f = dinov2_features((real_imgs + 1) / 2, device, image_size=image_size)
    gen_f = dinov2_features((gen_imgs + 1) / 2, device, image_size=image_size)
    prdc, _ = compute_prdc_per_class(real_f.to(device), real_labels.to(device),
                                     gen_f.to(device), gen_labels.to(device),
                                     nearest_k=NEAREST_K, num_classes=NUM_CLASSES)
    return prdc


def est_gflops(nfe, guide_size):
    """粗估總 GFLOPs（matched-budget 口徑討論用；標為估計）。"""
    per_img = nfe["diffusion_fwd_per_img"] * GFLOPS_DIFF_FWD
    per_img += nfe["guidance_fwd_bwd_per_img"] * GFLOPS_DINO_FWD.get(guide_size, GFLOPS_DINO_FWD[112])
    return round(per_img * nfe["images"] / 1e3, 1)   # TFLOPs 級（÷1e3）


def main():
    p = argparse.ArgumentParser(description="H3 公平化 v2（雙向 Chamfer、G_freq、FLOPs 記帳；T8/A5）。")
    p.add_argument("--weights", type=float, nargs="+", default=[0.05, 0.1, 0.3, 1.0])
    p.add_argument("--guide-every", type=int, default=5, help="導引頻率（官方 G_freq=5）")
    p.add_argument("--reps", type=int, default=5, help="每臂 TSTR 重訓次數（對稱 vanilla 之 5）")
    p.add_argument("--chamfer-features", choices=["dinov2", "judge"], default="dinov2")
    p.add_argument("--exemplars-per-class", type=int, default=16)
    p.add_argument("--batch", type=int, default=96, help="導引生成較重（DINOv2 反傳），建議 96")
    p.add_argument("--guide-image-size", type=int, default=112, help="導引 DINOv2 解析度（量測仍另用 224）")
    p.add_argument("--unidir-ablation-weight", type=float, default=1.0,
                   help="對 v1 的單向 ablation 於此 weight（設 0 關閉）")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    out_path = args.output or f"results/cifar100_h3_duel_v2_{args.chamfer_features}.json"
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = json.load(open(CONF, encoding="utf-8"))
    van_reps_frozen, van_reps_list, van_mean, van_cov = frozen_w15_tstr(d)
    van_seed10 = van_reps_frozen if isinstance(van_reps_frozen, (int, float)) else \
        (sum(van_reps_list) / len(van_reps_list) if van_reps_list else None)
    print(f"device={device} vanilla w1.5 TSTR seed10={van_seed10} cov224(frozen)={van_cov}", flush=True)

    model, schedule, _ = load_cfg_model("checkpoints/cifar100_cfg.pt", device)
    if args.chamfer_features == "judge":
        judge = ResNet18(num_classes=NUM_CLASSES).to(device)
        judge.load_state_dict(torch.load("checkpoints/cifar100_judge.pt", map_location=device, weights_only=True))
        judge.eval()
        feature_fn = cifar_penultimate_feature_fn(judge)
    else:
        feature_fn = make_dinov2_feature_fn(device, image_size=args.guide_image_size)

    real_imgs, real_labels = load_real_per_class("cifar100", 500, seed=0)
    test_loader = build_test_loader("cifar100")

    # 臂清單：每個 weight 一個雙向（主）；再加一個單向 ablation（對 v1）
    plan = [(w, "chamfer") for w in args.weights]
    if args.unidir_ablation_weight > 0:
        plan.append((args.unidir_ablation_weight, "coverage"))

    arms = []
    for weight, term in plan:
        print(f"\n[arm] term={term} weight={weight} guide_every={args.guide_every} ...", flush=True)
        t0 = time.time()
        imgs, labels, nfe = generate_chamfer(model, schedule, feature_fn, real_imgs, real_labels,
                                             weight, term, args.guide_every,
                                             args.exemplars_per_class, args.batch, device)
        gen_seconds = time.time() - t0
        prdc224 = coverage_in_space(imgs, labels, real_imgs, real_labels, device, 224)
        prdc_guide = coverage_in_space(imgs, labels, real_imgs, real_labels, device, args.guide_image_size)
        tstr = []
        for r in range(args.reps):
            ov, _ = run_tstr(imgs, labels, test_loader, device, num_classes=NUM_CLASSES, epochs=TSTR_EPOCHS)
            tstr.append(ov)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        tstr_mean = sum(tstr) / len(tstr)
        arm = {"term": term, "direction": "bidirectional" if term == "chamfer" else "unidirectional",
               "weight": weight, "guide_every": args.guide_every, "feature_space": args.chamfer_features,
               "tstr_reps": [round(t, 2) for t in tstr], "tstr_mean": round(tstr_mean, 2),
               "coverage_dinov2_224": round(prdc224["coverage"], 4),
               f"coverage_guide_{args.guide_image_size}": round(prdc_guide["coverage"], 4),
               "precision_dinov2_224": round(prdc224["precision"], 4),
               "nfe": nfe, "est_tflops": est_gflops(nfe, args.guide_image_size),
               "gen_seconds": round(gen_seconds, 1),
               "tstr_minus_vanilla": round(tstr_mean - van_seed10, 2) if van_seed10 else None,
               "coverage224_minus_vanilla": round(prdc224["coverage"] - van_cov, 4) if van_cov else None}
        arms.append(arm)
        print(f"  → TSTR {tstr_mean:.2f} (vs vanilla {van_seed10}, {arm['tstr_minus_vanilla']:+}pp)"
              f"  cov224 {prdc224['coverage']:.4f} (vs {van_cov}, {arm['coverage224_minus_vanilla']:+})"
              f"  cov{args.guide_image_size} {prdc_guide['coverage']:.4f}  ~{arm['est_tflops']}TFLOPs", flush=True)

    # vanilla 臂 FLOPs（純擴散、無導引）
    van_nfe = {"images": NUM_CLASSES * PER_CLASS, "diffusion_fwd_per_img": STEPS,
               "guided_steps_per_img": 0, "guidance_fwd_bwd_per_img": 0}
    bidir = [a for a in arms if a["term"] == "chamfer"]
    cov_rises = any(a["coverage224_minus_vanilla"] and a["coverage224_minus_vanilla"] > 0 for a in bidir)
    out = {
        "analysis": "cifar100_h3_duel_v2", "seed": SEED, "chamfer_features": args.chamfer_features,
        "vanilla_w15": {"tstr_seed10": van_seed10, "tstr_reps_frozen": van_reps_list,
                        "tstr_8seed_mean": van_mean, "coverage_dinov2_224": van_cov,
                        "est_tflops": est_gflops(van_nfe, 224), "note": "凍結 confirmatory、對稱 rep 對比"},
        "arms": arms,
        "verdict": {
            "bidirectional_coverage224_rises_vs_vanilla": cov_rises,
            "note": ("若雙向 coverage224 上升（同官方 0.603→0.912 方向），則 v1 之「增益不見於 coverage」"
                     "為單向簡化假影，須撤回 §5.6 攻擊句；若仍下降，則攻擊句在雙向、G_freq 對齊下仍成立。"),
            "matched_budget_caliber": "張數相等（各 5 萬）；FLOPs 不等——Chamfer 臂另加導引反傳（見 est_tflops）。",
        },
        "metadata": {
            "guide_every": args.guide_every, "reps": args.reps, "per_class": PER_CLASS,
            "num_classes": NUM_CLASSES, "steps": STEPS, "eta": ETA, "nearest_k": NEAREST_K,
            "tstr_epochs": TSTR_EPOCHS, "exemplars_per_class": args.exemplars_per_class,
            "guide_image_size": args.guide_image_size,
            "gflops_assumptions": {"diff_fwd": GFLOPS_DIFF_FWD, "dino_fwd": GFLOPS_DINO_FWD,
                                   "note": "order-of-magnitude 估計，backward≈2×forward；僅供口徑討論"},
            "note": "exploratory、禁因果；v1 driver 與輸出凍結不動，本檔為新檔。",
            "start_timestamp": start_timestamp, "argv": " ".join(sys.argv),
            "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
                    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()},
        },
    }
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[H3 v2] 雙向 coverage224 上升 vs vanilla = {cov_rises}（決定是否撤回 §5.6 攻擊句）")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
