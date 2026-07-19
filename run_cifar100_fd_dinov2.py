"""CIFAR-100 per-config FD-DINOv2（Q5 P1：補 which-FID 之 DINOv2 空間，堵住只算 Inception 的缺口）。

confirmatory 只算了 Inception clean-fid 一個 FID 空間（`docs/verdict_cifar100.md` 記為口徑範圍限制：
未算 per-config FD-DINOv2）。本檔在**同一凍結生成路徑**上補算 seed-10 之 per-config FD-DINOv2，
與 CIFAR-10 §E3（P1 streaming）之 FD-DINOv2 對齊，使 which-FID 於兩尺度皆有 DINOv2 空間讀數。

生成路徑逐字沿 `regen_cifar100_cells.py`（已對帳 confirmatory 於 1e-4）：
  - gseed 用 `datasets.cifar100_gseed.gseed`（sha256 hash），num_classes=100、per_class/steps/eta 自
    confirmatory metadata 取、batch 同。
  - w1/w2.5 之 gen DINOv2 特徵已由 regen 落盤於 `p1_assets_cifar100/seed10_w{1,2.5}/dino_feat.pt`
    （即 confirmatory 之生成），直接載入；其餘 8 個 config 現生成並落盤（利斷點續跑與重用）。
  - FD-DINOv2 = `metrics_features.fd_from_features(real_dino, gen_dino)`（全集分佈 Frechet distance）。
  - 逐 config streaming 寫出（crash-safe）。

判讀（which-FID，D6 口徑）：FD-DINOv2 之 argmin 與 seed-10 TSTR 之 argmax 相隔 > 1 格才算分離。
單 seed（10）、exploratory；補 confirmatory 未算之 DINOv2 空間，與 Inception 側 0/8 不分離對照。

Usage:
    uv run python run_cifar100_fd_dinov2.py --dry-run     # 無 GPU：驗流程、印 gseed 與 grid
    uv run python run_cifar100_fd_dinov2.py               # 需 GPU：生成 8 config + 算 FD-DINOv2
"""
import argparse
import json
import os
import platform
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # 決定性 cublas（同 P0/P1/regen）

import torch

from datasets.cifar100_gseed import gseed as gseed_hash

CONF = "results/cifar100_cfg_confirmatory.json"
ASSET = "results/p1_assets_cifar100"
OUT = "results/cifar100_fd_dinov2.json"
SEED = 10
NUM_CLASSES = 100


def cell_name(w):
    return f"w{w:g}"


def seed10_tstr(d, name):
    """confirmatory per_seed seed-10 之該 config TSTR（單 seed，用於 which-FID 的 argmax）。"""
    sb = next(s for s in d["per_seed"] if s["seed"] == SEED)
    return next(c for c in sb["configs"] if c["name"] == name)["tstr"]


def separation(fd_by_w, tstr_by_w, grid):
    """FD-DINOv2 argmin 與 TSTR argmax 之格步；> 1 才算分離（D6 口徑）。"""
    fd_argmin = min(fd_by_w, key=fd_by_w.get)
    tstr_argmax = max(tstr_by_w, key=tstr_by_w.get)
    idx = {w: i for i, w in enumerate(grid)}
    step = abs(idx[fd_argmin] - idx[tstr_argmax])
    return fd_argmin, tstr_argmax, step, step > 1


def dry_run(d):
    m = d["metadata"]
    grid = m["guidance_grid"]
    print("=== dry-run（不載模型、不生成、不碰 GPU）===")
    print(f"confirmatory: per_class={m['per_class']} steps={m['steps']} eta={m['eta']} "
          f"batch={m.get('batch', 250)} grid={grid}")
    print(f"seed={SEED} num_classes={NUM_CLASSES}")
    for w in grid:
        name = cell_name(w)
        cached = os.path.exists(f"{ASSET}/seed{SEED}_{name}/dino_feat.pt")
        print(f"  {name}: gseed={gseed_hash(SEED, w)}  cached_dino={cached}  "
              f"seed10_TSTR={seed10_tstr(d, name)}")
    real_cached = os.path.exists(f"{ASSET}/real_dino_feat.pt")
    print(f"real_dino cached={real_cached}；輸出→{OUT}")
    print("dry-run OK：流程可解析。真跑需 GPU（生成無快取之 config）。")


def main():
    p = argparse.ArgumentParser(description="CIFAR-100 per-config FD-DINOv2（which-FID DINOv2 空間補算）。")
    p.add_argument("--ckpt", default="checkpoints/cifar100_cfg.pt")
    p.add_argument("--output", default=OUT)
    p.add_argument("--batch", type=int, default=500,
                   help="生成 batch；加大以加速（FD 為分佈量，不需與 confirmatory 之 batch=250 一致）")
    p.add_argument("--dry-run", action="store_true", help="不載模型、不生成、不碰 GPU；只驗流程")
    args = p.parse_args()
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    d = json.load(open(CONF, encoding="utf-8"))
    if args.dry_run:
        dry_run(d)
        return

    from cifar_cfg_sample import load_cfg_model, generate_balanced
    from metrics_features import dinov2_features, fd_from_features
    from datasets.cifar import load_real_per_class

    # FD-DINOv2 為全集分佈量、非逐位對帳，不需嚴格決定性；開 cudnn.benchmark 並加大 batch 以加速生成。
    # 已快取之 w1/w1.5/w2.5 為 regen 之決定性 batch=250 生成；本次新生成 config 用加速設定，混用對分佈量
    # FD-DINOv2 無實質影響（同模型、同 guidance、同 gseed）。
    torch.backends.cudnn.benchmark = True

    m = d["metadata"]
    grid = m["guidance_grid"]
    per_class, real_per_class = m["per_class"], m["real_per_class"]
    steps, eta, batch = m["steps"], m["eta"], args.batch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 真實 DINOv2 特徵：優先載入 regen 已落盤者（與 confirmatory 同源），免 50k 前向 OOM。
    rp = f"{ASSET}/real_dino_feat.pt"
    if os.path.exists(rp):
        real_dino = torch.load(rp, map_location="cpu", weights_only=True)
        print(f"loaded cached real DINOv2 feats {tuple(real_dino.shape)}", flush=True)
    else:
        real_imgs, _ = load_real_per_class("cifar100", real_per_class, seed=0)
        real_dino = dinov2_features((real_imgs + 1) / 2, device).cpu()

    model = schedule = None  # 延遲載入：全部 config 皆有快取時免載模型
    per_config, fd_by_w, tstr_by_w = [], {}, {}
    for w in grid:
        name = cell_name(w)
        cdir = f"{ASSET}/seed{SEED}_{name}"
        fpath = f"{cdir}/dino_feat.pt"
        t0 = time.time()
        if os.path.exists(fpath):
            gen_dino = torch.load(fpath, map_location="cpu", weights_only=True)
            source = "cached"
        else:
            if model is None:
                model, schedule, _ = load_cfg_model(args.ckpt, device)
            g = gseed_hash(SEED, w)
            gen, gen_labels = generate_balanced(model, schedule, per_class, device, steps, eta,
                                                guidance=w, num_classes=NUM_CLASSES, batch=batch, seed=g)
            gen_dino = dinov2_features((gen + 1) / 2, device).cpu()
            os.makedirs(cdir, exist_ok=True)
            torch.save(gen_dino, fpath)               # 落盤利斷點續跑與重用
            del gen
            if device.type == "cuda":
                torch.cuda.empty_cache()
            source = "generated"

        fd = float(fd_from_features(real_dino, gen_dino))
        tstr = seed10_tstr(d, name)
        fd_by_w[w], tstr_by_w[w] = fd, tstr
        per_config.append({"name": name, "guidance": w, "gseed": gseed_hash(SEED, w),
                           "fd_dinov2": fd, "tstr_seed10": tstr, "source": source,
                           "seconds": time.time() - t0})
        print(f"[{len(per_config)}/{len(grid)}] {name} fd_dinov2={fd:.3f} "
              f"tstr={tstr:.2f} ({source}, {time.time()-t0:.0f}s)", flush=True)

        # 逐 config streaming 寫出（crash-safe）
        json.dump({"per_config": per_config, "status": "running", "metadata": None},
                  open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        del gen_dino

    fd_argmin, tstr_argmax, sep_step, separated = separation(fd_by_w, tstr_by_w, grid)
    out = {
        "per_config": per_config,
        "which_fid_dinov2": {
            "fd_dinov2_argmin": cell_name(fd_argmin), "tstr_argmax_seed10": cell_name(tstr_argmax),
            "separation_step": sep_step, "separated_gt1": separated,
            "rule": "FD-DINOv2 argmin 與 seed-10 TSTR argmax 相隔 > 1 格才算分離（D6）",
        },
        "status": "complete",
        "metadata": {
            "analysis": "cifar100_fd_dinov2", "source": CONF, "asset": ASSET, "seed": SEED,
            "num_classes": NUM_CLASSES, "grid": grid, "per_class": per_class, "steps": steps, "eta": eta,
            "batch": batch, "gen_seed_formula": "cifar100_gseed sha256[:15]",
            "acceleration": "cudnn.benchmark=True、非嚴格決定性、batch 加大（FD-DINOv2 為全集分佈量，"
                            "不需逐位決定性；已快取之 w1/w1.5/w2.5 為 regen 決定性 batch=250，混用不影響分佈量）",
            "note": "which-FID 之 DINOv2 空間補算（單 seed 10，exploratory）；生成路徑沿 regen（同模型／"
                    "guidance／cifar100_gseed）；w1/w1.5/w2.5 之 gen 特徵載自落盤快取。FD-DINOv2 為全集分佈距離。",
            "start_timestamp": start_timestamp, "argv": " ".join(sys.argv),
            "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
                    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()},
        },
    }
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nWrote {args.output}. FD-DINOv2 argmin={cell_name(fd_argmin)} "
          f"TSTR argmax={cell_name(tstr_argmax)} sep_step={sep_step} separated={separated}", flush=True)


if __name__ == "__main__":
    main()
