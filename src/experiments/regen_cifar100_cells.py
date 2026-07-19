"""重生成 CIFAR-100 confirmatory 的兩個 cell（seed 10、w∈{1, 2.5}），供 D3 介入臂（C3
coverage-matched pruning）用。confirmatory 的合成影像未落盤（driver 於 measure() 後 del gen），
故 C3 前須以相同生成路徑重生成並落盤，且對帳確認重現。

依 run_p1_streaming.py 的持久化與決定性慣例，但改為 CIFAR-100：
  - gseed 用 datasets.cifar100_gseed.gseed（sha256 hash 公式），非 CIFAR-10 的 seed*1e7 公式。
  - num_classes=100、per_class=500、steps=50、eta=0（自 confirmatory metadata 取，不手打）。
  - 落盤 results/p1_assets_cifar100/seed10_w{1,2.5}/ 之 img_uint8/dino_feat/judge_out/labels.pt，
    及 root 的 real_dino_feat/real_labels.pt。
  - 對帳：重生成 cell 的 precision/coverage/near_boundary_frac 對 confirmatory per_seed seed-10 config
    於 TOL=1e-4 內相符，才算重現 confirmatory 合成集（C3 依賴 coverage，故對帳這三個承重 scalar）。

--dry-run 只解析 confirmatory、印 gseed 與對帳目標、查資產目錄，不載模型、不生成、不碰 GPU。

Usage:
    uv run python regen_cifar100_cells.py --dry-run     # 無 GPU：驗流程與 gseed
    uv run python regen_cifar100_cells.py               # 需 GPU：重生成 + 落盤 + 對帳
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py
import argparse
import json
import os
import platform
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # 決定性 cublas（同 P0/P1）

import torch

from datasets.cifar100_gseed import gseed as gseed_hash

CONF = "results/cifar100_cfg_confirmatory.json"
ASSET = "results/p1_assets_cifar100"
OUT = "results/cifar100_regen_reconcile.json"
SEED = 10
WS = [1.0, 2.5]                # C3 只需 w2.5（剪枝對象）與 w1（coverage 目標）兩個 cell
NEAREST_K = 5
NUM_CLASSES = 100
TOL = 1e-4                     # 相對容忍，同 P1
RECONCILE_KEYS = ["precision", "coverage", "near_boundary_frac"]  # C3 承重 scalar


def cell_name(w):
    return f"w{w:g}"


def frozen_seed_config(d, w):
    """confirmatory per_seed 之 seed-10 config（凍結對帳目標）。"""
    sb = next(s for s in d["per_seed"] if s["seed"] == SEED)
    return next(c for c in sb["configs"] if c["name"] == cell_name(w))


def reconcile(got, ref):
    """got vs 凍結 config 逐 scalar 判三態（同 P1 口徑）。"""
    recon, over = {}, False
    for k in RECONCILE_KEYS:
        v, r = got.get(k), ref.get(k)
        if v is None or r is None:
            recon[k] = {"got": v, "ref": r, "status": "skipped_none"}
            continue
        rel = abs(v - r) / (abs(r) + 1e-12)
        within = rel <= TOL
        recon[k] = {"got": v, "ref": r, "abs_delta": abs(v - r), "rel_delta": rel, "within_tol": within}
        if not within:
            over = True
    return recon, over


def dry_run(d):
    """無 GPU：印 gseed、對帳目標、資產目錄，驗流程不觸 confirmatory 口徑。"""
    m = d["metadata"]
    print("=== dry-run（不載模型、不生成、不碰 GPU）===")
    print(f"confirmatory metadata: per_class={m['per_class']} real_per_class={m['real_per_class']} "
          f"steps={m['steps']} eta={m['eta']} batch={m.get('batch', 250)}")
    print(f"seed={SEED} num_classes={NUM_CLASSES} nearest_k={NEAREST_K} TOL={TOL}")
    for w in WS:
        g = gseed_hash(SEED, w)
        ref = frozen_seed_config(d, w)
        cdir = f"{ASSET}/seed{SEED}_{cell_name(w)}"
        print(f"  w={w:g}: gseed={g}  資產→{cdir}")
        print(f"         對帳目標(seed-10): " +
              ", ".join(f"{k}={ref.get(k)}" for k in RECONCILE_KEYS))
    print(f"輸出對帳檔→{OUT}")
    print("dry-run OK：流程可解析，gseed 與對帳目標齊備。真跑需 GPU。")


def main():
    p = argparse.ArgumentParser(description="重生成 CIFAR-100 seed10 w1/w2.5 兩 cell（D3 介入臂用）。")
    p.add_argument("--ckpt", default="checkpoints/cifar100_cfg.pt")
    p.add_argument("--judge", default="checkpoints/cifar100_judge.pt")
    p.add_argument("--dry-run", action="store_true", help="不載模型、不生成、不碰 GPU；只驗流程與 gseed")
    args = p.parse_args()
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    d = json.load(open(CONF, encoding="utf-8"))
    if args.dry_run:
        dry_run(d)
        return

    # 以下需 GPU（Stage 3 執行）。
    from cifar_cfg_sample import load_cfg_model, generate_balanced
    from cifar_classifier import ResNet18
    from mechanism import compute_margins, near_boundary_fraction
    from metrics_prdc import compute_prdc_per_class
    from metrics_features import dinov2_features
    from datasets.cifar import load_real_per_class

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    m = d["metadata"]
    per_class, real_per_class = m["per_class"], m["real_per_class"]
    steps, eta, batch = m["steps"], m["eta"], m.get("batch", 250)
    threshold = m["near_boundary_threshold"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} seed={SEED} ws={WS} k={NEAREST_K}", flush=True)
    model, schedule, hp = load_cfg_model(args.ckpt, device)
    judge = ResNet18(num_classes=NUM_CLASSES).to(device)
    judge.load_state_dict(torch.load(args.judge, map_location=device, weights_only=True)); judge.eval()

    real_imgs, real_labels = load_real_per_class("cifar100", real_per_class, seed=0)
    real_dino = dinov2_features((real_imgs + 1) / 2, device)
    os.makedirs(ASSET, exist_ok=True)
    torch.save(real_dino.cpu(), f"{ASSET}/real_dino_feat.pt")
    torch.save(real_labels.cpu(), f"{ASSET}/real_labels.pt")

    results, any_over = [], False
    for w in WS:
        g = gseed_hash(SEED, w)
        t0 = time.time()
        gen, gen_labels = generate_balanced(model, schedule, per_class, device, steps, eta,
                                            guidance=w, num_classes=NUM_CLASSES, batch=batch, seed=g)
        gen_dino = dinov2_features((gen + 1) / 2, device)
        prdc, _ = compute_prdc_per_class(real_dino.to(device), real_labels.to(device),
                                         gen_dino.to(device), gen_labels.to(device),
                                         nearest_k=NEAREST_K, num_classes=NUM_CLASSES)
        margins, preds = compute_margins(judge, gen, device)
        nb = near_boundary_fraction(margins, threshold)
        got = {"precision": prdc["precision"], "coverage": prdc["coverage"], "near_boundary_frac": nb}
        recon, over = reconcile(got, frozen_seed_config(d, w))
        any_over = any_over or over

        cdir = f"{ASSET}/seed{SEED}_{cell_name(w)}"
        os.makedirs(cdir, exist_ok=True)
        torch.save(((gen + 1) / 2 * 255).round().clamp(0, 255).to(torch.uint8).cpu(), f"{cdir}/img_uint8.pt")
        torch.save(gen_dino.cpu(), f"{cdir}/dino_feat.pt")
        torch.save({"margins": margins.cpu(), "preds": preds.cpu()}, f"{cdir}/judge_out.pt")
        torch.save(gen_labels.cpu(), f"{cdir}/labels.pt")

        results.append({"w": w, "gseed": g, "reconcile": recon, "over_tol": over,
                        "gen_seconds": time.time() - t0})
        print(f"[w={w:g}] gseed={g} over_tol={over}  " +
              "  ".join(f"{k}: got={recon[k].get('got'):.4f} ref={recon[k].get('ref'):.4f} "
                        f"rel={recon[k].get('rel_delta'):.2e}" for k in RECONCILE_KEYS), flush=True)
        del gen, gen_dino, margins, preds
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = {"cells": results, "all_within_tol": not any_over,
           "metadata": {"analysis": "regen_cifar100_cells", "source": CONF, "seed": SEED, "ws": WS,
                        "reconcile_keys": RECONCILE_KEYS, "tol": TOL, "nearest_k": NEAREST_K,
                        "start_timestamp": start_timestamp, "argv": " ".join(sys.argv),
                        "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                                "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
                                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()},
                        "determinism": {"use_deterministic_algorithms": True,
                                        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}}}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nWrote {OUT}. all_within_tol={not any_over}", flush=True)
    if any_over:
        raise SystemExit("對帳超容忍：重生成 cell 與 confirmatory 不符，C3 不得在此資產上跑。請查生成路徑。")


if __name__ == "__main__":
    main()
