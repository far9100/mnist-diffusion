"""P1×C1：streaming 持久化與逐 config 對帳，並隨算 per-config FD-DINOv2（C1 揭盲之數字產出）。

依 R-2026-07-08-02 §3 與 R-2026-07-06-05 §4 之 P1×C1 規格：

- seed-major：seed 10 全 10 config → 11 → 12；grid 與 config 順序沿凍結 JSON。
- 生成路徑與 gseed 公式原樣（run_cifar_cfg_multiseed.py measure() 之 generate_balanced 呼叫，
  gseed = seed*10_000_000 + int(w*1000)*10_000）；P0（R-2026-07-08-04）已證此路徑逐位決定性。
- 每 config：重生成 → 落盤 uint8 影像＋DINOv2 per-sample 特徵 → 即時以 k=5（P0 實證）對帳凍結 JSON
  之 7 個確定性 scalar → 隨算該 config FD-DINOv2（C1，接 metrics_features.fd_from_features）。
- 任一 config 任一 scalar 超容忍（相對 >1e-4）即 STOP：寫出已完成部分＋標記失敗 config，退出。
- 不對帳 TSTR（分類器訓練非決定性、不在 P2 gate，凍結 JSON 之 TSTR 續用，判決三以其為準）。
- 不計算/不記錄 recall、density（§3 不預授權；D8 若納 v2 再另落 C0 規則後計並標 exploratory）。
- C2/C3/C5 之後一律取自本檔落盤影像，禁第二次重生成。
- seed 10 完成之 1-seed FD 曲線僅供故障偵測與進度，禁詮釋；which-FID 裁決為 γ（三 seed 全量、凍結口徑）。

凍結 JSON 唯讀（輸出寫新檔）；不因對帳結果回改任何凍結數字。

Usage:
    uv run python run_p1_streaming.py
    uv run python run_p1_streaming.py --seeds 10 11 12
"""
import argparse
import json
import os
import platform
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # 決定性 cublas（同 P0）

import torch

from cifar_cfg_sample import load_cfg_model, generate_balanced
from cifar_classifier import ResNet18
from mechanism import compute_margins, near_boundary_fraction
from metrics_prdc import compute_prdc_per_class
from metrics_features import dinov2_features, fd_from_features
from datasets.cifar import load_real_per_class
from run_cifar_cfg_scout import load_inception_detector
from run_cifar_selector import inception_feats
from fid_clean import clean_fid_vs_dataset

CONF = "results/cifar10_cfg_confirmatory.json"
ASSET = "results/p1_assets"
OUT = "results/cifar10_p1_streaming.json"
TOL = 1e-4                    # 相對容忍，與 R-2026-07-06-05 §4 決定性三態一致
NEAREST_K = 5                 # P0（R-2026-07-08-04）實證：k=5 逐位重現凍結 JSON
NUM_CLASSES = 10
REAL_REF_SEED = 0             # load_real_per_class 常數 seed（driver:56,89）

# 對帳 scalar 集：僅確定性、且在 P2 gate 內者（precision/coverage 兩側、char_clean_fid、
# near_boundary_frac、label_noise_excess_mean）。TSTR、recall、density 不在此集。
RECONCILE_KEYS = ["precision", "coverage", "coverage_inception", "precision_inception",
                  "char_clean_fid", "near_boundary_frac", "label_noise_excess_mean"]


def set_determinism():
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def env_metadata(argv, start_timestamp):
    return {
        "nearest_k": NEAREST_K, "effective_nearest_k": NEAREST_K,
        "argv": " ".join(argv), "start_timestamp": start_timestamp,
        "reconcile_keys": RECONCILE_KEYS, "tol": TOL,
        "note": "TSTR/recall/density 不對帳/不記錄（§3）；FD-DINOv2 為 C1 新量、不對帳",
        "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
                "python": platform.python_version()},
        "determinism": {"use_deterministic_algorithms": True, "warn_only": True,
                        "cudnn_benchmark": False, "cudnn_deterministic": True,
                        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
    }


def measure_config(model, schedule, judge, detector, real_imgs, real_dino, real_labels,
                   judge_floor, threshold, per_class, steps, eta, batch, w, gseed, device, timing):
    """單 config：重生成＋量 7 個對帳 scalar＋FD-DINOv2；回傳 (scalars, fd_dinov2, artifacts)。

    生成呼叫與 run_cifar_cfg_multiseed.measure() 逐字相同（byte-identical 前提）。
    """
    t = time.time()
    gen, gen_labels = generate_balanced(model, schedule, per_class, device, steps, eta,
                                        guidance=w, num_classes=NUM_CLASSES, batch=batch, seed=gseed)
    timing['gen'] = time.time() - t

    t = time.time(); gen_dino = dinov2_features((gen + 1) / 2, device); timing['dino_feat'] = time.time() - t
    dino_prdc, _ = compute_prdc_per_class(real_dino.to(device), real_labels.to(device),
                                          gen_dino.to(device), gen_labels.to(device),
                                          nearest_k=NEAREST_K, num_classes=NUM_CLASSES)
    # 僅取 precision/coverage；recall/density 不取用不記錄（§3）
    precision, coverage = dino_prdc["precision"], dino_prdc["coverage"]

    # C1：per-config FD-DINOv2（全集分佈距離，非 per-class）。數字產出，裁決留 γ。
    t = time.time(); fd_dinov2 = float(fd_from_features(real_dino, gen_dino.cpu())); timing['fd_dinov2'] = time.time() - t

    t = time.time(); margins, preds = compute_margins(judge, gen, device); timing['judge'] = time.time() - t
    nb = near_boundary_fraction(margins, threshold)
    per_class_excess = []
    for c in range(NUM_CLASSES):
        mmask = (gen_labels == c)
        ln_c = (preds[mmask] != c).float().mean().item() if bool(mmask.any()) else float('nan')
        per_class_excess.append(ln_c - judge_floor[c])
    excess_mean = sum(per_class_excess) / len(per_class_excess)

    t = time.time()
    char_fid = float(clean_fid_vs_dataset((gen + 1) / 2, dataset_name="cifar10",
                                          dataset_split="train", dataset_res=32))
    timing['clean_fid'] = time.time() - t

    # Inception 側 coverage/precision（reconcile 用）；per-sample 特徵不落盤，需要時由影像重算（§3）。
    incep_cov = incep_prec = None
    if detector is not None:
        t = time.time()
        rf = inception_feats(real_imgs, detector, device)
        gf = inception_feats(gen, detector, device)
        incep_prdc, _ = compute_prdc_per_class(rf.to(device), real_labels.to(device),
                                               gf.to(device), gen_labels.to(device),
                                               nearest_k=NEAREST_K, num_classes=NUM_CLASSES)
        incep_cov, incep_prec = incep_prdc["coverage"], incep_prdc["precision"]
        timing['incep_feat'] = time.time() - t

    scalars = {"precision": precision, "coverage": coverage,
               "coverage_inception": incep_cov, "precision_inception": incep_prec,
               "char_clean_fid": char_fid, "near_boundary_frac": nb,
               "label_noise_excess_mean": excess_mean}
    artifacts = {"img_uint8": ((gen + 1) / 2 * 255).round().clamp(0, 255).to(torch.uint8).cpu(),
                 "dino_feat": gen_dino.cpu(), "labels": gen_labels.cpu(),
                 "margins": margins.cpu(), "preds": preds.cpu()}
    del gen, gen_dino, margins, preds
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scalars, fd_dinov2, artifacts


def reconcile(got, conf):
    """got vs 凍結 JSON config，逐項判三態；回傳 (recon_dict, over_tol_bool)。"""
    recon, over = {}, False
    for k in RECONCILE_KEYS:
        v, ref = got.get(k), conf.get(k)
        if v is None or ref is None:
            recon[k] = {"got": v, "ref": ref, "status": "skipped_none"}
            continue
        rel = abs(v - ref) / (abs(ref) + 1e-12)
        within = rel <= TOL
        recon[k] = {"got": v, "ref": ref, "abs_delta": abs(v - ref), "rel_delta": rel,
                    "bitexact": v == ref, "within_tol": within}
        if not within:
            over = True
    return recon, over


def main():
    p = argparse.ArgumentParser(description="P1×C1 streaming 持久化＋逐 config 對帳＋FD-DINOv2。")
    p.add_argument("--ckpt", default="checkpoints/cifar10_cfg.pt")
    p.add_argument("--judge", default="checkpoints/cifar10_judge.pt")
    p.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12])
    p.add_argument("--no-inception", action="store_true")
    args = p.parse_args()

    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    set_determinism()

    d = json.load(open(CONF, encoding="utf-8"))
    m = d["metadata"]
    per_class      = m["per_class"]                   # [JSON] 1000
    real_per_class = m["real_per_class"]              # [JSON] 1000
    steps          = m["steps"]                        # [JSON] 50
    eta            = m["eta"]                          # [JSON] 0.0
    batch          = m.get("batch", 250)               # metadata 未存則 driver 預設 250（P0 已證 bitexact）
    threshold      = m["near_boundary_threshold"]      # [JSON] 0.9525
    judge_floor    = m["judge_floor_per_class"]        # [JSON]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} seeds={args.seeds} k={NEAREST_K}", flush=True)
    model, schedule, hp = load_cfg_model(args.ckpt, device)
    judge = ResNet18(num_classes=NUM_CLASSES).to(device)
    judge.load_state_dict(torch.load(args.judge, map_location=device, weights_only=True)); judge.eval()
    real_imgs, real_labels = load_real_per_class("cifar10", real_per_class, seed=REAL_REF_SEED)
    real_dino = dinov2_features((real_imgs + 1) / 2, device)
    detector = None if args.no_inception else load_inception_detector(device)

    os.makedirs(ASSET, exist_ok=True)
    torch.save(real_dino.cpu(), f"{ASSET}/real_dino_feat.pt")   # 參考特徵存一次
    torch.save(real_labels.cpu(), f"{ASSET}/real_labels.pt")

    out = {"metadata": env_metadata(sys.argv, start_timestamp),
           "per_seed": [], "status": "running", "stopped_at": None}
    completed = 0
    t_all = time.time()

    for seed in args.seeds:
        seed_block = next(s for s in d["per_seed"] if s["seed"] == seed)
        seed_out = {"seed": seed, "configs": []}
        for conf in seed_block["configs"]:
            w = conf["guidance"]; name = conf["name"]
            gseed = seed * 10_000_000 + int(w * 1000) * 10_000   # 公式原樣（§2 格殺：P1 不改 gseed）
            timing = {}
            scalars, fd_dinov2, art = measure_config(
                model, schedule, judge, detector, real_imgs, real_dino, real_labels,
                judge_floor, threshold, per_class, steps, eta, batch, w, gseed, device, timing)
            recon, over = reconcile(scalars, conf)

            # 落盤影像＋DINOv2 特徵＋judge 輸出＋labels（供 C2/C3/C5，禁二次重生成）
            cdir = f"{ASSET}/seed{seed}_{name}"
            os.makedirs(cdir, exist_ok=True)
            torch.save(art["img_uint8"], f"{cdir}/img_uint8.pt")
            torch.save(art["dino_feat"], f"{cdir}/dino_feat.pt")
            torch.save({"margins": art["margins"], "preds": art["preds"]}, f"{cdir}/judge_out.pt")
            torch.save(art["labels"], f"{cdir}/labels.pt")

            cfg_result = {"name": name, "guidance": w, "gseed": gseed,
                          "reconcile": recon, "over_tol": over,
                          "fd_dinov2": fd_dinov2,          # C1 新量（不對帳，裁決留 γ）
                          "timing_seconds": timing}
            seed_out["configs"].append(cfg_result)
            completed += 1
            bitexact = all(r.get("bitexact") for r in recon.values() if r.get("status") != "skipped_none")
            print(f"[{completed}/30] seed{seed} {name} over_tol={over} bitexact={bitexact} "
                  f"fd_dinov2={fd_dinov2:.3f} gen={timing.get('gen',0):.0f}s", flush=True)

            # 每 config 落盤 streaming 輸出（crash-safe）
            tmp = out.copy(); tmp["per_seed"] = out["per_seed"] + [seed_out]
            json.dump(tmp, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

            if over:
                # 任一 config 超容忍即 STOP（§3、§4）
                out["per_seed"].append(seed_out)
                out["status"] = "STOP_over_tol"
                out["stopped_at"] = {"seed": seed, "name": name}
                json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
                print(f"STOP：seed{seed} {name} 超容忍。已寫 {OUT}，退出。", flush=True)
                return
        out["per_seed"].append(seed_out)
        # seed 完成：僅進度，禁詮釋（§3）。seed 10 之 1-seed FD 曲線不作 which-FID 判讀。
        print(f"seed {seed} 完成（{completed}/30）。[進度用，禁詮釋]", flush=True)

    out["status"] = "complete"
    out["total_seconds"] = time.time() - t_all
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"P1 完成，{completed}/30 全通過對帳。已寫 {OUT}。總時 {out['total_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
