"""P0 探針：單 cell (seed10, w1) 之生成決定性隔離 ＋ 整鏈 scalar 重現，供 β P1 全量 greenlight 前之 gate。

不修改凍結生成/度量路徑，只呼叫之（driver byte-identical 是 P0 前提）。設計要點：

- 輸入來源標註：凡影響對帳 scalar 之非生成輸入，一律從 confirmatory metadata/JSON 讀出對齊，標 [JSON]；
  driver 常數標 [DRV]；metadata 未存、只能沿用 driver 預設者標 [NEW]（＝對帳失效之候選源，呈報時明列）。
- 路 A（生成決定性隔離）：同 gseed 生成兩次，先驗兩次影像逐位相同（＝本環境生成決定性，P1 有效性唯一所繫），
  再對 scalar vs 凍結 JSON（＝整鏈）。confirmatory 已刪 per-sample，故只能以此隔離生成環。
- 三因歸因（非二值）：
  路 A 不等                       → 生成非決定性 → 真 STOP（P1 無效）。
  路 A 等 ＋ 全 scalar 逐位相等   → 生成/參考/數值三者全一致。
  路 A 等 ＋ scalar 容忍內(≤1e-4) → 整鏈 1e-4 內；生成已由路 A 證逐位，殘差在參考側/數值，非生成。
  路 A 等 ＋ scalar 超容忍        → 查參考側對齊([NEW] 清單)或跨環境生成漂移，不查生成決定性。
  注：路 A 證「本環境自我決定性」，非「與 confirmatory 跨環境相同」（per-sample 已刪、無法直比）；
      scalar-vs-JSON 是唯一跨環境相同之 proxy，故 [NEW] 輸入全對齊時，路A等+scalar超容忍指向跨環境生成漂移。

k 溯源（依 records/2026-07-08-02 §2.1 增修）：confirmatory metadata 未存 nearest_k，唯一來源為 driver
argparse 預設 5。precision/coverage（DINOv2 與 Inception 兩側）依賴 k；char_clean_fid、near_boundary_frac、
label_noise_excess 三個 k-free scalar 不依賴。本探針落盤 per-sample 特徵（DINOv2＋Inception），支援離線
k-sweep（--k-sweep-only）：k∈{1..15} 於同一落盤特徵重算 precision/coverage，兩側同步與凍結 JSON 比對，
找回唯一 k*（§2.3b）。凍結 JSON 不動、數字不回改；sweep 只重算已登記定義，屬稽核性參數找回、非結果選購。

Usage:
    uv run python run_p0_probe.py                          # 路 A ＋ 整鏈對帳，落盤 artifacts
    uv run python run_p0_probe.py --k-sweep-only           # 離線 k-sweep（讀既有 artifacts，免重生成）
"""
import argparse
import json
import os
import platform
import sys
import time

# 決定性環境旗標（依 records/2026-07-08-02 §2.1e）：在 import/建 CUDA context 前設 cublas workspace，
# 使 P0 於決定性環境執行，最大化路 A 逐位重現之機率。與 confirmatory 同機同環境跑。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from cifar_cfg_sample import load_cfg_model, generate_balanced
from cifar_classifier import ResNet18
from mechanism import compute_margins, near_boundary_fraction
from metrics_prdc import compute_prdc_per_class
from metrics_features import dinov2_features
from datasets.cifar import load_real_per_class
from run_cifar_cfg_scout import load_inception_detector, inception_crosscheck  # noqa: F401 (crosscheck 保留供參)
from run_cifar_selector import inception_feats
from fid_clean import clean_fid_vs_dataset

CONF = "results/cifar10_cfg_confirmatory.json"
ART = "results/p0_probe_artifacts"
SEED, W = 10, 1.0            # 首個 cell：seed10 之第一 grid 點（sorted grid 之 w1）
TOL = 1e-4                    # 容忍內門檻（相對誤差），與 records/2026-07-06-05 §4 決定性三態一致
K_SWEEP_RANGE = range(1, 16)  # k∈{1..15}（§2.3b）

# 對帳 scalar 是否依賴 nearest_k：k-free 三臂是 §2.3 分診的控制臂，k-dependent 是 k-sweep 找回對象。
K_DEPENDENT = {"precision", "coverage", "precision_inception", "coverage_inception"}


def set_determinism():
    """設決定性演算法旗標並回報實測設定（依 §2.1e）。

    warn_only=True：部分 op（UNet 上採樣、DINOv2 內插）無決定性 kernel，硬拋會使探針無法起跑；
    以 warn_only 保留可執行性，同時把「是否真決定性」交由路 A 逐位比對實測，不靠旗標空言。
    """
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def env_metadata(argv, start_timestamp, nearest_k, per_class, real_per_class, tau_fraction, batch):
    """探針輸出 metadata：強制含 k 溯源與環境版本（依 §2.1c）。凍結 JSON 未存者於此補齊留痕。"""
    n = min(per_class, real_per_class)  # 類別內樣本數；kth_nn 之有效 k 受此上限約束
    return {
        "nearest_k": nearest_k,
        "effective_nearest_k": min(nearest_k, n - 1),   # metrics_prdc.kth_nn_distance 之 k=min(k, n-1)
        "tau_fraction": tau_fraction,
        "batch": batch,
        "per_class": per_class,
        "real_per_class": real_per_class,
        "argv": " ".join(argv),
        "start_timestamp": start_timestamp,
        "env": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
            "python": platform.python_version(),
        },
        "determinism": {"use_deterministic_algorithms": True, "warn_only": True,
                        "cudnn_benchmark": False, "cudnn_deterministic": True,
                        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
    }


def measure_scalars(gen, gen_labels, real_imgs, real_dino, real_labels, judge, detector,
                    judge_floor, threshold, nearest_k, num_classes, device, timing):
    """完全複製 driver measure() 之 scalar 計算路徑（run_cifar_cfg_multiseed.py:67-113）。

    回傳 (scalars, artifacts)：artifacts 含 per-sample 特徵（DINOv2＋Inception）供離線 k-sweep。
    Inception 側同名值（coverage_inception/precision_inception）一併對帳（§2.2）；detector=None 時留 None。
    """
    t = time.time(); gen_dino = dinov2_features((gen + 1) / 2, device); timing['dino_feat'] = time.time() - t
    dino_prdc, _ = compute_prdc_per_class(real_dino.to(device), real_labels.to(device),
                                          gen_dino.to(device), gen_labels.to(device),
                                          nearest_k=nearest_k, num_classes=num_classes)

    t = time.time(); margins, preds = compute_margins(judge, gen, device); timing['judge'] = time.time() - t
    nb = near_boundary_fraction(margins, threshold)
    per_class_excess = []
    for c in range(num_classes):
        m = (gen_labels == c)
        ln_c = (preds[m] != c).float().mean().item() if bool(m.any()) else float('nan')
        per_class_excess.append(ln_c - judge_floor[c])
    excess_mean = sum(per_class_excess) / len(per_class_excess)

    t = time.time()
    char_fid = float(clean_fid_vs_dataset((gen + 1) / 2, dataset_name="cifar10",
                                          dataset_split="train", dataset_res=32))
    timing['clean_fid'] = time.time() - t

    # Inception 交叉表徵：抽 per-sample 特徵並保留（供落盤 + k-sweep），再算同名 coverage/precision。
    # 加 detector is not None 守衛（§2.1a）；與 inception_crosscheck 內部同一 inception_feats 路徑，數值一致。
    incep_cov = incep_prec = None
    gen_incep = real_incep = None
    if detector is not None:
        t = time.time()
        real_incep = inception_feats(real_imgs, detector, device)
        gen_incep = inception_feats(gen, detector, device)
        incep_prdc, _ = compute_prdc_per_class(real_incep.to(device), real_labels.to(device),
                                               gen_incep.to(device), gen_labels.to(device),
                                               nearest_k=nearest_k, num_classes=num_classes)
        incep_cov, incep_prec = incep_prdc["coverage"], incep_prdc["precision"]
        timing['incep_feat'] = time.time() - t

    scalars = {"precision": dino_prdc["precision"], "coverage": dino_prdc["coverage"],
               "coverage_inception": incep_cov, "precision_inception": incep_prec,
               "char_clean_fid": char_fid, "near_boundary_frac": nb,
               "label_noise_excess_mean": excess_mean}
    artifacts = {"gen_dino": gen_dino, "gen_incep": gen_incep, "real_incep": real_incep,
                 "margins": margins, "preds": preds}
    return scalars, artifacts


def reconcile(got, conf):
    """got vs 凍結 JSON config；標 k_dependent 供 §2.3 分診。None（detector 缺/JSON 無）者跳過。"""
    recon = {}
    for k, v in got.items():
        if v is None or k not in conf or conf[k] is None:
            recon[k] = {"got": v, "ref": conf.get(k), "status": "skipped_none", "k_dependent": k in K_DEPENDENT}
            continue
        ref = conf[k]; rel = abs(v - ref) / (abs(ref) + 1e-12)
        recon[k] = {"got": v, "ref": ref, "abs_delta": abs(v - ref), "rel_delta": rel,
                    "bitexact": v == ref, "within_tol": rel <= TOL, "k_dependent": k in K_DEPENDENT}
    return recon


def classify_verdict(gen_bitexact, recon):
    """三因歸因（§2.3）：以 k-free 三臂與整體對帳決定 verdict。"""
    if not gen_bitexact:
        return "GEN_NONDETERMINISTIC_HARD_STOP"
    checked = [r for r in recon.values() if r.get("status") != "skipped_none"]
    all_bitexact = all(r["bitexact"] for r in checked)
    all_within = all(r["within_tol"] for r in checked)
    kfree = [r for k, r in recon.items() if not r["k_dependent"] and r.get("status") != "skipped_none"]
    kfree_within = all(r["within_tol"] for r in kfree)
    if all_bitexact:
        return "ALL_BITEXACT"
    if all_within:
        return "WITHIN_TOL_residual_in_reference_or_numeric_not_generation"
    if kfree_within:
        # k-free 三臂過、k-dependent 未過 → 指向 k 未對齊（§2.3b），非生成非決定性。觸發 k-sweep。
        return "KFREE_OK_KDEP_OVER_TOL_run_k_sweep"
    return "OVER_TOL_check_NEW_inputs_or_cross_env_drift_not_gen_determinism"


def k_sweep(conf, device):
    """離線 k-sweep（§2.3b）：讀既有 artifacts，k∈{1..15} 兩側重算 precision/coverage vs 凍結 JSON。

    找回唯一 k*（兩側 precision/coverage 皆容忍內之 k）。不重生成、不改凍結 JSON。
    """
    labels = torch.load(f"{ART}/labels.pt", map_location=device, weights_only=True)
    gen_labels, real_labels = labels["gen_labels"], labels["real_labels"]
    gen_dino = torch.load(f"{ART}/dino_feat.pt", map_location=device, weights_only=True)
    real_dino = torch.load(f"{ART}/real_dino_feat.pt", map_location=device, weights_only=True)

    sides = [("dino", gen_dino, real_dino, "precision", "coverage")]
    incep_path = f"{ART}/incep_feat.pt"
    if os.path.exists(incep_path):
        gen_incep = torch.load(incep_path, map_location=device, weights_only=True)
        real_incep = torch.load(f"{ART}/real_incep_feat.pt", map_location=device, weights_only=True)
        sides.append(("inception", gen_incep, real_incep, "precision_inception", "coverage_inception"))

    sweep = {}
    for k in K_SWEEP_RANGE:
        entry = {}
        for name, gf, rf, prec_key, cov_key in sides:
            prdc, _ = compute_prdc_per_class(rf.to(device), real_labels.to(device),
                                             gf.to(device), gen_labels.to(device),
                                             nearest_k=k, num_classes=10)
            for metric, key in (("precision", prec_key), ("coverage", cov_key)):
                ref = conf.get(key)
                got = prdc[metric]
                rel = abs(got - ref) / (abs(ref) + 1e-12) if ref is not None else None
                entry[key] = {"got": got, "ref": ref, "rel_delta": rel,
                              "within_tol": (rel is not None and rel <= TOL)}
        sweep[k] = entry

    # k*：所有非 None 對帳項皆容忍內之 k（兩側 precision/coverage 同步約束）
    matches = [k for k, e in sweep.items()
               if all(v["within_tol"] for v in e.values() if v["ref"] is not None)]
    return {"sweep": sweep, "k_star_candidates": matches,
            "k_star": matches[0] if len(matches) == 1 else None,
            "unique": len(matches) == 1}


def run_k_sweep_only():
    """離線模式：僅跑 k-sweep（讀 artifacts，免 GPU 生成）。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = json.load(open(CONF, encoding="utf-8"))
    conf = next(c for s in d["per_seed"] if s["seed"] == SEED for c in s["configs"] if c["name"] == "w1")
    result = k_sweep(conf, device)
    out_path = "results/cifar10_p0_ksweep.json"
    json.dump(result, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print(f"Wrote {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description="P0 探針：單 cell 生成決定性隔離＋整鏈 scalar 重現。")
    p.add_argument("--ckpt", default="checkpoints/cifar10_cfg.pt")
    p.add_argument("--judge", default="checkpoints/cifar10_judge.pt")
    p.add_argument("--nearest-k", type=int, default=5,
                   help="PRDC k；confirmatory metadata 未存，預設 5 為文件推定（非儲存值）")
    p.add_argument("--batch", type=int, default=250, help="生成批次；CUDA 下影響噪聲，須與 confirmatory 對齊")
    p.add_argument("--tau-fraction", type=float, default=0.9, help="僅供 metadata 留痕；探針不算 selector")
    p.add_argument("--no-inception", action="store_true", help="略過 Inception 交叉表徵（僅除錯用）")
    p.add_argument("--k-sweep-only", action="store_true", help="離線 k-sweep：讀既有 artifacts、免重生成")
    args = p.parse_args()

    if args.k_sweep_only:
        run_k_sweep_only()
        return

    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    set_determinism()

    d = json.load(open(CONF, encoding="utf-8"))
    m = d["metadata"]
    # === 輸入來源對齊 ===
    per_class      = m["per_class"]                   # [JSON] 1000
    real_per_class = m["real_per_class"]              # [JSON] 1000
    steps          = m["steps"]                       # [JSON] 50
    eta            = m["eta"]                          # [JSON] 0.0
    threshold      = m["near_boundary_threshold"]     # [JSON] 0.9525
    judge_floor    = m["judge_floor_per_class"]       # [JSON]
    num_classes    = 10                                # [DRV]
    real_ref_seed  = 0                                 # [DRV] load_real_per_class 常數 seed=0（driver:56,89）
    nearest_k      = args.nearest_k                    # [NEW] metadata 未存 → 對帳失效候選；k-sweep 找回
    batch          = args.batch                        # [NEW] metadata 未存；CUDA 下 batch 影響噪聲 → 必對齊
    gseed = SEED * 10_000_000 + int(W * 1000) * 10_000  # = 110000000 = confirmatory (seed10, w1)

    sources = {"per_class":"JSON","real_per_class":"JSON","steps":"JSON","eta":"JSON",
               "threshold":"JSON","judge_floor":"JSON","num_classes":"DRV","real_ref_seed":"DRV",
               "nearest_k":"NEW","batch":"NEW","ckpt":"NEW","judge_ckpt":"NEW"}
    new_inputs = [k for k, v in sources.items() if v == "NEW"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} gseed={gseed} [NEW 對帳失效候選]={new_inputs}", flush=True)
    model, schedule, hp = load_cfg_model(args.ckpt, device)
    judge = ResNet18(num_classes=10).to(device)
    judge.load_state_dict(torch.load(args.judge, map_location=device, weights_only=True)); judge.eval()
    real_imgs, real_labels = load_real_per_class("cifar10", real_per_class, seed=real_ref_seed)
    real_dino = dinov2_features((real_imgs + 1) / 2, device)
    detector = None if args.no_inception else load_inception_detector(device)

    timing = {}
    # === 路 A：同 gseed 生成兩次，驗逐位相同 ===
    t = time.time(); gen1, lab1 = generate_balanced(model, schedule, per_class, device, steps, eta,
                                                    guidance=W, num_classes=num_classes, batch=batch, seed=gseed)
    timing['gen_1'] = time.time() - t
    t = time.time(); gen2, lab2 = generate_balanced(model, schedule, per_class, device, steps, eta,
                                                    guidance=W, num_classes=num_classes, batch=batch, seed=gseed)
    timing['gen_2'] = time.time() - t
    gen_bitexact = bool(torch.equal(gen1, gen2)) and bool(torch.equal(lab1, lab2))
    gen_max_abs_diff = float((gen1 - gen2).abs().max()) if gen1.shape == gen2.shape else None

    # === 整鏈：scalar on gen1 vs 凍結 JSON (seed10, w1) ===
    got, art = measure_scalars(gen1, lab1, real_imgs, real_dino, real_labels, judge, detector,
                               judge_floor, threshold, nearest_k, num_classes, device, timing)
    # 解除 per_seed 存檔順序耦合（§2.1b）：以 seed 值查，不假設 per_seed[0]=seed10。
    seed_block = next(s for s in d["per_seed"] if s["seed"] == SEED)
    conf = next(c for c in seed_block["configs"] if c["name"] == "w1")
    recon = reconcile(got, conf)
    verdict = classify_verdict(gen_bitexact, recon)

    # === 落盤 artifacts（uint8 影像＋DINOv2/Inception per-sample 特徵＋judge 輸出＋labels），供 k-sweep ===
    os.makedirs(ART, exist_ok=True)
    imgs_u8 = ((gen1 + 1) / 2 * 255).round().clamp(0, 255).to(torch.uint8)
    torch.save(imgs_u8, f"{ART}/img_uint8.pt")
    torch.save(art["gen_dino"].cpu(), f"{ART}/dino_feat.pt")
    torch.save(real_dino.cpu(), f"{ART}/real_dino_feat.pt")
    torch.save({"gen_labels": lab1.cpu(), "real_labels": real_labels.cpu()}, f"{ART}/labels.pt")
    torch.save({"margins": art["margins"].cpu(), "preds": art["preds"].cpu()}, f"{ART}/judge_out.pt")
    if art["gen_incep"] is not None:
        torch.save(art["gen_incep"].cpu(), f"{ART}/incep_feat.pt")
        torch.save(art["real_incep"].cpu(), f"{ART}/real_incep_feat.pt")
    storage = {f: os.path.getsize(f"{ART}/{f}") for f in os.listdir(ART)}
    storage_cell_bytes = sum(storage.values())

    # === k-sweep（僅在 k-free 過、k-dep 未過時觸發；否則附完整 sweep 供稽核）===
    ksweep = None
    if verdict in ("KFREE_OK_KDEP_OVER_TOL_run_k_sweep", "OVER_TOL_check_NEW_inputs_or_cross_env_drift_not_gen_determinism"):
        ksweep = k_sweep(conf, device)

    out = {"cell": {"seed": SEED, "w": W, "gseed": gseed},
           "metadata": env_metadata(sys.argv, start_timestamp, nearest_k, per_class,
                                     real_per_class, args.tau_fraction, batch),
           "input_sources": sources, "new_inputs": new_inputs,
           "path_A_generation": {"bitexact": gen_bitexact, "max_abs_diff": gen_max_abs_diff},
           "scalar_reconcile_vs_frozen_JSON": recon,
           "verdict": verdict,
           "k_sweep": ksweep,
           "timing_seconds": timing,
           "storage_bytes_cell": storage, "storage_bytes_cell_total": storage_cell_bytes,
           "storage_estimate_30cell_GB": round(storage_cell_bytes * 30 / 1e9, 3)}
    json.dump(out, open("results/cifar10_p0_probe.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    print("Wrote results/cifar10_p0_probe.json", flush=True)


if __name__ == "__main__":
    main()
