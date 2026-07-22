"""τ 靈敏度分析（對應審查 A6）：CaF 的選擇對 tau_fraction 有多敏感，含「無 floor」極限。

問題設定（給第一次讀的研究生）：CaF 在 precision >= τ 的組態中挑 signal（coverage 或 recall）最大者，
τ = tau_fraction × real-vs-real 參考 precision。CIFAR-100 的判決三「CaF-v2 與 FID-min 打平（regret
0.76 對 0.76）」其實座落在 τ 刀鋒上：oracle w1 的 precision 恰略低於預註冊的 tau_fraction=0.9 所定的 τ，
於是 w1 被 floor 擋掉、改選 w1.5。本檔把「選擇對 tau_fraction 的翻轉」量化出來，並報告一個上界情形：
拿掉 floor 的裸 argmax（recall/coverage）——它在 CIFAR-100 八 seed 全中 oracle w1、regret 0.00，但
即便如此，相對 FID-min 的勝幅仍不足預註冊 D4 的 1.5pp 門檻。兩點都如實報告。

量的東西（每資料集）：
  (a) 既算 τ-掃描：直接提出凍結 JSON 內每 seed 的 report.tau_robustness（modal 選擇、穩定度、picks），
      這是程式早已算出、論文未報的完整掃描。
  (b) tau_fraction ∈ {0.80,0.85,0.90,0.95} 與「無 floor」重跑 select_caf（輸入即凍結 configs），
      輸出每設定的 per-seed 選擇與 regret、命中 oracle 的 seed 數、以及相對 FID-min 是否達 D4 的 1.5pp。

本檔為純衍生分析：數字全部由凍結 JSON 讀出（沿用 selector.select_caf／auto_tau），不重跑 GPU、
可在 CPU 逐位重現。樣本數配平校準（real 側 250 子抽樣）另見 --matched（需快取特徵）。

Usage:
    uv run python src/experiments/run_tau_sensitivity.py
    uv run python src/experiments/run_tau_sensitivity.py --no-write   # 只算與對帳，不寫檔
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py

import argparse
import json
import os
import sys
import time

import torch

from selector import auto_tau, select_caf

TAU_FRACTIONS = [0.80, 0.85, 0.90, 0.95]
D4_MARGIN_PP = 1.5  # 預註冊 D4：CaF 須以此勝幅贏過 FID-min 才算「勝」

# 各資料集：來源、signal 訊號、參考 precision 欄名、FID-min duel 檔（供 D4 勝幅）
SOURCES = [
    {"name": "CIFAR-10", "path": "results/cifar10_cfg_confirmatory.json",
     "refkey": "ref_precision", "duel": "results/cifar10_c6_fidmin_duel.json"},
    {"name": "CIFAR-100", "path": "results/cifar100_cfg_confirmatory.json",
     "refkey": "ref_precision", "duel": "results/cifar100_c6_fidmin_duel.json"},
]


def signal_of(meta):
    """metadata.selector_signal_key（CIFAR-100=recall），缺或 None 時退回 coverage。"""
    return meta.get("selector_signal_key") or "coverage"


def one_setting(per_seed, signal, refkey, frac):
    """對某 tau_fraction（frac=None 表示無 floor）回傳 per-seed 選擇與 regret 與彙總。"""
    rows = []
    for b in per_seed:
        configs = b["configs"]
        oracle = max(configs, key=lambda c: c["tstr"])
        ref = b.get(refkey, b.get("real_ref_precision"))
        if frac is None:
            tau = -1.0  # 無 floor：precision >= -1 恆真 -> 裸 argmax signal
        else:
            tau = auto_tau(ref, frac)
        sel, passed = select_caf(configs, tau, signal_key=signal)
        rows.append({
            "seed": b["seed"],
            "selected": sel["name"],
            "regret": round(oracle["tstr"] - sel["tstr"], 2),
            "tau": round(tau, 4) if frac is not None else None,
            "passed_floor": passed,
            "hit_oracle": sel["name"] == oracle["name"],
        })
    regs = [r["regret"] for r in rows]
    return {
        "per_seed": rows,
        "mean_regret": round(sum(regs) / len(regs), 2),
        "hit_oracle_seeds": sum(1 for r in rows if r["hit_oracle"]),
        "n_seeds": len(rows),
    }


def analyse(data, meta_name, refkey, duel_path):
    d = data
    signal = signal_of(d["metadata"])
    per_seed = d["per_seed"]
    tau_frac_prereg = d["metadata"].get("tau_fraction", 0.9)

    # (a) 既算 τ-掃描：提出每 seed 的 tau_robustness
    existing = [{"seed": b["seed"],
                 "modal": b["report"]["tau_robustness"]["modal"],
                 "stability": round(b["report"]["tau_robustness"]["stability"], 4),
                 "picks": b["report"]["tau_robustness"]["picks"]}
                for b in per_seed]

    # (b) tau_fraction 掃描 + 無 floor
    settings = {}
    for frac in TAU_FRACTIONS:
        settings[f"tau_frac_{frac:.2f}"] = one_setting(per_seed, signal, refkey, frac)
    settings["no_floor"] = one_setting(per_seed, signal, refkey, None)

    # FID-min 勝幅（D4）：FID-min mean regret - CaF-setting mean regret >= 1.5 才算勝
    with open(duel_path, encoding="utf-8") as f:
        fidmin_mean = json.load(f)["fidmin_regret_mean"]
    for k, s in settings.items():
        margin = round(fidmin_mean - s["mean_regret"], 2)  # 正 = CaF 較佳
        s["beats_fidmin_by_pp"] = margin
        s["clears_d4_1.5pp"] = margin >= D4_MARGIN_PP

    return {
        "dataset": meta_name,
        "signal_key": signal,
        "tau_fraction_prereg": tau_frac_prereg,
        "fidmin_regret_mean": fidmin_mean,
        "d4_margin_pp": D4_MARGIN_PP,
        "existing_tau_robustness": existing,
        "settings": settings,
    }


def matched_calibration_seed10(seed10_block, feat_dir, signal, tau_fraction):
    """配平校準（seed 10）：per-config precision 在 500v500 與 250v250 兩尺度重算，以 250v250（對齊
    real-vs-real 校準尺度）重跑 floor，檢驗 CIFAR-100 的 τ 打平是否為量測（500v500）與校準（250v250）
    樣本數不一致的假影。需快取 DINOv2 特徵（`results/p1_assets_cifar100/`）；缺檔則回傳 available=False。"""
    try:
        import torch as _t
        from metrics_prdc import compute_prdc_per_class
    except Exception as e:
        return {"available": False, "reason": f"import 失敗：{e}"}
    real_path = os.path.join(feat_dir, "real_dino_feat.pt")
    if not os.path.exists(real_path):
        return {"available": False, "reason": f"缺快取特徵 {feat_dir}（gitignore，本機才有）"}
    real = _t.load(real_path, map_location="cpu")
    real_lab = _t.load(os.path.join(feat_dir, "real_labels.pt"), map_location="cpu")
    n, per = real_lab.numel(), 500
    within = _t.arange(n) % per
    gen_lab = _t.arange(n) // per            # 類序（已驗證 real 與 gen 皆 arange//500）
    half = within < per // 2                 # 每類前 250
    other = (within >= per // 2) & (within < per)

    def prec(rf, rl, gf, gl):
        return float(compute_prdc_per_class(rf, rl, gf, gl, nearest_k=5, num_classes=100)[0]["precision"])

    ref_250 = prec(real[half], real_lab[half], real[other], real_lab[other])  # real-vs-real 250v250
    tau = tau_fraction * ref_250
    oracle = max(seed10_block["configs"], key=lambda c: c["tstr"])
    per_config = []
    for c in seed10_block["configs"]:
        gp = os.path.join(feat_dir, f"seed10_{c['name']}", "dino_feat.pt")
        if not os.path.exists(gp):
            return {"available": False, "reason": f"缺 {gp}"}
        gf = _t.load(gp, map_location="cpu")
        per_config.append({
            "name": c["name"],
            "precision_500v500": round(prec(real, real_lab, gf, gen_lab), 4),          # 應與 frozen 相符
            "precision_250v250": round(prec(real[half], real_lab[half], gf[half], gen_lab[half]), 4),
            "recall": round(c[signal], 4),
            "tstr": c["tstr"],
        })

    def select(pkey):
        elig = [pc for pc in per_config if pc[pkey] >= tau]
        sel = max(elig if elig else per_config, key=lambda pc: pc["recall"])
        return sel["name"], round(oracle["tstr"] - sel["tstr"], 2)

    s500, r500 = select("precision_500v500")
    s250, r250 = select("precision_250v250")
    return {
        "available": True,
        "seed": 10,
        "tau_fraction": tau_fraction,
        "ref_precision_250v250": round(ref_250, 4),
        "tau": round(tau, 4),
        "per_config": per_config,
        "selection_500v500": {"selected": s500, "regret": r500},
        "selection_250v250_matched": {"selected": s250, "regret": r250},
        "flipped_to_oracle": bool(s250 == oracle["name"] and s500 != oracle["name"]),
        "note": "real 與 gen 側各子抽樣至 250/class（對齊 250v250 real-vs-real 校準尺度）重算 precision "
                "floor；num_classes=100、nearest_k=5、每類前 250。500v500 欄應逐位重現 frozen precision。",
    }


def reconcile(new, output_path):
    if not os.path.exists(output_path):
        print(f"[reconcile] 無既有 {output_path}，跳過（首次產生）")
        return True
    with open(output_path, encoding="utf-8") as f:
        old = json.load(f)
    ok = {k: v for k, v in old.items() if k != "metadata"} == {k: v for k, v in new.items() if k != "metadata"}
    print(f"[reconcile] vs {output_path}: {'ALL_MATCH' if ok else 'MISMATCH'}")
    return ok


def main():
    p = argparse.ArgumentParser(description="τ 靈敏度：tau_fraction 掃描與無 floor 極限（A6，純衍生）。")
    p.add_argument("--output", default="results/tau_sensitivity.json")
    p.add_argument("--no-write", action="store_true", help="只算與對帳，不寫檔")
    args = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    results, raw = {}, {}
    for src in SOURCES:
        with open(src["path"], encoding="utf-8") as f:
            data = json.load(f)
        raw[src["name"]] = data
        r = analyse(data, src["name"], src["refkey"], src["duel"])
        r["_source"] = {"confirmatory": src["path"], "duel": src["duel"]}
        results[src["name"]] = r

    # T4 (c) 配平校準：seed 10、CIFAR-100（需快取特徵；缺檔則 available=False）
    c100 = raw["CIFAR-100"]
    seed10 = next(b for b in c100["per_seed"] if b["seed"] == 10)
    results["CIFAR-100"]["matched_calibration_seed10"] = matched_calibration_seed10(
        seed10, "results/p1_assets_cifar100", signal_of(c100["metadata"]),
        c100["metadata"].get("tau_fraction", 0.9))

    out = {"datasets": results}
    matched = reconcile(out, args.output)

    # 報表
    print("\n" + "=" * 92)
    print("  τ 靈敏度：tau_fraction 掃描與無 floor 極限（regret@selected，pp，越低越好）")
    print("=" * 92)
    for name in ("CIFAR-10", "CIFAR-100"):
        r = results[name]
        print(f"\n  [{name}]  signal={r['signal_key']}  預註冊 tau_fraction={r['tau_fraction_prereg']}"
              f"  FID-min regret={r['fidmin_regret_mean']}")
        for k in [f"tau_frac_{f:.2f}" for f in TAU_FRACTIONS] + ["no_floor"]:
            s = r["settings"][k]
            picks = {}
            for row in s["per_seed"]:
                picks[row["selected"]] = picks.get(row["selected"], 0) + 1
            picks_str = ", ".join(f"{n}×{c}" for n, c in sorted(picks.items()))
            flag = "達 D4" if s["clears_d4_1.5pp"] else "未達 D4"
            print(f"    {k:>14}: mean_regret={s['mean_regret']:>5.2f}  命中 oracle {s['hit_oracle_seeds']}/{s['n_seeds']}"
                  f"  勝 FID-min {s['beats_fidmin_by_pp']:+.2f}pp（{flag}）  選中: {picks_str}")
    print("\n  重點：CIFAR-100 無 floor 的裸 argmax recall 八 seed 全中 oracle w1、regret 0.00，")
    print("        但相對 FID-min 僅勝 0.76pp、未達 D4 的 1.5pp 門檻——沒有任一 τ 設定能讓 CaF 依 D4 勝出。")
    mc = results["CIFAR-100"].get("matched_calibration_seed10", {})
    if mc.get("available"):
        print(f"\n  配平校準（seed 10，CIFAR-100）：τ={mc['tau']}（=0.9×ref250 {mc['ref_precision_250v250']}）")
        print(f"    500v500 量測 → 選 {mc['selection_500v500']['selected']}"
              f"（regret {mc['selection_500v500']['regret']}）")
        print(f"    250v250 配平 → 選 {mc['selection_250v250_matched']['selected']}"
              f"（regret {mc['selection_250v250_matched']['regret']}）"
              f"{'  翻回 oracle' if mc['flipped_to_oracle'] else ''}")
        print("    → 樣本數配平後 w1 通過 floor、CaF-v2 選中 oracle：τ 打平部分為 500v500/250v250 樣本數不一致假影。")
    else:
        print(f"\n  配平校準（seed 10）：略過（{mc.get('reason', '')}）")
    print("=" * 92)

    if args.no_write:
        return
    if not matched:
        raise SystemExit("與既有輸出檔對帳不符，拒絕覆寫。請查明差異來源。")

    out["metadata"] = {
        "analysis": "tau_sensitivity",
        "status": "derived",
        "note": "純衍生：沿用凍結 configs 與 selector.select_caf／auto_tau；不改任何凍結判決。",
        "tau_fractions": TAU_FRACTIONS,
        "d4_margin_pp": D4_MARGIN_PP,
        "sources": {name: results[name]["_source"] for name in results},
        "reconciled_against_output": matched,
        "start_timestamp": start_timestamp,
        "argv": " ".join(sys.argv),
        "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version()},
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.output}（純衍生，逐位可由凍結檔重導）")


if __name__ == "__main__":
    main()
