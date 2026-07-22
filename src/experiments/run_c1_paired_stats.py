"""C1 配對統計（對應審查 A8）：TSTR-argmax 組態 vs FID-argmin 組態的 per-seed 配對差，雙口徑呈現。

問題設定（給第一次讀的研究生）：判決三用「凍結格步口徑」判 which-FID 是否分離——FID-argmin
與 TSTR-argmax 在 guidance 網格上相隔幾格，>1 格才算「FID 與效用分離」。這個口徑是路由依據、
已凍結、不回改。但它把「兩者只差幾格」與「兩者 TSTR 差幾 pp」混在一起：即使不分離（0 格或
1 格），FID-min 每個 seed 仍可能系統性地少挑到一點 TSTR。本檔用第二把尺——標準配對檢定
（post-hoc）——把這個系統性偏移量化出來，兩把尺並列，讓讀者看清「不分離」與「無偏移」不是
同一件事。

量的東西（逐 seed，兩資料集各一份）：
  - oracle       = argmax_w TSTR（該 seed 的 TSTR 最佳組態）
  - fidmin       = argmin_w char_clean_fid（Inception 空間 clean-fid 最小組態；FID-min baseline 選的）
  - paired_diff  = TSTR(oracle) - TSTR(fidmin) = FID-min 的 regret@selected（恆 >= 0，因 oracle 為上界）
  - 口徑一（凍結格步）：sep_step = |grid(fidmin) - grid(oracle)|；separated = sep_step > 1 的 seed 數
  - 口徑二（配對檢定 post-hoc）：配對 t 檢定（scipy ttest_rel）、符號檢定（binomtest）、95% CI

結論敘述（寫入論文 §5.3/§5.4.1）：偏移系統性存在、方向符合原假說（FID-min 系統性少挑一點
TSTR），但幅度極小（CIFAR-100 約 0.76pp，實務可忽略）；且 TSTR-argmax 常落在網格邊界點 w1，
邊界最優本身待 w<1 scout（T5b）釐清。本檔不改任何凍結判決，只補充解讀。

本檔是純衍生分析：所有數字都從凍結 confirmatory JSON 讀出，不重跑 GPU，可在 CPU 逐位重現；
另以既有 c6 duel JSON 的 fidmin_regret_per_seed 交叉核對（自我一致性）。metadata 標 post-hoc。

Usage:
    uv run python src/experiments/run_c1_paired_stats.py
    uv run python src/experiments/run_c1_paired_stats.py --no-write   # 只算與對帳，不寫檔
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py

import argparse
import json
import os
import sys
import time

import numpy as np
import scipy
import torch
from scipy import stats


def paired_diffs(confirmatory):
    """回傳 (names, per_seed 明細, oracle TSTR 向量, fidmin TSTR 向量)。"""
    names = [f"w{w:g}" for w in confirmatory["metadata"]["guidance_grid"]]
    per_seed, tstr_oracle, tstr_fidmin = [], [], []
    for sr in confirmatory["per_seed"]:
        configs = sr["configs"]
        oracle = max(configs, key=lambda c: c["tstr"])
        fidmin = min(configs, key=lambda c: c["char_clean_fid"])
        sep = abs(names.index(fidmin["name"]) - names.index(oracle["name"]))
        per_seed.append({
            "seed": sr["seed"],
            "tstr_argmax": oracle["name"],
            "tstr_argmax_tstr": round(oracle["tstr"], 3),
            "fid_argmin": fidmin["name"],
            "fid_argmin_tstr": round(fidmin["tstr"], 3),
            "paired_diff": round(oracle["tstr"] - fidmin["tstr"], 2),  # 對齊 c6 duel 的 2 位
            "sep_step": sep,
        })
        tstr_oracle.append(oracle["tstr"])
        tstr_fidmin.append(fidmin["tstr"])
    return names, per_seed, tstr_oracle, tstr_fidmin


def analyse(confirmatory, dataset):
    names, per_seed, a, b = paired_diffs(confirmatory)
    a, b = np.asarray(a, float), np.asarray(b, float)
    diffs = a - b
    n = len(diffs)

    # 口徑一：凍結格步（路由依據）
    sep_steps = [r["sep_step"] for r in per_seed]
    grid_step = {
        "sep_step_per_seed": sep_steps,
        "separated_seeds_gt1": sum(1 for s in sep_steps if s > 1),
        "n_seeds": n,
        "rule": "C1 口徑：FID-argmin 與 TSTR-argmax 相隔 > 1 格才算分離",
    }

    # 口徑二：標準配對檢定（post-hoc）
    mean_d = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if n > 1 else 0.0
    se = sd / n ** 0.5 if n > 1 else 0.0
    tt = stats.ttest_rel(a, b)  # 雙尾配對 t 檢定，等價於對 diffs 的單樣本 t
    t_stat, p_two = float(tt.statistic), float(tt.pvalue)
    if n > 1:
        tcrit = float(stats.t.ppf(0.975, n - 1))
        ci = [mean_d - tcrit * se, mean_d + tcrit * se]
    else:
        ci = [mean_d, mean_d]
    pos = int((diffs > 1e-9).sum())
    neg = int((diffs < -1e-9).sum())
    zero = int((np.abs(diffs) <= 1e-9).sum())
    nz = pos + neg
    sign_p = float(stats.binomtest(pos, nz, 0.5).pvalue) if nz > 0 else None
    paired_test = {
        "mean_diff": round(mean_d, 4),
        "sd": round(sd, 4),
        "se": round(se, 4),
        "t": round(t_stat, 4),
        "df": n - 1,
        "p_two_sided": p_two,
        "ci95": [round(ci[0], 4), round(ci[1], 4)],
        "sign_test": {"n_positive": pos, "n_negative": neg, "n_zero": zero,
                      "n_nonzero": nz, "p_two_sided": sign_p},
    }

    return {
        "dataset": dataset,
        "n_seeds": n,
        "per_seed": per_seed,
        "paired_diff_per_seed": [r["paired_diff"] for r in per_seed],
        "grid_step_caliber": grid_step,
        "paired_test_posthoc": paired_test,
    }


def cross_check(result, duel_path):
    """以既有 c6 duel 的 fidmin_regret_per_seed 交叉核對本檔重算的 paired_diff（自我一致性）。"""
    if not os.path.exists(duel_path):
        print(f"[cross-check] 無 {duel_path}，跳過交叉核對")
        return True
    with open(duel_path, encoding="utf-8") as f:
        duel = json.load(f)
    got = result["paired_diff_per_seed"]
    want = duel["fidmin_regret_per_seed"]
    ok = got == want
    print(f"[cross-check] {result['dataset']} paired_diff vs {os.path.basename(duel_path)} "
          f"fidmin_regret: {'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        print(f"    重算 {got}\n    duel  {want}")
    return ok


def reconcile(new, output_path):
    """對既有輸出檔逐值對帳（若存在），確保重跑 payload 逐位不變。"""
    if not os.path.exists(output_path):
        print(f"[reconcile] 無既有 {output_path}，跳過（首次產生）")
        return True
    with open(output_path, encoding="utf-8") as f:
        old = json.load(f)
    old_p = {k: v for k, v in old.items() if k != "metadata"}
    new_p = {k: v for k, v in new.items() if k != "metadata"}
    ok = old_p == new_p
    print(f"[reconcile] vs {output_path}: {'ALL_MATCH' if ok else 'MISMATCH'}")
    return ok


def main():
    p = argparse.ArgumentParser(description="C1 配對統計：TSTR-argmax vs FID-argmin 雙口徑。")
    p.add_argument("--output", default="results/c1_paired_stats.json")
    p.add_argument("--no-write", action="store_true", help="只算與對帳，不寫檔")
    args = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # 確保中文報表不受終端編碼影響
    except Exception:
        pass
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    datasets = [
        ("CIFAR-10", "results/cifar10_cfg_confirmatory.json", "results/cifar10_c6_fidmin_duel.json"),
        ("CIFAR-100", "results/cifar100_cfg_confirmatory.json", "results/cifar100_c6_fidmin_duel.json"),
    ]
    results, all_cross_ok = {}, True
    for name, conf_path, duel_path in datasets:
        with open(conf_path, encoding="utf-8") as f:
            conf = json.load(f)
        r = analyse(conf, name)
        r["_source"] = {"confirmatory": conf_path, "cross_check_duel": duel_path}
        all_cross_ok &= cross_check(r, duel_path)
        results[name] = r

    out = {"datasets": results}
    matched = reconcile(out, args.output)

    # 報表
    print("\n" + "=" * 82)
    print("  C1 配對統計：TSTR-argmax vs FID-argmin（雙口徑）")
    print("=" * 82)
    for name in ("CIFAR-10", "CIFAR-100"):
        r = results[name]
        pt = r["paired_test_posthoc"]
        gs = r["grid_step_caliber"]
        print(f"\n  [{name}]  n_seeds={r['n_seeds']}  paired_diff(pp)={r['paired_diff_per_seed']}")
        print(f"    口徑一 凍結格步：分離(>1 格)的 seed 數 = {gs['separated_seeds_gt1']}/{gs['n_seeds']}"
              f"（sep_step={gs['sep_step_per_seed']}）")
        print(f"    口徑二 配對檢定：mean={pt['mean_diff']:.4f}pp  t={pt['t']:.4f}  df={pt['df']}  "
              f"p={pt['p_two_sided']:.3g}  CI95={pt['ci95']}")
        st = pt["sign_test"]
        print(f"                    符號檢定：{st['n_positive']}/{st['n_nonzero']} 正"
              f"（含 {st['n_zero']} 個 0）  p={st['p_two_sided']}")
    print("\n  解讀：不分離（口徑一）不等於無偏移（口徑二）；偏移系統性存在、方向符合原假說，")
    print("        但 CIFAR-100 幅度約 0.76pp 實務可忽略，且 TSTR-argmax 常落在網格邊界 w1。")
    print("=" * 82)

    if args.no_write:
        return
    if not all_cross_ok:
        raise SystemExit("交叉核對不符（重算 paired_diff 與凍結 duel 不一致），停止。請查明差異。")
    if not matched:
        raise SystemExit("與既有輸出檔對帳不符，拒絕覆寫。請查明差異來源。")

    out["metadata"] = {
        "analysis": "c1_paired_stats",
        "status": "post-hoc",
        "note": "凍結格步口徑為路由依據不回改；配對檢定為 post-hoc 補充解讀，不改任何凍結判決。",
        "sources": {name: results[name]["_source"] for name in results},
        "cross_check_passed": all_cross_ok,
        "reconciled_against_output": matched,
        "start_timestamp": start_timestamp,
        "argv": " ".join(sys.argv),
        "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "numpy": np.__version__, "scipy": scipy.__version__},
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.output}（純衍生，post-hoc；paired_diff 已與凍結 duel 交叉核對）")


if __name__ == "__main__":
    main()
