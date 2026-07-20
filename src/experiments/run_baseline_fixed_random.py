"""固定 w 與隨機可行點 baseline（對應審查 A7）：補齊預註冊 D5 明列而論文未報的兩個對照選擇器。

問題設定（給第一次讀的研究生）：CaF 與 FID-min 要證明「值得用」，得先贏過最笨的選法。預註冊 D5
列了兩個下限 baseline，但論文只報了 CaF 與 FID-min，沒報這兩個：
  - 固定 w（fixed-w）：不看任何訊號，永遠挑同一個 guidance。它的 per-seed regret = 該 seed 的
    oracle TSTR 減去該固定 w 的 TSTR。為避免事後挑一個剛好好看的 w（HARKing），本檔對網格上
    「每一個」w 都輸出整欄 per-seed regret，不預先挑點；論文表再標出三個有依據的值：網格最低
    w（w1／g1）、w1.5、w2（文獻慣例值，SD scale 2.0）。
  - 隨機可行點（random-feasible）：在「precision >= tau」的可行集中均勻抽一個組態的期望 regret。
    用該 seed report 內的 tau 與各 config 的 precision 重建可行集，取可行集內 TSTR 的解析平均
    （不抽樣），期望 regret = oracle TSTR - 可行集平均 TSTR。

三個資料集（MNIST/CIFAR-10/CIFAR-100）各算一份，對齊論文表 5.3 的三列。本檔是純衍生分析：
數字全部由凍結 JSON 讀出，不重跑 GPU，可在 CPU 逐位重現。

Usage:
    uv run python src/experiments/run_baseline_fixed_random.py
    uv run python src/experiments/run_baseline_fixed_random.py --no-write   # 只算與對帳，不寫檔
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py

import argparse
import json
import os
import sys
import time

import torch


def analyse(data, dataset):
    """對單一資料集算 fixed-w 整欄與 random-feasible baseline。data 為凍結 JSON。"""
    per_seed_blocks = data["per_seed"]
    seeds = [b["seed"] for b in per_seed_blocks]
    # 以 per_seed[0] 的 config 順序當網格；各 seed 以 name 對齊
    grid = [c["name"] for c in per_seed_blocks[0]["configs"]]

    oracle_tstr = []
    fixed = {name: [] for name in grid}       # name -> per-seed regret
    rf_regret, rf_sizes, rf_tau = [], [], []  # random-feasible

    for blk in per_seed_blocks:
        by_name = {c["name"]: c for c in blk["configs"]}
        tstr = {n: by_name[n]["tstr"] for n in grid}
        prec = {n: by_name[n]["precision"] for n in grid}
        o = max(tstr.values())
        oracle_tstr.append(round(o, 3))
        for n in grid:
            fixed[n].append(round(o - tstr[n], 2))
        # 隨機可行點：precision >= 該 seed 的 tau
        tau = blk["report"]["tau"]
        feasible = [n for n in grid if prec[n] >= tau]
        if not feasible:  # 理論上不會發生；保底退回整體網格
            feasible = list(grid)
        mean_feasible_tstr = sum(tstr[n] for n in feasible) / len(feasible)
        rf_regret.append(round(o - mean_feasible_tstr, 2))
        rf_sizes.append(len(feasible))
        rf_tau.append(round(tau, 4))

    def col_mean(v):
        return round(sum(v) / len(v), 2)

    fixed_out = {n: {"regret_per_seed": fixed[n], "mean_regret": col_mean(fixed[n])} for n in grid}
    return {
        "dataset": dataset,
        "grid": grid,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "oracle_tstr_per_seed": oracle_tstr,
        "fixed_w": fixed_out,
        "random_feasible": {
            "regret_per_seed": rf_regret,
            "mean_regret": col_mean(rf_regret),
            "feasible_size_per_seed": rf_sizes,
            "tau_per_seed": rf_tau,
            "note": "可行集 = {precision >= 該 seed report.tau}；期望 regret 為可行集 TSTR 解析平均。",
        },
    }


def sanity(result):
    """內部一致性檢查：所有 regret >= 0；隨機可行點 regret 不超過最差固定 w。"""
    grid = result["grid"]
    for n in grid:
        for r in result["fixed_w"][n]["regret_per_seed"]:
            assert r >= -1e-9, f"{result['dataset']} fixed {n} 出現負 regret {r}"
    worst_fixed = max(result["fixed_w"][n]["mean_regret"] for n in grid)
    for r in result["random_feasible"]["regret_per_seed"]:
        assert r >= -1e-9, f"{result['dataset']} random-feasible 負 regret {r}"
    rf_mean = result["random_feasible"]["mean_regret"]
    assert rf_mean <= worst_fixed + 1e-9, \
        f"{result['dataset']} random-feasible 平均 {rf_mean} > 最差固定 w {worst_fixed}"


def reconcile(new, output_path):
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
    p = argparse.ArgumentParser(description="固定 w 與隨機可行點 baseline（A7，純衍生）。")
    p.add_argument("--output", default="results/baseline_fixed_random.json")
    p.add_argument("--no-write", action="store_true", help="只算與對帳，不寫檔")
    args = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    sources = [
        ("MNIST", "results/selector_signal_multiseed.json"),
        ("CIFAR-10", "results/cifar10_cfg_confirmatory.json"),
        ("CIFAR-100", "results/cifar100_cfg_confirmatory.json"),
    ]
    results = {}
    for name, path in sources:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        r = analyse(data, name)
        r["_source"] = path
        sanity(r)
        results[name] = r

    out = {"datasets": results}
    matched = reconcile(out, args.output)

    # 報表
    print("\n" + "=" * 88)
    print("  固定 w／隨機可行點 baseline（regret@selected，pp，越低越好）")
    print("=" * 88)
    for name, _ in sources:
        r = results[name]
        print(f"\n  [{name}]  n_seeds={r['n_seeds']}  grid={r['grid']}")
        # 標出網格最低 w、w1.5、w2（若該資料集網格有）
        for n in r["grid"]:
            fw = r["fixed_w"][n]
            mark = "  <<" if n in ("w1", "g1", "w1.5", "w2", "g2") else ""
            print(f"    fixed {n:>5}: mean_regret={fw['mean_regret']:>6.2f}  per_seed={fw['regret_per_seed']}{mark}")
        rf = r["random_feasible"]
        print(f"    random-feasible: mean_regret={rf['mean_regret']:>6.2f}  per_seed={rf['regret_per_seed']}"
              f"  可行集大小={rf['feasible_size_per_seed']}")
    print("=" * 88)

    if args.no_write:
        return
    if not matched:
        raise SystemExit("與既有輸出檔對帳不符，拒絕覆寫。請查明差異來源。")

    out["metadata"] = {
        "analysis": "baseline_fixed_random",
        "status": "derived",
        "prereg_ref": "D5：固定 w 與隨機可行點 baseline",
        "harking_guard": "fixed-w 對網格每一個 w 都報整欄 per-seed regret，不預先挑點。",
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
