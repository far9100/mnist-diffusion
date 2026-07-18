"""C6：matched-budget FID-min 對決 CaF（判決三的來源）。

本檔重建 results/cifar10_c6_fidmin_duel.json 的產生程序；driver 經 --confirmatory/--output 一般化，
同一支也產出 results/cifar100_c6_fidmin_duel.json。該 JSON 原先無對應 driver、也無
metadata（違反 claude.md §5.2），使 README 引用的頭條數字「FID-min regret 0.91pp 勝 CaF
3.69pp」無法從工作樹重現。此處補回 driver 並對凍結 JSON 做逐值對帳。

問題設定：CaF 是免訓練選擇器，但它要贏，得先贏過一個更便宜的 baseline——直接挑 characterization
clean-fid 最小的組態（FID-min）。兩者都不看 TSTR，故預算相當（matched-budget）。評分用
regret@selected：oracle（該 seed 上 TSTR 最高的組態）的 TSTR 減去被選中組態的 TSTR，越小越好。

本檔是純衍生分析：所有數字都從凍結的 confirmatory JSON 讀出，不重跑 GPU，因此可逐位重現。

量的東西（逐 seed）：
  - oracle       = argmax_w TSTR
  - fidmin       = argmin_w char_clean_fid（Inception 空間的 clean-fid）
  - fidmin_regret = TSTR(oracle) - TSTR(fidmin)
  - caf_regret   = confirmatory 內 CaF 的 regret@selected（同一份資料，直接沿用）
  - incep_sep_step = FID-argmin 與 TSTR-argmax 在凍結 grid 上相隔幾格。C1 的分離口徑要求
                     > 1 格才算「FID 與效用分離」；0 或 1 格即判不分離。

Usage:
    uv run python run_c6_fidmin_duel.py
    uv run python run_c6_fidmin_duel.py --no-write   # 只對帳，不覆寫
"""

import argparse
import json
import os
import sys
import time

import torch


def grid_index(name, names):
    """組態名（如 "w1.5"）在凍結 grid 上的位置，供算分離格步。"""
    return names.index(name)


def duel_one_seed(seed_result, names):
    """單一 seed 的 FID-min vs CaF 對決。"""
    configs = seed_result["configs"]
    by_name = {c["name"]: c for c in configs}

    oracle = max(configs, key=lambda c: c["tstr"])
    fidmin = min(configs, key=lambda c: c["char_clean_fid"])

    fidmin_regret = oracle["tstr"] - fidmin["tstr"]
    # FID-argmin 與 TSTR-argmax 相隔的 grid 格數（C1 分離口徑：需 > 1 才算分離）
    sep_step = abs(grid_index(fidmin["name"], names) - grid_index(oracle["name"], names))

    return {
        "seed": seed_result["seed"],
        "oracle": oracle["name"],
        "oracle_tstr": round(oracle["tstr"], 2),
        "fidmin": fidmin["name"],
        "fidmin_fid": round(fidmin["char_clean_fid"], 2),
        "tstr_at_fidmin": round(by_name[fidmin["name"]]["tstr"], 2),
        "fidmin_regret": round(fidmin_regret, 2),
        "incep_sep_step": sep_step,
        "caf_regret": round(seed_result["report"]["regret_at_selected"], 2),
    }


def mean_curve_duel(per_config, names):
    """跨 seed 均值曲線上的對決（判決三的彙總列）。"""
    tstr = {pc["name"]: pc["tstr"]["mean"] for pc in per_config}
    fid = {pc["name"]: pc["char_clean_fid"]["mean"] for pc in per_config}
    tstr_argmax = max(names, key=lambda n: tstr[n])
    fid_argmin = min(names, key=lambda n: fid[n])
    return {
        "mean_tstr_argmax": float(tstr_argmax[1:]),
        "mean_fid_argmin": float(fid_argmin[1:]),
        "mean_sep_step": abs(grid_index(fid_argmin, names) - grid_index(tstr_argmax, names)),
    }


def build(confirmatory):
    names = [f"w{w:g}" for w in confirmatory["metadata"]["guidance_grid"]]
    per_seed = [duel_one_seed(sr, names) for sr in confirmatory["per_seed"]]

    fidmin_regrets = [r["fidmin_regret"] for r in per_seed]
    caf_regrets = [r["caf_regret"] for r in per_seed]
    sep_steps = [r["incep_sep_step"] for r in per_seed]

    out = {
        "per_seed": per_seed,
        "fidmin_regret_mean": round(sum(fidmin_regrets) / len(fidmin_regrets), 2),
        "fidmin_regret_per_seed": fidmin_regrets,
        "caf_regret_mean": round(sum(caf_regrets) / len(caf_regrets), 2),
        "caf_regret_per_seed": caf_regrets,
        "incep_sep_steps": sep_steps,
        # C1 分離口徑：> 1 格才算分離。此欄計有幾個 seed 達標（0 = 全數不分離）。
        "incep_separated_seeds_gt1": sum(1 for s in sep_steps if s > 1),
    }
    out.update(mean_curve_duel(confirmatory["aggregate"]["per_config"], names))
    return out


def reconcile(new, frozen_path):
    """對凍結 JSON 逐值對帳。凍結檔無 metadata，故只比數值 payload。"""
    if not os.path.exists(frozen_path):
        print(f"[reconcile] 無既有 {frozen_path}，跳過對帳（首次產生）")
        return True
    with open(frozen_path, encoding="utf-8") as f:
        old = json.load(f)
    old_payload = {k: v for k, v in old.items() if k != "metadata"}
    new_payload = {k: v for k, v in new.items() if k != "metadata"}
    ok = old_payload == new_payload
    print(f"[reconcile] vs {frozen_path}: {'ALL_MATCH' if ok else 'MISMATCH'}")
    if not ok:
        for k in sorted(set(old_payload) | set(new_payload)):
            if old_payload.get(k) != new_payload.get(k):
                print(f"  {k}:\n    凍結 {old_payload.get(k)}\n    重算 {new_payload.get(k)}")
    return ok


def main():
    p = argparse.ArgumentParser(description="C6 matched-budget FID-min vs CaF duel.")
    p.add_argument("--confirmatory", default="results/cifar10_cfg_confirmatory.json")
    p.add_argument("--output", default="results/cifar10_c6_fidmin_duel.json")
    p.add_argument("--no-write", action="store_true", help="只對帳，不寫檔")
    args = p.parse_args()
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(args.confirmatory, encoding="utf-8") as f:
        confirmatory = json.load(f)

    out = build(confirmatory)
    matched = reconcile(out, args.output)

    print("\n" + "=" * 78)
    print("  C6：matched-budget FID-min vs CaF（regret@selected，越低越好）")
    print("=" * 78)
    print(f"  {'seed':>5} {'oracle':>8} {'FID-min':>8} {'FID-min regret':>15} "
          f"{'CaF regret':>11} {'sep_step':>9}")
    for r in out["per_seed"]:
        print(f"  {r['seed']:>5} {r['oracle']:>8} {r['fidmin']:>8} "
              f"{r['fidmin_regret']:>15.2f} {r['caf_regret']:>11.2f} {r['incep_sep_step']:>9}")
    print("  " + "-" * 74)
    print(f"  FID-min regret 平均 : {out['fidmin_regret_mean']:.2f} pp {out['fidmin_regret_per_seed']}")
    print(f"  CaF     regret 平均 : {out['caf_regret_mean']:.2f} pp {out['caf_regret_per_seed']}")
    winner = "FID-min" if out["fidmin_regret_mean"] < out["caf_regret_mean"] else "CaF"
    print(f"  勝方（regret 較低）  : {winner}")
    print(f"  分離格步 > 1 的 seed 數 : {out['incep_separated_seeds_gt1']}/{len(out['per_seed'])}"
          f"（0 表示兩者在 grid 上不分離）")
    print("=" * 78)

    if args.no_write:
        return
    if not matched:
        raise SystemExit("對帳不符，拒絕覆寫凍結結果檔。請先查明差異來源。")

    out["metadata"] = {
        "analysis": "c6_fidmin_duel",
        "source": args.confirmatory,
        "source_metadata": confirmatory["metadata"],
        "separation_step_rule": "C1 口徑：FID-argmin 與 TSTR-argmax 相隔 > 1 格才算分離",
        "reconciled_against_frozen": matched,
        "start_timestamp": start_timestamp,
        "argv": " ".join(sys.argv),
        "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version()},
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.output}（數值 payload 與凍結檔逐值相同，僅補上 metadata）")


if __name__ == "__main__":
    main()
