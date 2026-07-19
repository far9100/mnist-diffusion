"""D3：CIFAR-100 機制複製的三觀察量（分支三診斷論文的獨立證據 driver）。

背景：預註冊 D3（`docs/prereg_cifar100.md`）以「三觀察量、三中二成立」判定 near-boundary 機制是否
在 CIFAR-100 複製。confirmatory 完成後（`2026-07-16-01`），這三個判讀原先只以內嵌 python 從
confirmatory JSON 讀出，沒有獨立 driver、也沒有 §5.2 metadata，使該裁決無法從工作樹逐值重現。
本檔補上這支 driver，把三觀察量落成可重現的純衍生分析。

本檔是純衍生分析：所有數字都從凍結的 confirmatory JSON 的 aggregate 均值讀出，不重跑 GPU、不重算
任何特徵（與 run_c6_fidmin_duel.py 同一類；不同於 run_c0_recall_density.py——後者會重算 PRDC 用 GPU）。

判讀口徑（凍程序不凍數字）：先以 coverage 均值曲線的峰位（argmax）把 grid 切成「升段」與「高段」，
再在段上判三觀察量。峰位由資料決定、非硬編，故可跨 seed 或改資料集沿用。
  - 觀察量(i) 低中段 near-boundary 單調降：在 coverage 升段 [grid 起點, coverage 峰]，near-boundary
    逐點嚴格下降。對應「guidance 上升、樣本往類原型集中，near-boundary 樣本被抽走」。
  - 觀察量(ii) 高段 coverage 與 TSTR 同崩：在 coverage 高段 [coverage 峰, grid 終點]，coverage 與
    TSTR 皆逐點嚴格下降。對應「過度 guidance 同時毀掉多樣性與下游效用」。
  - 觀察量(iii) 高段 near-boundary 脫鉤：在高段，near-boundary 出現內部谷底後回升（末點高於谷底），
    而同段 coverage 持續下降——near-boundary 不再跟隨 coverage，兩者脫鉤。
  - 判準：三中二成立即判機制複製。

Usage:
    uv run python run_d3_observables.py
    uv run python run_d3_observables.py --no-write   # 只算與對帳，不寫檔
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py

import argparse
import json
import os
import sys
import time

import torch


def _seg_strictly_decreasing(values):
    """逐點嚴格下降（a[k+1] < a[k]）。均值取自 8 seed，無平手之虞。"""
    return all(values[k + 1] < values[k] for k in range(len(values) - 1))


def build(confirmatory):
    """從 confirmatory 的 aggregate 均值計三觀察量，回傳可逐值對帳的 payload。"""
    grid = confirmatory["metadata"]["guidance_grid"]
    names = [f"w{w:g}" for w in grid]
    per_config = confirmatory["aggregate"]["per_config"]
    by_name = {pc["name"]: pc for pc in per_config}

    # 三個量的跨 seed 均值曲線（依 grid 順序）。
    coverage = [by_name[n]["coverage"]["mean"] for n in names]
    near_bnd = [by_name[n]["near_boundary_frac"]["mean"] for n in names]
    tstr = [by_name[n]["tstr"]["mean"] for n in names]

    # coverage 峰位（argmax）把 grid 切成升段與高段。峰點同屬兩段的邊界。
    peak = max(range(len(grid)), key=lambda i: coverage[i])

    # 觀察量(i)：升段 [0, peak] 的 near-boundary 嚴格下降。
    rise_idx = list(range(0, peak + 1))
    obs_i = _seg_strictly_decreasing([near_bnd[i] for i in rise_idx])

    # 觀察量(ii)：高段 [peak, 末] 的 coverage 與 TSTR 皆嚴格下降。
    high_idx = list(range(peak, len(grid)))
    cov_high = [coverage[i] for i in high_idx]
    tstr_high = [tstr[i] for i in high_idx]
    cov_down = _seg_strictly_decreasing(cov_high)
    tstr_down = _seg_strictly_decreasing(tstr_high)
    obs_ii = cov_down and tstr_down

    # 觀察量(iii)：高段 near-boundary 有內部谷底後回升，且同段 coverage 持續下降（脫鉤）。
    nb_high = [near_bnd[i] for i in high_idx]
    vmin = min(nb_high)
    vmin_local = nb_high.index(vmin)                       # 谷底在高段內的位置
    interior_valley = 0 < vmin_local < len(nb_high) - 1    # 谷底須落在內部（非段首、非段尾）
    rebound = nb_high[-1] > vmin                           # 末點高於谷底＝回升
    obs_iii = interior_valley and rebound and cov_down

    holding = sum([obs_i, obs_ii, obs_iii])
    r4 = lambda xs: [round(x, 4) for x in xs]

    return {
        "grid": grid,
        "coverage_peak": names[peak],
        "observable_i_near_boundary_decline": {
            "segment": [names[i] for i in rise_idx],
            "near_boundary": r4([near_bnd[i] for i in rise_idx]),
            "holds": obs_i,
        },
        "observable_ii_coverage_tstr_collapse": {
            "segment": [names[i] for i in high_idx],
            "coverage": r4(cov_high),
            "tstr": [round(x, 2) for x in tstr_high],
            "coverage_strictly_down": cov_down,
            "tstr_strictly_down": tstr_down,
            "holds": obs_ii,
        },
        "observable_iii_near_boundary_decoupling": {
            "segment": [names[i] for i in high_idx],
            "near_boundary": r4(nb_high),
            "valley": names[high_idx[vmin_local]],
            "rebound_end": names[high_idx[-1]],
            "interior_valley": interior_valley,
            "rebound": rebound,
            "coverage_strictly_down": cov_down,
            "holds": obs_iii,
        },
        "observables_holding": holding,
        # 判準：三中二（>= 2）成立即機制複製。
        "mechanism_replicates": holding >= 2,
    }


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
    p = argparse.ArgumentParser(description="D3 三觀察量：CIFAR-100 機制複製（純衍生）。")
    p.add_argument("--confirmatory", default="results/cifar100_cfg_confirmatory.json")
    p.add_argument("--output", default="results/cifar100_d3_observables.json")
    p.add_argument("--no-write", action="store_true", help="只算與對帳，不寫檔")
    args = p.parse_args()
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(args.confirmatory, encoding="utf-8") as f:
        confirmatory = json.load(f)

    out = build(confirmatory)
    matched = reconcile(out, args.output)

    oi = out["observable_i_near_boundary_decline"]
    oii = out["observable_ii_coverage_tstr_collapse"]
    oiii = out["observable_iii_near_boundary_decoupling"]
    yn = lambda b: "成立" if b else "不成立"
    print("\n" + "=" * 78)
    print("  D3 三觀察量：CIFAR-100 near-boundary 機制是否複製（三中二）")
    print("=" * 78)
    print(f"  coverage 峰位 : {out['coverage_peak']}")
    print(f"  (i)   低中段 near-boundary 單調降 [{oi['segment'][0]}..{oi['segment'][-1]}] : {yn(oi['holds'])}")
    print(f"         near-boundary {oi['near_boundary']}")
    print(f"  (ii)  高段 coverage 與 TSTR 同崩 [{oii['segment'][0]}..{oii['segment'][-1]}] : {yn(oii['holds'])}")
    print(f"         coverage {oii['coverage']}")
    print(f"         tstr     {oii['tstr']}")
    print(f"  (iii) 高段 near-boundary 脫鉤（谷 {oiii['valley']} 後回升至 {oiii['rebound_end']}）: {yn(oiii['holds'])}")
    print(f"         near-boundary {oiii['near_boundary']}")
    print("  " + "-" * 74)
    print(f"  成立數 : {out['observables_holding']}/3  ->  機制{'複製' if out['mechanism_replicates'] else '未複製'}")
    print("=" * 78)

    if args.no_write:
        return
    if not matched:
        raise SystemExit("對帳不符，拒絕覆寫凍結結果檔。請先查明差異來源。")

    out["metadata"] = {
        "analysis": "d3_three_observables",
        "source": args.confirmatory,
        "source_metadata": confirmatory["metadata"],
        # 判讀規則寫進 metadata，使裁決可重現（§5.2）：規則凍結、數字隨資料。
        "rule": {
            "segment_split": "以 coverage 均值曲線 argmax（峰位）切升段/高段；峰點屬兩段邊界",
            "observable_i": "升段 [grid 起, 峰] near-boundary 逐點嚴格下降",
            "observable_ii": "高段 [峰, grid 終] coverage 與 TSTR 皆逐點嚴格下降",
            "observable_iii": "高段 near-boundary 有內部谷底且末點高於谷底（回升），同段 coverage 嚴格下降",
            "decision": "三中二（>=2）成立即判機制複製",
        },
        "reconciled_against_frozen": matched,
        "start_timestamp": start_timestamp,
        "argv": " ".join(sys.argv),
        "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version()},
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
