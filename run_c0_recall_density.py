"""C0：recall 與 density 能否打破 Pareto 支配（CaF-v2 第三訊號的選定依據）。

本檔重建 results/cifar10_recall_density_c0.json 的產生程序。該 JSON 原先無對應 driver、也無
metadata（違反 claude.md §5.2），而它是預先登記 D8「CaF-v2 改用 recall」的唯一依據，必須可重現。

背景（C8 Pareto 失明引理，docs/c8_pareto_blindness.md）：CIFAR-10 confirmatory 上存在一個組態
c* = w2.5，它在 (precision, coverage) 平面上「嚴格支配」所有 TSTR-oracle——precision 與 coverage
都比 oracle 高。CaF 選的是「precision >= tau 之中 coverage 最大者」，所以只要 c* 通過 tau，它就
一定被選中；只要 c* 不通過，coverage 更低的 oracle 也不會通過。因此任何 tau 都選不到 oracle：
這不是校準問題，是選擇器的結構性盲點。

要救 CaF，就得換一個「不會被 c* 支配」的訊號——找一個訊號 s，使 s(c*) 低於所有 oracle：

    nondominating(s) = 對每個 oracle o，s(c*) < s(o)

recall/density 兩個候選各檢查一次。recall 過關、density 不過（c* 的 density 反而偏高），故 D8
採 recall。

資料來源：confirmatory 的輸出 JSON 當初沒有存 recall/density（只存 precision/coverage/tstr…），
所以本檔不是純衍生分析——它從 P1 落盤的 DINOv2 特徵（results/p1_assets/）重算 PRDC，一次拿到
四個量。P1 已證 30/30 cell 逐位可重現，故此路徑與 confirmatory 用的是同一批特徵。

內建對帳：重算出的 precision/coverage 必須與 confirmatory 的 aggregate 均值完全相同。這是本檔
最重要的護欄——它同時證明 nearest_k=5（confirmatory metadata 未存 k，即 P0 source-tracing 事件；
k=5 是 P0 以逐位重現反證確立的）與特徵來源都對得上。對不上就不是「recall 算錯」，是取錯了資料。

Usage:
    uv run python run_c0_recall_density.py
    uv run python run_c0_recall_density.py --no-write   # 只對帳，不覆寫
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch

from metrics_prdc import compute_prdc_per_class

SIGNALS = ("recall", "density")
PARETO_AXES = ("precision", "coverage")   # CaF 所看的兩軸，支配關係就定義在這個平面上
PRDC_KEYS = ("recall", "density", "precision", "coverage")


def load_cell(assets_dir, seed, w, device):
    """載入單一 (seed, w) cell 的 P1 落盤 DINOv2 特徵與標籤。"""
    cell = os.path.join(assets_dir, f"seed{seed}_w{w:g}")
    feat = torch.load(os.path.join(cell, "dino_feat.pt"), map_location=device, weights_only=True)
    labels = torch.load(os.path.join(cell, "labels.pt"), map_location=device, weights_only=True)
    return feat, labels


def prdc_over_grid(assets_dir, seeds, grid, nearest_k, num_classes, device):
    """逐 (seed, w) 重算 PRDC，回傳 {name: {key: [每個 seed 的值]}}。跨 seed 歸約留給呼叫端。"""
    real_dino = torch.load(os.path.join(assets_dir, "real_dino_feat.pt"),
                           map_location=device, weights_only=True)
    real_labels = torch.load(os.path.join(assets_dir, "real_labels.pt"),
                             map_location=device, weights_only=True)
    print(f"real ref: {real_dino.size(0)} 張，nearest_k={nearest_k}", flush=True)

    per_seed = {}
    for w in grid:
        name = f"w{w:g}"
        acc = {k: [] for k in PRDC_KEYS}
        for seed in seeds:
            gen_dino, gen_labels = load_cell(assets_dir, seed, w, device)
            m, _ = compute_prdc_per_class(real_dino, real_labels, gen_dino, gen_labels,
                                          nearest_k=nearest_k, num_classes=num_classes)
            for k in acc:
                acc[k].append(m[k])
        per_seed[name] = acc
        print(f"  {name:>5} prec={statistics.mean(acc['precision']):.3f} "
              f"cov={statistics.mean(acc['coverage']):.3f} "
              f"recall={statistics.mean(acc['recall']):.3f} "
              f"density={statistics.mean(acc['density']):.3f}", flush=True)
    return per_seed


def crosscheck_against_confirmatory(per_seed, confirmatory):
    """逐 seed 的 precision/coverage 必須能對回 confirmatory 的 aggregate 均值。

    這裡刻意用 confirmatory 自己的歸約（run_cifar_selector.summarize 的 sum/len），而不是本檔
    payload 用的 statistics.mean。兩者在 40 個 scalar 中有 9 個差 1 ULP（約 1e-16）：statistics.mean
    走精確有理數、正確捨入，sum/len 是逐次浮點累加。差異純屬歸約方式，不是資料不同——用對方的
    歸約去比對方的數字，才驗得到「特徵來源與 nearest_k 都對得上」這件事。
    """
    agg = {pc["name"]: pc for pc in confirmatory["aggregate"]["per_config"]}
    bad = []
    for name, acc in per_seed.items():
        for ax in PARETO_AXES:
            mine = sum(acc[ax]) / len(acc[ax])          # 與 summarize() 同一種歸約
            if mine != agg[name][ax]["mean"]:
                bad.append(f"{name}.{ax}: 重算 {mine!r} != confirmatory {agg[name][ax]['mean']!r}")
    ok = not bad
    print(f"[crosscheck] 重算 precision/coverage vs confirmatory（同 sum/len 歸約）: "
          f"{'ALL_BITEXACT' if ok else 'MISMATCH'}")
    for line in bad:
        print(f"  {line}")
    return ok


def find_dominator(per_config, oracles):
    """找出在 (precision, coverage) 平面上嚴格支配所有 oracle 的組態 c*（若有）。

    嚴格支配 = 兩軸都嚴格大於。c* 本身不能是 oracle。
    """
    for name, m in per_config.items():
        if name in oracles:
            continue
        if all(all(m[ax] > per_config[o][ax] for ax in PARETO_AXES) for o in oracles):
            return name
    return None


def signal_breaks_dominance(per_config, cstar, oracles, signal):
    """訊號 s 是否對 c* 非支配：s(c*) 必須低於每一個 oracle，CaF-v2 才可能選到 oracle。"""
    return all(per_config[cstar][signal] < per_config[o][signal] for o in oracles)


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
    p = argparse.ArgumentParser(description="C0 recall/density vs Pareto dominance (CaF-v2 signal).")
    p.add_argument("--confirmatory", default="results/cifar10_cfg_confirmatory.json")
    p.add_argument("--assets", default="results/p1_assets", help="P1 落盤的 DINOv2 特徵")
    p.add_argument("--output", default="results/cifar10_recall_density_c0.json")
    # k=5 未存於 confirmatory metadata（P0 source-tracing 事件），由 P0 逐位重現反證確立。
    p.add_argument("--nearest-k", type=int, default=5)
    p.add_argument("--no-write", action="store_true", help="只對帳，不寫檔")
    args = p.parse_args()
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.confirmatory, encoding="utf-8") as f:
        confirmatory = json.load(f)
    meta = confirmatory["metadata"]
    seeds, grid = meta["seeds"], meta["guidance_grid"]
    num_classes = meta.get("num_classes", 10)
    print(f"Using device: {device}  seeds={seeds} grid={grid}", flush=True)

    per_seed = prdc_over_grid(args.assets, seeds, grid, args.nearest_k, num_classes, device)
    bitexact = crosscheck_against_confirmatory(per_seed, confirmatory)
    if not bitexact:
        raise SystemExit("重算的 precision/coverage 對不上 confirmatory，資料來源或 k 有誤，中止。")

    # 跨 seed 歸約用 statistics.mean（精確有理數、正確捨入），與凍結 C0 逐位相符。
    per_config = {name: {k: statistics.mean(acc[k]) for k in PRDC_KEYS}
                  for name, acc in per_seed.items()}

    # 逐 seed 的 TSTR-oracle。支配關係只看集合，順序純屬呈現；去重後依 guidance 遞減排列。
    oracles = sorted(set(confirmatory["aggregate"]["oracle_best_per_seed"]),
                     key=lambda n: float(n[1:]), reverse=True)
    cstar = find_dominator(per_config, set(oracles))
    if cstar is None:
        raise SystemExit("找不到嚴格支配所有 oracle 的組態；C8 引理的前提在此資料上不成立。")
    verdict = {s: signal_breaks_dominance(per_config, cstar, oracles, s) for s in SIGNALS}

    out = {"per_config": per_config, "cstar": cstar, "oracles": oracles,
           # 沿用凍結檔欄名：recall 是否為非支配訊號（D8 的判準）
           "nondominating_signal": verdict["recall"]}
    matched = reconcile(out, args.output)

    print("\n" + "=" * 82)
    print(f"  C0：支配者 c* = {cstar}，逐 seed oracle = {oracles}")
    print("=" * 82)
    print(f"  {'組態':>6} {'precision':>10} {'coverage':>10} {'recall':>10} {'density':>10}")
    for name in [cstar] + sorted(set(oracles)):
        m = per_config[name]
        tag = "c*" if name == cstar else "oracle"
        print(f"  {name:>6} {m['precision']:>10.3f} {m['coverage']:>10.3f} "
              f"{m['recall']:>10.3f} {m['density']:>10.3f}   {tag}")
    print("  " + "-" * 78)
    print("  c* 在 (precision, coverage) 平面嚴格支配所有 oracle -> CaF 任何 tau 都選不到 oracle")
    for s in SIGNALS:
        state = "可打破支配（c* 低於所有 oracle）" if verdict[s] else "不能打破支配"
        print(f"  第三訊號候選 {s:>8} : {state}")
    chosen = [s for s in SIGNALS if verdict[s]]
    print(f"  D8 採用 : {chosen[0] if chosen else '無可用訊號'}")
    print("=" * 82)

    if args.no_write:
        return
    if not matched:
        raise SystemExit("對帳不符，拒絕覆寫凍結結果檔。請先查明差異來源。")

    out["metadata"] = {
        "analysis": "c0_recall_density_pareto",
        "feature_source": args.assets,
        "confirmatory": args.confirmatory,
        "seeds": seeds, "guidance_grid": grid, "num_classes": num_classes,
        "nearest_k": args.nearest_k,
        "effective_nearest_k": min(args.nearest_k, meta["real_per_class"] - 1),
        "cross_seed_reduction": "statistics.mean",
        "pareto_axes": list(PARETO_AXES),
        "signal_candidates": list(SIGNALS),
        "signal_verdict": verdict,
        "rule": "非支配訊號 s：對每個 oracle o 皆 s(c*) < s(o)，CaF-v2 才可能選到 oracle",
        "precision_coverage_bitexact_vs_confirmatory": bitexact,
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
