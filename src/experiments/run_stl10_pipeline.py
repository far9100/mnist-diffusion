"""P1-2：STL-10（96×96）規模門檻管線 dry-run 規劃器（骨架）。

把三判決推到 32×32 以上的最小尺度 STL-10（96×96、10 類）。此為 fix_tasks P1-2、成本最高、任務書列
「最後做」。完整規格見 `docs/prereg_stl10.md`（草稿，須定稿並 commit 後才可真跑）。

依 fix_tasks §3.3 與 §5.1 凍結四要件：backbone 訓練與 confirmatory 取樣屬需授權的高成本新跑。本檔
**只做 dry-run 規劃**——檢查資料/相依可用性、枚舉 cell、印各階段 ETA，不訓練、不取樣。四階段
（backbone→judge→base_gate→confirmatory）之真跑，待 prereg 定稿與作者授權後於後續 commit 實作。

Usage:
    uv run python src/experiments/run_stl10_pipeline.py     # dry-run 規劃＋資料檢查＋ETA
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py

import argparse
import json
import os
import sys
import time

# 凍結草稿值（見 docs/prereg_stl10.md；定稿前為暫定）。
RESOLUTION, NUM_CLASSES, PER_CLASS = 96, 10, 500
GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
SEEDS = [10, 11, 12, 13, 14, 15, 16, 17]     # 8 seed（D4 功效）

# 四階段與粗略 ETA（GPU 小時量級；定稿時以計時探針回填）。
STAGE_ETA = {
    "backbone": "96×96 CFG UNet 自訓（unlabeled 100k＋labeled），數日級 ~48–96h",
    "judge": "ResNet judge 於 labeled 5000 訓練，~1–2h",
    "base_gate": "50k 生成 + clean-fid（需自建 STL-10 參考 stats），~3–6h",
    "confirmatory": "sweep + reps×TSTR + coverage + 介入，數十 h",
}


def check_data(data_dir):
    """驗 STL-10 是否可經 torchvision 取得（不下載，只看 API 與本機快取）。"""
    try:
        import torchvision
        cached = os.path.isdir(os.path.join(data_dir, "stl10_binary"))
        return {"torchvision": torchvision.__version__, "stl10_cached": cached,
                "note": "已有本機快取" if cached else "未下載；真跑以 STL10(download=True) 取得"}
    except Exception as e:
        return {"torchvision": None, "stl10_cached": False, "note": f"torchvision 不可用：{e}"}


def main():
    p = argparse.ArgumentParser(description="STL-10 規模門檻管線 dry-run 規劃器（P1-2 骨架）。")
    p.add_argument("--grid", type=float, nargs="+", default=GRID)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--per-class", type=int, default=PER_CLASS)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--output", default="results/stl10_pipeline_dryrun.json")
    args = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    cells = len(args.seeds) * len(args.grid)
    data = check_data(args.data_dir)
    print("=" * 74)
    print("  P1-2 STL-10（96×96）管線規劃（dry-run，不訓練/不取樣）")
    print("=" * 74)
    print(f"  {RESOLUTION}×{RESOLUTION}｜類數 {NUM_CLASSES}｜grid {args.grid}")
    print(f"  seeds {args.seeds}｜per_class {args.per_class}｜confirmatory cells {cells}")
    print(f"  資料：torchvision={data['torchvision']} STL-10 快取={data['stl10_cached']}（{data['note']}）")
    for st, eta in STAGE_ETA.items():
        print(f"  [階段 {st:>12}] ETA {eta}")
    print("  prereg：docs/prereg_stl10.md（草稿，須定稿＋commit 後方可真跑）")
    print("=" * 74)

    report = {"metadata": {"analysis": "stl10_pipeline", "mode": "dry_run", "status": "skeleton",
                           "resolution": RESOLUTION, "num_classes": NUM_CLASSES, "grid": args.grid,
                           "seeds": args.seeds, "per_class": args.per_class,
                           "confirmatory_cells": cells, "prereg": "docs/prereg_stl10.md",
                           "start_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "argv": " ".join(sys.argv)},
              "data_check": data, "stage_eta": STAGE_ETA}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output}（dry-run 規劃，未真跑）")


if __name__ == "__main__":
    main()
