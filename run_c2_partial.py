"""C2 全網格偏相關裁決（confirmatory），依 R-2026-07-05-13。

讀 confirmatory 多 seed 輸出（run_cifar_cfg_multiseed 的 JSON），以 config 為觀測單位（grid 10 點），
每 config 取 fresh seeds 的均值（n=10），計兩個偏 Spearman 相關並以 permutation test 判定：

  H-C2a：partial ρ(TSTR, coverage | precision, 超額 label-noise) > 0（單尾），α=0.05。
  H-C2b：partial ρ(TSTR, precision | coverage, 超額 label-noise) 不顯著或 ≤ 0（雙尾）。
  通過準則：C2a 顯著為正 且 C2b 不顯著。

偏 Spearman＝先對各變數取秩，殘差化（對控制變數做最小二乘後取殘差），再算殘差的 Pearson 相關。
信賴區間以 bootstrap over seeds 給出（seeds 少時解析度有限，據實報告）。n=10 功效有限，「C2b 不顯著」
屬 absence of evidence，不得當「precision 不驅動」的強證據——一律連同效果量與 CI 報告。

Usage:
    uv run python run_c2_partial.py --input results/cifar10_cfg_confirmatory.json
"""

import argparse
import json

import numpy as np
from scipy.stats import rankdata


def _residualize(y_rank, Z_rank):
    """對控制變數（已取秩）做含截距的最小二乘，回傳殘差。"""
    n = len(y_rank)
    design = np.column_stack([np.ones(n)] + [z for z in Z_rank.T])
    beta, *_ = np.linalg.lstsq(design, y_rank, rcond=None)
    return y_rank - design @ beta


def partial_spearman(x, y, Z):
    """partial Spearman ρ(x, y | Z)：秩化 → 殘差化 → 殘差 Pearson 相關。"""
    rx, ry = rankdata(x), rankdata(y)
    rZ = np.column_stack([rankdata(z) for z in np.asarray(Z).T])
    ex, ey = _residualize(rx, rZ), _residualize(ry, rZ)
    if ex.std() < 1e-12 or ey.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(ex, ey)[0, 1])


def perm_pvalue(x, y, Z, observed, sided, B=10000, seed=0):
    """對殘差做排列的 permutation p 值。sided='greater' 單尾（C2a）或 'two'（C2b）。"""
    rng = np.random.default_rng(seed)
    rx, ry = rankdata(x), rankdata(y)
    rZ = np.column_stack([rankdata(z) for z in np.asarray(Z).T])
    ex, ey = _residualize(rx, rZ), _residualize(ry, rZ)
    if ex.std() < 1e-12 or ey.std() < 1e-12:
        return 1.0
    exn, eyn = ex / ex.std(), ey / ey.std()
    count = 0
    for _ in range(B):
        p = rng.permutation(len(ey))
        stat = float(np.mean(exn * eyn[p]))  # 排列殘差的相關
        if sided == "greater":
            count += stat >= observed
        else:
            count += abs(stat) >= abs(observed)
    return (1 + count) / (B + 1)


def per_config_means(data, key):
    """取 aggregate.per_config 各 config 的均值向量（n=10），依 guidance 排序。"""
    pcs = data["aggregate"]["per_config"]
    grid = data["metadata"]["guidance_grid"]
    order = sorted(range(len(pcs)), key=lambda i: grid[i])
    return np.array([pcs[i][key]["mean"] for i in order])


def bootstrap_ci(data, target, controls, B=2000, seed=1):
    """bootstrap over seeds 的偏相關 CI。每次重抽 seeds、逐 config 重算均值、重算偏相關。"""
    per_seed = data["per_seed"]
    grid = data["metadata"]["guidance_grid"]
    names = [f"w{w:g}" for w in sorted(grid)]
    seeds_idx = list(range(len(per_seed)))
    rng = np.random.default_rng(seed)

    def col(seed_subset, key):
        vals = []
        for nm in names:
            xs = []
            for s in seed_subset:
                row = next(c for c in per_seed[s]["configs"] if c["name"] == nm)
                xs.append(row[key])
            vals.append(np.mean(xs))
        return np.array(vals)

    stats = []
    for _ in range(B):
        subset = rng.choice(seeds_idx, size=len(seeds_idx), replace=True).tolist()
        y = col(subset, "tstr")
        x = col(subset, target)
        Z = np.column_stack([col(subset, c) for c in controls])
        stats.append(partial_spearman(x, y, Z))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    p = argparse.ArgumentParser(description="C2 全網格偏相關裁決（依 2026-07-05-13）。")
    p.add_argument("--input", default="results/cifar10_cfg_confirmatory.json")
    p.add_argument("--output", default="results/cifar10_c2_partial.json")
    p.add_argument("--perm", type=int, default=10000)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    tstr = per_config_means(data, "tstr")
    cov = per_config_means(data, "coverage")
    prec = per_config_means(data, "precision")
    excess = per_config_means(data, "label_noise_excess_mean")
    n = len(tstr)

    # C2a：coverage 驅動效用（控制 precision、超額 label-noise）
    rho_a = partial_spearman(cov, tstr, np.column_stack([prec, excess]))
    p_a = perm_pvalue(cov, tstr, np.column_stack([prec, excess]), rho_a, "greater", B=args.perm)
    ci_a = bootstrap_ci(data, "coverage", ["precision", "label_noise_excess_mean"], B=args.boot)

    # C2b：precision 不驅動效用（控制 coverage、超額 label-noise）
    rho_b = partial_spearman(prec, tstr, np.column_stack([cov, excess]))
    p_b = perm_pvalue(prec, tstr, np.column_stack([cov, excess]), rho_b, "two", B=args.perm)
    ci_b = bootstrap_ci(data, "precision", ["coverage", "label_noise_excess_mean"], B=args.boot)

    c2a_pass = (rho_a > 0) and (p_a < args.alpha)
    c2b_pass = not (p_b < args.alpha and rho_b > 0)  # 不顯著、或顯著但非正
    verdict = c2a_pass and c2b_pass

    print("\n" + "=" * 72)
    print(f"  C2 全網格偏相關裁決（n={n} configs，觀測單位=config，seeds 均值）")
    print("=" * 72)
    print(f"  H-C2a  partial rho(TSTR, coverage | precision, excess_ln) = {rho_a:+.3f}")
    print(f"         permutation p(單尾 >0) = {p_a:.4f}   95% CI [{ci_a[0]:+.3f}, {ci_a[1]:+.3f}]"
          f"   {'顯著正' if c2a_pass else '未達'}")
    print(f"  H-C2b  partial rho(TSTR, precision | coverage, excess_ln) = {rho_b:+.3f}")
    print(f"         permutation p(雙尾) = {p_b:.4f}   95% CI [{ci_b[0]:+.3f}, {ci_b[1]:+.3f}]"
          f"   {'不顯著（符合）' if c2b_pass else '顯著正（不符）'}")
    print("  " + "-" * 68)
    print(f"  C2 裁決：{'通過（C2a 顯著正 且 C2b 不顯著）' if verdict else '未通過'}")
    print(f"  註：n={n} 功效有限；C2b「不顯著」屬 absence of evidence，非「precision 不驅動」的強證據。")
    print("=" * 72)

    out = {"input": args.input, "n_configs": n, "alpha": args.alpha,
           "c2a": {"rho": rho_a, "perm_p_greater": p_a, "ci95": ci_a, "pass": bool(c2a_pass)},
           "c2b": {"rho": rho_b, "perm_p_two_sided": p_b, "ci95": ci_b, "pass": bool(c2b_pass)},
           "verdict_pass": bool(verdict),
           "note": "偏相關依 2026-07-05-13；n=10 功效有限，C2b 不顯著為 absence of evidence。"}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
