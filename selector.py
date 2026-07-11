"""CaF——Coverage-at-Fidelity：免訓練的擴散取樣組態選擇器。

論文角色（C3a）：給定數個候選 sampler 組態（steps x eta x guidance），在不為
每個組態各訓練一個分類器的前提下，挑出能帶來最佳 downstream 訓練 utility 的那一
個。CaF 選擇

    argmax_config  coverage(config)   subject to   precision(config) >= tau

其中 precision/coverage 是由一組小型真實 probe set 計算的流形（manifold）度量
（metrics_prdc.py）。coverage 是驅動 utility 的類別內多樣性；precision 下限 tau
會拒絕那些樣本已偏離真實流形的組態。

內建的護欄（見 R-2026-07-03-07_plan_research-revision-brief.md 第 3 節）：
  - tau 不可在 TSTR 上調整（那會破壞「training-free」的宣稱）。
    `auto_tau` 僅由真對真的參考 precision 得出 tau。
  - 我們回報 tau-robustness（在一次 tau 掃描中選出哪個組態）以及所選組態對 tau
    的敏感度。
  - 評估採用 `regret@selected` 與 top-k 命中率，而非全域 Spearman
    （少數不良組態會拉高 rho，但最頂端的選擇仍可能是錯的）。

一個「config」是至少含有 name、precision、coverage 的 dict。選用的 utility key
（預設為 "tstr"）是 oracle 的 downstream 準確率，僅用於事後為選擇器評分。
"""


def select_caf(configs, tau, signal_key="coverage"):
    """回傳 (selected_config, passed_floor)。

    在 precision >= tau 的組態中，回傳 signal_key 最大的那個（CaF 用 "coverage"；
    CaF-v2 用 "recall"，見 D8 R-2026-07-09-13）。若沒有任何組態通過下限，則
    退回整體 signal_key 最大的組態並加以標記。
    """
    eligible = [c for c in configs if c["precision"] >= tau]
    passed = bool(eligible)
    pool = eligible if eligible else configs
    selected = max(pool, key=lambda c: c[signal_key])
    return selected, passed


def auto_tau(real_ref_precision, fraction=0.9):
    """由真對真的參考 precision 得出不依賴 TSTR 的 tau。

    `real_ref_precision` = 一份保留的真實切分對照真實流形量得的 precision（對完美
    生成器而言可達到的上限）。設 tau = fraction * 該上限，即要求各組態達到真實自我
    一致性的某個比例，且完全不使用合成 utility 的資訊。
    """
    return fraction * real_ref_precision


def tau_robustness(configs, tau_grid, signal_key="coverage"):
    """對網格中每個 tau，記錄所選組態的名稱。

    回傳 dict：{tau: name}，另加一份穩定度摘要：眾數的選擇，以及 tau 網格中與其
    一致的比例。
    """
    picks = {}
    for tau in tau_grid:
        sel, _ = select_caf(configs, tau, signal_key=signal_key)
        picks[float(tau)] = sel["name"]
    names = list(picks.values())
    modal = max(set(names), key=names.count)
    stability = names.count(modal) / len(names)
    return {"picks": picks, "modal": modal, "stability": stability}


def regret_at_selected(configs, selected, utility_key="tstr"):
    """網格上的最大 utility 減去所選組態的 utility（>= 0）。"""
    utils = [c[utility_key] for c in configs if utility_key in c]
    if not utils or utility_key not in selected:
        return None
    return max(utils) - selected[utility_key]


def rank_of_selected(configs, selected, utility_key="tstr"):
    """所選組態依 utility 的排名（由 1 起算，1 = oracle 最佳）。"""
    ranked = sorted((c for c in configs if utility_key in c),
                    key=lambda c: c[utility_key], reverse=True)
    for i, c in enumerate(ranked, start=1):
        if c["name"] == selected["name"]:
            return i, len(ranked)
    return None, len(ranked)


def select_and_report(configs, real_ref_precision=None, tau=None,
                      tau_fraction=0.9, utility_key="tstr", topk=3,
                      tau_grid=None, signal_key="coverage"):
    """執行 CaF 並對照 oracle utility 為其評分。

    提供明確的 `tau`，或提供 `real_ref_precision`（走 auto_tau）二擇一。`signal_key`
    決定選擇訊號：CaF 用 "coverage"、CaF-v2 用 "recall"（D8）。回傳一份 report dict，
    含所選組態、regret、rank、top-k 命中與 tau robustness。
    """
    if tau is None:
        if real_ref_precision is None:
            raise ValueError("Provide tau or real_ref_precision.")
        tau = auto_tau(real_ref_precision, tau_fraction)

    selected, passed = select_caf(configs, tau, signal_key=signal_key)
    regret = regret_at_selected(configs, selected, utility_key)
    rank, n = rank_of_selected(configs, selected, utility_key)

    if tau_grid is None:
        # 在觀測到的 precision 範圍內掃描 tau，以評估 robustness
        precs = sorted(c["precision"] for c in configs)
        lo, hi = precs[0], precs[-1]
        tau_grid = [lo + (hi - lo) * i / 10 for i in range(11)]
    robustness = tau_robustness(configs, tau_grid, signal_key=signal_key)

    oracle = None
    if any(utility_key in c for c in configs):
        oracle = max((c for c in configs if utility_key in c),
                     key=lambda c: c[utility_key])["name"]

    return {
        "tau": tau,
        "tau_passed_floor": passed,
        "signal_key": signal_key,
        "selected": selected["name"],
        "oracle_best": oracle,
        "regret_at_selected": regret,
        "rank": rank,
        "n_configs": n,
        "topk_hit": (rank is not None and rank <= topk),
        "topk": topk,
        "tau_robustness": robustness,
    }


def _self_check():
    # 合成的組態：coverage 與 utility 同向，但 coverage 最大的組態的 precision
    # 略微偏低；CaF 的下限應避開偏離流形的高 coverage 陷阱，落在真正 utility
    # 最佳的組態上。
    configs = [
        {"name": "g1", "precision": 0.72, "coverage": 0.95, "tstr": 97.2},
        {"name": "g2", "precision": 0.80, "coverage": 0.80, "tstr": 95.9},
        {"name": "g3", "precision": 0.85, "coverage": 0.72, "tstr": 95.3},
        {"name": "g5", "precision": 0.90, "coverage": 0.65, "tstr": 91.9},
        {"name": "noise", "precision": 0.30, "coverage": 0.99, "tstr": 60.0},
    ]
    # 真對真的參考 precision ~0.75 -> tau = 0.9*0.75 = 0.675
    rep = select_and_report(configs, real_ref_precision=0.75, tau_fraction=0.9)
    print("CaF self-check:")
    for k in ("tau", "selected", "oracle_best", "regret_at_selected", "rank",
              "n_configs", "topk_hit"):
        print(f"  {k:20s}: {rep[k]}")
    print(f"  tau_robustness      : modal={rep['tau_robustness']['modal']} "
          f"stability={rep['tau_robustness']['stability']:.2f}")
    # g1 既是 utility 最佳，也通過 0.675 的下限；noise 會被過濾掉。
    assert rep["selected"] == "g1", f"expected g1, got {rep['selected']}"
    assert rep["regret_at_selected"] == 0.0
    print("  OK")


if __name__ == "__main__":
    _self_check()
