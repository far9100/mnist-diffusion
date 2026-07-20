# 用途：由 results/*.json 產生碩論 docs/thesis_draft.md 所需之發表級圖（P0，純畫既有數字、不跑新實驗）。
# 圖內文字用英文以避免中文字型缺字；markdown 圖說用中文。輸出至 docs/figures/。
#
# 產出：
#   fig_mnist.png            圖 5.1 MNIST：coverage 與 TSTR 同向、precision 峰錯位
#   fig_selector_reversal.png 圖 5.2 三尺度選擇器判決反轉（regret）
#   fig_pareto.png           圖 5.3 CIFAR-10 (precision, coverage) 平面上 w2.5 支配 oracle（§5.4.2）
#   fig_variance.png         圖 5.4 變異分解 σ_cls vs σ_gen 與上升肢 Δ（§5.4.3）
#   fig_two_stage.png        圖 5.5 CIFAR-10 與 CIFAR-100 雙段機制（TSTR/coverage/near-boundary）
#   fig_h3_duel.png          圖 5.6 H3 三臂：Chamfer 勝 vanilla 但 coverage 反而低

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py
import json
import os

import matplotlib

matplotlib.use("Agg")  # 無視窗後端，供批次產圖
import matplotlib.pyplot as plt

ROOT = _pathfix.ROOT  # 專案根（含 results/ 與 docs/figures/），由墊片提供
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
})

# 一致配色
C_TSTR = "#1f77b4"
C_COV = "#2ca02c"
C_PREC = "#ff7f0e"
C_NB = "#d62728"
C_REC = "#9467bd"


def load(name):
    with open(os.path.join(RES, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


def sc(v):
    """scalar 或 {'mean':...} 一律取數值。"""
    if isinstance(v, dict):
        return v.get("mean", v.get("value"))
    return v


def wnum(name):
    """'w2.5' -> 2.5，'g3' -> 3。"""
    return float(name.lstrip("wg"))


def fig_mnist():
    d = load("selector_signal_multiseed.json")
    pc = d["aggregate"]["per_config"]
    g = [wnum(c["name"]) for c in pc]
    tstr = [sc(c["tstr"]) for c in pc]
    cov = [sc(c["coverage"]) for c in pc]
    prec = [sc(c["precision"]) for c in pc]

    fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
    ax2 = ax1.twinx()
    l1, = ax1.plot(g, tstr, "o-", color=C_TSTR, label="TSTR (%)")
    l2, = ax2.plot(g, cov, "s--", color=C_COV, label="coverage")
    l3, = ax2.plot(g, prec, "^:", color=C_PREC, label="precision")

    # 標 TSTR 峰 (g1) 與 precision 峰 (g3) 的錯位
    i_tstr = max(range(len(tstr)), key=lambda i: tstr[i])
    i_prec = max(range(len(prec)), key=lambda i: prec[i])
    ax1.axvline(g[i_tstr], color=C_TSTR, alpha=0.25, lw=1)
    ax2.axvline(g[i_prec], color=C_PREC, alpha=0.25, lw=1)
    ax1.annotate("TSTR peak (g1)", (g[i_tstr], tstr[i_tstr]),
                 textcoords="offset points", xytext=(8, -4), color=C_TSTR)
    ax2.annotate("precision peak (g3)", (g[i_prec], prec[i_prec]),
                 textcoords="offset points", xytext=(6, 6), color=C_PREC)

    ax1.set_xlabel("CFG guidance g")
    ax1.set_ylabel("TSTR (%)", color=C_TSTR)
    ax2.set_ylabel("coverage / precision", color="k")
    ax1.set_title("MNIST: coverage tracks TSTR, fidelity (precision) does not")
    ax1.legend(handles=[l1, l2, l3], loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_mnist.png"))
    plt.close(fig)


def fig_selector_reversal():
    # coverage 選擇器 regret：MNIST 0（multiseed）、CIFAR-10/100 由 duel 檔
    mnist = load("selector_signal_multiseed.json")
    caf_mnist = sc(mnist["aggregate"]["regret_at_selected"])
    # MNIST 保真代理：T1a 實測 MNIST FID-min 之 regret（classifier-Fréchet，非再用 precision-argmax）
    fid_mnist = load("mnist_fid_arm.json")["aggregate"]["mnist"]["fidmin_regret_mean"]

    c10 = load("cifar10_c6_fidmin_duel.json")
    c100 = load("cifar100_c6_fidmin_duel.json")

    datasets = ["MNIST", "CIFAR-10", "CIFAR-100"]
    caf = [caf_mnist, c10["caf_regret_mean"], c100["caf_regret_mean"]]
    fidmin = [fid_mnist, c10["fidmin_regret_mean"], c100["fidmin_regret_mean"]]

    x = range(len(datasets))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    b1 = ax.bar([i - w / 2 for i in x], caf, w, color=C_COV,
                label="Coverage selector (CaF / CaF-v2)")
    b2 = ax.bar([i + w / 2 for i in x], fidmin, w, color=C_PREC,
                label="Fidelity proxy (FID-min)")
    ax.bar_label(b1, fmt="%.2f", padding=2)
    ax.bar_label(b2, fmt="%.2f", padding=2)

    # 標每個資料集哪個較可靠（regret 較低）
    for i in x:
        better = "coverage" if caf[i] < fidmin[i] else ("tie" if abs(caf[i] - fidmin[i]) < 1e-9 else "fidelity")
        ax.annotate(f"reliable: {better}", (i, max(caf[i], fidmin[i])),
                    textcoords="offset points", xytext=(0, 14), ha="center",
                    fontsize=9, color="#444")

    ax.set_xticks(list(x))
    ax.set_xticklabels(datasets)
    ax.set_ylabel("regret@selected (pp, lower is better)")
    ax.set_ylim(0, max(max(caf), max(fidmin)) * 1.20)
    ax.set_title("FID-min near-optimal across all three; coverage-CaF gated by\n"
                 "feature space (task-aligned works, task-agnostic fails)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
              framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_selector_reversal.png"), bbox_inches="tight")
    plt.close(fig)


def fig_pareto():
    d = load("cifar10_recall_density_c0.json")
    pc = d["per_config"]
    cstar = d["cstar"]
    oracles = set(d["oracles"])
    names = list(pc.keys())
    prec = {k: pc[k]["precision"] for k in names}
    cov = {k: pc[k]["coverage"] for k in names}

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    px, py = prec[cstar], cov[cstar]
    # 縮放到資料範圍，使 (precision, coverage) 平面與支配區可讀
    xmin = min(prec.values()) - 0.012
    xmax = max(prec.values()) + 0.022
    ymin = min(cov.values()) - 0.02
    ymax = max(cov.values()) + 0.03
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # 支配區：w2.5 左下方（precision 與 coverage 皆較低者被 w2.5 嚴格支配）
    ax.add_patch(plt.Rectangle((xmin, ymin), px - xmin, py - ymin,
                               color="#d62728", alpha=0.10, zorder=0,
                               label="dominated by w2.5"))
    for k in names:
        if k == cstar:
            ax.scatter(prec[k], cov[k], s=130, color=C_NB, marker="*", zorder=5)
            ax.annotate(f"{k} (c*)", (prec[k], cov[k]),
                        textcoords="offset points", xytext=(8, 4), color=C_NB)
        elif k in oracles:
            ax.scatter(prec[k], cov[k], s=70, color=C_TSTR, marker="o", zorder=4)
            ax.annotate(f"{k} (oracle)", (prec[k], cov[k]),
                        textcoords="offset points", xytext=(6, -12), color=C_TSTR)
        else:
            ax.scatter(prec[k], cov[k], s=40, color="#999", marker="o", zorder=3)
            ax.annotate(k, (prec[k], cov[k]), textcoords="offset points",
                        xytext=(5, 3), color="#999", fontsize=8)

    ax.set_xlabel("precision (DINOv2)")
    ax.set_ylabel("coverage (DINOv2)")
    ax.set_title("CIFAR-10: w2.5 strictly dominates every TSTR-oracle\n"
                 "(Pareto blindness — any monotone selector misses the oracle)")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_pareto.png"))
    plt.close(fig)


def fig_variance():
    d = load("cifar10_c4_variance.json")
    s_cls = d["variance"]["sigma_cls"]
    s_gen = d["variance"]["sigma_gen"]
    # 上升肢 Δ：w1 -> w1.5
    rising = d["per_w"]["w1.5"]["w_mean"] - d["per_w"]["w1"]["w_mean"]

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    bars = ax.bar(["σ_cls\n(classifier)", "σ_gen\n(generation)"], [s_cls, s_gen],
                  color=[C_NB, C_COV], width=0.6)
    ax.bar_label(bars, fmt="%.2f pp", padding=3)
    ax.axhline(rising, color=C_TSTR, ls="--", lw=1.5,
               label=f"rising-limb Δ (w1→w1.5) = {rising:.2f} pp")
    ax.set_ylabel("std of TSTR (pp)")
    ax.set_title("Variance decomposition: classifier noise (σ_cls)\n"
                 "dominates and exceeds the rising-limb Δ")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, max(s_cls, rising) * 1.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_variance.png"))
    plt.close(fig)


def _two_stage_panel(ax, per_config, title):
    w = [wnum(c["name"]) for c in per_config]
    tstr = [sc(c["tstr"]) for c in per_config]
    cov = [sc(c["coverage"]) for c in per_config]
    nb = [sc(c["near_boundary_frac"]) for c in per_config]
    ax2 = ax.twinx()
    l1, = ax.plot(w, tstr, "o-", color=C_TSTR, label="TSTR (%)")
    l2, = ax2.plot(w, cov, "s--", color=C_COV, label="coverage")
    l3, = ax2.plot(w, nb, "^:", color=C_NB, label="near-boundary")
    # coverage 峰位
    i_cov = max(range(len(cov)), key=lambda i: cov[i])
    ax2.axvline(w[i_cov], color=C_COV, alpha=0.25, lw=1)
    ax2.annotate(f"coverage peak (w{w[i_cov]:g})", (w[i_cov], cov[i_cov]),
                 textcoords="offset points", xytext=(6, 6), color=C_COV, fontsize=9)
    ax.set_xlabel("CFG guidance w")
    ax.set_ylabel("TSTR (%)", color=C_TSTR)
    ax2.set_ylabel("coverage / near-boundary", color="k")
    ax.set_title(title)
    return [l1, l2, l3]


def fig_two_stage():
    c10 = load("cifar10_cfg_confirmatory.json")["aggregate"]["per_config"]
    c100 = load("cifar100_cfg_confirmatory.json")["aggregate"]["per_config"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    _two_stage_panel(axes[0], c10, "CIFAR-10 confirmatory")
    handles = _two_stage_panel(axes[1], c100, "CIFAR-100 confirmatory")
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.05), framealpha=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT, "fig_two_stage.png"), bbox_inches="tight")
    plt.close(fig)


def fig_h3_duel():
    dv = load("cifar100_h3_duel_dinov2.json")
    jg = load("cifar100_h3_duel_judge.json")
    vanilla_tstr = dv["arms"]["fidmin"]["tstr_seed10"]
    vanilla_cov = 0.643  # vanilla w1.5 DINOv2 coverage（confirmatory 均值曲線）
    arms = ["vanilla w1.5\n(=FID-min=CaF-v2)",
            "Chamfer\n(DINOv2, task-agnostic)",
            "Chamfer\n(judge, task-aligned)"]
    tstr = [vanilla_tstr, dv["arms"]["chamfer"]["tstr_mean"], jg["arms"]["chamfer"]["tstr_mean"]]
    cov = [vanilla_cov, dv["arms"]["chamfer"]["coverage_dinov2"], jg["arms"]["chamfer"]["coverage_dinov2"]]

    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    x = range(len(arms))
    w = 0.38
    b1 = ax1.bar([i - w / 2 for i in x], tstr, w, color=C_TSTR, label="TSTR (%)")
    ax2 = ax1.twinx()
    b2 = ax2.bar([i + w / 2 for i in x], cov, w, color=C_COV, label="DINOv2 coverage")
    ax1.bar_label(b1, fmt="%.2f", padding=2)
    ax2.bar_label(b2, fmt="%.3f", padding=2)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(arms, fontsize=9)
    ax1.set_ylabel("TSTR (%)", color=C_TSTR)
    ax2.set_ylabel("DINOv2 coverage", color=C_COV)
    ax1.set_ylim(55, max(tstr) + 2)
    ax2.set_ylim(0, 0.8)
    ax1.set_title("H3 matched-budget comparison: Chamfer beats vanilla in utility,\n"
                  "yet its coverage is lower (cheap proxy misses the high-utility set)")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax2.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_h3_duel.png"))
    plt.close(fig)


def main():
    fig_mnist()
    fig_selector_reversal()
    fig_pareto()
    fig_variance()
    fig_two_stage()
    fig_h3_duel()
    print("wrote figures to", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  ", f)


if __name__ == "__main__":
    main()
