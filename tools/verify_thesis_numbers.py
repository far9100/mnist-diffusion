# 用途：把 docs/thesis_draft.md 第五章各表（表 5.1–5.5）與附錄 E（表 E.1–E.3）以及若干
# 承重內文 scalar，逐一對 results/*.json 逐位對帳，產出對帳報告。對帳只讀不寫，絕不改動任何
# results/*.json 或凍結檔；若論文數字與 JSON 不符一律報 MISMATCH，由人以 JSON 為準改論文。
#
# 設計說明（給第一次讀的研究生）：
#   - 每個表格的「真值來源」是對應的 results/*.json；本腳本重算出該表每格應有的值，再和論文
#     裡實際寫的值比對。
#   - 比對精度由「論文那格自己寫了幾位小數」決定：例如論文寫 97.30 就比到 2 位小數、寫 0.0269
#     就比到 4 位。這樣不必為每個指標另訂精度，且對日後 g→w 之類的欄名改寫免疫（以 guidance
#     數值或 seed 當鍵，不靠欄位標籤文字）。
#   - 差距分三級：完全相符（OK）、只差最後一位的進位邊界（ROUNDING-EDGE，視為通過但列出）、
#     其餘（MISMATCH，視為失敗）。
#
# 用法：uv run python tools/verify_thesis_numbers.py
#   結束碼 0＝全部通過（含 rounding-edge）；非 0＝存在 MISMATCH 或來源缺漏。

import json
import os
import re
import sys
from statistics import mean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
THESIS = os.path.join(ROOT, "docs", "thesis_draft.md")


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 數字正規化與比對
# ---------------------------------------------------------------------------

def norm_num(s):
    """把論文表格內的一格文字轉成 (float 值, 小數位數)。

    處理全形負號、前導點、正號、markdown 粗體 (**x**)，以及數字後接說明文字的情形
    （例如 '0.91（2 勝 1 負）' 只取前導的 0.91）。取不到數字回 (None, 0)。
    """
    t = s.strip().replace("−", "-").replace("–", "-")  # 全形/連字號負號 → ASCII
    t = t.replace("*", "").replace(" ", "")            # 去 markdown 粗體與空白
    t = t.lstrip("+")
    # 取開頭的數字 token：支援 0.91 / .645 / -.038 / 13606
    m = re.match(r"-?(?:\d+\.?\d*|\.\d+)", t)
    if not m:
        return None, 0
    tok = m.group(0)
    tok = re.sub(r"^(-?)\.", r"\g<1>0.", tok)  # .645 → 0.645
    val = float(tok)
    ndig = len(tok.split(".")[1]) if "." in tok else 0
    return val, ndig


RESULTS = {"ok": 0, "edge": 0, "mismatch": 0, "missing": 0}
LINES = []


def check(label, thesis_str, source_val, ndigits=None):
    """比對一格。thesis_str 為論文原文字；source_val 為由 JSON 重算之值。"""
    tval, tdig = norm_num(thesis_str)
    if tval is None:
        LINES.append(f"  [SKIP ] {label}: 論文格非數值（{thesis_str!r}）")
        return
    if source_val is None:
        RESULTS["missing"] += 1
        LINES.append(f"  [MISS ] {label}: 來源缺值（論文 {thesis_str}）")
        return
    nd = tdig if ndigits is None else ndigits
    r = round(float(source_val), nd)
    unit = 10 ** (-nd)
    diff = abs(tval - r)
    if diff < 1e-9:
        RESULTS["ok"] += 1
        # 通過者不逐行印，避免報告過長；需要細節可解除下一行註解
        # LINES.append(f"  [OK   ] {label}: {thesis_str} == {r}")
    elif diff <= unit + 1e-9:
        RESULTS["edge"] += 1
        LINES.append(f"  [EDGE ] {label}: 論文 {thesis_str} vs 來源 {r}（差一位進位，源始值 {source_val}）")
    else:
        RESULTS["mismatch"] += 1
        LINES.append(f"  [DIFF ] {label}: 論文 {thesis_str} vs 來源 {r}（源始值 {source_val}）")


def check_str(label, thesis_str, source_str):
    if thesis_str.strip() == str(source_str).strip():
        RESULTS["ok"] += 1
    else:
        RESULTS["mismatch"] += 1
        LINES.append(f"  [DIFF ] {label}: 論文 {thesis_str!r} vs 來源 {source_str!r}")


# ---------------------------------------------------------------------------
# markdown 表格擷取
# ---------------------------------------------------------------------------

def parse_tables(md):
    """回傳所有 markdown 表格；每個表格為 list[list[str]]（已去分隔列）。"""
    tables, cur = [], []
    for line in md.splitlines():
        if line.lstrip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            # 分隔列（---|:--）跳過
            if all(re.fullmatch(r":?-{2,}:?", c or "-") for c in cells):
                continue
            cur.append(cells)
        else:
            if cur:
                tables.append(cur)
                cur = []
    if cur:
        tables.append(cur)
    return tables


def gnum(name):
    """'w1.5' / 'g10' / '1.5' → float。"""
    return float(re.sub(r"^[wg]", "", str(name)))


# ---------------------------------------------------------------------------
# 由 JSON 建每個表的真值
# ---------------------------------------------------------------------------

def curve_mnist():
    d = load("selector_signal_multiseed.json")
    out = {}
    for c in d["aggregate"]["per_config"]:
        g = c["guidance"]
        out[g] = {
            "precision": c["precision"]["mean"],
            "coverage": c["coverage"]["mean"],
            "tstr": c["tstr"]["mean"],
            "near_boundary": c["near_boundary_frac"]["mean"],
        }
    return out


def curve_cifar10():
    d = load("cifar10_cfg_confirmatory.json")
    out = {}
    for c in d["aggregate"]["per_config"]:
        g = gnum(c["name"])
        out[g] = {
            "tstr": c["tstr"]["mean"],
            "coverage": c["coverage"]["mean"],
            "precision": c["precision"]["mean"],
            "near_boundary": c["near_boundary_frac"]["mean"],
            "ln_excess": c["label_noise_excess_mean"]["mean"],
        }
    # FD-DINOv2：表 5.2 之列，源自 cifar10_p1_streaming.json 之逐 seed 均值
    s = load("cifar10_p1_streaming.json")
    acc = {}
    for seed_blk in s["per_seed"]:
        for c in seed_blk["configs"]:
            acc.setdefault(gnum(c["name"]), []).append(c["fd_dinov2"])
    for g, vals in acc.items():
        out.setdefault(g, {})["fd_dinov2"] = mean(vals)
    return out


def curve_cifar100():
    d = load("cifar100_cfg_confirmatory.json")
    out = {}
    for c in d["aggregate"]["per_config"]:
        g = gnum(c["name"])
        out[g] = {
            "tstr": c["tstr"]["mean"],
            "coverage": c["coverage"]["mean"],
            "precision": c["precision"]["mean"],
            "recall": c["recall"]["mean"],
            "near_boundary": c["near_boundary_frac"]["mean"],
            "char_clean_fid": c["char_clean_fid"]["mean"],
        }
    return out


# ---------------------------------------------------------------------------
# 各表對帳
# ---------------------------------------------------------------------------

def find_table(tables, pred):
    for t in tables:
        if t and pred(t[0], t):
            return t
    return None


def verify_table_5_1(tables):
    t = find_table(tables, lambda h, _t: len(h) == 5 and h[0] in ("g", "w")
                   and "precision" in h[1] and "near-boundary" in h[4])
    if not t:
        LINES.append("  [MISS ] 表 5.1 未找到")
        RESULTS["missing"] += 1
        return
    src = curve_mnist()
    for row in t[1:]:
        g = gnum(row[0])
        s = src.get(g)
        if not s:
            RESULTS["missing"] += 1
            LINES.append(f"  [MISS ] 表5.1 g={g} 來源缺")
            continue
        check(f"表5.1 g{g} precision", row[1], s["precision"])
        check(f"表5.1 g{g} coverage", row[2], s["coverage"])
        check(f"表5.1 g{g} TSTR", row[3], s["tstr"])
        check(f"表5.1 g{g} near-boundary", row[4], s["near_boundary"])


def verify_wide_table(tables, caption_metrics, src, tag):
    """表 5.2 / 5.4：橫向（欄=guidance，列=指標）。caption_metrics: {列標籤關鍵字: src 鍵}。"""
    def is_it(h, tbl):
        if not (h[0] == "w" and len(h) >= 6 and h[1] == "1"):
            return False
        labels = {r[0] for r in tbl[1:]}
        return all(any(k in lab for lab in labels) for k in caption_metrics)
    t = find_table(tables, is_it)
    if not t:
        LINES.append(f"  [MISS ] {tag} 未找到")
        RESULTS["missing"] += 1
        return
    guid = [gnum(x) for x in t[0][1:]]
    for row in t[1:]:
        label = row[0]
        key = next((v for k, v in caption_metrics.items() if k in label), None)
        if key is None:
            continue
        for gi, cell in zip(guid, row[1:]):
            s = src.get(gi, {})
            check(f"{tag} {label} w{gi}", cell, s.get(key))


def verify_table_5_3(tables):
    t = find_table(tables, lambda h, _t: "TSTR-oracle" in " ".join(h))
    if not t:
        LINES.append("  [MISS ] 表 5.3 未找到")
        RESULTS["missing"] += 1
        return
    c10 = load("cifar10_c6_fidmin_duel.json")
    c100 = load("cifar100_c6_fidmin_duel.json")
    mnist = load("selector_signal_multiseed.json")
    base = load("baseline_fixed_random.json")["datasets"]  # T2：D5 baseline
    # 欄序：資料集 | TSTR-oracle | coverage 選擇器選中 | coverage 選擇器 regret | FID-min regret
    #        | fixed-w1 regret | fixed-w2 regret | random-feasible regret | 可靠代理
    reg = {
        "MNIST": mnist["aggregate"]["regret_at_selected"]["mean"],
        "CIFAR-10": c10["caf_regret_mean"],
        "CIFAR-100": c100["caf_regret_mean"],
    }
    fidmin = {"CIFAR-10": c10["fidmin_regret_mean"], "CIFAR-100": c100["fidmin_regret_mean"]}
    if os.path.exists(os.path.join(RES, "mnist_fid_arm.json")):  # T1a：MNIST 實測 FID-min
        fidmin["MNIST"] = load("mnist_fid_arm.json")["aggregate"]["mnist"]["fidmin_regret_mean"]

    def fixed_at(ds, guid):
        """該資料集固定於某 guidance（1.0=fixed-w1、2.0=fixed-w2）的平均 regret；MNIST 網格用 g 命名。"""
        d = base[ds]
        name = next(n for n in d["grid"] if gnum(n) == guid)
        return d["fixed_w"][name]["mean_regret"]

    for row in t[1:]:
        ds = row[0].strip()
        if ds in reg:
            check(f"表5.3 {ds} coverage選擇器 regret", row[3], reg[ds])
        if ds in fidmin:
            check(f"表5.3 {ds} FID-min regret", row[4], fidmin[ds])
        if ds in base:
            check(f"表5.3 {ds} fixed-w1 regret", row[5], fixed_at(ds, 1.0))
            check(f"表5.3 {ds} fixed-w2 regret", row[6], fixed_at(ds, 2.0))
            check(f"表5.3 {ds} random-feasible regret", row[7], base[ds]["random_feasible"]["mean_regret"])


def verify_table_5_5(tables):
    t = find_table(tables, lambda h, _t: h[0] == "臂" and "TSTR" in h[1])
    if not t:
        LINES.append("  [MISS ] 表 5.5 未找到")
        RESULTS["missing"] += 1
        return
    dv = load("cifar100_h3_duel_dinov2.json")
    jd = load("cifar100_h3_duel_judge.json")
    van_cov = load("cifar100_cfg_confirmatory.json")
    van_cov_w15 = next(c["coverage"]["mean"] for c in van_cov["aggregate"]["per_config"] if c["name"] == "w1.5")
    for row in t[1:]:
        name = row[0]
        if "vanilla" in name:
            check("表5.5 vanilla TSTR", row[1], dv["arms"]["fidmin"]["tstr_seed10"])
            check("表5.5 vanilla coverage", row[2], van_cov_w15)
        elif "DINOv2" in name:
            check("表5.5 Chamfer-DINOv2 TSTR", row[1], dv["arms"]["chamfer"]["tstr_mean"])
            check("表5.5 Chamfer-DINOv2 coverage", row[2], dv["arms"]["chamfer"]["coverage_dinov2"])
        elif "judge" in name:
            check("表5.5 Chamfer-judge TSTR", row[1], jd["arms"]["chamfer"]["tstr_mean"])
            check("表5.5 Chamfer-judge coverage", row[2], jd["arms"]["chamfer"]["coverage_dinov2"])


def verify_etable(tables, header_key, duel_json, caf_key, tag):
    t = find_table(tables, lambda h, _t: h[0] == "seed" and any(header_key in c for c in h))
    if not t:
        LINES.append(f"  [MISS ] {tag} 未找到")
        RESULTS["missing"] += 1
        return
    d = load(duel_json)
    by_seed = {p["seed"]: p for p in d["per_seed"]}
    for row in t[1:]:
        if row[0] in ("均值", "mean"):
            check(f"{tag} 均值 FID-min regret", row[4], d["fidmin_regret_mean"])
            check(f"{tag} 均值 CaF regret", row[5], d[caf_key.replace('_per_seed', '_mean')])
            continue
        try:
            seed = int(row[0])
        except ValueError:
            continue
        p = by_seed.get(seed)
        if not p:
            continue
        check_str(f"{tag} seed{seed} oracle", row[1], p["oracle"])
        check(f"{tag} seed{seed} oracle TSTR", row[2], p["oracle_tstr"])
        check_str(f"{tag} seed{seed} FID-min 選中", row[3], p["fidmin"])
        check(f"{tag} seed{seed} FID-min regret", row[4], p["fidmin_regret"])
        check(f"{tag} seed{seed} CaF regret", row[5], p["caf_regret"])


def verify_table_e3(tables):
    t = find_table(tables, lambda h, _t: h[0] == "seed" and "w1" in h and "w8" in h)
    if not t:
        LINES.append("  [MISS ] 表 E.3 未找到")
        RESULTS["missing"] += 1
        return
    d = load("cifar100_cfg_confirmatory.json")
    guid = [gnum(x) for x in t[0][1:]]
    per_seed = {}
    for blk in d["per_seed"]:
        per_seed[blk["seed"]] = {gnum(c["name"]): c["tstr"] for c in blk["configs"]}
    agg = {gnum(c["name"]): c["tstr"]["mean"] for c in d["aggregate"]["per_config"]}
    for row in t[1:]:
        if row[0] in ("均值", "mean"):
            for gi, cell in zip(guid, row[1:]):
                check(f"表E.3 均值 w{gi}", cell, agg.get(gi))
            continue
        try:
            seed = int(row[0])
        except ValueError:
            continue
        s = per_seed.get(seed, {})
        for gi, cell in zip(guid, row[1:]):
            check(f"表E.3 seed{seed} w{gi}", cell, s.get(gi))


def verify_inline():
    """承重內文 scalar：以來源重算，與論文寫法比對（論文原文寫法列於 label）。"""
    var = load("cifar10_c4_variance.json")["variance"]
    check("內文 σ_cls=2.963", "2.963", var["sigma_cls"])
    check("內文 σ_gen=1.182", "1.182", var["sigma_gen"])
    check("內文 σ_cls=2.96", "2.96", var["sigma_cls"])

    itv = load("cifar100_d3_intervention.json")["c3_coverage_matched"]
    cov_m = mean(itv["tstr_w2.5_cov_matched"])
    rnd = mean(itv["tstr_w2.5_rand_pruned"])
    check("內文 N=2 cov-matched TSTR 45.75", "45.75", cov_m)
    check("內文 N=2 random-pruned TSTR 45.86", "45.86", rnd)
    check("內文 N=2 兩者差 -0.11", "-0.11", cov_m - rnd)
    check("內文 w2.5 base coverage 0.700", "0.700", itv["w2.5_base_coverage"])
    check("內文 w1 target coverage 0.481", "0.481", itv["w1_target_coverage"])
    check("內文 n_pruned 13606", "13606", itv["n_pruned_to_match"], ndigits=0)
    check("內文 w2.5 frozen TSTR 50.66", "50.66", itv["w2.5_frozen_tstr"])
    # N=8 更高功效 follow-up（§5.5 主要報告值；結構量 0.700/0.481/13606/50.66 兩檔相同）
    itv8 = load("cifar100_d3_intervention_n8.json")["c3_coverage_matched"]
    cov8 = mean(itv8["tstr_w2.5_cov_matched"])
    rnd8 = mean(itv8["tstr_w2.5_rand_pruned"])
    check("內文 N=8 cov-matched TSTR 46.30", "46.30", cov8)
    check("內文 N=8 random-pruned TSTR 46.63", "46.63", rnd8)
    check("內文 N=8 兩者差 -0.33", "-0.33", cov8 - rnd8)

    c0 = load("cifar10_recall_density_c0.json")["per_config"]
    check("內文 recall w2.5 .493", ".493", c0["w2.5"]["recall"])
    check("內文 recall w2 .528", ".528", c0["w2"]["recall"])
    check("內文 recall w1.5 .555", ".555", c0["w1.5"]["recall"])
    check("內文 recall w1 .579", ".579", c0["w1"]["recall"])

    dv = load("cifar100_h3_duel_dinov2.json")
    jd = load("cifar100_h3_duel_judge.json")
    check("內文 H3 Chamfer-DINOv2 +2.54", "2.54", dv["duel"]["chamfer_minus_vanilla_seed10"])
    check("內文 H3 Chamfer-judge +3.11", "3.11", jd["duel"]["chamfer_minus_vanilla_seed10"])

    thr = load("cifar100_cfg_confirmatory.json")["metadata"]["near_boundary_threshold"]
    check("內文 near-boundary 門檻 0.3622", "0.3622", thr)

    # CIFAR-100 which-FID 之 DINOv2 空間（§5.4.1，事後、單 seed 10）
    fd = load("cifar100_fd_dinov2.json")
    wf = fd["which_fid_dinov2"]
    check("內文 FD-DINOv2 分離格步 3", "3", wf["separation_step"], ndigits=0)
    check_str("內文 FD-DINOv2 argmin w2.5", "w2.5", wf["fd_dinov2_argmin"])
    fpc = {c["name"]: c for c in fd["per_config"]}
    check("內文 FD-DINOv2 w2 基底 153.8", "153.8", fpc["w2"]["fd_dinov2"])
    check("內文 FD-DINOv2 w2.5 基底 153.6", "153.6", fpc["w2.5"]["fd_dinov2"])
    oracle_tstr = max(c["tstr_seed10"] for c in fd["per_config"])
    check("內文 FD-DINOv2 選擇器 regret 8.8", "8.8",
          oracle_tstr - fpc[wf["fd_dinov2_argmin"]]["tstr_seed10"])

    # C1 配對統計（§5.4.1，post-hoc；來源 c1_paired_stats.json）
    cps = load("c1_paired_stats.json")["datasets"]
    check("內文 C1 CIFAR-100 配對均值 0.76", "0.76", cps["CIFAR-100"]["paired_test_posthoc"]["mean_diff"])
    check("內文 C1 CIFAR-100 配對 t=9.71", "9.71", cps["CIFAR-100"]["paired_test_posthoc"]["t"])
    check("內文 C1 CIFAR-10 配對均值 0.91", "0.91", cps["CIFAR-10"]["paired_test_posthoc"]["mean_diff"])
    check("內文 C1 CIFAR-10 配對 t=1.18", "1.18", cps["CIFAR-10"]["paired_test_posthoc"]["t"])

    # τ 靈敏度（§6.3，post-hoc；來源 tau_sensitivity.json）
    ts = load("tau_sensitivity.json")["datasets"]["CIFAR-100"]["settings"]
    check("內文 τ tau_frac0.85 regret 0.00", "0.00", ts["tau_frac_0.85"]["mean_regret"])
    check("內文 τ tau_frac0.90 regret 0.76", "0.76", ts["tau_frac_0.90"]["mean_regret"])
    check("內文 τ tau_frac0.95 regret 5.40", "5.40", ts["tau_frac_0.95"]["mean_regret"])
    check("內文 τ no_floor regret 0.00", "0.00", ts["no_floor"]["mean_regret"])
    check("內文 τ no_floor 勝 FID-min 0.76", "0.76", ts["no_floor"]["beats_fidmin_by_pp"])

    # 配平校準（§5.4.1，seed 10，需快取特徵；缺檔則 available=False，略過）
    mc = load("tau_sensitivity.json")["datasets"]["CIFAR-100"].get("matched_calibration_seed10", {})
    if mc.get("available"):
        pcw1 = next(pc for pc in mc["per_config"] if pc["name"] == "w1")
        check("內文 配平 w1 precision 500v500 0.78", "0.78", pcw1["precision_500v500"])
        check("內文 配平 w1 precision 250v250 0.8106", "0.8106", pcw1["precision_250v250"])
        check_str("內文 配平 500v500 選 w1.5", "w1.5", mc["selection_500v500"]["selected"])
        check_str("內文 配平 250v250 選 w1", "w1", mc["selection_250v250_matched"]["selected"])

    # T1a/T1b MNIST 臂（§5.3／§6.1；來源 mnist_fid_arm.json、mnist_dinov2_stack.json）
    if os.path.exists(os.path.join(RES, "mnist_fid_arm.json")):
        mfa = load("mnist_fid_arm.json")["aggregate"]["mnist"]
        check("內文 T1a MNIST FID-min regret 1.02", "1.02", mfa["fidmin_regret_mean"])
        mds = load("mnist_dinov2_stack.json")["aggregate"]
        check("內文 T1b DINOv2 CaF(coverage) regret 1.02", "1.02", mds["caf"]["regret_mean"])
        check("內文 T1b DINOv2 CaF-v2(recall) regret 0.0", "0.0", mds["caf_v2"]["regret_mean"])
        cf = load("mnist_fid_arm.json")["aggregate"].get("clean")   # T1a clean-fid 第二讀數
        if cf:
            check("內文 T1a clean-fid regret 1.02", "1.02", cf["fidmin_regret_mean"])
    # T1c CIFAR judge 特徵堆疊（§6.1 六格 2×2；來源 cifar_judgefeat_stack.json）
    if os.path.exists(os.path.join(RES, "cifar_judgefeat_stack.json")):
        jfd = load("cifar_judgefeat_stack.json")["datasets"]
        jf10 = jfd.get("cifar10", {})
        if jf10.get("complete") and jf10.get("reports"):
            check("內文 T1c CIFAR-10 judge CaF regret 2.45", "2.45", jf10["reports"]["caf"]["regret_at_selected"])
        jf100 = jfd.get("cifar100", {})
        if jf100.get("complete") and jf100.get("reports"):
            check("內文 T1c CIFAR-100 judge CaF regret 0.79", "0.79", jf100["reports"]["caf"]["regret_at_selected"])
            # 2×2 之 CIFAR-100 DINOv2 coverage-CaF（signal=coverage，由凍結 confirmatory 重算）
            for _p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "src", "core")):
                if _p not in sys.path:
                    sys.path.insert(0, _p)
            from selector import select_caf, auto_tau
            c100 = load("cifar100_cfg_confirmatory.json")
            regs = []
            for b in c100["per_seed"]:
                oracle = max(b["configs"], key=lambda c: c["tstr"])
                sel, _ = select_caf(b["configs"], auto_tau(b["ref_precision"], 0.9), signal_key="coverage")
                regs.append(oracle["tstr"] - sel["tstr"])
            check("內文 2×2 CIFAR-100 DINOv2 coverage-CaF 6.10", "6.10", sum(regs) / len(regs))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # 確保中文報告不受終端編碼影響
    except Exception:
        pass
    with open(THESIS, encoding="utf-8") as f:
        md = f.read()
    tables = parse_tables(md)

    verify_table_5_1(tables)
    verify_wide_table(tables, {"TSTR": "tstr", "coverage(DINOv2)": "coverage",
                               "precision": "precision", "near-boundary": "near_boundary",
                               "ln_excess": "ln_excess", "FD-DINOv2": "fd_dinov2"},
                      curve_cifar10(), "表5.2")
    verify_table_5_3(tables)
    verify_wide_table(tables, {"TSTR": "tstr", "coverage(DINOv2)": "coverage",
                               "precision": "precision", "recall": "recall",
                               "near-boundary": "near_boundary", "char_clean_fid": "char_clean_fid"},
                      curve_cifar100(), "表5.4")
    verify_table_5_5(tables)
    verify_etable(tables, "CaF regret", "cifar10_c6_fidmin_duel.json", "caf_regret_per_seed", "表E.1")
    verify_etable(tables, "CaF-v2 regret", "cifar100_c6_fidmin_duel.json", "caf_regret_per_seed", "表E.2")
    verify_table_e3(tables)
    verify_inline()

    print("=" * 72)
    print("thesis_draft.md 數值對帳報告（來源＝results/*.json；不改 JSON）")
    print("=" * 72)
    for ln in LINES:
        print(ln)
    if not LINES:
        print("  （無任何 EDGE / DIFF / MISS，全部逐位相符）")
    print("-" * 72)
    print(f"OK={RESULTS['ok']}  ROUNDING-EDGE={RESULTS['edge']}  "
          f"MISMATCH={RESULTS['mismatch']}  MISSING={RESULTS['missing']}")
    print("=" * 72)
    return 1 if (RESULTS["mismatch"] or RESULTS["missing"]) else 0


if __name__ == "__main__":
    sys.exit(main())
