<!-- 用途:論文 intro 草稿。主張順序依 records/2026-07-05-12 §5;範圍聲明與護城河變薄依 §3;related work 差異化依 §3 後果。confirmatory 資料出來前為草稿。 -->

> **2026-07-09 狀態（E4，本文依 -12 §9 第 5 項不回改）**：主張 2 之頭條版本（效用最優偏離保真最優）已於兩表徵空間被 CIFAR-10 confirmatory 反證（FID/TSTR 重合於 w1.5、C1 不分離）；主張 3 全稱句撤下；主張 5 之 CaF 於 confirmatory 敗於 FID-min（per-seed regret 3.69 對 0.91，2 勝 1 負）且具結構性 Pareto 失明；主張 1 在 CIFAR-10 上因 FID-min 近最優而失去操作區辨力。三判決見 records/2026-07-09-03；CIFAR-100 分支裁決前本文僅供決策軌跡參考。

（E4 banner 依 CLAUDE.md §3.3「不以符號作狀態標記」去除原文案之警告圖示，文字內容照載。）

# Intro 草稿（Sampling for Utility, not Fidelity）

## 主張順序（依定位 v2 §5）

1. 合成影像被當成下游分類器的訓練資料時，取樣組態的選擇該為下游效用（TSTR）而非視覺保真度（FID）
   服務。本文聚焦 CFG guidance 軸。
2. 在 guidance 軸上，下游效用對 guidance **非單調、存在內部最優**：低 guidance 引入離類/模糊樣本壓低
   效用，高 guidance 崩多樣性（coverage）同樣壓低效用，最優落在中間。
3. 因此**任何固定的 guidance 值**跨資料集與任務必然次優——沒有一個通用配方。
4. 故需要一個**組態 selector**：在既有取樣組態中挑效用最優者，而非再調一個固定值。
5. 貢獻本體 **CaF（Coverage-at-Fidelity）**：免訓練、低成本、有機制解釋，以一小份真實 probe 的
   precision/coverage 定位效用最優組態（argmax coverage s.t. precision ≥ τ，auto-τ 不依賴 TSTR）。
   甜蜜點的存在是「為什麼需要 selector」的動機；CaF 是貢獻本體。

## 範圍聲明與護城河（依 §3，誠實陳述，不得中性帶過）

本文 CIFAR 尺度的主張收緊到 CFG guidance 軸。steps 與 η 的效用行為只在 MNIST sandbox 尺度得證（η 對
效用為 null、steps 次要），不在 CIFAR 尺度宣稱聯合曲面。此收緊使護城河變薄：η 與聯合曲面支點由 CIFAR
尺度退為 sandbox 證據，CIFAR 尺度的差異化剩三項——**非單調性、CaF、機制**。此三項相對 Chamfer、Fan、
DP 擴散文獻是否充分，由 CIFAR-100 機制 gate 與 Chamfer matched-budget 對決回答；在該兩者有資料前，
充分性為未決。CIFAR-100 的 η spot-check 提供 CIFAR 尺度的部分回補。

## Related work 差異化（依 §3 後果）

- **Fan et al.（固定 low-CFG 配方）**：給一個固定 guidance 值。本文的非單調性顯示任何固定值跨資料集/
  任務不可靠，故需逐情境選組態，而非套配方。
- **Chamfer Guidance（NeurIPS 2025）**：取樣時的 guidance 方法，須擁有 sampler、需真實參考影像逐步導引。
  CaF 是**組態層的 selector**，不改取樣、可疊在任何產生器輸出上（包括 Chamfer 的輸出），屬不同操作層次、
  正交可互補。差異化押在 selector≠guidance、選組態成本、遷移性，以及 CaF 附的機制（coverage 而非 fidelity
  驅動效用），這是 Chamfer 未給的因果說法。
- **DP 擴散文獻（η↔下游先導觀察）**：η 的單點觀察散見文獻。本文不在 CIFAR 尺度宣稱 η 貢獻，差異化不押
  在 η 軸。

## 機制（C2）

因果鏈：guidance 上升 → coverage 下降 → near-boundary 訓練樣本變少 → 下游 margin 變弱 → TSTR 下降。
競爭機制（低 guidance 的離類/模糊樣本，即 label-noise）自始內建為對照。C2 的 confirmatory 裁決以全網格
偏相關 partial ρ(TSTR, coverage | precision, label-noise) 進行（見 records/2026-07-05-13），不分段。

## 狀態

本草稿於 confirmatory 資料產生前撰寫。confirmatory 與 CIFAR-100、Chamfer 對決的結果出來後，依協定裁決
更新；本文定位不因結果回改，若衝突則據實報告。
