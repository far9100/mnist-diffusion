<!-- 用途：彙整並分析各階段實驗結果（量測錨點、MNIST sandbox、CIFAR-10 預覽）。 -->

# 實驗結果分析

本文件彙整目前已產出的實驗數據與其解讀。逐次執行的原始記錄見 `records/`；此處只保留可作結論或
待確認的結果分析。CIFAR-10 confirmatory 已於 2026-07-06 完成（`results/cifar10_cfg_confirmatory.json`），
其三判決分析見下方「CIFAR-10 confirmatory（三判決定稿，E3）」段，鏡像 B 定稿 `records/2026-07-09-03`。

範圍註記（依 `records/2026-07-05-12` 定位 v2）：CIFAR 尺度的主張收緊到 CFG guidance 軸。取樣步數 steps
與 DDIM 隨機性 η 的效用結論只保留在 MNIST sandbox 尺度，不在 CIFAR 尺度宣稱聯合曲面。下方 MNIST 表
為 sandbox 動機證據，CIFAR 自訓 CFG 的 pilot 與 scout 為 exploratory，confirmatory 三判決已定稿（見下方
E3 段）。

## 量測堆疊正確性（backbone 已驗證）

在做任何 CIFAR 組態掃描前，先證明本專案的 EDM 生成到 FID 的量測路徑能重現論文數字。EDM CIFAR-10
（條件、VP）在 50000 張下 FID 為 1.848，論文參考值 1.79，差 0.058，通過正確性 gate。略高於 1.79
最可能來自 Windows 上以純 PyTorch 參考實作取代 custom CUDA ops 的良性數值差異，對組態之間的相對
比較不構成問題。詳見 `records/2026-07-03-03_test_edm-fid-gate.md`。

## MNIST sandbox（乾淨案例，Phase 0）

設定：DDIM、η=0、steps=50，guidance g 屬於 {1,2,3,5,7,10}，每類 1000 張，seeds {0,1,2}，auto-τ
為 real-vs-real 參考 precision 的 0.9 倍（約 0.857）。

| g | precision | coverage | TSTR% | near-boundary 比例 |
|---|---|---|---|---|
| 1 | 0.916±0.003 | 0.875±0.006 | 97.30±0.15 | 0.0269 |
| 2 | 0.961±0.001 | 0.851±0.003 | 96.28±0.06 | 0.0008 |
| 3 | 0.966±0.001 | 0.732±0.004 | 95.01±0.20 | 0.0001 |
| 5 | 0.964±0.001 | 0.556±0.005 | 92.31±0.06 | 0.0 |
| 7 | 0.951±0.003 | 0.436±0.004 | 88.85±0.16 | 0.0 |
| 10 | 0.919±0.001 | 0.319±0.005 | 78.51±1.00 | 0.0 |

分析：

- **coverage 驅動效用**：coverage 隨 guidance 單調下降（0.875→0.319），TSTR 也單調下降
  （97.30→78.51），兩者近乎完美同向。
- **precision 不追蹤效用**：precision 在 g3 達峰（0.966），TSTR 卻在 g1 達峰。保真最佳與效用最佳
  落在不同組態，這是相對 Chamfer 在機制層的乾淨差異。
- **CaF 可行**：免訓練的 CaF 在三個 seed 全部選中 g1，與 oracle TSTR 最佳一致；regret 為
  0.00±0.00 pp、rank 1/6、top-3 命中 100%。信賴區間很緊、排序跨 seed 從不翻轉，代表「效用最佳點
  不等於保真最佳點」不是單 seed 雜訊。
- **機制在 MNIST 飽和**：near-boundary 比例隨 guidance 單調枯竭（0.027→0），方向正確；但 MNIST
  近乎可分，真實資料的 near-boundary 比例僅約 0.009，能被枯竭的邊界質量很少，機制的強證據需要更難、
  非可分的 CIFAR-100。
- **τ 誠實註記**：auto-τ 選中 g1（regret 0），但 τ 掃描的眾數選擇是 g2；g1 與 g2 對 τ 敏感，惟
  兩者皆在前二名、robust 區間內 regret 不超過約 1.3pp。

原始記錄見 `records/2026-07-03-02_test_phase0-caf-signal.md` 與
`records/2026-07-03-04_test_gateA-verdict.md`。

## CIFAR-10 預覽（初步，較複雜，尚未定案）

設定刻意求快：1 seed、每類 500 張、生成步數 8、Inception 特徵做 PRDC。CFG guidance 軸 w 屬於
{1,1.5,2,3}。

| w | precision | coverage | recall | TSTR% |
|---|---|---|---|---|
| 1.0 | 0.767 | 0.825 | 0.720 | 47.50 |
| 1.5 | 0.807 | 0.842 | 0.689 | 49.74（oracle 最佳） |
| 2.0 | 0.819 | 0.844 | 0.656 | 49.23 |
| 3.0 | 0.803 | 0.807 | 0.571 | 48.06 |

分析：

- 與 MNIST 不同，CIFAR-10 的 TSTR 似乎有**內部最優**（在中間 guidance w≈1.5–2 達峰，而非 w=1），
  coverage 也非單調。CaF（auto-τ 約 0.697）選中 w2，oracle 最佳為 w1.5；regret 0.51pp、rank 2/4、
  top-3 命中。
- 兩種解讀，需全品質才能分辨：（a）真實 CIFAR 行為——CIFAR 不像 MNIST 近乎可分，適度 guidance 先
  提升保真與代表性使 TSTR 上升，只有過高 guidance 才崩多樣性使 TSTR 下降，選擇器的任務變成找甜蜜點；
  （b）8 步的低步數假影——為求快把生成步數由 18 降到 8，對 w=1（無 CFG）端傷害最大，可能人為壓低
  低 guidance 的 TSTR、製造出內部最優。必須用 18 步全品質重跑後才能下結論。
- TSTR 偏低（約 48–50%）是因為只有 5000 張合成且用 8 步，屬弱訊號預覽。**目前不主張 CIFAR 結果。**

原始記錄見 `records/2026-07-03-06_test_cifar-preview.md`。

## 自訓 CFG CIFAR-10（exploratory pilot 與上緣 scout）

以下為自訓 CFG 主軸的 exploratory 結果（pilot 用 seeds 0/1/2，grid 於看過 scout 後才鎖定，故只能定位為
exploratory；confirmatory 以 fresh seeds 10/11/12、10 點 grid 另跑）。

Stage 4 多 seed pilot（steps=50、η=0，grid {1,2,3,4,5,8}，3 seed 均值 ± std）：

| w | precision | coverage(DINOv2) | TSTR% | label_noise | near_bnd |
|---|---|---|---|---|---|
| 1 | 0.827±0.009 | 0.710±0.006 | 41.18±1.15 | 0.112±0.004 | 0.252±0.008 |
| 2 | 0.884±0.004 | 0.834±0.009 | 46.01±1.85 | 0.013±0.003 | 0.063±0.005 |
| 3 | 0.903±0.004 | 0.834±0.002 | 43.88±1.49 | 0.005±0.002 | 0.037±0.004 |
| 4 | 0.903±0.003 | 0.806±0.005 | 40.59±0.91 | 0.003±0.002 | 0.032±0.001 |
| 5 | 0.897±0.001 | 0.758±0.008 | 37.43±2.39 | 0.003±0.001 | 0.031±0.002 |
| 8 | 0.871±0.003 | 0.622±0.012 | 28.43±0.70 | 0.008±0.002 | 0.055±0.004 |

pilot 判讀（exploratory，敘事描述，不承擔裁決功能）：TSTR 峰在 w=2、三 seed oracle 一致，內部最優在
自訓主軸成立；CaF 每 seed 選 [w2,w3,w2]、regret 0.28pp、top-3 100%。三段結構——低段（w1→2）precision/
coverage 同升而 label-noise 降（三力同向、混淆）、中段（w2→3）coverage 平坦而 TSTR 微降、高段（w3→8）
coverage 崩且 precision 高原（兩者分離）——只作敘事，不作 C2 裁決。C2 的 confirmatory 裁決改以全網格偏
相關進行，納入全部 10 點含混淆段，不分段（見 `records/2026-07-05-13`）。

上緣 coverage-only scout（1 seed、coverage-only、DINOv2 特徵）：coverage 於 w∈[8,20] 由 0.533 單調降至
0.259，依判準 X=0.02 未觸底。此為描述性觀察，緊貼其條件標注；依封頂 amendment（`records/2026-07-05-11`）
以先驗 CFG 實用範圍封頂 w=8，confirmatory 不重跑 w>8 區段。併入雙力敘事：低段 fidelity 上升、高段
coverage 單調下降，兩力交會產生內部最優。

## CIFAR-10 confirmatory（三判決定稿，E3）

鏡像 B 定稿 `records/2026-07-09-03`。三判決互不裁決；本段為觀察性描述，禁因果措辭。
設定：steps=50、η=0、grid {1,1.5,2,2.5,3,4,5,6,7,8}、fresh seeds {10,11,12}、per_class 1000。

### 判決一（thesis）

FID-opt 與 TSTR-opt 重合於 w1.5（clean-fid 均值 8.82／TSTR 均值 63.96）。排序（n=10 均值曲線 Spearman）：
ρ(−char_clean_fid, TSTR)=0.964、ρ(coverage, TSTR)=0.624。README 頭條「FID-opt 偏離 TSTR-opt」在本資料上被
反證。which-FID 交叉裁決（C1，`records/2026-07-09-02`）：FD-DINOv2 均值 argmin=w2、TSTR argmax=w1.5，格步 1、
依凍結口徑（>1 才算分離）判不分離；疊加 Inception／clean-fid 側 0 格步不分離——兩表徵空間皆不分離。「內部
最優」為未登記 exploratory 觀察（上升肢 w1→w1.5 +0.80pp、SE 1.88，3 gen seeds 下不可判定）；「必然次優」
全稱句撤下。災難性非單調（w≥3 TSTR 崩 11–30pp）為存活觀察。

### 判決二（H-C2）

A 批（`records/2026-07-06-09`）：DINOv2 側 C2a ρ=+0.658 p=0.0188、Inception 側 ρ=+0.859 p=0.0008（跨表徵
一致），機械通過；但穩健性沿種子軸（bootstrap CI 跨零）與可交換性軸（gseed 碰撞）無法內部驗證，最終須
CIFAR-100 獨立複製回答。H-C2a 顯著亦不寫「coverage 驅動效用」等因果措辭。

### 判決三（selector，描述性）

協定未凍門檻，不作過／敗判定。FID-min per-seed regret 0.91（2.45/0.28/0.00）對 CaF 3.69（0.54/5.03/5.49）；
FID-min 為 3 seed 2 勝 1 負（seed10 上 CaF 0.54 反勝 FID-min 2.45）。regret 主 3.69（per-seed 均值）、並列
2.77（mean-curve 口徑：均值 oracle w1.5 之 63.96 減均值 CaF w2.5 之 61.19）。CaF 於本網格結構性 Pareto 失明
（w2.5 (.873,.792) 嚴格支配三 oracle，C8 引理 `docs/c8_pareto_blindness.md`）。可辯護措辭：CaF 為可靠避崖器、
糟糕平台優化器——FID-min 同樣避崖且成本結構相同。

### 雙段機制（觀察性，禁因果；均值曲線）

| w | 1 | 1.5 | 2 | 2.5 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| TSTR | 63.16 | 63.96 | 61.16 | 61.19 | 53.08 | 47.20 | 44.10 | 39.10 | 36.06 | 33.46 |
| coverage(DINOv2) | .645 | .751 | .777 | .792 | .778 | .745 | .698 | .648 | .604 | .559 |
| precision | .806 | .841 | .858 | .873 | .876 | .879 | .869 | .859 | .848 | .836 |
| near-boundary | .256 | .114 | .063 | .046 | .037 | .032 | .037 | .039 | .048 | .059 |
| ln_excess | +.044 | −.038 | −.057 | −.062 | −.065 | −.065 | −.065 | −.064 | −.063 | −.062 |
| FD-DINOv2 | 282.4 | 195.2 | 175.4 | 176.7 | 188.7 | 223.9 | 261.1 | 302.6 | 341.9 | 379.4 |

觀察（禁因果）：低中段（w1→w2.5）near-boundary 比例 .256→.046 下降、coverage .645→.792 上升（coverage 峰
落 w2.5）；w1 之 ln_excess +.044 為全網格唯一正值。高段（w2.5→w8）coverage 與 TSTR 同降、near-boundary 由
.032 回升至 .059、precision 同降。全網格 ρ 將雙段抹平（C2 裁決照跑之凍結義務，見判決二）。

### P 對帳

P0（單 cell）與 P1（全 30 config）對帳（`records/2026-07-08-04`、`records/2026-07-09-01`）：全 30 config 之
量測對帳 scalar（precision、coverage 之 DINOv2 與 Inception 兩側、char_clean_fid、near_boundary_frac、
label_noise_excess_mean）逐位重現凍結 JSON；k=5 獲 P0 探針反證支持。TSTR 依協定含未種子化 shuffle、非決定性、
不在對帳集、不宣稱逐位重現。

## C 批補遺（exploratory，非判決輸入，禁因果）

以 P1 落盤資產計算，標 exploratory；不改三判決（`records/2026-07-09-03` 不回改）。觀察性描述、禁因果句。
噪聲脈絡：C4 測得 σ_cls=2.963pp，下列單 seed／少重訓之差異多在 1–2 σ_cls 內。

- **C4 變異分解**（`records/2026-07-09-07`）：σ_cls=2.963pp（分類器訓練變異，主導）對 σ_gen=1.182pp（生成
  變異）；上升肢 w1→w1.5（+2.52pp）小於 σ_cls、維持 unresolved。餵 D4 功效規劃。
- **C7 small-probe FID 穩定性**（`records/2026-07-09-06`）：Kendall τ 最小 0.911（100/class）→ 1.000
  （500/class），FID-argmin 於 15/15 個 probe draw 全落 w1.5。依凍結判定 FID-min baseline 於小 probe 可靠，
  餵 D5。
- **C2/C3/C5 介入式證據**（`records/2026-07-09-08`）：C2 near-boundary 移除之 TSTR 掉幅未大於等數隨機移除；
  C5 TSTR 未隨 near-boundary 純度單調上升；C3 降 w2.5 coverage 至 w1 水準之 TSTR 代價略大於隨機剪枝，但
  w2.5≈w1 之等 TSTR 使橋接複雜。三項未對 near-boundary 機制提供強支持，與 C1 反證方向一致；機制之
  confirmatory 回答移 CIFAR-100（D3）。
- **C8 Pareto 失明引理**（`docs/c8_pareto_blindness.md`）：w2.5 嚴格支配三 oracle → CaF 結構性選不到 oracle。

## 待確認
- **CIFAR-100（主線）**：coverage 主導是否複製（難集不翻轉），near-boundary 機制是否可量測（不飽和）；
  which-FID 是否於更難資料集才分離。
- CaF 與簡化 Chamfer 在 matched-budget 下的對決與可組合性展示（D5/D7 三臂：FID-min／CaF／Chamfer）。
