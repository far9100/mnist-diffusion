<!-- 用途：彙整並分析各階段實驗結果（量測錨點、MNIST sandbox、CIFAR-10 預覽）。 -->

# 實驗結果分析

本文件彙整目前已產出的實驗數據與其解讀。逐次執行的原始記錄見 `records/`；此處只保留可作結論或
待確認的結果分析。CIFAR-10 confirmatory 已於 2026-07-06 完成（`results/cifar10_cfg_confirmatory.json`），
其結果分析與三判決依定稿計畫 `records/2026-07-06-05` 工作包 B 定稿後補入本檔（E3），此處先不列結論。

範圍註記（依 `records/2026-07-05-12` 定位 v2）：CIFAR 尺度的主張收緊到 CFG guidance 軸。取樣步數 steps
與 DDIM 隨機性 η 的效用結論只保留在 MNIST sandbox 尺度，不在 CIFAR 尺度宣稱聯合曲面。下方 MNIST 表
為 sandbox 動機證據，CIFAR 自訓 CFG 的 pilot 與 scout 為 exploratory，confirmatory 主結果已完成（見上），
其分析待工作包 B 定稿後補入。

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

## 待確認（confirmatory）

- CIFAR-10 confirmatory（steps=50、η=0、10 點 grid、fresh seeds 10/11/12）：內部最優是否持續，CaF regret
  是否維持小；C2 全網格偏相關是否 C2a 顯著正、C2b 不顯著。
- CIFAR-100：coverage 主導是否複製（難集不翻轉），near-boundary 機制是否可量測（不飽和）。
- 第二特徵表徵交叉驗證（Inception/CLIP），破除 DINOv2 雙重使用。
- CaF 與簡化 Chamfer 在 matched-budget 下的對決與可組合性展示。
