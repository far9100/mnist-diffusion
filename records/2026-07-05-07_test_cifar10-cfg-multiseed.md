<!-- 用途：記錄 Stage 4（自訓 CFG CIFAR-10 多 seed 全量）的目標、結果、機制判讀與後續。 -->

# 2026-07-05-07 自訓 CFG CIFAR-10 多 seed 全量

## 目標

在 Stage 3 定死的 grid（w∈{1,2,3,4,5,8}，固定 steps=50 eta=0）上跑 3 seed，取 confirmatory 主結果：
每組 precision/coverage（DINOv2 主 + Inception 交叉）/TSTR/標籤噪音帶信賴區間，並以 CaF 在完整網格
（不剪枝）計 regret@selected / rank / top-k。

## 結果

3 seed 彙總（mean ± std）：

| w | precision | coverage(DINOv2) | TSTR% | label_noise | near_bnd |
|---|---|---|---|---|---|
| 1 | 0.827±0.009 | 0.710±0.006 | 41.18±1.15 | 0.112±0.004 | 0.252±0.008 |
| 2 | 0.884±0.004 | 0.834±0.009 | 46.01±1.85 | 0.013±0.003 | 0.063±0.005 |
| 3 | 0.903±0.004 | 0.834±0.002 | 43.88±1.49 | 0.005±0.002 | 0.037±0.004 |
| 4 | 0.903±0.003 | 0.806±0.005 | 40.59±0.91 | 0.003±0.002 | 0.032±0.001 |
| 5 | 0.897±0.001 | 0.758±0.008 | 37.43±2.39 | 0.003±0.001 | 0.031±0.002 |
| 8 | 0.871±0.003 | 0.622±0.012 | 28.43±0.70 | 0.008±0.002 | 0.055±0.004 |

CaF：每 seed 選 [w2, w3, w2]（modal w2, 67%）；oracle TSTR-best 三 seed 全 w2；regret@selected
0.280±0.485 pp（max 0.840）；rank [1,2,1]；top-3 命中 100%。

## 機制判讀

- 內部最優確認：TSTR 峰在 w=2，三 seed oracle 一致。效用對 guidance 非單調、有甜蜜點，在自訓主軸站住。
- CaF 免訓練找到甜蜜點：regret 0.28pp、top-3 100%，H2 在 CIFAR-10 確認。
- coverage 驅動效用、precision 不驅動（MNIST 機制複製）：precision 峰在 w3–4（0.903）但 TSTR 峰在
  w2（precision 不追 TSTR）；coverage 峰在 w2–3、與 TSTR 同步（coverage 追 TSTR）。C2 核心在 CIFAR-10 成立。

## 對先前 C2 低段假設的更正（重要）

早期預警（EDM proxy、w∈[1,3]）曾提「C2 coverage 驅動效用在低段要改寫」。完整寬 grid + 多 seed 後
資料不支持該改寫：早期預警的「coverage 平坦、不驅動」是範圍太低的假象。掃寬後 coverage 非單調
（w1→2 升、w2→8 崩）且確實追蹤 TSTR（兩者皆 w2 達峰）。故：

- C2「coverage 驅動、precision 不驅動」在 CIFAR-10 仍成立。
- 相對 MNIST 的真正差異是 coverage 對 guidance 非單調（有甜蜜點），而非單調遞減。
- label_noise 是低段的額外競爭力（w=1 有 11% 離類），與「w=1 coverage 已較低」共同壓低 w=1 效用，
  屬低段附加懲罰、非「coverage 不驅動」。

正確的 exploratory 修訂：C2 核心存活；CIFAR 的新意是「coverage 非單調 → 甜蜜點」＋「低段 label-noise
附加力」。此修訂仍為 exploratory，最終 confirmatory 裁決以此多 seed 資料與後續 CIFAR-100 為準。

## go/no-go

- CIFAR-10：regret@selected 0.28pp（低）+ top-3 100% + coverage 追蹤效用（含非單調）——通過。
- 未結：coverage 主導須在 CIFAR-100 也成立（難集不翻＝硬門檻）；CIFAR-100 的 near-boundary 未飽和，
  才是機制承重牆。matched-budget 對 Chamfer 的對決亦待做。

## 後續

- Stage 5：更新 docs/results_analysis.md 與 README 的機制定位（coverage 驅動 + 非單調甜蜜點 + label-noise
  低段附加力；CaF 找甜蜜點）。
- 之後：CIFAR-100（機制承重牆 + 翻轉硬門檻）、CaF vs 簡化 Chamfer matched-budget 對決。
