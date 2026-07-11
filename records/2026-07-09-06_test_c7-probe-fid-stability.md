<!-- 用途：C7 small-probe FID 排序穩定性之執行與結果。依凍結規則段 records/2026-07-06-15、-09-04 §2.1。gen 用 P1 落盤影像（禁重生成）。exploratory（C0）；數字餵 D5（matched-probe FID-min baseline），D5 規格數字於 δ 決策單。 -->

# C7：small-probe FID 排序穩定性

## Goal

依凍結規則段 `records/2026-07-06-15`（DINOv2 揭盲前凍結、規則先於計算）與執行指令（二）`records/2026-07-09-04`
§2.1，量 clean-fid 對 guidance 組態之排序在小 probe 下是否穩定，餵 D5 之 matched-probe FID-min baseline。
exploratory（C0）：不因結果回改。gen 用 P1 落盤影像（`results/p1_assets`，seed 10 之 10 組態），禁二次重生成。

凍結設計：probe {100, 250, 500}/class、每尺寸 5 draws（seed 事前定 draw 0..4）；穩定性度量＝小 probe 排序對
全參考排序之 Kendall τ、並報 FID-argmin 是否隨 probe 尺寸改變。判定（事前）：τ 高且 FID-argmin 穩定 →
FID-min baseline 於小 probe 可靠（餵 D5）；否則須報 probe 敏感度。

## Result

全參考排序（seed 10 之 char_clean_fid，gen vs 全 CIFAR train，凍結 JSON）：argmin = **w1.5**
（值 [10.93, 8.77, 10.18, 12.26, 14.56, 18.95, 22.50, 25.84, 28.62, 31.38] 對 grid [1..8]）。

小 probe 結果（clean-fid，cleanfid clean 特徵層）：

| probe/class | 平均 Kendall τ | 最小 τ | FID-argmin 命中全參考 | argmin 各 draw |
|---|---|---|---|---|
| 100 | 0.956 | 0.911 | 5/5 | 全 w1.5 |
| 250 | 0.991 | 0.956 | 5/5 | 全 w1.5 |
| 500 | 1.000 | 1.000 | 5/5 | 全 w1.5 |

**裁定（依凍結判定）**：Kendall τ 高（最小 0.911＠100/class、隨 probe 增至 1.000）且 FID-argmin 於 15/15
個 probe draw 全落 w1.5（與全參考一致、完全穩定）。兩條件滿足 → **FID-min baseline 於小 probe 可靠**。
小 probe 之 FID 絕對值較高（100/class 約 36–56 對全參考約 9–31）為小樣本 FID 正偏誤，屬預期、不影響排序。

**良性重複計數註記**：cleanfid 於 Windows（不分大小寫 NTFS）以大小寫兩種副檔名 glob，每張 .png 被數兩次
（特徵陣列列數為影像數 2 倍）。此對 FID 數學上無影響：精確複製每樣本使 μ 不變、Σ 僅縮放
(2N−2)/(2N−1)≈1，且對 gen 與所有 probe 一致，故 FID 值與排序完全保留；小 probe 之共變異秩虧（本測之標的）
因唯一樣本數不變而保留。保留 folder-glob 之忠實 clean-fid 前處理（換 tensor 路徑反可能失真），數字有效。

## Follow-up

- **餵 D5（δ 決策單，`records/2026-07-09-04` §3.1）**：C7 判定 FID-min baseline 於小 probe 可靠；D5 之
  matched-probe FID-min baseline 規格數字（probe 尺寸選擇等）於 D4/D8 決策單一併備，作者填。「τ 高」之
  數值門檻若 D5 需明訂，於決策單設；本檔 τ 最小 0.911、argmin 15/15 穩定，明列供判。
- 與判決三 FID-min 對決（C6，全參考）並讀但獨立：C6 為全參考 regret、C7 為小 probe 排序穩定性。
- 本檔不觸凍結 JSON、無數字回改。原始輸出 `results/cifar10_c7_probe_fid_stability.json`；資產
  `results/p1_assets/`（不入 git）。
