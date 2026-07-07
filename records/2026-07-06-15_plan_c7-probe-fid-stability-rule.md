<!-- 用途：C7（small-probe FID 排序穩定性）規則段——測 clean-fid 對組態之排序在小 probe 下是否穩定，餵 D5 之 matched-probe FID-min baseline。計算前凍結。exploratory（C0），計算留 P1/β。依 records/2026-07-06-05 §6 C7、D5。 -->

# C7：small-probe FID 排序穩定性規則段

## 目標

D5 之 matched-probe FID-min baseline 需知：clean-fid 對組態之排序在小 probe（少量真實參考）下是否穩定。
C7 量此穩定性、餵 D5。規則先於計算。exploratory（C0）。

## 結果（規則，凍結）

- **干預**：真實參考子抽樣至小 probe 尺寸（凍：{100, 250, 500}/class），各尺寸重抽數次（凍：每尺寸 5 次、seed 事前定），
  重算各組態 clean-fid、得排序。
- **穩定性度量（凍）**：小 probe 排序對全參考排序之 Kendall τ；並報 FID-argmin 是否隨 probe 尺寸改變。
- **判定（事前）**：τ 高且 FID-argmin 穩定 → FID-min baseline 於小 probe 可靠（餵 D5）；否則 → D5 之 matched-probe
  baseline 須報其 probe 敏感度。
- **計算**：需各組態落盤生成集，留 P1/β；禁第二次重生成。exploratory。

## 後續

P1/β 後計算，結果餵 D5（matched-probe FID-min baseline）。與判決三 FID-min 對決（C6）並讀但獨立（C6 為全參考、
C7 為小 probe 穩定性）。
