<!-- 用途：D8 第三訊號之 C0 規則計算——recall/density 是否提供 (precision,coverage) 以外之非支配訊號，供 CaF-v2 脫離 Pareto 失明。規則凍於 records/2026-07-09-11 §5（作者 -09-04 後續點頭啟動）。離線自 P1 落盤 DINOv2 特徵、零 GPU、標 exploratory（C0），CIFAR-10=discovery。依 -09-11 §5、C8。 -->

# C0：recall/density 第三訊號（D8 v2 候選）

## Goal

依 D8 決策單 `records/2026-07-09-11` §5 之凍結 C0 規則（作者已裁 D8＝留、第三訊號用 recall/density、啟動
C0），離線自 P1 落盤 DINOv2 特徵（`results/p1_assets`）重算 per-config recall、density（compute_prdc_per_class，
k=5，同 confirmatory 口徑，3 seed 均值），套事前判定：recall/density 是否提供 (precision, coverage) 以外之
非支配訊號——即某 oracle 於 recall 或 density 是否 ≥ 支配組態 w2.5（若是＝該訊號可讓 v2 脫離失明）。
零 GPU、標 exploratory（C0）；CIFAR-10＝discovery，不入裁決、不因結果回改。

## Result

per-config（3 seed 均值，DINOv2 空間，k=5）：

| w | precision | coverage | recall | density |
|---|---|---|---|---|
| 1 | 0.806 | 0.645 | **0.579** | 1.015 |
| 1.5 | 0.841 | 0.751 | **0.555** | 1.057 |
| 2 | 0.858 | 0.777 | **0.528** | 1.106 |
| 2.5 | 0.873 | 0.792 | 0.493 | 1.138 |
| 3 | 0.876 | 0.778 | 0.459 | 1.143 |
| … | | | | |
| 8 | 0.836 | 0.559 | 0.247 | 0.885 |

**支配檢查（w2.5＝Pareto 失明之支配組態，vs 三 oracle w2/w1.5/w1）**：

- precision：w2.5 (.873) > 三 oracle → 支配。
- coverage：w2.5 (.792) > 三 oracle → 支配。
- **recall：w2.5 (.493) < 三 oracle（w2 .528、w1.5 .555、w1 .579）→ 不支配（oracle 皆較高）。**
- density：w2.5 (1.138) > 三 oracle → 支配。

**裁定（依事前判定）**：**recall 提供非支配訊號**——recall 隨 guidance 單調下降（.579→.247），偏好低
guidance（oracle 所在），w2.5 於 recall 被三 oracle 全數超過。故納入 recall 之 selector 不再被 w2.5 之
(precision, coverage, density) 支配所迫，**可脫離 Pareto 失明**。**density 無效**——w2.5 於 density 仍支配三
oracle，density 與被支配面冗餘。

## Follow-up

- **餵 D8 v2 規格**：CaF-v2 之第三訊號採 **recall（非 density）**。此為 discovery（CIFAR-10）之必要條件觀察
  （recall 打破 w2.5 支配），**非充分**：v2 selector 之具體形式、及其 matched-budget regret 是否勝 FID-min
  ≥1.5pp（D4 候選 B、X=1.5），為 CIFAR-100 之 validation 問題（D8 discovery/validation split）。
- v2 selector 候選形式（供 D 包）：如 `argmax recall s.t. precision ≥ τ`，或 (coverage, recall) 之非單調
  組合；具體形式與門檻隨 D 包登記、CIFAR-100 驗。CIFAR-10 上永遠 exploratory。
- 不觸凍結 JSON、無數字回改。原始輸出 `results/cifar10_recall_density_c0.json`。
