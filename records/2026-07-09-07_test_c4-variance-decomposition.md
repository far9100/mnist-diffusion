<!-- 用途：C4 變異分解執行與結果。σ_cls（分類器訓練變異）對 σ_gen（生成變異），交付 D4 功效規劃。依凍結規則 records/2026-07-06-05 §6 C4、-09-04 §2.2。gen 用 P1 落盤影像（禁重生成）。exploratory（C0）；上升肢 unresolved 為事前寫死之預期。 -->

# C4：變異分解（σ_cls 對 σ_gen）

## Goal

依凍結規則 `records/2026-07-06-05` §6 C4 與執行指令（二）`records/2026-07-09-04` §2.2，分離分類器訓練變異
σ_cls（同一合成集、不同訓練 RNG）與生成變異 σ_gen（不同 gen seed），交付 D4 決策單之功效規劃。exploratory
（C0）：不因結果回改；上升肢 unresolved 為事前寫死之預期。gen 用 P1 落盤影像（`results/p1_assets`），禁重生成。

凍結設計：w ∈ {1, 1.5, 2, 2.5} × seed ∈ {10, 11, 12} = 12 cells；每 cell +2 新 TSTR 重訓（fresh ResNet18、
未種子化 shuffle、15 epoch、augment on，同 confirmatory 協定）；混池規則——P1 全逐位（`records/2026-07-09-01`）
→ 凍結 confirmatory TSTR 計為第 3 replicate，3/cell、within df = 24。2 層 nested random-effects ANOVA。

## Result

| 量 | 值 |
|---|---|
| σ_cls（分類器訓練變異） | **2.963 pp** |
| σ_gen（生成變異） | **1.182 pp** |
| MS_within（rep 內，df=24） | 8.782 |
| MS_between_seed（w 內 seed 間，df=8） | 12.974 |

σ_cls² = MS_within；σ_gen² = (MS_between_seed − MS_within) / 3。**σ_cls 主導**（約 2.5× σ_gen）：分類器
訓練噪聲大於生成噪聲。

per-cell replicates（第 0 為凍結 confirmatory TSTR，後 2 為新重訓）：

| w | seed10 | seed11 | seed12 |
|---|---|---|---|
| w1 | [60.49, 60.02, 52.14] | [67.52, 56.81, 64.72] | [61.47, 65.40, 64.66] |
| w1.5 | [58.71, 60.84, 67.04] | [67.24, 64.26, 66.16] | [65.94, 64.33, 61.43] |
| w2 | [61.16, 60.19, 59.01] | [62.78, 57.51, 57.54] | [59.54, 59.89, 61.70] |
| w2.5 | [60.62, 56.56, 57.63] | [62.49, 61.42, 58.87] | [60.45, 57.42, 59.31] |

w-均值：w1 61.47、w1.5 63.99、w2 59.92、w2.5 59.42。

**上升肢 unresolved（事前寫死之預期，達成）**：w1→w1.5 之 w-均值差 +2.52pp 小於 σ_cls（2.96pp）；per-cell
replicate 跨度大（如 w1 seed10 [60.49, 60.02, 52.14] 跨約 8pp）。cell-均值（3 rep）之等效噪聲
√(σ_gen² + σ_cls²/3) ≈ 2.08pp，與 w1→w1.5 上升肢同量級，故上升肢於 3 gen seeds 下不可判定，與重訓次數
無關（σ_cls 主導）。此與 dossier 乙-6（配對差 +0.80±SE1.88）一致。

## Follow-up

- **餵 D4 決策單（δ，`records/2026-07-09-04` §3.1）**：σ_cls=2.963、σ_gen=1.182 供更新 MDE 表；σ_cls 主導
  之意涵（同一 cell 增重訓比增 seed 更能定 TSTR）供 D4 功效規劃；「若 CIFAR-100 σ_gen 同量級，3 seeds 定
  不住峰位 ±1 格 → 登記主張須為峰位噪聲穩健形式（高原＋懸崖，不押點峰）」之判斷於 D4 決策單依 CIFAR-10
  σ 量級外推，數字門檻由作者填。
- C4 為描述性變異交付，不下判決；與判決一/三獨立。不觸凍結 JSON、無數字回改。原始輸出
  `results/cifar10_c4_variance.json`。
