<!-- 用途：記錄 Gate A（投入 CIFAR 前的判定）三準則的目標、結果與後續。 -->

# 2026-07-03 Gate A 判定

## 目標

在投入 EDM/CIFAR 基礎建設前，用 MNIST 與現有資產以最低成本判定三個準則：對必打贏對手 Chamfer 是否還有差異空間、選擇器訊號是否穩健、機制方向是否正確。三者皆過才繼續。

## 結果

三準則通過。

準則 a（相對 Chamfer 的差異空間）：詳見 `2026-07-03-05_proofread_chamfer-positioning.md`。結論是有空間但賣點需轉向，核心差異為可組合性、免任務分類器的操作點、以及為選擇證明的機制。

準則 b（多 seed 選擇器訊號）：DDIM、η=0、steps=50，guidance 網格 {1,2,3,5,7,10}，每類 1000 張，seeds {0,1,2}，auto-τ = 0.9 乘以真實對真實的參考 precision。

| g | precision | coverage | TSTR% | near-boundary 比例 |
|---|---|---|---|---|
| 1 | 0.916±0.003 | 0.875±0.006 | 97.30±0.15 | 0.0269 |
| 2 | 0.961±0.001 | 0.851±0.003 | 96.28±0.06 | 0.0008 |
| 3 | 0.966±0.001 | 0.732±0.004 | 95.01±0.20 | 0.0001 |
| 5 | 0.964±0.001 | 0.556±0.005 | 92.31±0.06 | 0.0 |
| 7 | 0.951±0.003 | 0.436±0.004 | 88.85±0.16 | 0.0 |
| 10 | 0.919±0.001 | 0.319±0.005 | 78.51±1.00 | 0.0 |

CaF 在三個 seed 全部選中 g1，oracle TSTR 最佳也都是 g1；regret 為 0.000±0.000 pp、rank 1/6、top-3 命中 100%。信賴區間很緊，coverage 與 TSTR 的排序跨 seed 從不翻轉，代表「效用最佳組態不等於保真最佳組態」不是單 seed 雜訊。coverage 隨 guidance 單調下降、TSTR 也單調下降、兩者近乎同向；precision 不追蹤 TSTR（在 g3 達峰，TSTR 在 g1 達峰），即保真最佳與效用最佳落在不同組態，這是相對 Chamfer 在機制層的乾淨差異。

誠實的 τ 註記：auto-τ 約 0.857，每個 seed 都落在 g1（regret 0），但 τ 掃描的眾數選擇是 g2（穩定度約 0.79）。若 τ 抬高到 g1 的 precision（約 0.916）之上，g1 被排除、改選 g2（regret 約 1.0pp）。故 g1 與 g2 對 τ 敏感，但兩者皆在前二名、robust 帶內 regret 不超過 1.3pp。此點如實記錄。

準則 c（機制方向）：near-boundary 比例隨 guidance 單調枯竭（0.027 到 0），label noise 不是主因。但 MNIST 近乎可分，真實資料的 near-boundary 比例僅 0.0087，生成資料在 g2 起就趨近 0，能被枯竭的邊界質量很少。因此 MNIST 上機制主要靠「coverage 與效用同向、precision 與效用脫鉤」承載；真正的 margin 枯竭證據需要更難、非可分的 CIFAR-100。另註：g1 的 label noise 最高（0.041）卻有最佳 TSTR，代表 coverage 的好處在 MNIST 上壓過 label noise 成本，此細節需交代而非隱藏。

## 後續

判定為可繼續。進入 Phase 1（見 `2026-07-03-09_plan_phase1-cifar.md`）：EDM CIFAR-10 FID 重現（已完成，見 `2026-07-03-03_test_edm-fid-gate.md`），再做 CaF 與簡化 Chamfer 的對決，以及 coverage 主導在 CIFAR-10 與 CIFAR-100 的複製，機制證據在 CIFAR-100 上才能完整呈現。重現指令：`uv run python run_selector_signal.py --seeds 0 1 2`。
