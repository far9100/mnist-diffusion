<!-- 用途：Phase 1 的預先登記（pre-register）實驗協定，於任何 CIFAR 掃描結果進來前鎖定。 -->

# 2026-07-05-02 Phase 1 預先登記協定

## 目標

在看到任何 CIFAR 組態掃描結果之前，把實驗設計、指標、go/no-go 門檻、matched-budget 定義與
分析決策全部寫死，避免事後選擇性分析（p-hacking / HARKing）以及被 reviewer 質疑協定為結果量身
訂做。本協定一旦鎖定，confirmatory 部分不得再改；任何偏離都須在最終記錄中明列並標為 exploratory。
鎖定時點：CIFAR-10 完整網格 TSTR 首次執行之前（即目前 CIFAR-10 訓練收斂前的空窗內）。

沿用不變的研究決策見 records/2026-07-03-09_plan_phase1-cifar.md 與
records/2026-07-05-01_plan_phase1-execution.md。

## 協定（鎖定內容）

### 1. 研究問題與假設

- H1（機制，confirmatory）：驅動下游效用（TSTR）的是 coverage（多樣性）而非 fidelity（precision）。
- H2（選擇器，confirmatory）：免訓練的 CaF 能在候選組態中選到近效用最佳者（regret@selected 低）。
- H3（對決，confirmatory）：在 matched-budget 下，CaF 選 vanilla 組態的下游準確率打平或優於（簡化）
  Chamfer，且 CaF 的「選組態」成本顯著較低。
- 探索性（exploratory，不受本協定約束、須明確標示）：η 在 CIFAR 的行為、guidance 內部最優是否存在、
  第二 guidance 方法（autoguidance/ICG）的表現。

### 2. 資料集與 backbone

- 資料集：CIFAR-10 與 CIFAR-100。CIFAR-100 為機制承重牆與翻轉檢查的難集。
- 主 guidance 軸 backbone：自訓 CFG-capable 擴散模型，取樣用 DDIM(η)（ddpm.ddim_sample_loop）。
- 量測錨點：預訓練 EDM，僅用於驗證量測流程（已過 FID 1.848 vs 1.79），不用於 CFG。
- 第二 guidance 方法：autoguidance / ICG（exploratory）。
- ImageNet 延後，不在本協定範圍。

### 3. 取樣組態網格

- 三軸 steps × η × guidance，於自訓 CFG 模型上以 DDIM(η) 掃描。
- 各軸候選值於 CIFAR-10 模型收斂後、掃描開始前一次定死並記錄；定死規則：steps 取涵蓋 few-step 到
  full 的等比點、η 取 {0, 中, 1}、guidance 取涵蓋無 CFG 到明顯崩多樣性的範圍。定死後不得增刪點以
  迎合結果。
- 預覽一律使用與最終跑相同的 sampler 與步數，只以「減類別數或減 seed」加速，per-class 樣本維持在
  消除 coverage 樣本數假影的水準（約 1000，見 records/2026-07-03-02）。

### 4. 指標與特徵空間

- 主要效用指標：TSTR（僅用合成圖訓練下游分類器、於真實測試集量準確率），下游分類器與訓練設定固定。
- 多樣性/保真度：PRDC 的 coverage（主要多樣性把手）與 precision（fidelity 下限），metrics_prdc。
- 保真度對照：clean-fid（Inception 錨點）與 FD-DINOv2，僅作報告與對照，不作為選擇或 go/no-go 主指標。
- 特徵空間：selector 的 coverage 以 DINOv2 為主，另以 CLIP 或 Inception 特徵重算一次做交叉驗證
  （破除 DINOv2 雙重使用循環）。兩表徵結論須一致方採信。

### 5. CaF 選擇器

- 規則：在候選組態上選 argmax coverage s.t. precision ≥ τ。
- τ 決定：以真實 probe 的 real-vs-real precision 乘一比例自動決定，不依賴 TSTR（免訓練）。
- 須報告 τ 穩健性掃描與 selected config 對 τ 的敏感度；不得用 TSTR 反推 τ。

### 6. 主要結果指標與 go/no-go

- 選擇器主指標：regret@selected（選中組態 TSTR 與完整網格最佳 TSTR 之差）與 top-k 命中，取代全域
  Spearman ρ。regret 必須以完整網格（含 CaF 不會選的組態）計得，掃描期間不得先剪枝。
- go/no-go 門檻：
  1. CIFAR 上 regret@selected 低、top-k 命中。
  2. coverage 主導效用跨 CIFAR-10 與 CIFAR-100 皆成立（難集不翻為硬門檻；若翻則走第 9 節兩結果框架）。
  3. CIFAR-100 上以 coverage 受控的 margin 介入分析（mechanism.coverage_controlled_margin）顯示邊界
     抽乾，而非只單調曲線。
  4. matched-budget 下 CaF 打平或優於 Chamfer，帶多 seed 信賴區間與顯著性檢定。
  5. 正確性 gate：新 backbone 先重現其 FID 再掃描（EDM 1.848 已過；自訓 CFG 的堪用 FID 範圍於 repo 內
     以自身量測確立後方採用）。

### 7. matched-budget 定義與 compute 帳本

- 對齊軸：兩方（CaF 選 vanilla 組態 vs 簡化 Chamfer 改取樣）產出的下游訓練集「影像張數」相同、每張的
  取樣 NFE（number of function evaluations）相同、下游分類器訓練設定完全相同。
- 主要比較量：下游測試準確率（TSTR），多 seed。
- 不宣稱 CaF 取樣比 Chamfer 便宜（不同成本軸）。差異化押在「選組態成本」：另以獨立帳本呈現 CaF 的
  選組態成本（一次 probe 的 PRDC 前傳）對比 Chamfer 每步 guidance 的特徵抽取器/解碼器額外開銷。
- compute 帳本欄位（每次 run 輸出並保存）：每張影像 NFE、生成張數、特徵抽取器前傳次數、下游分類器
  訓練次數與 epoch、seed 數、wall-clock、GPU。

### 8. 分析決策鎖定

- 多 seed：所有 confirmatory 主張以 seed 數不少於 3 執行，帶信賴區間或顯著性檢定；不得單 seed argmax
  下結論。
- 停訓準則（排序穩定，非絕對 FID）：需同時滿足三條件方可提早停——FID 落在 repo 內自證的堪用範圍、
  排序在足以分辨近平手差異（記錄顯示 g1/g2 約 1pp、w1.5/w2 約 0.5pp）的精度下穩定、且比較的是充分
  暖身的 EMA(0.9999) 模型排序；任一不滿足即續訓。CIFAR-10 讓其跑完，本準則主要用於 CIFAR-100。
- 驗證期不剪枝：CIFAR-10 這個驗證資料集跑完整網格 TSTR 取誠實 regret；CaF 的省算力價值以事後反事實
  呈現，剪枝只用於部署或後續資料集，不用於當前驗證。
- CIFAR-100 範疇：先以 2 至 3 點 spot-check 確認 η 行為與 CIFAR-10 一致，再固定 (steps,η)、只掃
  guidance 軸並上機制儀器；不假設 η 遷移。

### 9. 兩結果框架（預先宣告，非事後）

CIFAR-100 若 coverage 主導不翻轉，即原 thesis（H1/H2/H3）。若翻轉，則預先宣告的替代結論為：效用最優
guidance 隨任務難度而變，存在 coverage 對標籤噪音的取捨，CaF 能逐資料集選對該操作點。標籤噪音診斷
（mechanism.analyze_dataset 的 label_noise_frac）自始內建，使兩個方向皆能即時解釋，不臨時補儀器。此
框架於看到 CIFAR-100 結果前登記，故任一走向皆非事後合理化。

### 10. confirmatory 與 exploratory 界線

第 1 節列為 confirmatory 者受本協定完全約束。其餘（η 曲線細節、內部最優、第二 guidance 方法、
ImageNet）為 exploratory，可自由分析但最終須明確標示，不得混入 confirmatory 主張。

## 後續

- 本協定於 CIFAR-10 完整網格 TSTR 首次執行前鎖定；鎖定後第 3 節的網格候選值一經定死即附於本檔末並
  不再更動。
- 掃描 harness（run_comparison / run_cifar_selector / run_guidance_study）須輸出第 7 節的 compute 帳本
  欄位。
- 若後續決策涉及 backbone 路線或範疇變更，須新增記錄說明對本協定 confirmatory 部分的影響。
