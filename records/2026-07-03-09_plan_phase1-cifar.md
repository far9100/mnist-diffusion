# Phase 1 Plan — CIFAR Pivot（含 backbone fork 決議）

> **給 Claude Code 的說明**：本文件記錄 Phase 0→1 檢視後的**已決策**與 Phase 1 執行順序。
> 它**解掉**了先前卡住的 backbone/guidance fork——請依此執行，**不要再停下來問 backbone 該走哪條**。
> 與 `records/2026-07-03-07_plan_research-revision-brief.md` 或原 roadmap 衝突處，以本文件為準。本檔位於 `records/2026-07-03-09_plan_phase1-cifar.md`。

---

## 0. TL;DR（本次決策）
- **Phase 0 通過**：CaF 在全 3D 網格 regret 0.16pp / rank 2-of-18 / top-3；C1 三主張全部成立；機制方向正確。
- **Backbone fork 已決**：**自訓 CFG-capable CIFAR 模型**為主軸（與已驗證 sandbox 同旋鈕）；**預訓練 EDM 重現 FID≈1.79** 只當量測堆疊錨點；**autoguidance/ICG on EDM** 當第二 guidance 方法。**EDM2/ImageNet 延後**。
- **下一步順序**：先跑量測堆疊 gate（最便宜）→ 立 CFG backbone（背景訓練 ~2 天）→ CIFAR-10 複製 C1 → **CIFAR-100 拿 C2 機制證據（最重要）** → CaF vs Chamfer 對決。
- **現在就啟動**：multi-seed Gate A（seeds 0 1 2, ~3h，平行不搶資源）。

---

## 1. 現況解讀（Phase 0）

### 已被證實（sandbox，全 3D 網格）
- **C1-a**：FID 最佳 ≠ TSTR 最佳。best FID `s50_eta0_g2` (FID 8.81)；best TSTR `s50_eta0_g1` (97.33)。
- **C1-b（差異化軸）**：**η 買保真度、買不到效用**。`s50/g3`：η 0→1 → FID 20.4→17.2，TSTR 平在 ~95。
- **C1-c（新賣點）**：**few-step 對效用幾乎免費**。`g1`：50→10 步 TSTR 97.33→96.76，但 FID 惡化更多。
  → CaF 傾向選便宜組態（挑 `s20` 而非 oracle `s50`），pp regret **低估**其實用價值。**升級為獨立賣點：「CaF 選對又選便宜」。**
- **機制**：coverage 追蹤 TSTR、precision 不追蹤（全網格確認）。
- **CaF**：3D 網格 regret 0.16pp、rank 2/18、top-3 hit；TSTR-free τ、τ-robustness、regret@selected 均已在 `selector.py` 落地。

### sandbox 到此為止的邊界（下一階段要關的洞）
- **[關鍵] MNIST margin 飽和**（real near-boundary frac ~0.6%）→ **C2「高 guidance 抽乾決策邊界」在 MNIST 無空間顯現，必須靠 CIFAR-100**。這是 Phase 1 最優先的科學缺口，**優先於 CIFAR-10 的 C1 複製**。
- CaF regret 從單軸 0.000 升到 3D 0.16pp → 不是完美 oracle；**要盯難度上升時 regret 是否擴大**。
- 一切「可發表」的證據仍在 CIFAR 之後——sandbox 只完成了機制與 harness 的 de-risk。

---

## 2. Backbone / Guidance fork —— 已決議

**背景問題**：整條 guidance 軸研究的是 CFG；CFG 需 label-dropout 的無條件路徑；標準預訓練 **EDM CIFAR 條件 checkpoint 沒有這條路徑**（著名 FID 1.79 就是無 CFG 的條件模型）。所以「直接載 EDM」給不了 CFG 旋鈕。**CIFAR 的 CFG 沒有單一權威預訓練 checkpoint。**

**決議（把難題變成「跨 guidance 方法的一張圖」，反而是相對 Chamfer/Fan 的加分）：**

| 角色 | 選擇 | 理由 |
|---|---|---|
| **主 guidance 軸（與 sandbox 連續）** | **自訓 CFG-capable CIFAR-10 模型**（沿用現有訓練 code + label dropout；或叉 `FutureXiang/Diffusion` 類 repo） | 保住與已驗證 sandbox **相同旋鈕語意**；與必打贏的 Chamfer/Fan **同軸可比**。單卡 ~2 天，可背景跑 |
| **量測堆疊錨點** | **預訓練 EDM，重現 FID ≈ 1.79** | 用公認數字驗證 clean-fid/PRDC/FD-DINOv2 **量測正確性**；把「pipeline 對不對」與「我的 CFG 模型合不合理」解耦。**不拿它做 CFG** |
| **第二 guidance 方法** | **autoguidance / ICG on 預訓練 EDM** | 給「C1 跨 guidance 方法」（非只跨 scale）；autoguidance 的弱模型可用自訓早期 checkpoint；ICG 對「未用 dropout 訓練的條件模型」免訓練即得 guidance 旋鈕。直接餵 C3b |
| **延後** | **EDM2 / ImageNet** | 違反單卡優先；留 Phase 3 / stretch，現在不叉 |

**Fallback（僅在 ~2 天訓練排不進預算時）**：**ICG-only on 預訓練 EDM**——零訓練、純預訓練 backbone，代價是旋鈕語意與 sandbox 的 CFG 略異，需在文中交代此 confound。

**實作備註**
- 自訓：優先評估直接叉 `FutureXiang/Diffusion` / `zhaisf/cDiffusion-cifar10`（極簡 CFG CIFAR，`cifar_guide.py`，有回報 FID）；能拿到其 checkpoint 就免訓。
- 目標 FID **不需 SOTA**：研究的是組態間的**相對**差異，FID ~5–10 級即可用；但仍要落在該類模型的合理範圍以避免「home-grown artifact」質疑——用預訓練 EDM 的 1.79 錨點 + repo 回報值雙重佐證。
- 訓練即存**早期 checkpoint**（供 autoguidance 當弱模型用）。

---

## 3. Phase 1 執行順序（最便宜、最高價值先做）

1. **[gate, 最便宜] 量測堆疊正確性**
   - 載預訓練 EDM CIFAR-10，`clean-fid` 重現 **FID ≈ 1.79**（correctness gate，先過再掃）。
   - 跑通 `metrics_prdc.py` 的真實 PRDC + `dgm-eval` 的 FD-DINOv2。
   - `fid.py` 由 MNIST-FID 升級為 clean-fid / FD-DINOv2 介面。新增 `backbones/`（EDM 適配，經 `diffusers`/`k-diffusion`）、`datasets/`（CIFAR-10/100）。

2. **[背景] 立 CFG backbone**
   - 叉 repo 取 checkpoint，或啟動一次背景訓練（~2 天）。訓練期間不阻塞第 1、5 步的準備。
   - 存早期 checkpoint 供 autoguidance。

3. **CIFAR-10 複製 C1**
   - CFG 模型就緒 → 跑 3D `(steps×η×guidance)` 曲面（`run_comparison.py`），每類數千張真實生成、CIFAR ResNet 量 TSTR、clean-fid/FD-DINOv2/PRDC 並報。
   - **go/no-go**：coverage 主導效用在 CIFAR-10 複製；**η-null 是否複製（要驗不要假設，見 §5）**。

4. **[最重要] CIFAR-100 拿 C2 機制證據**
   - margin 未飽和 → `mechanism.py`：near-boundary 支持度 vs guidance、**coverage 受控下的 margin 條件分析**（介入式，非只給單調曲線）。
   - **go/no-go**：coverage 主導在更難的 100 類**不翻轉**（硬門檻）；margin 抽乾曲線隨 guidance 單調。

5. **CaF vs Chamfer 對決**
   - 實作簡化 Chamfer 基線，`run_guidance_study.py` 走 **matched-budget** 頭對頭（pre-registered 協定 + compute 帳本）。
   - selector 用第二表徵（CLIP/Inception）重算 coverage 做 robustness（破 DINOv2 雙重使用循環）。

---

## 4. Phase 1 go/no-go（取代舊的全域 Spearman）
- CIFAR 上 **regret@selected 低 + top-k 命中**（非全域 ρ）。
- **coverage 主導效用跨 CIFAR-10 與 CIFAR-100 都成立**（難集不翻轉＝硬門檻）。
- CIFAR-100 上看得到 **margin 抽乾**（介入式證據）。
- CaF 在 matched-budget 下**打平或贏 Chamfer**（帶多 seed CI / 顯著性）。
- 正確性 gate：新 backbone 先重現其 FID 再掃描。

---

## 5. Watch-list（Phase 1 要主動盯的風險）
- **別假設 η-null 複製到 CIFAR**：CIFAR 紋理多，stochastic churn 的誤差校正可能比 MNIST 更有作用。DP 擴散先導偏向它會複製，但這是**要實證驗、不是假設**的。
- **難度上升時 CaF regret 是否擴大**：CIFAR-100 是壓力測試。
- **競爭機制（標籤噪音）**：低 guidance 在難集可能引入離類/模糊樣本，壓過多樣性好處 → 可能使 coverage 主導翻轉。這正是第 4 步要盯的。
- **τ 循環 / DINOv2 雙重使用**：見 `records/2026-07-03-07_plan_research-revision-brief.md` §3，實作要求不變。

---

## 6. 現在就啟動（this session）
1. **launch multi-seed Gate A**：`run_selector_signal.py --seeds 0 1 2 --per-digit 1000`（~3h，平行、不搶資源），替 sandbox 三主張補 CI。
2. 依 §2 決議，開始 §3 第 1 步（量測堆疊 gate）與第 2 步（CFG backbone 準備）。

---

## 7. 決策所依據的關鍵事實（供查證）
- EDM CIFAR 官方 checkpoint 為無 CFG 條件模型；CFG on CIFAR 需自訓 label-dropout。CIFAR 無單一權威 CFG 預訓練 checkpoint。
- 極簡 CFG CIFAR repo：`FutureXiang/Diffusion`、`zhaisf/cDiffusion-cifar10`（35.7M UNet ~14h on 4×3080ti，有回報 FID）。單卡自訓到堪用 FID 約 ~2 天。
- **ICG（Independent Condition Guidance）**：對未用 dropout 訓練的條件模型，免訓練即得 guidance 旋鈕（負項取隨機條件）。
- **autoguidance（Karras 2024, NeurIPS）**：用同模型弱版本（早期 checkpoint / 小模型）當負項，不需 CFG-dropout；需超參搜尋。
- 相關 arXiv：Chamfer Guidance 2508.10631；deliberate-practice 2502.15588；DPDM 2210.09929（η↔下游先導）；interval-guidance / autoguidance NeurIPS 2024。

## 8. 一句話
> backbone 難題已轉為「跨 guidance 方法的一張圖」＝相對 Chamfer/Fan 的加分。主軸自訓 CFG（連續性），EDM 當量測錨點＋第二方法，ImageNet 延後。先過量測 gate，機制證據靠 CIFAR-100。
