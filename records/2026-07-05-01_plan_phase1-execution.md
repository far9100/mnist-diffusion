<!-- 用途：Phase 1 執行計畫（嚴謹綜合版），signal-forward 安全子集與完整網格誠實驗證。 -->

# Phase 1 執行計畫（嚴謹綜合版）— Sampling for Utility, not Fidelity

狀態：現行計畫（supersedes 排程優化版的排程主張；沿用其前一版之所有研究決策）
適用分支：研究主線（CaF selector）
撰寫緣由：排程優化版被 coding agent 依 repo 記錄審查後，確認其中 L4/L5/L6 以方法論嚴謹換速度、且與既有記錄或自身 Watch-list 矛盾。本版採其安全子集，保留排程好處而不賠嚴謹。

備註：本檔已依 repo 記錄體例落庫（檔名 2026-07-05-01_plan_phase1-execution.md、補檔頭、無表情符號與勾選框）。文中對「前一版 Phase 1 計畫」與「修訂 brief」的引用，對應當前樹中的 records/2026-07-03-09_plan_phase1-cifar.md 與 records/2026-07-03-07_plan_research-revision-brief.md。

---

## 0. 本版定位

排程優化版的元洞察成立：目前做法把全案生死門檻（CIFAR-100 coverage 主導是否翻轉）排在長序列鏈最末端、1000 epoch 很可能過訓、8 步預覽自製假影。但其換取速度的手段中有三條不安全：L4 使 regret@selected 循環、L5 重新引入 coverage 樣本數假影並混淆 backbone、L6 內建被 Watch-list 禁止的 η 遷移假設。本版的原則是：保留「signal-forward + 並行」帶來的時程好處，但不開任何方法論借據——凡是會導致重跑或被 reviewer 拆的捷徑一律不採。

沿用不變的研究決策（見 records/2026-07-03-09_plan_phase1-cifar.md）：backbone fork（自訓 CFG 為主軸、EDM 僅量測錨點、autoguidance/ICG 為第二 guidance 方法、ImageNet 延後）；go/no-go 以完整網格的 regret@selected 為準；τ 循環與 DINOv2 雙重使用護欄。

---

## 1. 純上手項目（無爭議，先做）

一、今天提交研究主線，消存亡風險。HEAD 目前在 experiment/var-mini，而研究主線（selector、mechanism、train_cifar、run_comparison 等）全部 untracked，一次意外即抹除數週進度。自當前狀態開研究主線分支，提交這些檔案，var-mini 留作側分支。此為擱置已久、且與任何實驗結果無關的動作，應先於一切新 run。

二、並行實作進 GPU 空窗（原 L3，保留）。GPU 的訓練與生成是唯一硬瓶頸；把純 CPU/寫程式工作（簡化 Chamfer 基線、pre-register 協定、clean-fid 的 Windows pickling 修復、mechanism 的介入分析與標籤噪音診斷）排進 CIFAR-10 收尾與 CIFAR-100 訓練的空窗。守則：GPU 永不閒置、寫程式永不擋 GPU。這是唯一「免費」把 A（算力）與 B（實作）從相加逼近取大的槓桿。

---

## 2. 修正後的加速項目（保留直覺、補足規格）

### 2.1 早期翻轉預警（原 L1，降級為「預警」而非 gate）

用現有 CIFAR-10 模型，在難子集（易混類：貓/狗、汽車/卡車、鹿/馬，margin 未飽和）量 coverage 主導是否鬆動；並在 mechanism 內建標籤噪音診斷（數個 guidance 值下量離類/模糊質量佔比）。

嚴謹界線：此訊號單向。若出現鬆動或高標籤噪音，是有效早期預警，可提早準備兩結果框架（見 2.5）；若未出現，什麼都不能清除——CIFAR-100 的機制 gate 仍須完整跑。因此本項只把「預警」提前，不把「gate」提前；時程上不得據此宣稱 gate 前移。

### 2.2 排序穩定停訓（原 L2，補條件）

相對研究看組態排序穩定、非絕對 FID，直覺正確，但節省是有條件的，需同時滿足三點才可提早停：一，量到的 FID 明確落在堪用範圍（此範圍須在 repo 內以自身量測確立，勿沿用外部單點數字）；二，排序在足以分辨近平手差異的量測精度下穩定——記錄顯示 g1/g2 常差約 1pp、w1.5/w2 約 0.5pp，故 ranking-stability 檢查本身不能用過低的樣本/seed，否則「穩定」是量測噪音；三，比較的是成熟 EMA(0.9999) 模型的排序，並確認 EMA 已充分暖身，否則提早停訓量到的排序未必等於最終模型。三者有一不滿足，就讓它續訓。CIFAR-100 同準則收斂。

### 2.3 CaF 剪枝改為事後反事實（原 L4，去循環）

驗證期不得先剪枝。在 CIFAR-10 這個驗證資料集上跑完整網格 TSTR（含 CaF 不會選的組態），以取得誠實的 regret@selected——這正是全案 go/no-go 指標，必須以真正網格最佳為基準。CaF 的省算力價值改以事後反事實呈現：從完整網格計算「若採 CaF 只跑其 probe，會省下多少、選中組態的 regret 為何」。如此既保住誠實 regret，又保留效率賣點。實際剪枝只在 CaF 於 CIFAR 尺度驗證通過之後，用於部署或後續資料集，不用於當前驗證。附帶理由：CIFAR 尺度上 CaF 尚未驗證（預覽仍曖昧），先剪枝即先射箭再畫靶。

### 2.4 預覽方法修正（原 L5，去假影、釐清 backbone）

保留「不在非研究網格上預覽」的正確直覺，但兩處修正。其一，不得為求快而砍 per-class 樣本：記錄 2026-07-03-02 已證實 coverage 對樣本數有假影（per-digit-50 非單調、mis-pick，per-digit-1000 平衡取樣後消失），而 coverage 正是 CaF 最依賴的量。要更快就砍類別數或 seed，per-class 樣本維持在消除該假影的水準（約 1000）。其二，預覽預算須以 CFG 軸的真實 backbone 定義：CFG 軸跑自訓 CFG 模型 + DDIM(η)（ddim_sample_loop），故 canonical 預覽預算應以該模型的 DDIM 步數與 η 表述，不可用 EDM 18-step Heun——EDM 的隨機性是 S_churn 而非 DDIM η，無法表達 steps×η 曲面，且 EDM 僅為量測錨點、非 CFG backbone。

### 2.5 CIFAR-100 針對性 + η 遷移 spot-check（原 L6，去內建假設）

CIFAR-10 負責畫 steps×η×guidance（C1）；CIFAR-100 的任務是 C2 機制與翻轉檢查，方向正確，不必在 100 類複製整個立方體。但收斂 (steps,η) 之前，須先在 CIFAR-100 上以 2 至 3 點 spot-check 確認 η 行為與 CIFAR-10 一致（Watch-list 明寫「別假設 η 遷移」）；確認後才固定 (steps,η)、聚焦 guidance 軸與機制儀器。如此保住範疇裁減，又不內建未驗證的遷移假設。

### 2.6 翻轉兩結果化（保留）

把 CIFAR-100 翻轉從硬 No-Go 改為兩種可發表結果：不翻，即原 thesis；翻，則結論為「效用最優 guidance 隨任務難度而變，存在 coverage 對標籤噪音的取捨，CaF 逐資料集選對該點」，是更豐富的貢獻。標籤噪音診斷（2.1）從一開始內建，使兩個方向都能即時解釋，不臨時補儀器。此項誠實且能 de-risk thesis，予以保留。

---

## 3. 建議執行順序（本 session 起）

先提交研究主線（1）。接著量現行 CIFAR-10 CFG 的 FID 與成熟 EMA 排序，依 2.2 三條件判斷是否可停訓；同一時間，CPU 端並行實作 Chamfer 基線、標籤噪音診斷、clean-fid 修復、協定草稿（2）。CIFAR-10 一空出，跑難子集翻轉預警（2.1），取得早期方向性訊號。決定停訓後，GPU 排 CIFAR-100 訓練，同樣以排序穩定收斂。CIFAR-10 完整網格 TSTR 跑滿以取得誠實 regret（2.3）。CIFAR-100 先做 η spot-check 再聚焦 guidance 與機制（2.5）。最後 CaF vs Chamfer 於 matched-budget 對決。

---

## 4. go/no-go（沿用，附早期預警之定位）

早期預警（非 gate，2.1）：難子集鬆動或高標籤噪音出現與否，只提前準備、不清除任何 gate。

正式 gate：CIFAR 上以完整網格計得的 regret@selected 低、top-k 命中；coverage 主導須跨 CIFAR-10 與 CIFAR-100 皆成立（難集不翻為硬門檻，若翻則走 2.6 兩結果框架）；CIFAR-100 上以介入式證據顯示 margin 抽乾；CaF 於 matched-budget 打平或贏 Chamfer，帶多 seed 信賴區間。正確性 gate：新 backbone 先重現其 FID 再掃描（EDM 1.848 已過）。

---

## 5. 時程（誠實化）

移除不安全捷徑後，go/no-go 定論不再是樂觀的 10 至 14 天。合理區間介於保守序列的 2 至 3 週與該樂觀值之間，取決於並行是否順利、提早停訓三條件是否滿足、CIFAR-100 是否翻轉。可靠先行的只有「翻轉早期預警」——約第 2 至 3 天可得方向性訊號，但它不縮短正式 gate。可投稿完整初稿仍約 2 至 3 個月，受 Phase 2 主結果表與撰稿主導，並行救不了，不予壓縮。

---

## 6. Watch-list

η 是否遷移到 CIFAR，要驗不要假設（DP 擴散先導偏向會複製）。難度上升時 CaF regret 是否擴大，CIFAR-100 為壓力測試。標籤噪音是否在難集壓過多樣性（即翻轉），診斷已內建。CIFAR-100 是否過訓，以 2.2 準則收斂。近平手排序與 EMA 成熟度是否使排序穩定判斷失真。τ 循環與 DINOv2 雙重使用護欄。

---

## 7. 外部參照（未經 repo 驗證，僅供起點，勿當規劃事實）

無現成 CIFAR-100 CFG 預訓練 checkpoint，CIFAR-100 需自訓。外部曾見 CIFAR-100 conditional teacher 以約 35.8M 參數達 FID 約 6.8、部分 CFG CIFAR 設置約 100 epoch 出結果——這些為單次搜尋所得，堪用 FID 範圍與收斂點須在 repo 內以自身量測確立後方可據以停訓。

---

## 8. 待你裁決

排程優化版審查結尾提到「一個決定會影響接下來怎麼做」但未完整帶出。請補上該決策點；本計畫的結構（signal-forward 的安全子集 + 完整網格誠實驗證）應能容納多數走向，但若該決策涉及 backbone 路線或範疇，2.3 至 2.5 可能需再調。
