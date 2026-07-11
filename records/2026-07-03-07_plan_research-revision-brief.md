# 研究方向修訂 Brief — Sampling for Utility, not Fidelity

> **給 Claude Code 的說明**：本文件是對原 roadmap（`Sampling for Utility, not Fidelity`）的**修訂與註記**，
> 依據 2025–2026 文獻重掃結果調整。原 roadmap 的核心直覺保留，但**貢獻定位、優先順序、go/no-go 準則**
> 需依本文件更新。遇到與原 roadmap 衝突處，**以本文件為準**。實作時請把本文件當成 working spec。

---

## 0. TL;DR（決策摘要）

- **核心直覺仍成立**：取樣器應為「訓練資料效用」最佳化，而非 FID。此角度有前人零星證據支持。
- **白區已被侵蝕**：原 roadmap 鎖定的三個「開放白區」中，C3a/C3b（免訓練 selector、utility-guidance）
  已被 **Chamfer Guidance (NeurIPS 2025)** 大幅佔據；η 軸的單點觀察已見於 **DP 擴散文獻**。
- **贏面轉移**：價值不再是「發現 FID≠效用」，而是「**比 Chamfer 那批更便宜、機制更清楚、能免訓練地選組態**」。
- **策略轉向**：**C3-first**（不要 sandbox→定律→方法慢慢走）；**砍範疇**到最小可發表核心；
  **go/no-go 改用 regret@selected**（非全域 Spearman）。
- **第一篇 = CIFAR-10/100 + CaF + 一條 utility-guidance，正面對打 Chamfer**。ImageNet/其餘 guidance 降級為 thesis 後續。

---

## 1. 為什麼要轉向：2025–2026 撞區（novelty threats）

| 撞區工作 | arXiv / 出處 | 撞到原 roadmap 的哪裡 | 影響 |
|---|---|---|---|
| **Chamfer Guidance** (Dall'Asen, Askari-Hemmat, Romero-Soriano 等) | arXiv **2508.10631**，NeurIPS 2025 | **C3a + C3b** | **最嚴重**。免訓練 guidance，用少量真實範例刻畫 precision + coverage，下游分類器準確率 ID +15% / OOD +16% over CFG。與「免訓練 + coverage/precision + 下游效用」高度重疊，且出自你點名的 Hemmat 同團隊 |
| **Deliberate practice scaling laws** (Askari-Hemmat et al.) | arXiv **2502.15588**，2025 | **對手 Fan 那條線的後續** | Fan 的 scaling-laws 已往「動態/主動決定生成什麼」演進，逼近 C3b 地盤。不能再把 Fan 當靜態靶 |
| **DP 擴散的隨機性觀察** (Dockhorn et al.) | arXiv **2210.09929**（+ Ghalebikesabi 2302.13861） | **C1 的 η 軸** | 已報「stochastic 取樣對 FID 關鍵、對下游準確率不重要甚至有害；下游受益於多樣性」。你的 MNIST 先導結論非首見 |

**結論**：原 roadmap 中「無人擁有此貢獻」「現有保多樣性 guidance 全部只優化 FID」等句子**已不成立**，
必須改寫。護城河從「無人做過」壓縮為「別人做了相鄰版本，我要證明更便宜/機制更清楚/可遷移」。

**行動**：投稿前需精讀 Chamfer Guidance 全文與 code，逐維度列差異（見 §3 差異表框架）。
若讀完差異太薄 → **現在轉向遠比 Phase 2 才發現便宜**。

---

## 2. 修訂後的定位（positioning）

### 2.1 護城河四點重估（原 roadmap 的四點，逐一標記現況）
1. ~~η 軸幾乎無人做過~~ → **改寫**：η 單點觀察見於 DP 文獻；差異化在「**(steps×η×guidance) 聯合曲面 +
   可利用規則**」，且非 DP 情境。措辭：「η 的單點觀察散見文獻，但聯合曲面與可利用的組態選擇規則無人給出」。
2. ~~免訓練 selector 無人擁有~~ → **改寫**：Chamfer 已用 coverage/precision 做免訓練 guidance。
   我方差異必須落在「**selector（選組態）≠ guidance（改取樣）**」這個層次 + 跨資料集/backbone 的**遷移性** + **成本**。
3. utility-targeted guidance → **仍可做，但必須正面對打 Chamfer**，不能只對打舊的 feedback-guided。
4. 機制解釋（coverage 而非 fidelity 驅動效用 + margin 抽乾）→ **這是目前最乾淨的差異化**，Chamfer 沒給機制。
   **把機制當第一賣點之一。**

### 2.2 must-cite / must-beat（更新）
- **必須正面打贏並區隔**：Chamfer Guidance (2508.10631, NeurIPS 2025)、feedback-guided (Hemmat, arXiv 2310.00158/TMLR 2024)、
  deliberate-practice scaling (2502.15588)、Fan et al. CVPR 2024。
- **必引先導**：Ravuri & Vinyals NeurIPS 2019 (CAS/TSTR)、Dockhorn DPDM (2210.09929，**η↔下游先導觀察**)、
  Ghalebikesabi (2302.13861)、Lomurno 2024、Sariyildiz CVPR 2023、Azizi TMLR 2023、He ICLR 2023、Tian StableRep NeurIPS 2023。
- **機制/指標**：Ho & Salimans CFG (2022)、Kynkäänniemi interval-guidance NeurIPS 2024、Karras autoguidance NeurIPS 2024、
  Kynkäänniemi P&R NeurIPS 2019、Naeem D&C ICML 2020、Sorscher NeurIPS 2022 (coreset/margin)、
  Stein NeurIPS 2023 + Kynkäänniemi「Role of ImageNet classes in FID」ICLR 2023（論證別只信 Inception-FID）。

### 2.3 intro 要拆的 reviewer 陷阱（更新版）
- **不是**「FID 不完美」（舊）→ **是**「FID 對取樣組態的排序相對效用被反轉，且有便宜規則能選對」。
- **不是**「low-CFG 較好」（Fan 做過）→ **是**「聯合 (steps×η) 曲面 + 免訓練 selector + 機制」。
- **新增**：主動說明與 **Chamfer Guidance** 的區隔（selector vs guidance、有無機制、遷移性、成本），
  不要讓 reviewer 先問。

---

## 3. C3 是承重牆：方法論護欄（務必寫進實作）

> C1（FID≠效用）已被承認不新，全篇價值押在 C3。若 CaF / utility-guidance 沒在 matched-budget 下顯著贏，
> 整篇塌回「又一次 FID≠效用」。以下裂縫必須在 code 層面處理。

- **[τ 循環風險] CaF 的 `precision ≥ τ` 閾值本身是超參**。若要靠 TSTR 定 τ，就破壞「免訓練」賣點。
  → **實作要求**：`selector.py` 需提供「對 τ 掃描的穩健性曲線」+ 一個**不依賴 TSTR** 的 τ 自動決定法
  （例如以真實 probe set 的 precision 分位數自動定 τ），並報告 selected config 對 τ 的敏感度。
- **[DINOv2 雙重使用循環]** CaF 的 coverage 用 DINOv2、保真度指標又用 FD-DINOv2 → 被質疑「在自選度量上贏自己」。
  → **實作要求**：selector 與評估用**不同表徵**交叉驗證（例如 selector 用 DINOv2，另備 CLIP/Inception 特徵重算一次 coverage 做 robustness）。
- **[特徵空間錯配]** coverage 在 DINOv2 空間量、utility 在 from-scratch ResNet 空間實現，兩者未必對齊；
  在難資料集（CIFAR-100 / ImageNet-100）相關可能鬆掉。→ 必須在**難資料集上**單獨驗證 CaF↔TSTR，不能只靠 MNIST/CIFAR-10 外推。
- **[go/no-go 量錯]** 全域 Spearman ρ 會被爛組態撐高、卻仍選錯 top。
  → **改用 `regret@selected`（選中組態的 TSTR 與網格最佳 TSTR 之差）或 top-k 命中率**。`selector.py` / `evaluate.py` 需輸出此指標。
- **[選最大值偏誤 + 統計]** 網格 argmax 有一部分是雜訊；沒有多 seed，「效用最佳點位移」本身可能是雜訊。
  → **從一開始編列多 seed（≥3）**，所有主張帶信賴區間 / 顯著性檢定。
- **[競爭機制：標籤噪音]** 低 guidance 也會產生離類/模糊樣本；難資料集上標籤噪音可能壓過多樣性好處，
  使「coverage 主導」**翻轉**。→ Phase 1/3 的真正 go/no-go 要盯「coverage 主導是否跨資料集複製」，而非只盯「效用點是否偏離 FID 點」。
- **[matched-budget 比較會被拆]** 「matched fidelity」match 在哪個指標會偏袒某方；Chamfer/feedback 也需下游信號，compute 對齊微妙。
  → **實驗協定投稿前寫死並預先註記**（pre-register），`run_guidance_study.py` 固定協定、輸出 compute 帳本。

---

## 4. 機制（C2）：因果性要求
- 偏相關 `utility ~ coverage | precision` 是**相關非因果**；coverage 與 utility 可能同被潛在因子驅動。
- margin/near-boundary 故事需**介入式**證據：控制住 coverage 後 margin 效應是否消失；或直接證明高 guidance 丟掉的
  **正是**邊界相關樣本（不能只給一條隨 guidance 單調的曲線）。
- **實作**：`analyze_distribution.py` / 新 `mechanism.py` 需支援：(a) 用真實資料訓練的分類器量 synthetic 樣本
  near-boundary 佔比 vs guidance；(b) coverage 受控下的 margin 條件分析。

---

## 5. 範疇裁減與里程碑（C3-first）

**原 roadmap 的量 ≈ 2–3 篇論文**。第一篇明確收斂：

### 第一篇（目標）
> CIFAR-10/100 上的 (steps×η×guidance) 效用曲面 + CaF selector + 一條 utility-guidance，
> 正面對打 Chamfer Guidance，附機制解釋（coverage 而非 fidelity 驅動效用）。

### 降級為 thesis 後續 / stretch
- ImageNet-100（DiT/EDM2）：**降級**。若做，改為「用 CaF 先篩 3–5 個 promising 組態、只對這幾個跑完整下游」，
  並在文中誠實說明 ImageNet 上未跑滿網格、CaF 驗證靠 CIFAR 外推。
- 第二、三條 guidance（per-class adaptive、autoguidance-for-utility）：**降級**為 ablation / 後續章節。

### 修訂後階段
- **Phase 0（MNIST sandbox，1–2 週）**：原樣保留 sandbox 加固，但**同時原型 CaF + margin 分析**，
  盡快進到「CaF vs 簡化 Chamfer」的最小對決。
- **Phase 1（CIFAR，C3-first，3–5 週）**：**先做最小對決**（CaF + 1 guidance vs Chamfer，組態數可少），
  用它當真正 go/no-go；C1 曲面 / C2 機制作為支撐**同步**產出，非前置阻塞。
- **Phase 2（方法深化 + ablation，4–6 週）**：CaF 形式化 + matched-budget 主結果表。
- **Phase 3（規模化，降級/選做）**：ImageNet-100 如上處理。
- **Phase 4（撰稿/可複現，3–4 週）**：多指標（clean-fid + FD-DINOv2 + prdc）、pre-registered 協定、釋出 code/組態。

---

## 6. 具體任務（對應 repo 檔案）

### Phase 0 — sandbox 加固 + C3 原型
- [ ] `run_comparison.py` → 擴成**三維聯合掃描** (steps × η × guidance)，取代目前兩條獨立掃描。
- [ ] 新增 `metrics_prdc.py`（precision/recall/density/coverage，接 `clovaai/generative-evaluation-prdc`）。
- [ ] `analyze_distribution.py` → `extract_features`/`mean_pairwise_l2` 併入 prdc 度量。
- [ ] `evaluate.py` → 加 **per-class TSTR + 混淆矩陣**；**輸出 `regret@selected`、top-k 命中**（取代全域 ρ 當閘門）。
- [ ] 新增 `selector.py`（**CaF**）：`argmax coverage s.t. precision ≥ τ`；**含 τ 穩健性掃描 + 不依賴 TSTR 的 τ 自動決定法**。
- [ ] 新增 `mechanism.py`（或擴 `analyze_distribution.py`）：near-boundary 佔比 vs guidance、coverage 受控的 margin 分析。
- [ ] 加**資料增強情境**（TS+TR 混訓、low-data regime）到 `evaluate.py`，把故事推向「有沒有用」。
- [ ] 多 seed（≥3）基礎設施：所有 TSTR / selector 指標帶信賴區間。

### Phase 1 — CIFAR，C3-first
- [ ] 新增 `backbones/`：EDM 適配層（`NVlabs/edm` checkpoint，經 `diffusers` / `crowsonkb/k-diffusion`）。
      **先重現 EDM CIFAR-10 cond FID ≈ 1.79 再掃描**（正確性 gate）。
- [ ] 新增 `datasets/`：CIFAR-10/100 設定。
- [ ] `fid.py` → 由 MNIST-FID 升級為呼叫 `GaParmar/clean-fid`（避 resize 假影）+ `layer6ai-labs/dgm-eval` 的 **FD-DINOv2**。
- [ ] `run_guidance_study.py` → 演進為**方法評估 harness**：**最小對決 CaF+1 guidance vs Chamfer**（可先小組態集），
      固定 pre-registered 協定、輸出 compute 帳本。
- [ ] `guidance/`：**先實作一條** utility-targeted guidance（建議 interval-guidance 但區間由 coverage/CaF 決定）；
      per-class / autoguidance 版本標為後續。
- [ ] selector 用**第二表徵**（CLIP 或 Inception）重算 coverage 做 robustness（破 DINOv2 雙重使用循環）。

### Phase 2+
- [ ] CaF 形式化 + 跨 CIFAR-10/100「CaF 選組態 ≈ 全 TSTR 選組態」驗證 + 成本節省量化。
- [ ] matched-budget 主結果表：本方法下游準確率 vs Chamfer / feedback-guided / autoguidance(FID 目標)。
- [ ] `records/`：持續研究日誌（每階段一份 md）。

---

## 7. 修訂後的 go/no-go 閘門

| 階段 | 舊閘門 | **新閘門（取代）** |
|---|---|---|
| Phase 0 | CaF↔TSTR 全域 Spearman ρ ≥ 0.8 | **`regret@selected` 低 + top-k 命中高**；per-class 效應與 margin 機制方向一致 |
| Phase 1 | 效用最佳點顯著偏離 FID 最佳點且跨 CIFAR 一致 | 上述 **+ 「coverage 主導」跨 CIFAR-10/100 複製（含難資料集不翻轉）** **+ 最小對決能打平/贏 Chamfer** |
| Phase 2 | CaF↔TSTR 高且成本低 | CaF `regret@selected` 低 + 成本顯著低 **+ matched-budget 下顯著贏 Chamfer（帶統計檢定）** |

**正確性 gate（不變）**：新 backbone 取樣先重現其論文 FID 再掃描。

---

## 8. 立即動作（this week）
1. 精讀 **Chamfer Guidance (2508.10631)** 全文 + code，產出逐維度差異表（方法機制 / 是否免訓練 / 目標函數 / 算力 / 驗證資料集）。
2. 把 `run_comparison.py` 改成三維掃描 + 接 `metrics_prdc.py`。
3. 原型 `selector.py`（CaF）與 `evaluate.py` 的 `regret@selected` 輸出。
4. 在 MNIST 上跑「CaF vs 簡化 Chamfer」最小對決，作為是否值得繼續的早期訊號。
5. 重寫 intro 的 η 敘事與 Chamfer 區隔段落（草稿即可）。

---

## 9. 參考（帶 arXiv id，供查證）
- Chamfer Guidance — arXiv 2508.10631（NeurIPS 2025）
- Deliberate practice scaling laws — arXiv 2502.15588（2025）
- Feedback-guided synthesis (Hemmat) — arXiv 2310.00158（TMLR 2024）
- DPDM（η↔下游先導觀察，Dockhorn） — arXiv 2210.09929
- DP diffusion useful synthetic images（Ghalebikesabi） — arXiv 2302.13861
- Fan et al. scaling laws — CVPR 2024
- Ravuri & Vinyals CAS/TSTR — NeurIPS 2019
- interval-guidance (Kynkäänniemi) — NeurIPS 2024；autoguidance (Karras) — NeurIPS 2024
- P&R (Kynkäänniemi) NeurIPS 2019；D&C (Naeem) ICML 2020；coreset/margin (Sorscher) NeurIPS 2022
- FID 假影：Stein NeurIPS 2023；Kynkäänniemi「Role of ImageNet classes in FID」ICLR 2023

---

## 10. 一句話收斂
> 贏面不在「發現 FID≠效用」，而在「**做得比 Chamfer 那批更便宜、機制更清楚、且能免訓練地選對組態**」。
> 全案圍繞這句話重新收斂；C3 是承重牆，C1/C2 是支撐。
