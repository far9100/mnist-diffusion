# 研究計畫（重生成）— Sampling for Utility, not Fidelity

> 依 `records/2026-07-03-07_plan_research-revision-brief.md`（working spec，衝突處以它為準）重新生成。
> 專案：`C:\Users\fartw\OneDrive\Desktop\github\mnist-diffusion\`。本檔為戰略主檔，即 `records/2026-07-03-08_plan_regenerated-roadmap.md`。

---

## Context — 為什麼是這份計畫

**專案校正：** 真正的研究不是 RISC-V，而是 `mnist-diffusion` 的「Sampling for Utility, not Fidelity」——擴散取樣組態（steps × η × guidance）應為**下游訓練效用（TSTR）**而非 FID 最佳化。

**現有程式碼有兩代：**
- **Gen-1（已跑、有結果）**：MNIST DDPM+DDIM、guidance sweep、sampler/η/steps sweep。核心發現已實證：**FID-最佳 g=2 ≠ TSTR-最佳 g=1**（`fid_and_tstr_optima_differ=true`）；guidance 一升 diversity 單調崩塌（div_ratio 1.05→0.55，g=10 時 TSTR 崩到 79%）；η 改善 FID 卻略傷 TSTR（效用偏好 deterministic η=0）。
- **Gen-2（已寫好、但**從未執行**）**：三維聯合網格 `run_comparison.py`、`metrics_prdc.py`、`selector.py`（CaF + TSTR-free `auto_tau` + `regret_at_selected` + top-k + τ-robustness）、`mechanism.py`（margin/near-boundary + label-noise 對照）、`run_selector_signal.py`。全對齊 brief §6 Phase 0，但 `results/selector_signal.log` 為 0 bytes——**選擇器訊號一次都還沒在真實資料上驗證**。

**Brief 的戰略要旨：** 贏面**不在**「發現 FID≠效用」（brief 自認 C1 已不新），而在「**比 Chamfer Guidance（NeurIPS 2025, 2508.10631）更便宜、機制更清楚、且能免訓練地選對組態**」。C3（C3a CaF selector + C3b utility-guidance）是承重牆；C2（機制：coverage 而非 fidelity 驅動效用 + margin 抽乾）是最乾淨的差異化、當第一賣點。策略＝**C3-first、砍到一篇、go/no-go 改用 `regret@selected`**。

**離線時採用的預設（皆可覆寫）：** 算力＝單張消費級 GPU（CIFAR 可行但吃緊、ImageNet 降級）；Chamfer 定位＝正交互補為主兼談競爭；範疇＝維持一篇 CIFAR 收斂。

---

## 論文脊椎（重新定調）

> **不是**「在 CIFAR 上再證一次 FID≠效用」，**而是**：一個**免訓練、可遷移、有機制解釋的組態選擇器（CaF）**，用零頭成本拿到 Chamfer 級（或更好）的下游效用，並解釋**為什麼**（coverage/margin）。**C3a（CaF）＋ C2（機制）共同掛帥，C1 曲面只是背景。** 每個實驗的存在理由都服務這句話。

**對 Chamfer 的定位（預設，可改）：正交互補為主、兼談競爭。**
CaF 是 *selector*（在既有取樣組態中選、不改取樣）；Chamfer 是 *guidance*（改取樣）。關鍵洞見：**selector 可套在任何產生器輸出上——包括 Chamfer 的輸出**。故框架為「CaF 與 guidance 正交、可疊加」，再在 matched-budget 下比「CaF 選 vanilla 組態」vs「Chamfer 改取樣」的下游準確率與**成本**。這比硬碰硬更能 disarm reviewer，且把最強對手變成可整合的元件。

---

## 重排後的計畫（C3-first，含硬 gate）

### ★ Gate A —「這週」的近零成本生死判定（投入 Phase 1 前的硬閘門）
三件事並行，全在現有 MNIST 資產上，不需新算力：

1. **Chamfer 拆解 → 逐維度差異表。** 精讀 Chamfer Guidance（2508.10631）全文 + 官方 code，逐維度列：方法機制 / 是否免訓練 / 目標函數 / 算力成本 / 驗證資料集 / selector-vs-guidance 層次。**若差異太薄 → 現在轉向，遠比 Phase 2 才發現便宜。**（brief §1/§8 明列。）
2. **實際執行 Gen-2 的 Phase-0 driver（作者設好卻沒拉的訊號）。** 跑 `run_selector_signal.py` + 改寫版 `run_comparison.py` 於 MNIST guidance grid，產出**第一個真實的 CaF regret@selected / rank / top-k / τ-robustness**。**CaF 若連在 MNIST（發現最強處）都選不中 TSTR-最佳組態，selector 故事在碰 CIFAR 前就該修。**
3. **先補多 seed（≥3）+ 信賴區間，再信任任何 regret 數字。** 整篇押在「效用-最佳點 ≠ FID-最佳點」，但現全是 single-seed argmax——**noisy 網格取 argmax 本身可能就是雜訊**。把 seed 迴圈 retrofit 進 Gen-2 driver（MNIST 上便宜、且是 brief §3 硬要求）。

**Gate A 通過準則：** (a) Chamfer 差異表顯示 selector≠guidance / 成本 / 遷移性至少一軸有實質空間；(b) MNIST 上 CaF 的 `regret@selected` 低、top-k 命中高（多 seed 下穩定）；(c) 機制方向一致（高 guidance 抽乾 near-boundary 樣本）。**三者不過 → 停下重想框架，不投 CIFAR。**

### Phase 1 — CIFAR，C3-first（brief 3–5 週）
1. **EDM CIFAR 重現 spike（技術長桿，先去風險）：** 建 `backbones/`（`NVlabs/edm` checkpoint 經 `diffusers`/`k-diffusion`）、`datasets/`（CIFAR-10/100）、`fid.py` 升級呼叫 `clean-fid` + `dgm-eval` FD-DINOv2。**正確性 gate：先重現 EDM CIFAR-10 cond FID ≈ 1.79 再掃描。**
2. **最小對決（真正的 go/no-go）：** CaF + **一條** utility-guidance vs（簡化）Chamfer，組態數可少、pre-registered 協定、輸出 compute 帳本（`run_guidance_study.py` 演進為方法評估 harness）。
3. **coverage 主導的跨資料集複製：** 在 CIFAR-10 **與** CIFAR-100（難）上驗 CaF↔TSTR，確認「coverage 主導」**不翻轉**（難資料集 label-noise 可能壓過多樣性好處——這才是真 go/no-go，非只看效用點是否偏離 FID 點）。

### Phase 2 — 方法深化 + 主結果（brief 4–6 週）
- CaF 形式化 + 跨 CIFAR-10/100「CaF 選組態 ≈ 全 TSTR 選組態」+ **成本節省量化**。
- **matched-budget 主結果表**：本方法下游準確率 vs Chamfer / feedback-guided / autoguidance(FID 目標)，**帶統計檢定**。
- C2 機制的**介入式**證據：coverage 受控後 margin 效應是否仍在；直接證明高 guidance 丟掉的正是 boundary 相關樣本。

### Phase 3 — ImageNet-100（降級/選做，視算力）
若做：用 CaF 先篩 3–5 個 promising 組態、只對這幾個跑完整下游；文中誠實說明未跑滿網格、CaF 驗證靠 CIFAR 外推。（單卡預設下多半降為 thesis 後續。）

### Phase 4 — 撰稿 / 可複現（brief 3–4 週）
多指標（clean-fid + FD-DINOv2 + prdc）、pre-registered 協定、釋出 code/組態。intro 重寫：從「FID 不完美」→「**FID 對取樣組態的排序相對效用被反轉，且有便宜規則能選對**」；主動寫 Chamfer 區隔段（selector vs guidance / 機制 / 遷移性 / 成本）。

---

## 方法論護欄（brief §3/§4，務必落到 code）
- **τ 循環：** `selector.py` 已有 `auto_tau`（TSTR-free）+ `tau_robustness`；報告 selected config 對 τ 的敏感度，不得用 TSTR 定 τ。
- **DINOv2 雙重使用：** selector 用 DINOv2，另備 **第二表徵（CLIP/Inception）**重算 coverage 做 robustness（破「在自選度量上贏自己」）。**目前只有單一 MNIST-CNN 特徵空間，需新建。**
- **特徵空間錯配：** 必須在**難資料集**（CIFAR-100 / ImageNet-100）單獨驗 CaF↔TSTR，不靠 MNIST/CIFAR-10 外推。
- **go/no-go 量錯：** 用 `regret@selected` / top-k，**不用**全域 Spearman ρ（爛組態會撐高 ρ 卻選錯 top）。
- **argmax 雜訊：** ≥3 seed + 信賴區間（**現有硬洞，Gate A 補**）。
- **競爭機制 label-noise：** `mechanism.py` 已含 label-noise proxy；go/no-go 盯「coverage 主導是否跨資料集複製」。
- **matched-budget：** 協定**投稿前寫死並 pre-register**，固定 compute 帳本輸出。

---

## Go/No-Go 閘門（brief §7，取代舊 Spearman 版）
| 階段 | 新閘門 |
|---|---|
| Gate A / Phase 0 | `regret@selected` 低 + top-k 命中高（多 seed 穩定）；per-class 效應與 margin 機制方向一致；Chamfer 差異表有實質空間 |
| Phase 1 | 上述 + 「coverage 主導」跨 CIFAR-10/100 複製（難資料集不翻轉）+ 最小對決能打平/贏 Chamfer |
| Phase 2 | CaF `regret@selected` 低 + 成本顯著低 + matched-budget 下顯著贏 Chamfer（帶統計檢定） |

**不變的正確性 gate：** 任何新 backbone 先重現其論文 FID 再掃描（EDM CIFAR-10 ≈ 1.79）。

---

## 具體任務對應 repo 檔案
**Gate A / Phase 0（多為執行 + 補洞，程式多已存在）**
- 執行 `run_selector_signal.py`、改寫版 `run_comparison.py`（三維網格 + PRDC + CaF + regret）於 MNIST，產出首組真實選擇器訊號。
- 於 `run_comparison.py` / `run_guidance_study.py` / `run_selector_signal.py` 加 **≥3 seed 迴圈 + 信賴區間**（現無）。
- `mechanism.py`：補齊 **coverage 受控的 margin 介入**（matched-coverage 子取樣函式目前只 stub 在 docstring）。
- `evaluate.py`：加**資料增強情境**（TS+TR 混訓、low-data regime）——把故事推向「到底有沒有用」。

**Phase 1（greenfield，最大工作量）**
- 新增 `backbones/`（EDM 適配）、`datasets/`（CIFAR-10/100）。
- `fid.py`：MNIST-FID → `clean-fid` + FD-DINOv2（新依賴 `clean-fid`/`dgm-eval`）。
- `metrics_prdc.py`：目前為 torch 自寫；決定是否對齊 `clovaai/generative-evaluation-prdc`（或保留自寫但加交叉驗證）。
- 新增 `guidance/`：**先實作一條** utility-targeted guidance（建議 interval-guidance，區間由 coverage/CaF 決定）。
- 新增 **Chamfer 基線**（官方 code 若釋出優先；否則忠實重寫核心；MNIST 最小對決可用簡化 proxy）。
- selector 第二表徵（CLIP/Inception）交叉驗 coverage。

---

## 立即動作（下一步，離線預設下我會先做）
1. **Chamfer 拆解 + 逐維度差異表**（deep-research + 讀官方 code）。
2. **實跑 Gen-2 Phase-0 driver 於 MNIST**（先補 ≥3 seed），拿到首個 `regret@selected` / top-k / τ-robustness。
3. 依 1+2 結果判 **Gate A**：過 → 進 EDM CIFAR 重現 spike；不過 → 停下，帶證據回報並重議框架。
4. 重寫 intro 的 η 敘事與 Chamfer 區隔段（草稿）。

---

## 主要風險與緩解
- **與 Chamfer 差異太薄** → Gate A 第 1 項為硬前置；差異表撐不起就轉向（selector≠guidance / 成本 / 遷移性 至少守住一軸）。
- **選擇器訊號在真實資料不成立** → Gate A 第 2 項用 MNIST 最便宜地暴露；不過就修 CaF 或降級為「效用曲面刻畫 + 機制」論文。
- **coverage 主導在難資料集翻轉（label-noise）** → Phase 1 明列為 go/no-go，跨 CIFAR-10/100 複製才算數。
- **EDM 重現卡住（技術長桿）** → Phase 1 第 1 步獨立 spike 去風險，先過 FID≈1.79 再掃。
- **單卡算力不足** → 維持一篇 CIFAR 收斂、ImageNet 降級；CaF 的價值正是「免訓練預篩」省算力，與資源限制同向。
- **single-seed argmax 是雜訊** → 多 seed + CI 為所有主張前置。

---

## 待同步 / 版控
- 本檔即 `records/2026-07-03-08_plan_regenerated-roadmap.md`，已納入版控。
- README「核心研究問題」仍是舊「FID vs TSTR 解耦」框架 → 後續同步為新脊椎（selector + 機制掛帥、對 Chamfer 定位）。
- 專案 memory（`project_status.md` 目前記的是 RISC-V）需更新為本專案的實際定位，避免跨專案敘事混淆。
