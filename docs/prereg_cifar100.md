<!-- 用途：CIFAR-100 預註冊全文（D 包 D0-D12 定稿，加 D10 第四閘 grid 凍結 amendment）。本文於揭盲前凍結、隨 repo 發布，供事後查驗登記內容早於任何 confirmatory 取樣。 -->

# CIFAR-100 預註冊

本文是 CIFAR-100 confirmatory 實驗的預註冊全文，由兩份文件合併：

| 段落 | 來源 | 登記日 | 狀態 |
| --- | --- | --- | --- |
| D 包 D0–D12 | `2026-07-09-13` | 2026-07-09（作者終簽 2026-07-10） | 凍結 |
| grid 凍結 amendment | `2026-07-11-05` | 2026-07-11 | 凍結 |

兩段皆逐字保留原文，未事後修訂。登記早於任何 CIFAR-100 confirmatory 合成樣本，git 提交時間可驗：
D 包草稿 `be16274`、定稿 `5cd3e8f`、grid 凍結 amendment `24e208e`。文中引用的其他記錄以 ID 標示，
摘要見 [CHANGELOG.md](../CHANGELOG.md)，全文留在 git 歷史（`git log -- records/`）。

confirmatory 真跑截至本文寫定時尚未執行。

---

## 第一部分：D 包（CIFAR-100 預註冊定稿，作者終簽 2026-07-10）

### D0：H1/H2/H3 原文與終審

逐字調出 `2026-07-05-02`：

- **H1（機制，confirmatory）**：驅動下游效用（TSTR）的是 coverage（多樣性）而非 fidelity（precision）。
- **H2（選擇器，confirmatory）**：免訓練的 CaF 能在候選組態中選到近效用最佳者（regret@selected 低）。
- **H3（對決，confirmatory）**：在 matched-budget 下，CaF 選 vanilla 組態的下游準確率打平或優於（簡化）Chamfer。

終審：CIFAR-10 上 H1 頭條被反證（C1 兩空間不分離）、H2 之 CaF 敗 FID-min（`2026-07-09-03` 判決三）。
H2 於 CIFAR-100 之數字門檻見 D4（前此無凍結門檻＝CIFAR-10 僅描述性）。H3 身分：確認為 confirmatory，但
其必跑性由 D7 條件化（分支相依）。CIFAR-100 之角色＝validation（`2026-07-05-02` L94：coverage 主導
不翻轉即原 thesis；翻轉則預宣告替代結論）。

### D1：四分支決策樹（承 `2026-07-08-03`）

1. 分離出現 且 CaF（v2）勝 FID-min → 邊界條件復活。
2. 分離出現 但 CaF 敗 FID-min → thesis 活、selector 死。
3. 不分離 但機制複製 → 診斷論文。
4. 皆否 → 負結果短文。

### D2：分支一先驗（已登記 `2026-07-08-03`）

整體約一成；誠實聲明、非裁決輸入；登記於 C1 揭盲前。路由由 D1 客觀觀察量決定，不由先驗。

### D3：分支三操作化＋介入臂

- **三觀察量（三中二判複製）**：(i) 低中段 near-boundary 於 coverage 升平區單調降；(ii) 高段 coverage 與
  TSTR 同崩；(iii) 高段 near-boundary 脫鉤。三中二成立＝機制複製。
- **介入臂（預註冊，D3 增列 `2026-07-08-02` §5.2.3）**：於 CIFAR-100 confirmatory 已生成合成集上跑一個
  C3 型 coverage-matched pruning，**定位為分支三論文宣稱之必要證據、非分支路由輸入**（路由仍由三觀察量判）。
  剪枝規則沿 CIFAR-10 C3（剪至最低 guidance 之 coverage 水準、移除離真實流形最近者、二分搜尋匹配，
  `2026-07-09-08`）；剪枝條件之具體 coverage 目標於 D10 網格凍結後定（隨網格、非終簽 blocker）。
  成本：無新取樣，僅剪枝＋重訓。**表徵口徑**：每觀察量標量測空間，DINOv2 主、Inception robustness
  （依 `2026-07-05-08` §4 型式）。
- CIFAR-10 之 C2/C3/C5 介入未強支持 near-boundary 機制（`2026-07-09-08`），故分支三之機制複製在
  CIFAR-100 屬真正待驗、非既定。

### D4：H2 數字門檻（作者已裁候選 B、X=1.5）

- **H2 通過 ⟺ CaF-v2 之 per-seed regret 至少比 matched-budget FID-min baseline 低 1.5pp**（候選 B、X=1.5，
  作者裁 `2026-07-09-04`）。錨定：CIFAR-10 上 FID-min（0.91）勝 CaF（3.69），selector 存在理由須為
  勝過更便宜之 FID-min；X=1.5 為勝幅門檻。與 D8 v2 價值判準同數（1.5）。
- **登記主張形式**：峰位噪聲穩健（高原＋懸崖，不押點峰）——C4 MDE 表示 3seed×1rep MDE 7.29pp 無法解析
  ~2.5pp 峰位（`2026-07-09-10`）。
- **功效配置（作者終簽）**：n_seed=8 × n_rep=5（MDE 2.49pp，`2026-07-09-10` 表），可解析 ~2.5pp
  峰位偏移。

### D5：強制 baseline 集與三臂對決

- baseline：matched-probe FID-min（引 C7 `2026-07-09-06`：小 probe 排序穩定、FID-min 可靠）、固定 w
  慣例值（登記出處）、隨機可行點。
- **三臂對決（matched-budget，同表）**：FID-min／CaF-v2／Chamfer；缺 FID-min 臂之「勝 Chamfer」無證明力。
- matched-probe 尺寸（作者終簽）：1000/class（與 CaF probe 同、遠在 C7 穩定區）。

### D6：復活條件與分離空間口徑

- 復活＝分離出現 且 CaF-v2 勝 FID-min（≥1.5pp）。
- **分離空間口徑（`2026-07-09-04` §0.1.5）**：雙空間（Inception＋DINOv2）皆分離＝完全復活；僅 DINOv2
  分離＝表徵條件弱復活。事前寫死。

### D7：Chamfer 牆條件化（作者核定裁量）

作者裁定（`2026-07-09-04` §5.2.7 核定文案）：「-12 §9 第 3 項之範圍例外延伸至 Chamfer 牆條件化規則
之事前登記（分支 1/2 必跑、3/4 選配），僅此一項。」CPU 空窗 Chamfer 適配照原排程。

### D8：CaF-v2（作者裁＝留、第三訊號 recall）

- **第三訊號＝recall**（C0 `2026-07-09-12`：recall 打破 w2.5 之 Pareto 支配——w2.5 recall .493 < 三
  oracle；density 無效）。**recall 為 DINOv2 量、不引 judge → 保住「免任務標籤」定位**（優於 near_bnd）。
- **v2 selector 形式（作者終簽）**：`argmax recall s.t. precision ≥ τ`（τ 沿 CaF 之 auto-τ＝0.9×real-vs-real
  precision）；於 CIFAR-100 validation 驗其 matched-budget regret 是否勝 FID-min ≥1.5pp。
- **價值判準**：v2 於 matched-budget 勝 FID-min regret ≥1.5pp（同 D4 X）。
- 框架：discovery（CIFAR-10，永遠 exploratory）／validation（CIFAR-100）。CIFAR-10 之 recall 打破支配為必要
  非充分。

### D9：量測程序凍結（凍程序不凍數字）＋種子公式＋metadata

- CIFAR-100 judge 品質 gate 定法、near-boundary 重校準程序、特徵空間依 `2026-07-05-08`。
- **metadata（承 E5/F1）**：driver 輸出強制含 nearest_k、有效 k=min(k,n−1)、tau_fraction、batch、完整 argv、
  start_timestamp、環境雜湊（torch/cuda/cudnn）。
- **種子公式無碰撞（作者終簽、已實作驗證）**：CIFAR-10 之 `seed*1e7+int(w*1e3)*1e4` 於本網格退化碰撞
  （`2026-07-06-05` §1.12，10 組碰撞、26 cells 涉入）。CIFAR-100 採 hash 派生
  `gseed(seed,w)=int(sha256(f"{seed}_{w:g}").hexdigest()[:15],16)`，實作 `datasets/cifar100_gseed.py`；
  全網格枚舉驗證（100 seeds × 91 ws=9100 cells → 9100 distinct，collision-free=True）。公式與驗證一併登記。
- **base-model gate（作者授權 agent 填，記為委派）**：clean-fid ≤ **20 @ 50k、w=1**。錨定：CIFAR-10 base
  8.95@50k（gate ≤10）；CIFAR-100 更難（100 類、500/類），約 2× headroom 為 round 且有意義之門檻。量測於 D10
  （取樣 D commit 後）；若量測值接近門檻，屬 backbone 品質之實質發現，非移動門檻之理由（凍程序不凍數字例外：
  此為委派之單一 gate 數字，明文留痕）。

### D10：時序（D commit 後）

D commit → judge 訓練與校準 → scout（僅定網格，讀數不回饋判準）→ 網格凍結 amendment → confirmatory。
各閘為作者 STOP 點（`2026-07-09-04` §0.2）。

### D11：凍結定義（承 E5 `claude.md` §5）

僅適用 D 包：prose＋程式＋已揭盲資料 dry-run＋輸出雜湊；P 資產為基底。

### D12：一頁論文骨架

- **分支三版**（`paper_skeleton_branch3.md`，本批成稿）：so-what 骨＝`2026-07-06-05` §1.10「無單一
  便宜代理普遍可靠」（MNIST 與 CIFAR-10 selector 判決相反）；C1 反證後其現實性升高。
- **分支四版（作者終簽：另立）**：`paper_skeleton_branch4.md`，負結果短文骨架。

---

## 第二部分：D10 第四閘 grid 凍結 amendment（2026-07-11 登記）

本 amendment 為 pre-registration 步驟，早於任何 confirmatory 合成樣本；grid 選擇不得以 scout 之 TSTR
讀數為據（D10：scout 讀數不回饋判準）。

### 凍結 grid（作者裁定）

confirmatory guidance grid = **{1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8}**（10 點，封頂 w8）。沿用 CIFAR-10
confirmatory 之凍結 grid（`2026-07-05-11`）。

grid 選擇依據（不含 TSTR 讀數）：

- **coverage 崩點定位**：scout 之 DINOv2 coverage 峰在 w2、崩點（峰後首個跌破 90% 峰值）在 w5；Inception
  交叉檢查同形。低段密點 {1,1.5,2,2.5,3} 解析 coverage 峰與轉折，{4,5,6,7,8} 涵蓋崩尾。
- **跨資料集可比性**：與 CIFAR-10 同 grid，使 CIFAR-100 之 coverage/TSTR 曲線可與 CIFAR-10 直接對照
  （D0 之 CIFAR-100=validation 角色）。
- **CFG 實用範圍封頂**：w8 上限沿 `2026-07-05-11`。
- 明文排除：scout 之 TSTR 峰落 w1 為 exploratory 觀察，不作 grid 選擇依據；grid 選擇僅用 coverage 幾何
  與可比性。

### 一併登記（多數已於 D 包凍結，此處彙整並補 grid 相依項）

- 量測 sampler：steps=50、eta=0（固定）。
- 量測儀器：judge `checkpoints/cifar100_judge.pt`（真實測試 74.25%）、near-boundary threshold 0.3622
  （p20 相對分位，`2026-07-11-03`）；coverage 特徵 DINOv2 主、Inception robustness。
- selector：CaF-v2 = `argmax recall s.t. precision ≥ τ`，auto-τ = 0.9 × real-vs-real precision（D8）。
- 功效配置：8 seed × 5 rep（D4，MDE 2.49pp）。matched-probe FID-min baseline 1000/class（D5）。
- H2 門檻：CaF-v2 之 per-seed regret 至少低於 matched-budget FID-min 1.5pp（D4，X=1.5）。
- 種子公式：`gseed(seed,w)=int(sha256(f"{seed}_{w:g}").hexdigest()[:15],16)`，`datasets/cifar100_gseed.py`
  （D9，全網格無碰撞已驗）。此 10 點 grid × seeds 之無碰撞於 confirmatory driver dry-run 再驗。
- D3 介入臂剪枝目標：剪至最低 guidance（w1）之 coverage 水準，沿 CIFAR-10 C3 規則（`2026-07-09-08`）；
  於 confirmatory 生成集落盤後執行，非路由輸入。

### 凍結落實（依 `claude.md §5.1`）

本 amendment 完成 (a) prose 登記。(b) grid 將寫入 CIFAR-100 confirmatory driver 之預設 config（confirmatory
閘實作）；(c) dry-run（含 gseed 無碰撞、grid 列舉）於真跑前執行；(d) 輸出 hash / 逐位對帳於 confirmatory
之 P 資產。(b)-(d) 屬下一閘工作，本閘只鎖 grid 與規格 prose。

### 未凍結項

- `--per-class`（每類生成張數）於本 amendment 未凍結，留至 confirmatory 閘定案。若偏離 CIFAR-10 的
  1000/class 口徑，須另以 pre-registration amendment 明記。
- grid 一經此 amendment 凍結，confirmatory 不得改點；若真跑後發現 grid 不足，屬另一輪 pre-registration。

### 規模

8 seed × 5 rep × 10 grid = 400 個 (seed, rep, w) cell，每 cell 需平衡生成一份 CIFAR-100 合成集並訓一個
TSTR 分類器；單 GPU 下屬數日級重計算，且為預註冊揭盲之不可逆步。
