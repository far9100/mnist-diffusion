<!-- 用途：D 包——CIFAR-100 預註冊（-12 §9 第 3 項例外，單檔）。承 D4（候選 B、X=1.5）、D8（留、第三訊號 recall、C0 -09-12）、-08-02 §5.2 全條款。D0-D12 逐項。仍待作者終簽之欄位（功效配置、D5 probe 尺寸、seed 公式、v2 形式、分支四骨架、base gate 數字）明標。D commit 早於任何 CIFAR-100 取樣。依 -06-05 §7、-08-02 §5、-09-04 §3、2026-07-05-02。 -->

# D 包：CIFAR-100 預註冊（定稿，作者終簽 2026-07-10）

## Goal

依定稿計畫 `records/2026-07-06-05` §7、執行指令 `records/2026-07-08-02` §5.2、執行指令（二）
`records/2026-07-09-04` §3.3 起草 CIFAR-100 預註冊包，納入作者已裁之 D4（候選 B、X=1.5）與 D8（留、第三
訊號 recall、C0 `records/2026-07-09-12`）。D commit 為 hard gate：早於任何 CIFAR-100 合成樣本（scout 含），
git 可驗；backbone 訓練不受阻（已完成 `checkpoints/cifar100_cfg.pt`）。仍待作者終簽之欄位於各條明標
「〔作者終簽〕」。正文限裁決必需項，論證引 B/C。

## Result

### D0：H1/H2/H3 原文與終審

逐字調出 `records/2026-07-05-02`：

- **H1（機制，confirmatory）**：驅動下游效用（TSTR）的是 coverage（多樣性）而非 fidelity（precision）。
- **H2（選擇器，confirmatory）**：免訓練的 CaF 能在候選組態中選到近效用最佳者（regret@selected 低）。
- **H3（對決，confirmatory）**：在 matched-budget 下，CaF 選 vanilla 組態的下游準確率打平或優於（簡化）Chamfer。

終審：CIFAR-10 上 H1 頭條被反證（C1 兩空間不分離）、H2 之 CaF 敗 FID-min（`records/2026-07-09-03` 判決三）。
H2 於 CIFAR-100 之數字門檻見 D4（前此無凍結門檻＝CIFAR-10 僅描述性）。H3 身分：確認為 confirmatory，但
其必跑性由 D7 條件化（分支相依）。CIFAR-100 之角色＝validation（`records/2026-07-05-02` L94：coverage 主導
不翻轉即原 thesis；翻轉則預宣告替代結論）。

### D1：四分支決策樹（承 `records/2026-07-08-03`）

1. 分離出現 且 CaF（v2）勝 FID-min → 邊界條件復活。
2. 分離出現 但 CaF 敗 FID-min → thesis 活、selector 死。
3. 不分離 但機制複製 → 診斷論文。
4. 皆否 → 負結果短文。

### D2：分支一先驗（已登記 `records/2026-07-08-03`）

整體約一成；誠實聲明、非裁決輸入；登記於 C1 揭盲前。路由由 D1 客觀觀察量決定，不由先驗。

### D3：分支三操作化＋介入臂

- **三觀察量（三中二判複製）**：(i) 低中段 near-boundary 於 coverage 升平區單調降；(ii) 高段 coverage 與
  TSTR 同崩；(iii) 高段 near-boundary 脫鉤。三中二成立＝機制複製。
- **介入臂（預註冊，D3 增列 `records/2026-07-08-02` §5.2.3）**：於 CIFAR-100 confirmatory 已生成合成集上跑一個
  C3 型 coverage-matched pruning，**定位為分支三論文宣稱之必要證據、非分支路由輸入**（路由仍由三觀察量判）。
  剪枝規則沿 CIFAR-10 C3（剪至最低 guidance 之 coverage 水準、移除離真實流形最近者、二分搜尋匹配，
  `records/2026-07-09-08`）；剪枝條件之具體 coverage 目標於 D10 網格凍結後定（隨網格、非終簽 blocker）。
  成本：無新取樣，僅剪枝＋重訓。**表徵口徑**：每觀察量標量測空間，DINOv2 主、Inception robustness
  （依 `records/2026-07-05-08` §4 型式）。
- CIFAR-10 之 C2/C3/C5 介入未強支持 near-boundary 機制（`records/2026-07-09-08`），故分支三之機制複製在
  CIFAR-100 屬真正待驗、非既定。

### D4：H2 數字門檻（作者已裁候選 B、X=1.5）

- **H2 通過 ⟺ CaF-v2 之 per-seed regret 至少比 matched-budget FID-min baseline 低 1.5pp**（候選 B、X=1.5，
  作者裁 `records/2026-07-09-04`）。錨定：CIFAR-10 上 FID-min（0.91）勝 CaF（3.69），selector 存在理由須為
  勝過更便宜之 FID-min；X=1.5 為勝幅門檻。與 D8 v2 價值判準同數（1.5）。
- **登記主張形式**：峰位噪聲穩健（高原＋懸崖，不押點峰）——C4 MDE 表示 3seed×1rep MDE 7.29pp 無法解析
  ~2.5pp 峰位（`records/2026-07-09-10`）。
- **功效配置（作者終簽）**：n_seed=8 × n_rep=5（MDE 2.49pp，`records/2026-07-09-10` 表），可解析 ~2.5pp
  峰位偏移。

### D5：強制 baseline 集與三臂對決

- baseline：matched-probe FID-min（引 C7 `records/2026-07-09-06`：小 probe 排序穩定、FID-min 可靠）、固定 w
  慣例值（登記出處）、隨機可行點。
- **三臂對決（matched-budget，同表）**：FID-min／CaF-v2／Chamfer；缺 FID-min 臂之「勝 Chamfer」無證明力。
- matched-probe 尺寸（作者終簽）：1000/class（與 CaF probe 同、遠在 C7 穩定區）。

### D6：復活條件與分離空間口徑

- 復活＝分離出現 且 CaF-v2 勝 FID-min（≥1.5pp）。
- **分離空間口徑（`records/2026-07-09-04` §0.1.5）**：雙空間（Inception＋DINOv2）皆分離＝完全復活；僅 DINOv2
  分離＝表徵條件弱復活。事前寫死。

### D7：Chamfer 牆條件化（作者核定裁量）

作者裁定（`records/2026-07-09-04` §5.2.7 核定文案）：「-12 §9 第 3 項之範圍例外延伸至 Chamfer 牆條件化規則
之事前登記（分支 1/2 必跑、3/4 選配），僅此一項。」CPU 空窗 Chamfer 適配照原排程。

### D8：CaF-v2（作者裁＝留、第三訊號 recall）

- **第三訊號＝recall**（C0 `records/2026-07-09-12`：recall 打破 w2.5 之 Pareto 支配——w2.5 recall .493 < 三
  oracle；density 無效）。**recall 為 DINOv2 量、不引 judge → 保住「免任務標籤」定位**（優於 near_bnd）。
- **v2 selector 形式（作者終簽）**：`argmax recall s.t. precision ≥ τ`（τ 沿 CaF 之 auto-τ＝0.9×real-vs-real
  precision）；於 CIFAR-100 validation 驗其 matched-budget regret 是否勝 FID-min ≥1.5pp。
- **價值判準**：v2 於 matched-budget 勝 FID-min regret ≥1.5pp（同 D4 X）。
- 框架：discovery（CIFAR-10，永遠 exploratory）／validation（CIFAR-100）。CIFAR-10 之 recall 打破支配為必要
  非充分。

### D9：量測程序凍結（凍程序不凍數字）＋種子公式＋metadata

- CIFAR-100 judge 品質 gate 定法、near-boundary 重校準程序、特徵空間依 `records/2026-07-05-08`。
- **metadata（承 E5/F1）**：driver 輸出強制含 nearest_k、有效 k=min(k,n−1)、tau_fraction、batch、完整 argv、
  start_timestamp、環境雜湊（torch/cuda/cudnn）。
- **種子公式無碰撞（作者終簽、已實作驗證）**：CIFAR-10 之 `seed*1e7+int(w*1e3)*1e4` 於本網格退化碰撞
  （`records/2026-07-06-05` §1.12，10 組碰撞、26 cells 涉入）。CIFAR-100 採 hash 派生
  `gseed(seed,w)=int(sha256(f"{seed}_{w:g}").hexdigest()[:15],16)`，實作 `datasets/cifar100_gseed.py`；
  全網格枚舉驗證（100 seeds × 91 ws=9100 cells → 9100 distinct，collision-free=True）。公式與驗證一併登記。
- **base-model gate（作者授權 agent 填，記為委派）**：clean-fid ≤ **20 @ 50k、w=1**。錨定：CIFAR-10 base
  8.95@50k（gate ≤10）；CIFAR-100 更難（100 類、500/類），約 2× headroom 為 round 且有意義之門檻。量測於 D10
  （取樣 D commit 後）；若量測值接近門檻，屬 backbone 品質之實質發現，非移動門檻之理由（凍程序不凍數字例外：
  此為委派之單一 gate 數字，明文留痕）。

### D10：時序（D commit 後）

D commit → judge 訓練與校準 → scout（僅定網格，讀數不回饋判準）→ 網格凍結 amendment → confirmatory。
各閘為作者 STOP 點（`records/2026-07-09-04` §0.2）。

### D11：凍結定義（承 E5 `claude.md` §5）

僅適用 D 包：prose＋程式＋已揭盲資料 dry-run＋輸出雜湊；P 資產為基底。

### D12：一頁論文骨架

- **分支三版**（`docs/paper_skeleton_branch3.md`，本批成稿）：so-what 骨＝`records/2026-07-06-05` §1.10「無單一
  便宜代理普遍可靠」（MNIST 與 CIFAR-10 selector 判決相反）；C1 反證後其現實性升高。
- **分支四版（作者終簽：另立）**：`docs/paper_skeleton_branch4.md`，負結果短文骨架。

## Follow-up

- **作者終簽完成（2026-07-10）**：D4 功效 8×5、D5 1000/class、D8 v2 形式 argmax recall s.t. precision≥τ、
  D12 分支四另立、base gate 委派 agent 填（≤20@50k）。D3 剪枝 coverage 目標於 D10 網格凍結後定（非終簽項）。
- **下一步（`records/2026-07-09-04` §3.4）**：main 併回（`records/2026-07-07-01` 掛帳 8）**需作者明確授權此 git
  步**（本 record 未預授）→ D commit（本定稿早於任何 CIFAR-100 取樣，git 可驗）→ E2 之 ⟨D⟩ 編號回填為
  `records/2026-07-09-13` → D10 時序啟動（judge 訓練校準 → scout 僅定網格 → 網格凍結 amendment →
  confirmatory；各閘為作者 STOP 點 `records/2026-07-09-04` §0.2）。
- 不觸凍結 JSON、不引任何 CIFAR-100 讀數（尚無）、無數字回改。
