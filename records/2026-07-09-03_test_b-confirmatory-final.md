<!-- 用途：B 定稿——confirmatory 三判決分立 record（定稿）。承 B 骨架 records/2026-07-06-17，填入 C1 which-FID 反證結果（-09-02）、P0/P1 對帳結論（-08-04/-09-01）、B1 排序複核。三判決互不裁決、禁交叉混寫。機制介入批 C2/C3/C5（需重訓）延後為 GPU 批，不入三判決。依 records/2026-07-06-05 §5、2026-07-08-02 §4.4、2026-07-06-09/-10/-16。 -->

# B 定稿：confirmatory 三判決分立 record

## Goal

依定稿計畫 `records/2026-07-06-05` §5 與執行指令 `records/2026-07-08-02` §4.4，將 B 骨架
（`records/2026-07-06-17`）定稿為 confirmatory record：三判決互不裁決、禁交叉混寫。相對骨架補入：判決一
之 C1 which-FID 結果（`records/2026-07-09-02`）、附錄之 P0/P1 對帳（`records/2026-07-08-04`、
`records/2026-07-09-01`）、B1 排序複核。判決二逐字引 A 批（`records/2026-07-06-09`），判決三引 C6
（`records/2026-07-06-10`）與 C8（`records/2026-07-09-02` 之前 -06-16 引理，一頁版 `docs/c8_pareto_blindness.md`）。

範圍註記：機制介入批 C2/C3/C5（near-boundary 剪枝、coverage-matched 剪枝、純度過濾——皆需分類器重訓）
與 C7（small-probe FID）為獨立 exploratory 機制/穩健性證據，非三判決之相依，延後為 GPU 批（見 Follow-up）。
本定稿之三判決以已完成之 C1/C6/C8/P 為據，完整成立。

## Result

### 判決一（thesis）

- **FID/TSTR 重合**：於 w1.5（clean-fid 均值 8.82 / TSTR 均值 63.96，兩者 argopt 同落 w1.5）。B1 排序複核
  （n=10 均值曲線，Spearman）：ρ(−char_clean_fid, TSTR) = 0.964；ρ(coverage, TSTR) = 0.624（骨架與 §1.1 之
  「≈0.64」即 0.624 之修約，非 Pearson——Pearson 為 0.719；計算式為 10 點均值曲線 Spearman，消歧完成）。
  README 頭條「FID-opt 偏離 TSTR-opt」在本資料上被反證。
- **災難性非單調**：w≥3 崩 11–30pp（TSTR 均值 w3 53.08 → w8 33.46）；「最優位置跨資料集移動」為存活觀察。
- **「內部最優」**：為未登記 exploratory 觀察，附上升肢 w1→w1.5 配對差 +0.80±3.3pp（SD 3.26、SE 1.88，
  3 gen seeds 下不可判定，dossier 乙-6）；明文從未受 confirmatory 保護。
- **「必然次優」全稱句撤下**（E2 執行，措辭待作者核）。
- **C1 which-FID（已裁，`records/2026-07-09-02`）**：強命題 pre-closed（Inception/clean-fid 側 0 格步不分離，
  `records/2026-07-06-11` §2、C6）。DINOv2 側揭盲後對號入座：FD-DINOv2 均值 argmin = w2（三 seed 全 w2）、
  TSTR argmax = w1.5，**格步 = 1，依凍結口徑（>1 才算分離）判不分離（反證確立）**，落「1 格步」點（DINOv2
  一格偏移、仍不分離，1 格偏移記誠實邊註、與 Inception 0 格步對照）。**兩特徵空間皆不分離 → CIFAR-10 尺度
  「FID≠效用」頭條命題反證確立**：FID（兩表徵）皆為近最優 selector。誠實邊註（輔助、不覆蓋主裁）：per-seed
  格步 [0,2,1] 方向非一致。thesis 於 CIFAR-10 尺度被資料反證，最終存活與否移 CIFAR-100。

### 判決二（H-C2）

逐字引 `records/2026-07-06-09`「B 判決二之凍結內容」全段（頂格 C2b 號翻經驗證據 ＋ 主體三重 caveat ＋
A2 增列 ＋ 三軸總結句 ＋ 結論），不重寫。淨結論：CIFAR-10 上機械通過（A1 DINOv2 C2a ρ=+0.658 p=0.0188；
A2 Inception C2a ρ=+0.859 p=0.0008，跨表徵一致），但穩健性沿種子軸（bootstrap CI 跨零）與可交換性軸
（gseed 碰撞）均無法內部驗證，最終須 CIFAR-100 獨立複製回答。H-C2a 顯著亦不得寫「coverage 驅動效用」等
因果措辭（`records/2026-07-03-07` §4、`records/2026-07-06-05` §2 格殺）。

### 判決三（selector，描述性）

- **開頭明文**：協定未凍門檻，本節不作過／敗判定。
- **Pareto 失明**：w2.5 (.873, .792) 嚴格支配三 oracle（w2 .858/.777、w1.5 .841/.751、w1 .806/.645）→ CaF
  結構性選不到 oracle，非校準問題（C8 引理 `docs/c8_pareto_blindness.md`；tau_robustness.picks 實證 τ 全段
  選 w2.5 或更高，dossier 乙-5）。
- **τ knife-edge**：w1 於 seed11/12 之可行邊際僅 .0034/.0092，可行性由 seed 級噪聲決定。
- **FID-min 對決（C6，`records/2026-07-06-10`）**：FID-min per-seed regret 0.91（2.45/0.28/0.00）vs CaF 3.69
  （0.54/5.03/5.49）。FID-min 更便宜（免 τ、免 coverage）卻近最優，打在 CaF 存在理由上。**per-seed 對稱句
  （防讀成全面碾壓）**：FID-min 為 3 seed 2 勝 1 負——seed10 上 CaF(0.54) 反勝 FID-min(2.45)，FID-min 負的
  那次輸得少、贏的兩次贏得多。壞消息不稀釋、CaF 那一格好消息不抹掉。
- **regret 數字口徑**：主 **3.69**（per-seed regret 均值）；並列 **2.77**（mean-curve oracle 口徑：均值 oracle
  w1.5 之 63.96 減均值 CaF w2.5 之 61.19 = 2.77）。兩數差異來源即口徑（per-seed vs mean-curve），明標。
- **可辯護措辭**：CaF 為可靠避崖器、糟糕平台優化器——FID-min 同樣避崖且成本結構相同。modal_fraction 1.0
  （三 seed 全選 w2.5）重讀為低變異高偏差。

### 附錄

- **P 對帳結果（填 `records/2026-07-06-17` 保留槽；`records/2026-07-08-04`、`records/2026-07-09-01`）**：
  P0 單 cell（seed10, w1）ALL_BITEXACT（路 A 同 gseed 生成逐位相同、7 scalar 逐位重現凍結 JSON）；
  P1 全 30 config ALL_BITEXACT（worst rel_delta = 0，零 over_tol、零 within-not-bitexact）。k 溯源：k=5
  下 k-dependent scalar 逐位重現 → k=5 正向反證（`records/2026-07-05-14` 文件推定實證確立），無降級語義
  觸發。整個 confirmatory 結果集本環境逐位可重現；判決一/三之資料基底與 C 批之 P 資產信任鏈確立。
- **時序鏈**：052492c（規格凍結 07-05 14:59）→ ec1f746（儀器 23:53）→ 推導起跑 ≈07-06 00:29（標推導值、
  非登記）→ mtime 08:45。
- **ec1f746 結論（F2，填保留槽；dossier 乙-2）**：ec1f746 @ 07-05 23:53:43 之 confirmatory 儀器 commit，
  3 檔 +295/−19（run_c2_partial.py 等，純實作），早於全量開跑；屬凍結後行使實作自由度之流程債，純事實入
  附錄、不代敘事。
- **gseed 碰撞（壞消息）**：反對角線共享噪聲（30 cells 僅 14 份獨立噪聲）、H-C2 可交換性受損、CIFAR-10 內部
  無乾淨 restricted 補救（A3 (c)、`records/2026-07-05-08` spec）。
- **scout 乾淨（好消息，不稀釋壞消息能見度）**：兩 scout 皆 flat seed=0，種子值域與 confirmatory 之
  (seed+w)×10⁷ 不相交、無噪聲重疊（§1.12 訂正）。與碰撞分述、不互稀釋。
- **MNIST 降級**：分離證據軼事級（一格步、單 seed argmax、bespoke FID、方向不轉移）。
- **fee419e 時序（純事實）**：fee419e commit 於 2026-07-05 22:37，早於 confirmatory 起跑約兩小時；定位重寫
  為 pre-data，與隨後揭盲的資料未經對賬。
- **揭盲時間線**：均值揭盲 → A0 凍結（permutation N/seed、分支敘事、block 結構）→ A1 p 值揭盲 → C1
  DINOv2 揭盲（P1）→ 對號入座（`records/2026-07-09-02`）。

## Follow-up

- **E2/E3/E4（帶結果文件同步，作者核措辭）**：README 頭條依判決一改寫（「FID-opt 偏離 TSTR-opt」在 CIFAR-10
  被反證、C1 兩空間不分離、「必然次優」撤下、最終定位待 CIFAR-100）；docs/results_analysis.md 鏡像三判決；
  intro banner。依 `records/2026-07-06-05` §8，等本定稿 + 作者核措辭後執行；E2 措辭先呈作者（`records/2026-07-08-02`
  §0.2、§6.1）。**本定稿為 §4.4 STOP 呈報點。**
- **機制介入批（延後 GPU 批）**：C2（near-boundary 剪枝重訓）、C3（coverage-matched 剪枝重訓）、C5（純度過濾
  重訓）皆需分類器重訓 TSTR，與 C4（變異分解重訓）同類；C7（small-probe FID 穩定性，需 GPU 不重訓）餵 D5。
  歸為一組 GPU 批另跑，為 exploratory 機制/穩健性補強，非三判決相依。C5 需與 run_flip_earlywarning.py 對賬
  （`records/2026-07-06-14` 掛帳）。
- **δ（D 起草）可並行**（`records/2026-07-08-02` §5）：C7 結果餵 D5、C8 單調 selector 推廣餵 D8；D4 門檻、
  D8 CaF-v2 去留為作者保留欄位。
- 三判決分立、C6 數字只進判決三、判決二逐字引 -09；不觸凍結 JSON、無數字回改。dossier（`records/2026-07-06-06`）
  為事實基 backing。
