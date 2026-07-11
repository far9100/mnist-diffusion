<!-- 用途：專案對外的更新歷史——依日期倒序列點記錄每一次計畫、實作、測試與裁決。 -->

# Changelog

本檔是專案的對外更新歷史，依日期倒序排列，新的在上。每列開頭的 ID 對應一份本機過程記錄
（`records/`，不隨工作樹發布）；原始全文留在 git 歷史，需要時以 `git log -- records/` 查閱。

動作字義：`plan` 計畫或規格凍結、`test` 執行與量測、`add` 新增程式或儀器、`refactor` 重構、
`proofread` 校對與事實同步、`audit` 稽核。

CIFAR-100 的預註冊全文另存於 `docs/prereg_cifar100.md`，該文件於揭盲前凍結、隨 repo 發布。

## 2026-07-11

- `2026-07-11-09` plan — D10 末閘：凍結 CIFAR-100 confirmatory 的 per-class 口徑為 gen=real=500。
  CIFAR-10 的 1000/class 匹配口徑物理不可行——CIFAR-100 訓練集每類僅 500 張，真實參考集上限即 500；
  500/500 是唯一能同時維持 gen=real 匹配並讓真實參考取滿的選擇。判準本體不動；coverage 絕對值跨
  資料集不可比一事於揭盲前登記為限制。
- `2026-07-11-08` debug — confirmatory 前置修正五項：凍結 grid/seeds/reps 寫入 driver 並對偏離者
  拒跑；`load_real_per_class` 樣本不足改為拋錯（原會靜默截斷，CIFAR-100 每類上限 500）；PRDC density
  分母改用夾限後的 k（現有路徑數值不變）；補回 `run_c6_fidmin_duel.py` 與 `run_c0_recall_density.py`
  兩支缺失 driver，對凍結結果逐值對帳皆 ALL_MATCH，並查明 C0 係自 P1 特徵重算、跨 seed 平均用
  `statistics.mean`（與 confirmatory 的 `sum/len` 差 1 ULP）；`cifar_data.py` 併入 `datasets/cifar.py`。
- `2026-07-11-07` refactor — 將 69 份過程記錄下架出版控（`records/` 進 .gitignore，本機保留、全文留在
  git 歷史），改以本檔為對外更新歷史；CIFAR-100 預註冊提升為 `docs/prereg_cifar100.md` 續留版控；
  62 處引用改寫（`.py` 溯源標記改純 ID、README 與 docs 內容性引用改指 CHANGELOG 錨點）；
  claude.md §1 改為 records 加 CHANGELOG 雙軌、§5.1 凍結定義改要求寫在版控的 `docs/` 文件。
- `2026-07-11-06` test — 將 confirmatory driver 一般化為 `--dataset cifar100`，接上 CaF-v2 recall
  selector 與 hash 派生 gseed；dry-run 驗得 80 cell 無碰撞、全鏈路 quick 通過。`--per-class` 待定，
  真跑未授權。
- `2026-07-11-05` plan — D10 第四閘：凍結 CIFAR-100 confirmatory grid 為 10 點
  {1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8}，僅依 coverage 幾何與跨資料集可比性，明文排除 scout 的 TSTR 讀數。
- `2026-07-11-04` test — D10 第三閘：1-seed 寬 grid scout（w 1 到 8）。coverage 峰在 w2
  （DINOv2 0.781）、崩點在 w5；TSTR 於 w1 最高 49.45 後單調降。
- `2026-07-11-03` test — D10 第二閘：在真實 CIFAR-100 訓 ResNet-18 judge，測試準確度 74.25%，
  以 margin p20 校準 near-boundary threshold 0.3622。margin 中位數 0.913，未如 CIFAR-10 飽和。
- `2026-07-11-02` test — D10 第一閘：CIFAR-100 base-model FID gate，w=1、50k 平衡樣本得
  clean-fid 11.226，落在事前門檻 ≤20 內，backbone 通過。
- `2026-07-11-01` proofread — 修正 claude.md §4 殘留他專案的措辭、§1.2 補列 audit action，
  並同步 results_analysis.md 的 CIFAR 預覽段與 CaF-v2 selector。未觸碰凍結敘述。

## 2026-07-09

- `2026-07-09-13` plan — 起草並定稿 CIFAR-100 預註冊 D 包（D0–D12）：D4 採候選 B 且 X=1.5、
  功效 8 seed × 5 rep、D8 保留 CaF-v2 且形式為 `argmax recall s.t. precision ≥ τ`、hash 派生無碰撞
  種子公式、base gate clean-fid ≤20@50k。作者於 2026-07-10 終簽。全文見 `docs/prereg_cifar100.md`。
- `2026-07-09-12` test — 依凍結 C0 規則離線重算 per-config recall/density。w2.5 的 recall 0.493 低於
  三個 oracle（w2 0.528、w1.5 0.555、w1 0.579），提供非支配訊號；density 仍被 w2.5 支配而無效。
  裁定 CaF-v2 的第三訊號採 recall。
- `2026-07-09-11` plan — D8 決策單：由單調 selector 不可能性推出 CaF-v2 必須引入第三訊號，
  列 near_bnd 與 recall/density 兩候選，附 C0 規則草稿。
- `2026-07-09-10` plan — D4 決策單：交付 σ_cls/σ_gen 與 MDE 表（3 seed × 1 rep 為 7.29pp、
  8 × 5 為 2.49pp）與 H2 候選門檻 A/B/C，建議登記主張採峰位噪聲穩健形式。
- `2026-07-09-09` add — 以補遺 record 將 C1、C4、C7、C2/C3/C5、C8 的 exploratory 結果連結入 B 定稿的
  閱讀脈絡，全數不改三判決，`2026-07-09-03` 原文不回改。
- `2026-07-09-08` test — C2/C3/C5 三項介入：near-boundary 剪枝的掉幅未大於隨機對照、純度過濾未單調
  上升，不支持 near-boundary 的因果角色；C3 顯示降 coverage 有超計數的 TSTR 代價，但橋接被 w2.5≈w1
  複雜化。全屬 exploratory。
- `2026-07-09-07` test — C4 變異分解（12 cells 加重訓）：σ_cls = 2.963pp、σ_gen = 1.182pp，分類器訓練
  噪聲主導約 2.5 倍。上升肢 w1→w1.5 的 +2.52pp 小於 σ_cls，維持 unresolved。
- `2026-07-09-06` test — C7 small-probe FID 排序穩定性：Kendall τ 最小 0.911，FID-argmin 於 15/15 個
  draw 全落 w1.5。裁定 FID-min baseline 於小 probe 可靠。
- `2026-07-09-05` proofread — 訂正 P0/P1/B 定稿「整個 confirmatory 逐位可重現」的過寬措辭：精確範圍
  限 7 項量測對帳 scalar；TSTR 因未種子化 shuffle 為非決定性，不在對帳集。
- `2026-07-09-04` plan — 作者第二份執行指令：追認 B 先於 C 批的次序偏差、放行 E2–E4、
  裁定 GPU 佇列為 C7 → C4 → C2/C3/C5 → backbone 訓練。
- `2026-07-09-03` test — **B 定稿，三判決**。判決一：thesis 在 CIFAR-10 尺度被本專案自家資料反證
  （FID-opt 與 TSTR-opt 重合於 w1.5、C1 於兩表徵空間皆不分離、「必然次優」全稱句撤下）。
  判決二：H-C2a 機械通過（DINOv2 偏 ρ = +0.658、Inception +0.859），但種子軸與可交換性軸無法內部
  驗證，須由 CIFAR-100 複製回答。判決三：selector 層為描述性，FID-min 的 per-seed regret 0.91pp
  勝 CaF 3.69pp（per-seed 2 勝 1 負），CaF 於本網格存在結構性 Pareto 失明。
- `2026-07-09-02` test — C1 which-FID 裁決：FD-DINOv2 的 argmin（w2）與 TSTR-argmax（w1.5）相距 1 格步、
  未超過 >1 門檻，判定不分離；疊加 Inception 側的 0 格步，兩個表徵空間皆不分離。
- `2026-07-09-01` test — P1 streaming 重生成全 30 config 並即時對帳：30/30 全部 ALL_BITEXACT、
  worst rel_delta 為 0，同時產出 per-config FD-DINOv2。總時約 8.35 小時、1.8 GB。

## 2026-07-08

- `2026-07-08-04` test — P0 決定性閘（單 cell：seed 10、w1）：verdict 為 ALL_BITEXACT，7 個 scalar
  逐位重現凍結 JSON，k=5 獲反證支持。單 cell 約 1225 秒、244 MiB。
- `2026-07-08-03` plan — 於 C1 揭盲前登記 CIFAR-100 四分支樹之分支一先驗約一成，明標為誠實聲明、
  非裁決輸入；路由仍由 D1 的客觀觀察量決定。
- `2026-07-08-02` plan — 作者首份執行指令：裁定 P0 溯源走路一並授權進 β 跑 P0，核定 D6 空間口徑與
  D7 文案，D4 數字門檻、D8 CaF-v2 去留、P1 greenlight 保留為作者欄位。
- `2026-07-08-01` proofread — 修正 cifar_data.py 檔頭矛盾與 results_analysis.md 過時陳述兩處結果無關的
  事實層問題，未寫入任何 confirmatory 數字與詮釋。

## 2026-07-07

- `2026-07-07-01` audit — 全專案健康掃描：39 個 .py 無阻斷 bug，現修 README 事實矛盾與 pyproject 依賴
  缺漏等四項；查出 P0 探針的 nearest_k=5 僅為文件推定、非儲存值，處置待作者裁決後才進 β。

## 2026-07-06

- `2026-07-06-17` plan — 搭出 B 骨架：三判決分立、禁止交叉混寫，判決三強制寫 per-seed 對稱句
  （FID-min 為 2 勝 1 負），判決一待 C1 補入。
- `2026-07-06-16` plan — C8 證出 Pareto 失明引理：只要有組態嚴格支配所有 oracle，
  `argmax coverage s.t. precision ≥ τ` 在任何 τ 下都選不到 oracle。CIFAR-10 上 w2.5 支配三個 oracle，
  故 CaF 失效是結構性的，不是校準問題。一頁版見 `docs/c8_pareto_blindness.md`。
- `2026-07-06-15` plan — C7 凍結 small-probe FID 排序穩定性規則：100/250/500 per-class 三種 probe 尺寸
  各重抽 5 次，以 Kendall τ 與 FID-argmin 是否改變判定。
- `2026-07-06-14` plan — C5 凍結 near-boundary 純度過濾規則：於 w1.5 按 margin 構造三個純度水準子集
  各重訓 2 次，測 TSTR 是否隨純度單調上升。
- `2026-07-06-13` plan — C3 凍結 coverage-matched pruning 規則：將 w2.5 剪至 w1 的 coverage 水位後重訓
  TSTR，若差距顯著縮小則 coverage 本身承載效用。
- `2026-07-06-12` plan — C2 凍結 boundary-targeted pruning 的干預規則：於 w1.5 與 w2.5 移除 margin 小於
  0.9525 的 near-boundary 樣本，以等量隨機移除為對照，每情境 2 次重訓。
- `2026-07-06-11` plan — C1 於 DINOv2 揭盲前凍結 which-FID 分離口徑（均值曲線的 argmin 與 argmax 相異
  且大於 1 格步才算分離），並預寫雙分支敘事。
- `2026-07-06-10` test — C6 per-seed FID-min 對決（純讀 JSON）：FID-min 的 regret 均值 0.91
  （2.45 / 0.28 / 0.00）勝 CaF 3.69（0.54 / 5.03 / 5.49）約 2.78pp，且 Inception 側格步 [1, 1, 0] 皆不分離。
- `2026-07-06-09` test — A 批收束，逐字鎖定判決二：C2a 偏 ρ = +0.658、p = 0.0188（Inception 側 +0.859、
  p = 0.0008），機械 verdict 為 pass；但 bootstrap CI [−0.601, +0.675] 跨零、可交換性受損，
  穩健性須由 CIFAR-100 複製回答。
- `2026-07-06-08` plan — A3 block-permutation 規格：restricted 置換空間 746,496 足夠大，但 per-w 中心化
  方案因低段 TSTR 的 SE 2.2 至 2.65pp 蓋過 0.80pp 的上升肢而不可行，裁定不硬跑、誠實收 CIFAR-100。
- `2026-07-06-07` plan — A0 於揭盲前凍結 H-C2 permutation 參數（N = 100,000、seed = 0、α = 0.05）
  與兩則分支敘事，並凍結 block-permutation 的分塊結構為 gseed 反對角線 14 塊。
- `2026-07-06-06` proofread — 交付稽核 dossier：彙整協定門檻、30 configs 的 per-seed 全量五指標、
  τ 與 real-vs-real 參考、per-config characterization FID 與時序，作為定稿計畫的唯讀 backing，本身不含裁決。
- `2026-07-06-05` plan — 合併 v1/v2/errata 與第三輪覆審的定稿修正計畫，列 12 項事實基（FID 與 TSTR
  同峰於 w1.5、FID-min regret 0.91 對 CaF 3.69、gseed 碰撞使 30 cells 只有 14 份初始噪聲），並宣告計畫凍結。
- `2026-07-06-04` plan — v2 的 errata，裁決五項殘留：P 重構為 streaming P1×C1、A2 解除 gate 直跑、
  影像儲存估從 2.8GB 更正為 0.92GB、新增 C4 混池規則。已由 `2026-07-06-05` 取代。
- `2026-07-06-03` plan — 修正計畫 v2：稽核推翻「零重生成」前提（driver 已刪 per-sample 資產），
  新增持久化工作包 P 並把 C 批除 C6 外全掛 P 閘之後。已由 `2026-07-06-05` 取代。
- `2026-07-06-02` plan — 揭盲後首版修正計畫，切出 A 到 F 六個工作包。已由 `2026-07-06-05` 取代。
- `2026-07-06-01` proofread — 校對 README，只同步與 confirmatory 結果無關的三處，補列 8 個漏列的研究
  主線腳本；帶結果數字的進度節延後至 C2 裁決後處理。

## 2026-07-05

- `2026-07-05-14` add — 補齊三項凍結規格的儀器（per-class 超額 label-noise、per-config characterization
  FID、C2 偏相關腳本），合成資料驗證裁決路徑可跑；1-config 計時探針約 1000 秒，估 30 configs 約 6 至 8 小時。
- `2026-07-05-13` plan — 將 C2 裁決落為協定的事前統計規格：n = 10 config 為觀測單位，
  C2a 偏相關 ρ(TSTR, coverage | precision, 超額 label-noise) 顯著為正且 C2b 不顯著方算通過，
  並要求報效果量、CI 與 n = 10 的功效限制。
- `2026-07-05-12` plan — 研究定位 v2（現行定位依據）：thesis 收緊到 CFG guidance 軸、重心移到 CaF、
  C2 裁決改為全網格偏相關並廢除分段、明文承認護城河變薄，宣告治理凍結至 CIFAR-100 閘與 Chamfer 對決取得資料為止。
- `2026-07-05-11` plan — 上緣 coverage-only scout 顯示 coverage 由 0.533 單調降到 0.259、未觸底，
  故以先驗理由（CFG 實用範圍至多 w8）裁決 confirmatory grid 封頂於 w = 8，凍結 10 點 grid 與 steps = 50、η = 0。
- `2026-07-05-10` plan — 研究定位調整 v1 草稿。稽核出幽靈引用、C2 分段構成 HARKing 漏洞等四項缺陷，
  判定 superseded，留作決策軌跡。已由 `2026-07-05-12` 取代。
- `2026-07-05-09` test — 依 amendment 用 50k 樣本重量 base model 的 clean-fid，得 8.950，通過事前帶 ≤10；
  證實先前 5k 的 13.95 是小樣本正偏誤而非模型問題。
- `2026-07-05-08` plan — 預註冊稽核認定 Stage 4 主結果因 grid 在看過 scout 後才鎖定、seed 與 scout 重疊，
  只能算 exploratory；故在任何 confirmatory 資料前凍結新規格：50k FID 閘 ≤10、judge 凍結逐類扣 floor、
  上緣 coverage-only scout 定崩點、fresh seeds 10/11/12。
- `2026-07-05-07` test — 在定死 grid 上跑 3 seed 全量：TSTR 峰在 w = 2（46.01%），
  CaF regret@selected 0.280pp、top-3 命中 100%。判定 coverage 驅動效用而 precision 不驅動，
  C2 核心存活，新意是 coverage 非單調。
- `2026-07-05-06` test — 1 seed 寬 grid（w 1 至 8）scout 自訓 CFG：TSTR 峰移到 w ≈ 3、coverage 於 w5 與 w8
  崩（DINOv2 0.836 降至 0.607），確認雙力拉鋸，據此定死 grid {1, 2, 3, 4, 5, 8}。
- `2026-07-05-05` add — 在真實 CIFAR-10 訓 ResNet-18 judge，測試準確度 93.08%，以真實 margin 的 p20
  分位校準 near-boundary threshold 0.9525；同時指出 CIFAR-10 margin 中位數 0.999、訊號偏飽和。
- `2026-07-05-04` test — 以 5000 張、steps = 50、η = 0 量自訓 CFG CIFAR-10 的 FID：clean-fid 13.952 落在
  自證堪用帶 5 至 15，FD-DINOv2 324.262，通過。固定量測 sampler 為 (steps = 50, η = 0)。
- `2026-07-05-03` plan — 依早期預警的 EDM proxy 對協定做 amendment：加寬 guidance grid 至 coverage 會崩的
  高段、把標籤噪音與 precision 升為第一等量測，並把 C2 低段修訂定位為 exploratory，不放寬既有門檻。
- `2026-07-05-02` plan — 在任何 CIFAR 掃描結果進來前預註冊 Phase 1 協定，鎖定 H1 機制、H2 CaF 選擇器、
  H3 matched-budget 對決三項 confirmatory 假設、指標、網格定死規則與五條 go/no-go 閘。
- `2026-07-05-01` plan — Phase 1 執行計畫，採 signal-forward 的安全子集：保留並行與早期預警，剔除會使
  regret 循環的先剪枝與內建 η 遷移假設，go/no-go 仍以完整網格的 regret@selected 為準。

## 2026-07-04

- `2026-07-04-03` plan — Phase 1 排程優化版，以 signal-forward 與並行把 CIFAR-100 翻轉的生死信號提前到
  第 2 至 3 天。已由 `2026-07-05-01` 取代，其 L4 至 L6 判定為以嚴謹換速度而不採用。
- `2026-07-04-02` refactor — 輸出安全整理與文件重組：samples_cifar 加入 gitignore，前言與方法併入 README、
  實驗結果數字移到 `docs/results_analysis.md`。
- `2026-07-04-01` refactor — 依 claude.md 規範整理專案：11 份 records 改名為 `YYYY-MM-DD-NN` 格式，
  30 個 .py 檔頭與註解中譯並經 tokenize 驗證程式碼未變，README 改寫為研究主線。

## 2026-07-03

- `2026-07-03-10` add — 啟動 Phase 1 建置：擴充自有 ddpm.py 到 CIFAR 而非叉外部 repo，完成 train_cifar.py
  等模組並啟動 47.9M 參數的 CFG CIFAR-10 訓練。量測堆疊核心通過。
- `2026-07-03-09` plan — 解掉 backbone 分岔：自訓 CFG-capable CIFAR 模型為主軸、預訓練 EDM 只當量測錨點、
  ImageNet 延後，並把 CIFAR-100 的 C2 機制證據列為最優先。
- `2026-07-03-08` plan — 重生成戰略主檔：論文脊椎定為 CaF 選擇器加機制掛帥、C1 曲面降為背景，
  並訂出 Gate A 三項近零成本生死判定與各階段閘門。
- `2026-07-03-07` plan — 依文獻重掃改寫研究定位：判定白區已被 Chamfer Guidance 侵蝕，策略轉為 C3-first、
  砍到一篇 CIFAR 論文，go/no-go 改用 regret@selected 取代全域 Spearman。
- `2026-07-03-06` test — CIFAR-10 快速預覽（1 seed、每類 500 張、8 步）：CaF 選 w2、oracle 最佳為 w1.5、
  regret 0.51pp；coverage 呈非單調而效用有內部最優，與 MNIST 不同。
- `2026-07-03-05` proofread — 拆解對手 Chamfer Guidance 並逐維度比對定位：差異空間仍在，但賣點必須轉向
  可組合性、免任務分類器的操作點與為選擇證明的機制，不再宣稱發現「FID 不等於效用」或比 CFG 便宜。
- `2026-07-03-04` test — 以 MNIST 現有資產判定 Gate A 三準則：Chamfer 差異空間、多 seed 選擇器訊號
  （CaF regret 0.000pp、rank 1/6）與機制方向皆通過，判決為可進 Phase 1；但 MNIST margin 飽和使機制證據
  須留待 CIFAR-100。
- `2026-07-03-03` test — EDM CIFAR-10 條件模型 50000 張的 FID 重現作為量測正確性閘：得 1.848、
  論文參考 1.79、差 0.058，通過。EDM 僅作量測錨點，不用於 CFG。
- `2026-07-03-02` test — 建 PRDC、CaF selector 與機制分析基礎設施並取 Phase 0 早期訊號：CaF 在三個 seed
  全選中 oracle 最佳的 g1、regret 0.000pp，coverage 與 TSTR 同向而 precision 不追蹤，go/no-go 綠燈。
- `2026-07-03-01` test — 全量跑完取樣器與 guidance 兩組掃描：FID 最佳的 guidance 2.0 不等於 TSTR 最佳的
  guidance 1.0；DDIM 50 步比 DDPM 1000 步快 20 倍而 TSTR 更高（97.22%），預設組態被 Pareto 支配。

## 2026-07-02

- `2026-07-02-01` plan — 規劃以同一組 DDPM 權重接上 DDIM 取樣器，掃 steps、η 與 guidance 以解耦 FID 與
  TSTR。煙霧測試通過，DDIM 10 步達 92 倍加速，多樣性隨 guidance 單調崩塌。

## 2026-05-03 以前（無過程記錄，僅 git 提交）

本階段為 MNIST sandbox 的建置期，尚未採用 records 慣例，內容以 git 提交訊息為準：

- 2026-05-03 — 為 MNIST 上的 VAR-mini 訓練實作多尺度殘差 VQ-VAE。
- 2026-05-01 — 評估器改為 TSTR、精簡推論期視覺化，新增 TSTR 診斷腳本並以正式語氣重寫 README。
- 2026-05-01 — 建立 CNN 評估器管線與端到端分析報告，產出自述式的 .txt 與 .json 評估報告。
- 2026-05-01 — 重構程式結構、移除失效檔案，並將生成影像持久化以免重跑。
- 2026-02-28 — 專案初始化。
