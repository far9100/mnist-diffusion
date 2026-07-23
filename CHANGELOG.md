<!-- 用途：專案對外的更新歷史——依日期倒序列點記錄每一次計畫、實作、測試與裁決。 -->

# Changelog

本檔是專案的對外更新歷史與唯一記錄，依日期倒序排列，新的在上，逐列記錄每次計畫與更新。每列開頭的
ID 為該列識別碼（`YYYY-MM-DD-NN`）。2026-07-19 起停用本機 `records/`；2026-07-11 以前的舊記錄仍留在
git 歷史，可以 `git log -- records/` 查閱。

動作字義：`plan` 計畫或規格凍結、`test` 執行與量測、`add` 新增程式或儀器、`refactor` 重構、
`proofread` 校對與事實同步、`audit` 稽核。

CIFAR-100 的預註冊全文另存於 `docs/prereg_cifar100.md`，該文件於揭盲前凍結、隨 repo 發布。

## 2026-07-23

- `2026-07-23-01` test — 完整版 fix_tasks T8 全 weight 掃描（seed10、weights {0.05,0.1,0.3,1.0}、雙向＋單向
  ablation、G_freq=5）＋§5.6.1 robustness：`results/cifar100_h3_duel_v2_dinov2_sweep.json`。cov@224 於每個
  weight 皆低（0.46–0.49 << vanilla 0.642）、cov@112 皆高（0.85–0.88）、TSTR 增益隨 weight 遞增
  （+0.74／+0.67／+2.24／+3.03pp）；即量測解析度落差非 weight 1.0 特例、於整個範圍穩健，雙向 cov@224 於任一
  weight 皆未上升。導引生成非決定性使 weight 1.0 TSTR 本掃描 61.68 vs MVP 62.22（差 ~0.5pp、σ_cls 內）。
  §5.6.1 增 weight 掃描段；verifier 增 2 條，重跑 OK 383／MISMATCH 0；凍結與 MVP 結果未動。
- `2026-07-23-02` test — 完整版 fix_tasks T5b 3-seed（seeds 10/11/12、續用 seed10 partial 只補 11/12）＋§6.3：
  `results/cifar100_subunity_scout.json`（現 3 seed）。三 seed 皆單峰、峰皆在 w1——w0.5 42.07–42.74 < w0.75
  55.80–56.36 < w1 59.44–59.98 > w1.5 58.65–59.52，排序穩健，坐實「w1 為真實內部峰、非網格邊界假影」跨 seed。
  §6.3 由單 seed 改 3-seed；verifier 增 2 條（seed11/12 w0.75），重跑 OK 385／MISMATCH 0；seed10 數字與凍結
  未動。完整版兩項（T8 全 weight 掃描＋T5b 3-seed）完成。
- `2026-07-23-03` add — 完整版 fix_tasks T9 第二架構（作者授權）之程式：`src/core/cifar_classifier.py` 增
  `WideResNet16_4`（WRN-16-4、2.77M 參數、feat 256 維）與 `_WRNBasicBlock`；`run_tstr` 增 `arch` 參數
  （預設 resnet18 保現行逐位行為，wrn16_4 走 WRN）。`run_tstr_protocol.py` 增 `--arch {resnet18,wrn16_4}`
  並串至 ablation。dry-run：py_compile／WRN forward（feat 256、logits 100、2.77M）／--help／cifar_classifier
  self-check（ResNet18 路徑逐位不變、T6b 決定性）皆過。隨即以 `--arch wrn16_4 --reps 3 --epochs 15 50
  --generate-missing` 跑第二架構 ablation → `tstr_protocol_ablation_wrn.json`（驗 cell 排序之架構穩健性）。

## 2026-07-22

- `2026-07-22-01` test — 任務書 fix_tasks T11（B1）ViT-L/14 PRDC 真跑完成（CIFAR-100 seed10 全 10 config、
  8 格 GPU regen＋2 快取，約 9.4h）：`results/cifar100_prdc_vitl14_seed10.json`。which-FID 與 coverage 排序對
  DINOv2 backbone **穩健**——ViT-L（1024 維）與 ViT-B（768 維）之 coverage-argmax 皆 w2.5、FD-argmin 皆 w2.5、
  FD-argmin 與 TSTR-argmax（w1）皆相隔 3 格判分離；絕對值隨 backbone 改變（ViT-L coverage 系統性低約 0.15、
  FD 較高）但排序不變。故 Pareto 失明與 which-FID 診斷不依賴特定 DINOv2 尺寸（回應 Stein et al. 之 ViT-L
  建議）。整合 §6.3；verifier 增 1 條（ViT-L 分離格步 3），重跑 OK 368／MISMATCH 0；凍結未動（commit a814b94 之記錄）。
- `2026-07-22-02` test — 任務書 fix_tasks T8（A5）H3 公平化 v2 真跑（MVP：seed10、weight 1.0、雙向 vs 單向、
  G_freq=5）＋§5.6 重寫：`results/cifar100_h3_duel_v2_dinov2.json`。雙向 Chamfer TSTR 62.22（+3.57pp）、單向
  61.79（+3.14pp），兩者 cov@224 皆低（0.460／0.462，對 vanilla 0.642）、cov@112（導引空間）皆高（0.853／
  0.829）。**「coverage 反低」非單向簡化假影**（雙向與單向 224-coverage 幾乎相同），**而是量測解析度相依**
  （於導引所用 112 解析度 coverage 反高）；官方 0.603→0.912 應為導引/較低解析度所量。據此修正 T7.5 之「單向
  假影」推測（原係未測前之保守 hedge）：新增 thesis §5.6.1、§5.6 標題/bullet/意涵/圖說＋摘要/§6.3/§6.4/§7 與
  paper_branch3／results_analysis 一致改為「量測解析度相依」。FLOPs：雙向約 21750 TFLOPs 對 vanilla 15000
  （張數 matched、FLOPs 不 matched）。全 weight 掃描為後續。verifier 增 5 條，重跑 OK 373／MISMATCH 0；凍結
  v1 H3／prereg／verdict 未動。
- `2026-07-22-03` test — 任務書 fix_tasks T5b（A8）w<1 網格 scout 真跑（MVP：seed10 w0.5/0.75、off_protocol）
  ＋§6.3 整合：`results/cifar100_subunity_scout.json`。TSTR 隨 w 為單峰——w0.5 42.74 < w0.75 55.80 < w1 59.44
  （峰）> w1.5 58.65 > w2 55.44。故 **w1 為真實內部峰、非網格邊界假影**（回應正文多處「w1 為網格邊界點」之
  保留）；w<1 因往無條件內插、類別條件性下降（w0.5 precision 0.66／coverage 0.23）而 TSTR 掉。整合 §6.3；
  verifier 增 2 條（w0.5／w0.75 TSTR），重跑 OK 375／MISMATCH 0；凍結未動。全 3-seed 擴充為後續。
- `2026-07-22-04` test — 任務書 fix_tasks T9（B2）TSTR 協定強化真跑＋§5.4.3 整合：`results/tstr_real_ceiling.json`、
  `tstr_protocol_ablation.json`。(1) real 上限線：train-on-real TSTR＝CIFAR-100 70.75（15ep）／76.22（50）、
  CIFAR-10 64.78／79.46；合成最佳距 15-epoch 上限約 6–11pp（CIFAR-10 real 15→50ep 大漲，示 15ep 未訓足）。
  (2) epochs 消融（seed10 w1/w1.5/w2.5，w1.5 regen）：cell 排序**隨 epochs 翻轉**——15ep w1（59.96）>w1.5
  （58.66）、50ep w1.5（61.52）>w1（61.34）、w2.5 恆最差；即「oracle w1」為 15-epoch 協定相依，訓足後 oracle
  反成 w1.5（＝FID-min 選中，與 §5.2.1 v2 一致），w1/w1.5 之差落在 σ_cls 內。此為「噪聲地板為協定內生」之
  直接證據。整合 §5.4.3；verifier 增 4 條，重跑 OK 379／MISMATCH 0；凍結未動。第二架構為後續。
- `2026-07-22-05` test — 任務書 fix_tasks T10（B3）margin-pruning 介入真跑＋§5.5 整合：
  `results/cifar100_margin_intervention.json`。於 w2.5 移除 judge margin 最低 n 個 vs 等計數隨機、N=8 重訓：
  n=13606 差（rand−margin）−1.03pp（SE 0.46、CI [−1.93,−0.12]、MDE 1.29）、n=6803 −0.87pp（SE 0.42、
  CI [−1.70,−0.05]、MDE 1.18）。兩檔位方向一致——移除低 margin 樣本反而略**提升** TSTR（與「near-boundary
  供給承載效用」相反、幅度皆 < MDE）。故直接介入同樣**未支持**機制之因果宣稱，與 D3 coverage-matched 介入之
  null 同向。§5.5 併述兩種介入；verifier 增 2 條，重跑 OK 381／MISMATCH 0；凍結未動。Stage 4 全 7 項真跑完成
  （commit c99ee8d 之記錄）。

## 2026-07-21

- `2026-07-21-01` proofread — 任務書 fix_tasks T7.5（A5）H3 主張降級：把「Chamfer 增益不見於 coverage」於
  摘要（中英）、§5.6（標題／bullet／意涵／圖 5.6 圖說）、§6.3 限制、§6.4、§7 一律改為條件句——保留穩健的
  「Chamfer 下游效用勝任何 vanilla（+2.54pp DINOv2／+3.11pp judge）」，但明記本文 Chamfer 為單向簡化重寫、
  單 seed，其 coverage 反低與官方雙向實作（報告 coverage 上升 0.603→0.912）方向相反，故該讀數可能為簡化
  假影、待雙向公平化後方可定論；`paper_branch3_diagnostic.md`、`results_analysis.md` 同步。§5.6 未實體搬
  附錄（避免破壞交叉引用與表 5.5 對帳，列為選配）。純措辭未動數字，凍結未動。
- `2026-07-21-02` proofread — 任務書 fix_tasks T12.4：EDM 正確性 gate 口徑差註記——`thesis_draft.md`（量測
  正確性錨點）、`results_analysis.md`、`docs/code_map.md` 明記本專案 EDM CIFAR-10 FID 1.848 為單次評估、
  官方 1.79 為 min-of-3（NVlabs EDM README），兩者評估口徑不同、非同口徑比較（僅作 backbone 正確性錨點）。
  純措辭未動數字。
- `2026-07-21-03` proofread — 任務書 fix_tasks T12.8：補齊參考文獻正式清單並統一格式——把 §2.4 已引但未列於
  清單的 14 篇（Ravuri & Vinyals、Azizi、Alaa、Kynkäänniemi 2019／2024、Sariyildiz、Shipard、StableRep、
  SynCLR、Astolfi、CADS、Karras autoguidance、Bartlett、Sorscher）依主題新增為條目 16–29、各附 arXiv；
  Dockhorn 條目 venue 統一為 TMLR 2023、去「順帶」殘留；Bradley & Nakkiran（2408.09000）經 web／dblp 查僅見
  arXiv／CoRR 記錄，正式 venue 由「NeurIPS 2024 M3L Workshop」改標「待核」不臆造。純文字未動數字。
- `2026-07-21-04` add — 任務書 fix_tasks T6a（A9）results 對外追蹤：`.gitignore` 由一刀切 `results/` 改為
  `results/*` + `!results/*.json`，首次把 53 個 results JSON（凍結 confirmatory／duel／intervention 與 6 個
  新衍生臂、共約 1.8MB、metadata 零本機路徑洩漏）納入版控，使 clone 者不依賴作者本機即可核對論文每個
  scalar；p1_assets 子目錄資產、*.pt 快取、*.png、*.jsonl、log 續排除。checkpoints 重現包與 README 重現
  指南另議。
- `2026-07-21-05` plan — 任務書 fix_tasks T3（A4）CIFAR-10 confirmatory v2 無碰撞重跑之凍結：新增
  pre-registration amendment `docs/amendment_cifar10_v2.md`（唯一實質改動＝生成種子公式 legacy→hash、
  reps 1→5、啟用 --tstr-seeded；grid／seeds{10,11,12}／per_class 1000／steps 50／eta 0／nearest_k 5／
  tau_fraction 0.9／tstr_epochs 15 一律不動），並於 `run_cifar_cfg_multiseed.py::gen_seed` 加
  `--gseed-formula {legacy,hash}`（cifar10 hash 走 gseed_hash、預設 legacy 保 v1 逐位重現、metadata 記
  gseed_formula_choice）。dry-run 枚舉 30 cell：hash 30/30 相異（無碰撞）、legacy 14/30（10 群、26 cell
  共用，重現舊碰撞結構）、legacy(10,1.0)=110000000 逐位不變。amendment commit 先於真跑（凍結四要件 a）；
  commit 後以 detached 啟動 v2 真跑 → `results/cifar10_cfg_confirmatory_v2.json`（30 cell，ETA 待單 cell
  實測補）。凍結 v1 不動。
- `2026-07-21-06` add — 任務書 fix_tasks T11（B1）ViT-L/14 PRDC 複算之程式就緒：新增
  `src/experiments/run_prdc_vitl14_seed10.py`，對 CIFAR-100 seed10 以 dinov2_vitl14（1024 維）重算
  per-class PRDC 與 FD，對照凍結 confirmatory 之 ViT-B coverage 與 `cifar100_fd_dinov2.json` 之 ViT-B FD，
  判 coverage-argmax 與 FD-argmin（which-FID）是否 backbone 相依。影像來源：p1_assets 快取（seed10 僅
  w1／w2.5 有影像）＋`--generate-missing` streaming 補 8 格（GPU regen、決定性 hash gseed）。dry-run：
  --help import／argparse 過、參照 JSON 結構核對（confirmatory config 有 coverage、fd_dinov2 per_config
  list）。最小版免授權；全 10 格 which-FID 讀數待 T3 釋放 GPU 後以 --generate-missing 跑。
- `2026-07-21-07` add — 任務書 fix_tasks T8（A5）H3 公平化 v2 之程式就緒：`src/core/chamfer.py` 之
  `chamfer_guided_ddim_sample` 加 `guide_every`（官方 G_freq；預設 1＝每步保舊行為，self-check 過），
  雙向項 `term="chamfer"` 早已存在。新增 `src/experiments/run_cifar100_h3_duel_v2.py`（v1 driver 與其
  輸出凍結不動、寫新檔）：weight 掃描 {0.05,0.1,0.3,1.0} 雙向為主＋單向 ablation、`--guide-every 5`、與
  vanilla 對稱 reps=5、coverage 量 DINOv2-224（量測空間）＋導引空間（112）、FLOPs 記帳（生成 NFE＋導引
  反傳次數×特徵網路成本，matched-budget 口徑張數 vs FLOPs 明寫）。判準：若雙向 coverage224 上升（同官方
  0.603→0.912）則撤回 §5.6 攻擊句。dry-run：py_compile／--help／chamfer self-check 過。待 T3／T11 後跑。
- `2026-07-21-08` add — 任務書 fix_tasks T5b（A8）w<1 scout 之 CFG 條件擴充：`src/gen1_mnist/ddpm.py` 之
  `predict_eps` CFG 判斷由 `guidance_scale > 1.0` 改 `!= 1.0`，使 w<1 也走 CFG 分支（往無條件內插、降類別
  銳化）；w==1.0 仍純條件、w>1 不變。回歸（決定性小模型、CPU）：w=1 逐位等於純條件、w=2 符 CFG 公式、
  w=0.5 現走 CFG 且異於純條件——三項 PASS，凍結資料（全 w≥1）不受影響、verify 不涉此路徑。scout 真跑
  （`--off-protocol --guidance 0.5 0.75 --seeds 10 11 12 --reps 5` → `cifar100_subunity_scout.json`，
  off_protocol=True）待 T3 後於 GPU 跑；MNIST g∈{0.5,0.75} 併後補。
- `2026-07-21-09` add — 任務書 fix_tasks T9（B2）TSTR 協定強化之程式就緒：新增
  `src/experiments/run_tstr_protocol.py`（--ceiling／--ablation 兩模式）。ceiling＝對 real 資料跑 TSTR
  （CIFAR-100 500/class、CIFAR-10 1000/class）epochs∈{15,50} 各 5 reps → `tstr_real_ceiling.json`
  （train-on-real 上限，表 5.2/5.4 加行）；ablation＝CIFAR-100 seed10 {w1,w1.5,w2.5}（w1/w2.5 快取、w1.5
  `--generate-missing` regen）epochs∈{15,50} 各 5 reps、量 σ_cls 與排序穩定 → `tstr_protocol_ablation.json`
  （回應「噪聲地板為協定內生」）。reps 以 sha256 決定性種子（T6b）。dry-run：py_compile／--help 過、
  run_tstr 簽名相符。凍結不動、待 GPU。
- `2026-07-21-10` add — 任務書 fix_tasks T10（B3）margin-pruning 介入之程式就緒：新增
  `src/experiments/run_cifar100_margin_intervention.py`（沿 D3 骨架、seed10 w2.5 資產，judge_out.pt 含
  margins 50000）。直接介入中介量：移除 judge margin 最低（最近邊界）n 個 vs 等計數隨機移除，
  n∈{13606, 6803}、每檔位各 N=8 from-scratch 重訓（T6b 決定性種子、每類保底 6），輸出差（rand−margin）／
  SE／t／CI95／MDE（同 n8 口徑 2.8×SE）。判準：差為正且 >MDE 支持 near-boundary 承載效用。dry-run：
  py_compile／--help 過、judge_out.pt margins 結構核對。凍結不動、待 GPU；§5.5 併述兩介入。
- `2026-07-21-11` test — 任務書 fix_tasks T3 CIFAR-10 confirmatory v2 真跑完成（30 cell、hash 無碰撞、
  reps=5、tstr-seeded；約 11 小時）：`results/cifar10_cfg_confirmatory_v2.json` 與重導之
  `cifar10_c6_fidmin_duel_v2.json`。結果——**FID-min regret 0.00（三 seed 全中，v1 為 0.91）**、CaF regret
  6.70（v1 3.69）、C1 sep=0 三 seed 全不分離（v1 為 1/1/0）、oracle 三 seed 穩定 w1.5（v1 漂移 w2/w1/w1.5）。
  判決方向不變（便宜 FID-min 勝 coverage-CaF）且更強：v1 的 oracle 漂移與 FID-min 殘餘 regret 主為種子碰撞
  （26/30 cell 共用 latent）＋單 rep 噪聲之假影，無碰撞後 FID-min 完全等於 oracle。v1 保留凍結；C8/C2 複核與
  正文改引 v2（附錄註 v1 碰撞）進行中。
- `2026-07-21-12` proofread — 任務書 fix_tasks T3 v2 分析與正文改引：C8 支配複核（v2 seed10 w2.5 於
  coverage .792≥.749 與 precision .869≥.842 雙雙支配 oracle w1.5、其 TSTR 56.61 為網格最差之一，Pareto
  失明較 v1 更清晰）、C2 偏相關複算（`cifar10_c2_partial_v2.json`：partial ρ(TSTR,coverage|precision)
  =+0.526、p=0.058、未通過、n=10 功效有限，直接回應 §5.2 判決二之碰撞保留）。thesis 新增 §5.2.1「無碰撞
  重跑（v2）」、§5.2 判決二／判決三加 v2 指標；v1 數字與既有引用保留凍結不動、正文 CIFAR-10 confirmatory
  主張以 v2 為準。verifier 增 9 條 v2 對帳（FID-min 0.00／CaF 6.70／oracle 65.59／w2.5 支配數字），重跑
  OK 367／MISMATCH 0；凍結 v1／prereg／verdict／duel 全 ALL_MATCH 未動。

## 2026-07-20

- `2026-07-20-07` add — 任務書 fix_tasks T1c CIFAR-100 側補齊（streaming）＋2×2 完整六格＋T7 頭條定案：
  `run_cifar_judgefeat_stack.py` 增 `--generate-missing`——CIFAR-100 缺 8 格以逐類生成、邊算 judge 特徵
  （只留 512 維、不累積影像，避 CPU-RAM OOM/kill；smoke 驗峰值 1.18GB），seed 化分佈匹配。完整
  coverage-CaF regret 2×2：MNIST judge 0.00／DINOv2 1.02；CIFAR-10 2.45／3.69；CIFAR-100 0.79／6.10。
  judge 每列皆低於 DINOv2（特徵效應恆正、CIFAR-100 達 +5.31），故頭條**定案**為「coverage 代理可靠主由
  **特徵空間**閘定、資料集調幅度」（先前 CIFAR-10 單格致之「資料集主」修正被全六格推翻）；摘要／§1.3／§6.1
  （標題＋開場＋六格表＋兩型）／§5.3／§6.2／§7／圖 5.2 一致。verifier 增 CIFAR-100 judge 0.79（JSON）＋DINOv2
  6.10（由 confirmatory 重算），重跑 OK 358／MISMATCH 0；凍結未動。
- `2026-07-20-06` proofread — 任務書 fix_tasks T7 尾項（B&N 定理降範圍＋§3.5 matched-probe 校準）：§2.1 加
  B&N（Bradley & Nakkiran 2024）引用範圍界定——其 predictor-corrector 定理朝 gamma-powered 中間分佈銳化
  （≠「朝 class prototype 集中」）、且覆蓋隨機 CFG_DDPM，本文全程 η=0 DDIM 落其範圍外、MNIST η-null 與
  「隨機性關鍵」有張力，故僅作銳化方向動機、非機制證明；§3.5 明寫 FID-min 之 clean-fid 用 cleanfid 全訓練集
  stats（與 CaF probe 非同份）、C7 小 probe 排序穩定為佐證而非實作。純措辭，verify OK 356／MISMATCH 0。
  未竟：參考文獻正式清單格式統一（T12.8）與 B&N 正式 venue 核對，列排版定稿階段。
- `2026-07-20-05` add — 任務書 fix_tasks T7 相關工作補全（§2.4）：路線三改寫——加 Ravuri & Vinyals 2019
  （arXiv:1905.10887，FID/IS 不預測 CAS 之完整先行，非僅 DPDM 旁註）、強化 DPDM（2210.09929）刻畫為並列
  FID／Acc 兩套 sampler 設定＋diversity 機制（去「順帶」弱化措辭）、加 Azizi 2023（2304.08466，FID-argmin
  為好選擇器之正面先例、與 T1a 實測一致）；本文定位加 Astolfi 2024（2406.10429，PRDC 前沿選組態、與 CaF
  操作點最近，明確差異化）；補其他相關工作清單附 arXiv（Sariyildiz 2212.08420、Shipard 2302.03298、
  StableRep 2306.00984／SynCLR 2312.17742、CADS 2310.17347、Kynkäänniemi 2404.07724／1904.06991、Karras
  2406.02507、Alaa 2102.08921、Bartlett 2017、Sorscher 2022）。純文字未動數字，verify OK 356／MISMATCH 0。
- `2026-07-20-04` add — 任務書 fix_tasks T1c（CIFAR judge 特徵堆疊，A1b 反向交叉）＋T1a clean-fid 第二讀數
  ＋T7 頭條再修正：新增 `src/experiments/run_cifar_judgefeat_stack.py`，CIFAR-10 seed10 十格自快取影像算
  judge 512 維 PRDC 與 CaF——coverage-CaF 選 w1.5（oracle w2、regret 2.45），judge 任務對齊特徵下仍未中
  oracle。完整 2×2（coverage-CaF regret）：MNIST judge 0.00／DINOv2 1.02；CIFAR-10 judge 2.45／DINOv2 3.69
  ——資料集效應（約 2.5pp）大於特徵空間（約 1pp），coverage-CaF 只在 MNIST+judge 成立。據此把 T7 頭條由
  先前偏「特徵空間相依」**再修正**為「資料集主、特徵次之合取」（摘要／§1.3／§6.1 標題＋開場＋2×2 補充＋
  兩型／§5.3／§6.2／§7／圖 5.2 一致）。T1a clean-fid（Inception 空間、重用 cache）亦選 g2、regret 1.02，與自製
  classifier-FID 一致，破「自製 FID 有利」質疑。`regen_cifar100_cells.py` 增 `--guidance`；CIFAR-100 8 格 regen
  進行中（補 2×2 第六格 CIFAR-100+judge）。verify OK 356／MISMATCH 0；凍結未動。
- `2026-07-20-03` add — 任務書 fix_tasks T6b TSTR 種子化：`src/core/cifar_classifier.py` 之
  `run_tstr`／`train_classifier`／`_AugmentedTensorDataset` 增 `seed` 參數——`seed=None` 維持現行未種子化
  行為（凍結對帳語意不變）；非 None 時 `torch.manual_seed` 控模型初始化、`DataLoader(generator=)` 控
  shuffle、增強改用傳入 CPU `Generator`。self-check 增決定性測試（CPU 同 seed 兩次逐位相同）通過。
  `run_cifar_cfg_multiseed.py` 增 `--tstr-seeded` 旗標，`measure()` reps 迴圈以 sha256(tstr_<ds>_<seed>_<w>_<rep>)
  衍生種子（T3 v2 啟用、凍結 v1 不加旗標故不受影響）。CUDA 殘留 cuDNN 非決定性另議、不由此保證。
- `2026-07-20-02` proofread — 任務書 fix_tasks T7 頭條重寫（依 T1a/T1b 實測、保守 hedge）：摘要、§1.3、
  §6.1（標題「代理可靠性之資料集相依性」→「coverage 代理可靠性之特徵空間相依性（FID-min 相對普適）」、
  開場、兩型段）、圖 5.2 標題與圖說、§5.1（precision-argmax 與 FID-min 區辨）、§6.2／§7 結論一致改寫：由
  「代理可靠性之資料集相依反轉」改為「coverage 型代理可靠性之特徵空間相依、FID-min 三尺度近最優相對普適」；
  「反轉」限於 coverage 選擇器且沿特徵空間、完整 2×2 歸因待 T1c；刪去被反證之「FID-min 在 MNIST 選錯」。
  附錄 B 逐字內嵌凍結 verdict（含舊框架）刻意不動；verify 重跑 OK 354／MISMATCH 0，凍結 verdict/prereg diff 空。
- `2026-07-20-01` add — 任務書 fix_tasks T1a／T1b（A1）MNIST 兩臂：新增 `src/experiments/run_mnist_fid_arm.py`
  （實測 MNIST FID-min，classifier-Fréchet）與 `src/experiments/run_mnist_dinov2_stack.py`（DINOv2 堆疊
  CaF），共用 `results/mnist_gen_cache/` 同批影像（seeds 0/1/2、grid {1,2,3,5,7,10}、per_class 1000、DDIM
  steps 50 eta 0）。結果（3 seed 一致）：T1a MNIST FID-min 選 g2、regret 1.02、與 oracle g1 相隔 1 格、
  依 C1 口徑不分離——與 CIFAR 同型，推翻「FID-min 在 MNIST 選錯」（原係 precision-argmax 代打之假影）；
  T1b 換 DINOv2 後 MNIST coverage 轉非單調（峰 g2）、CaF(coverage) regret 1.02（0/3 中 oracle）、
  CaF-v2(recall) regret 0（3/3），即反轉主要跟著特徵空間。與 T2 fixed-g2（1.02）逐值交叉一致。落地：圖 5.2
  橘柱改實測 FID-min、表 5.3 MNIST FID-min 欄 1.02、§5.3／§6.1 歸因段；verifier 增 4 對帳，OK 354／MISMATCH 0；
  凍結未動。clean-fid 第二讀數（同快取再一遍 Inception）與摘要頭條之 T7 重寫待續。

## 2026-07-19

- `2026-07-19-14` add — 任務書 fix_tasks T4（A6）τ 靈敏度：新增 `src/experiments/run_tau_sensitivity.py`
  與 `results/tau_sensitivity.json`。(a) 提出凍結 `report.tau_robustness` 完整掃描；(b) tau_fraction
  ∈{.80,.85,.90,.95}＋無 floor 重跑——CIFAR-100 於 0.85→regret 0.00（中 oracle w1）、預註冊 0.90→0.76
  （改選 w1.5）、0.95→5.40，且無任一設定達 D4 的 1.5pp 勝幅；(c) seed-10 配平校準（快取 DINOv2 特徵）：
  precision 量測改 250v250（對齊 250v250 校準）後 w1 通過 floor、CaF-v2 由 w1.5 翻回 oracle w1（regret
  0.79→0.00），500v500 欄逐位重現 frozen。整合 §5.4.1／§6.3，verifier 增 9 對帳，重跑 OK 350／MISMATCH 0；凍結未動。
- `2026-07-19-13` proofread — 任務書 fix_tasks T12.6：README 頭條保留「Sampling for Utility, not
  Fidelity」但其下加診斷結論註記（該敘事在 CIFAR 尺度被自家資料反證、C1 兩空間不分離、貢獻收斂於
  CaF）；純措辭、未動數字。
- `2026-07-19-12` refactor — 任務書 fix_tasks T12.2／T12.3 圖表：`src/figures/make_thesis_figures.py`
  圖 5.2（fig_selector_reversal）加「precision-argmax（非實測 FID）」過渡底註、圖 5.6（fig_h3_duel）
  標題去「moat duel」改「matched-budget comparison」、檔頭圖號更正（fig_selector_reversal=圖 5.2、
  fig_two_stage=圖 5.5）；重生成 `docs/figures/` 六張圖，純製圖未動任何數字。
- `2026-07-19-11` proofread — 任務書 fix_tasks T12.5 去代號化：活躍文件與程式之「護城河／moat duel」
  一律改「matched-budget 對照」（`thesis_draft.md`、`paper_branch3_diagnostic.md`、`results_analysis.md`、
  `README.md`、`run_cifar100_h3_duel.py` 檔頭）；README「範圍與護城河」之競爭定位語意改「差異化」。凍結
  `verdict_cifar100.md`、其於 thesis 附錄 B 之逐字內嵌副本（`thesis_draft.md:1153`）、CHANGELOG 歷史刻意保留不動。
- `2026-07-19-10` refactor — 任務書 fix_tasks T12.7：停放旁支 `var/`、`src/var_mini/` 以 git mv 移入
  `attic/`（`attic/var`、`attic/var_mini`），覆寫 `2026-07-19-05` 之「不動程式」裁定（作者本輪裁示）。三支
  var_mini 腳本墊片改為接回 `src/_pathfix` 並補 `attic/` 於 `sys.path`，`--help` 驗證三支 import 均通過；同步
  README、`docs/code_map.md`、`src/_pathfix.py` 註解。主線無任何模組 import var／var_mini，零影響。
- `2026-07-19-09` proofread — 修正任務書 fix_tasks T12.1：刪 `docs/thesis_draft.md:1259` 本機路徑
  洩漏（`.claude/plans/...`）改「依內部撰寫計畫」，為 git 追蹤檔中唯一之絕對路徑洩漏；純措辭、未動數字。
- `2026-07-19-08` add — 任務書 fix_tasks T5a（A8）C1 配對統計：新增 `src/experiments/run_c1_paired_stats.py`
  純衍生 post-hoc driver 與 `results/c1_paired_stats.json`。以雙口徑呈現 TSTR-argmax 對 FID-argmin：凍結
  格步口徑 CIFAR-100 判 0/8 不分離（路由依據不回改），標準配對檢定則 mean 0.76pp、t=9.71、p≈2.6e-5、
  符號 8/8——系統偏移存在但實務可忽略、TSTR-argmax 恆在網格邊界 w1；paired_diff 與凍結 c6 duel 逐值交叉
  核對。整合入 §5.4.1，`verify_thesis_numbers.py` 增 4 對帳，重跑 OK 341／MISMATCH 0；凍結判決未動。
- `2026-07-19-07` add — 任務書 fix_tasks T2（A7）固定 w／隨機可行點 baseline：新增
  `src/experiments/run_baseline_fixed_random.py` 純衍生 driver 與 `results/baseline_fixed_random.json`，
  補預註冊 D5 兩下限。fixed-w 對網格每一 w 報整欄 per-seed regret（防 HARKing），random-feasible 為可行集
  TSTR 解析平均。CaF／FID-min 三資料集皆遠勝隨機可行點（CIFAR-100 22.82／CIFAR-10 15.66／MNIST 5.92pp）；
  固定 w1 於 MNIST／CIFAR-100 為 0.00（oracle 恰在網格邊界 w1）但非自適應、CIFAR-10 為 1.71 劣於 FID-min。
  表 5.3 增 fixed-w1／fixed-w2／random-feasible 三欄，`verify_thesis_numbers.py` 增 9 對帳；凍結檔未動。
- `2026-07-19-06` proofread — 收尾被取代草稿與未來工作段措辭：`docs/paper_branch3_diagnostic.md` 檔頭與
  狀態段標明本稿已被定稿 `docs/thesis_draft.md`（含 6 張發表級圖）取代、保留作撰寫軌跡，去除「發表級圖
  後補／待補」開放待辦；`docs/results_analysis.md`「待確認」段改標「未來工作（本論文範圍外）」對齊論文
  第七章。純措辭同步、未動任何數字，`tools/verify_thesis_numbers.py` 重跑仍 OK 328／MISMATCH 0。
- `2026-07-19-05` proofread — VAR-mini 去留定案為停放旁支、不列入論文：README 與 `docs/code_map.md` 措辭
  由「去留待定」改為「停放、僅 smoke、不列入論文」（承 `2026-07-18-02`／`2026-07-18-05` 兩次保留現狀裁定）；
  `var/`、`src/var_mini/` 程式與 smoke 權重不動、零程式改動。
- `2026-07-19-04` refactor — 記錄慣例改為單軌：`CHANGELOG.md` 成為專案唯一記錄，停用本機 `records/`
  （已刪除，2026-07-11 前的舊記錄仍在 git 歷史）。改寫 `claude.md` §1（Records and Changelog → Changelog）、
  §3.1／§5.1 對 records 的引用，並同步 README 與本檔開頭說明。
- `2026-07-19-03` refactor — 根目錄 45 個 .py 依類別移入 `src/`（core／experiments／gen1_mnist／
  var_mini／figures）。因專案未打包成 package，扁平 import 以新增的 `src/_pathfix.py` 墊片在執行時把
  各 src 子資料夾補回 sys.path 維持，import 語句不變；修 EDM／製圖三處 `__file__` 相對路徑改用
  `_pathfix.ROOT`，墊片並把 argv[0] 正規化為裸檔名以保 results/*.json argv 逐字元相符（§5.2 慣例層）。
  更新 README 結構與執行指令、新增 `docs/code_map.md`；凍結 prereg／thesis 內嵌區與 CHANGELOG 歷史不動。
  驗證：最深 import 鏈與 EDM `--help`、fd_dinov2 `--dry-run`、metrics_prdc 自檢皆過，圖表重生成逐位相同，
  verify_thesis_numbers 仍 OK 328／MISMATCH 0。
- `2026-07-19-02` refactor — 清除過時檔案與收斂雜亂：刪除根目錄已落地審查建議書與 docs/ 三份被取代草稿
  （`paper_intro_draft`／`paper_skeleton_branch3`／`paper_skeleton_branch4`，後者為分支四死枝），更新
  README／results_analysis／paper_branch3_diagnostic 之可編輯引用，統一 thesis_draft 圖目錄路徑字面
  （P0-5），刪一個 0-byte 備份空檔；凍結 prereg 及其逐字內嵌副本之骨架引用刻意保留（僅存 git 歷史）。
  records 不刪。驗證工具重跑 OK 328／MISMATCH 0，承重數字未動。
- `2026-07-19-01` test — Q5 P1（二）CIFAR-100 per-config FD-DINOv2 補算（GPU，低佔用 batch128 跨夜、
  斷點續跑）：新增 `run_cifar100_fd_dinov2.py`，seed-10 FD-DINOv2 argmin=w2.5、離 TSTR-argmax w1 達 3 格
  **判分離**，與 Inception（0/8 不分離）**相反**——Inception-FID 選 w1.5 regret≈0.8pp、DINOv2-FID 選 w2.5
  regret≈8.8pp；以「which-FID 可靠性表徵相依」強化診斷（不復活普適 FID≠效用、不改凍結 Inception-only
  路由，單 seed 事後）。整合入 `docs/thesis_draft.md` §5.4.1／branch3 §3a／results_analysis 三檔，驗證工具
  增 5 個 FD-DINOv2 對帳、重跑 OK 328／MISMATCH 0；凍結檔未動。

## 2026-07-18

- `2026-07-18-12` proofread — N=8 介入結論三份文件一致化：把 `2026-07-18-11` 的 N=8 結果（cov-matched
  46.30 對 random 46.63、差 −0.33pp、MDE≈1.85pp、有功效 null）同步至 `docs/paper_branch3_diagnostic.md`
  §4＋數據表，並將 `docs/results_analysis.md` 過時的「待確認」（把已完成的 D3 介入與 H3 仍列待執行）改為
  已補結果段；三檔介入敘述一致，凍結檔未動。
- `2026-07-18-11` test — Q5 P1（一）CIFAR-100 D3 介入臂 N=8 更高功效重跑（GPU）：coverage-matched
  剪枝 TSTR 46.30 對等計數隨機剪枝 46.63、差 −0.33pp（SE 0.66、t=−0.50、MDE≈1.85pp，CI 跨零），把原
  N=2（−0.11、underpowered）升級為有功效之 null（仍限單 seed、單一介入型式，不宣稱普遍否證）；driver
  加 `--output`＋改載快取真實 DINOv2 特徵（免 OOM、與 regen 同源）另存 `cifar100_d3_intervention_n8.json`
  不覆寫原檔；整合入 `docs/thesis_draft.md` §5.5/§6.3/§7/附錄 D，驗證工具增 N=8 對帳，重跑 OK 323／MISMATCH 0。
- `2026-07-18-10` add — 診斷論文 P3＋P4（補強＋自足＋收斂）：新增 §3.6 量測與訓練細節（PRDC k／
  DINOv2 管線／TSTR／judge p20=0.3622／Chamfer 實作，可複製）；§2.4 相關工作擴為分主題回顧；附錄 A/B
  逐字內嵌凍結檔 prereg（249 行）與 verdict（72 行）使論文自足（凍結檔未動）；§4.2 形式化 H1/H2/H3 及
  go/no-go 門檻與結果；§6.1 補「反轉為何發生」因果綜述；§6.1/6.3/6.4 依 P4 收斂宣稱（兩型非連續譜、
  未證≠否證、度量堆疊混淆、CaF-v2 事後性、H3 削弱差異化後之三項剩餘貢獻），不改實驗結論。對帳仍 OK 320。
- `2026-07-18-09` refactor — 診斷論文 P2（結構）：預先登記四分支決策樹提前為 §1.3（原論文組織順延
  §1.4），§4.2 保留完整版；§5.2 判決前新增「兩把尺」前置段（Pareto 失明、σ_cls 雜訊地板）並加前向
  指標（§5.4.2 實例含表 5.2 資料依賴，故不整段上移、採 hybrid）；§2.1 steps/η 兩 bullet 合併精簡。
  數字未動，對帳仍 OK 320。
- `2026-07-18-08` proofread — 診斷論文 P1（句法展開＋去譬喻＋粗體酌減）：摘要／§5.2 判決三／§5.5 介入
  之電報式逗號串句展開為主謂完整句（數字核對一致）；去「押注／押在／頭條／避崖器／平台優化器／thesis 活
  selector 死」等非正式詞，護城河對決／moat duel 於標題與摘要改正式名「與 Chamfer 之 matched-budget
  對照」；§5 冗餘行內粗體酌減。數字未動，對帳仍 OK 320。
- `2026-07-18-07` proofread — 診斷論文 P1（去代號＋g/w 統一＋行話界定＋符號表）：去裸用代號
  （H-C2 併入 C2、C0／D3／D4 加描述名、C2/C3/C5 改述並去未定義之 C5、圖表目錄 H3 描述名前置），附錄 D
  新增 D.1 代號對照表；MNIST guidance 全文由 g 統一為 w；plan-of-record／matched-probe／matched-budget／
  ln_excess／ancestral sampling／probability-flow ODE／oracle／Pareto 失明／denoise-then-sharpen 補中文
  界定（HARKing／MDE／auto-τ 已定義不重複）；新增符號表。數字未動，對帳仍 OK 320／MISMATCH 0。
- `2026-07-18-06` audit — 診斷論文審查建議書 P0 落地：新增 `tools/verify_thesis_numbers.py` 對
  `docs/thesis_draft.md` 表 5.1–5.5／E.1–E.3 及承重內文 scalar 逐位對帳 `results/*.json`（OK 320、
  MISMATCH 0、MISSING 0、僅 1 進位邊界），進度句改為已逐位核對；README Phase 1-4／1-5 狀態同步至落
  分支三、消除與論文矛盾；缺漏引用補正（Deliberate Practice = Askari-Hemmat 2025 arXiv:2502.15588、
  DP-diffusion = Dockhorn TMLR 2023 arXiv:2210.09929 §5.2）；清 stale 待辦標記；審查書 P0-5 圖路徑判為
  誤判不改。凍結檔與 `results/*.json` 未動。
- `2026-07-18-05` proofread — 修正健檢 Q3 三處小瑕疵：刪 README 殘留的 `cifar_data.py` 行、`run_c6`
  檔頭補述資料集通用（也產 `cifar100_c6_fidmin_duel.json`）、VAR-mini 依「保留現狀」不動程式改在 README
  補 smoke 權重覆寫說明；皆事實同步、無研究行為變更，py_compile 通過、凍結檔未動。
- `2026-07-18-04` add — 碩論草稿 P0 收尾：新增 `make_thesis_figures.py`（純讀 `results/*.json`，dev 依賴
  加 matplotlib）產 6 張發表級圖並嵌入 `docs/thesis_draft.md` 第五章、參考文獻查證改正式格式 13 筆
  （附 arXiv）、附錄 E 補 per-seed 表 E.1–E.3；數字對 `results/*.json` 皆一致、凍結檔未動。
- `2026-07-18-03` add — 通讀專案回覆作者五問，並依裁定（繁中／新檔保留凍結證據／碩論章節）整合 docs
  散稿為單一 `docs/thesis_draft.md`（七章＋前置頁＋五附錄，結果章數字逐一對 `results/*.json`、凍結
  prereg/verdict 僅引用不動、被取代骨架留作決策軌跡）；發表級圖、參考文獻完整出處與 P1 加值實驗列為後續。
- `2026-07-18-02` proofread — 分支三診斷論文全文定稿：逐一對 results/*.json 核數字皆一致，摘要與 §5
  補 H3、§4.5 補 rep 不對稱限制；VAR-mini 裁定保留現狀；合併 PR #4（Stage 1–4）入 main。分支三收尾
  完成——D1 落分支三、機制觀察複製但因果未證、selector 敗/平便宜 baseline、Chamfer 勝 vanilla 但
  增益不見於 coverage。發表級圖後補。
- `2026-07-18-01` test — CIFAR-100 H3 護城河對決（Stage 4b，GPU）：新增 `run_cifar100_h3_duel.py`，
  matched-budget 比 Chamfer 對 vanilla w1.5（＝FID-min＝CaF-v2）。Chamfer 兩變體皆勝——任務無關
  DINOv2 特徵 TSTR 61.18（+2.54pp）、任務對齊 judge 特徵 61.75（+3.11pp），皆勝過 vanilla oracle w1
  （59.66）；但兩臂 DINOv2 coverage 皆低（0.44–0.48 < vanilla 0.643），效用增益不見於 coverage proxy。
  CaF「選 vanilla」在效用上敗於 Chamfer，強化「無普適便宜代理」診斷主軸；已整合入診斷論文 §4.5。

## 2026-07-17

- `2026-07-17-06` test — CIFAR-100 Chamfer 真模型 smoke（Stage 4a，GPU）：`chamfer.py` 補 `_cifar_smoke`
  （完成記載的 CIFAR smoke TODO），驗 `chamfer_guided_ddim_sample` 於真實 CFG 模型 + ResNet18 特徵端到端
  跑通；chamfer_weight 0→1 使 PRDC coverage 0.411→0.614、覆蓋項 9.304→7.792（w=10 飽和），Chamfer 可用、
  操作點 weight≈1.0。三臂 duel driver 待作者確認設計後建。
- `2026-07-17-05` test — CIFAR-100 D3 介入臂（Stage 3，GPU）：重生成 seed-10 w1/w2.5 兩 cell 對
  confirmatory 逐位重現（rel_delta 全 0）；C3 coverage-matched pruning 把 w2.5 剪至 w1 coverage 水準
  （移 13606 樣本）TSTR 45.75，等計數隨機剪枝 45.86，差 −0.11pp（N_retrain=2、遠在 σ_cls 內）——
  cov-matched 與 random 無法區分，介入臂未對 coverage 因果角色提供支持，與 CIFAR-10 C2/C3/C5 同向；
  已整合入診斷論文 §4。
- `2026-07-17-04` add — CIFAR-100 D3 介入臂前置程式（Stage 2，CPU）：新增 `regen_cifar100_cells.py`
  （重生成 seed-10 w1/w2.5 兩 cell、cifar100 gseed、對帳 confirmatory 於 1e-4）與
  `run_cifar100_d3_intervention.py`（C3-only coverage-matched pruning，target_cov 由重生成 w1 cross-env
  現算，另立新檔不動凍結的 CIFAR-10 driver）；py_compile 與 dry-run 通過，待 GPU 空出跑 Stage 3。
- `2026-07-17-03` add — CIFAR-100 揭盲裁決落地分支三：新增揭盲 verdict（`docs/verdict_cifar100.md`，
  客觀讀數 C1 不分離 0/8、H2 CaF-v2 平 FID-min 0.76pp、H1 機制 3/3，作者簽核）與診斷論文正文草稿
  （`docs/paper_branch3_diagnostic.md`，表格版，主結果 selector 判決跨資料集反轉）；`results_analysis.md`
  吸收 CIFAR-100 confirmatory；標註 branch3/4 骨架現行狀態。精確化「兩空間」措辭——CIFAR-100 僅算
  Inception clean-fid 一個 FID 空間（未算 FD-DINOv2），which-FID 單空間評估，不改路由。
- `2026-07-17-02` add — D3 三觀察量獨立 driver（`run_d3_observables.py`，純衍生、附 §5.2 metadata）：
  以 coverage 均值峰位切段判三觀察量，(i) 升段 near-boundary 單調降、(ii) 高段 coverage 與 TSTR 同崩、
  (iii) 高段 near-boundary 谷後回升脫鉤，三項全成立、3/3 機制複製，與 `2026-07-16-01` 內嵌判讀一致；
  輸出對帳 ALL_MATCH。
- `2026-07-17-01` debug — off-protocol 護欄補 nearest_k/tau_fraction/tstr_epochs/threshold 四個量測
  參數（原只攔 7 個凍結鍵），metadata 補記 tstr_epochs；凍結值與已完成 confirmatory 的 as-run 完全
  吻合，不重算、不影響既有輸出，只強化對未來 re-run（如 D3 介入臂重生成）誤帶量測參數的攔截；
  scratch 七情境驗證全通過。

## 2026-07-16

- `2026-07-16-01` test — CIFAR-100 confirmatory 下游裁決：matched-budget FID-min duel 顯示 CaF 與
  FID-min 逐 seed 同選 w1.5、regret 均 0.76pp 打平（未達 D4 的 ≥1.5pp，selector 主張不成立）；C1 分離
  格步 0/8（不分離）；D3 三觀察量三項全成立（機制複製）。客觀讀數對照 D1 第三分支（不分離但機制複製→
  診斷論文），揭盲路由待作者確認。輸出 `results/cifar100_c6_fidmin_duel.json`。
- `2026-07-11-10` test — CIFAR-100 confirmatory 真跑完成：80 cell（seeds 10–17 × grid 10 點、reps 5、
  per_class=real=500、CaF-v2 recall selector）於 2026-07-14 起跑、跨 GPU 搶佔中斷以斷點續跑接力、
  2026-07-16 22:58 完成；coverage(DINOv2) 峰 w2.5 0.698、TSTR 由 w1 59.66 單調降至 w8 15.88、
  char_fid 最小 w1.5 7.17、CaF 全 seed 選 w1.5（regret 0.76pp、rank 2/10、top-3 命中 100%）。

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
