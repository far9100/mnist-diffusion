# mnist-diffusion 修改任務書（供編碼代理執行）

來源：`review_mnist_diffusion.md`（2026-07-19 嚴格審查報告）。本文件把該報告的修改建議整理成
可直接執行的任務規格。每個任務標明：對應審查問題編號（A1–A9、B1–B8、C1–C6）、改動範圍、
具體步驟、輸出檔、驗收標準、GPU 成本等級（無／低／中／高）、以及是否需要作者授權。

---

## 0. 總則（編碼代理必讀，優先於一切任務內容）

1. **先讀 `claude.md`**，全程遵守專案慣例：繁體中文註解與文件、每個檔案開頭用途註解、
   每次改動在 `CHANGELOG.md` 同 commit 加一行（`YYYY-MM-DD-NN` + action）、MVP 優先、
   規格外功能先問作者。
2. **凍結資產不可觸碰**：`docs/prereg_cifar100.md`、`docs/verdict_cifar100.md` 為凍結證據，
   一個字都不能改；既有 `results/*.json`（confirmatory、duel、intervention 等）不可覆寫或
   重算後回填。所有新結果一律寫**新檔**（後綴 `_v2` / `_posthoc` / `_arm` 等），並在 metadata
   標明 `exploratory` 或 `post-hoc`。既有對帳腳本（`run_c6_fidmin_duel.py --no-write`、
   `tools/verify_thesis_numbers.py`）在任何改動後必須仍能通過。
3. **STOP 閘**：標注【需作者授權】的任務（所有中/高 GPU 任務、所有涉及新 confirmatory 的
   任務），代理只做到「程式完成 + `--dry-run` 通過 + ETA 估算寫入 CHANGELOG」為止，不得
   自行開跑。凡屬 confirmatory 等級的新跑（T3、T5b、P1 全部），須先在 `docs/` 新增
   pre-registration amendment 並 commit，滿足凍結四要件的 (a)，才能真跑。
4. **新 driver 一律輸出 §5.2 完整 metadata**（start_timestamp、完整 argv、nearest_k 與有效 k、
   tau_fraction、batch、seeds、torch/cuda/cudnn 版本）。
5. **新生成一律用 hash 種子**：`datasets/cifar100_gseed.gseed`（或同型式的
   `int(sha256(...)[:15],16)`），禁止再用 `seed*1e7+int(w*1e3)*1e4` 舊公式（唯一例外：
   為逐位重現既有凍結 CIFAR-10 資產而做的 regen）。
6. 新腳本放在 `src/experiments/`，開頭沿用 `_pathfix` 墊片；純衍生分析（讀 JSON 算數字）不
   依賴 GPU，須可在 CPU 直接執行。

---

## 1. 任務總覽

| ID | 名稱 | 對應審查 | GPU | 授權 | 產出 |
|---|---|---|---|---|---|
| T1a | MNIST 真 FID-min 臂 | A1(a) | 低 | 否 | `results/mnist_fid_arm.json` |
| T1b | MNIST DINOv2 堆疊版 CaF | A1(b) | 低 | 否 | `results/mnist_dinov2_stack.json` |
| T1c | CIFAR judge 特徵堆疊版 CaF | A1(b) | 中 | 是 | `results/cifar{10,100}_judgefeat_stack.json` |
| T2 | 固定 w／隨機可行點 baseline | A7 | 無 | 否 | `results/baseline_fixed_random.json` |
| T3 | CIFAR-10 confirmatory v2（無碰撞重跑） | A4 | 高 | 是 | `results/cifar10_cfg_confirmatory_v2.json` |
| T4 | τ 靈敏度 + 樣本數配平校準 | A6 | 低 | 否 | `results/tau_sensitivity.json` |
| T5a | C1 配對統計 | A8 | 無 | 否 | `results/c1_paired_stats.json` |
| T5b | w<1 網格延伸 scout | A8 | 中 | 是 | `results/cifar100_subunity_scout.json` |
| T6a | 開放 results JSON 與重現包 | A9 | 無 | 是 | `.gitignore` 修訂、release 說明 |
| T6b | TSTR 全面種子化 | A9 | 無 | 否 | `cifar_classifier.py` 增量修改 |
| T7 | 相關工作與主張重寫 | A3/B3 | 無 | 否 | `docs/thesis_draft.md` 等文字修訂 |
| T8 | H3 公平化或降級 | A5 | 中 | 是 | `chamfer.py`、`run_cifar100_h3_duel.py` v2 |
| T9 | TSTR 協定強化 + real 上限線 | B2 | 中 | 是 | `results/tstr_protocol_ablation.json` |
| T10 | margin-pruning 介入 | B3 | 中 | 是 | `results/cifar100_margin_intervention.json` |
| T11 | ViT-L/14 PRDC 複算（seed 10） | B1 | 低 | 否 | `results/cifar100_prdc_vitl14_seed10.json` |
| T12 | 文字/圖表/衛生批次修正 | B5,B7,B8,C1–C5 | 無 | 否 | 多檔文字修訂 |
| P1-1 | EDM 第二 backbone | A2/規模 | 高 | 是 | 規格見 §4 |
| P1-2 | 更高解析度資料集 | A2/規模 | 高 | 是 | 規格見 §4 |

依相依關係的建議順序：`T12 → T2 → T5a → T4 → T1a → T1b → T6b → T7 → T6a →（授權後）T1c → T3 → T11 → T10 → T9 → T5b → T8 → P1`。

---

## 2. P0 任務規格

### T1a. MNIST 真 FID-min 臂【對應 A1(a)；GPU 低；免授權】

**目的**：表 5.3 與圖 5.2 的 MNIST「保真代理」目前用 precision-argmax 代打，從未實測 FID-min。
補上每個 guidance 組態的 MNIST-FID（classifier-Fréchet），讓三個資料集的 FID-min 臂同構。

**步驟**：
1. 新增 `src/experiments/run_mnist_fid_arm.py`。重用 `run_selector_signal.py` 的生成路徑
   （同 seeds {0,1,2}、steps=50、eta=0、grid {1,2,3,5,7,10}、per_class=1000；該腳本以 seed
   統一設定 RNG，重生成應可重現原分佈——若 `results/selector_signal_multiseed.json` 中留有
   per-config 特徵資產則直接載入，先檢查再決定）。
2. 對每 (seed, g)：以 `fid.load_cnn` + `analyze_distribution.extract_features` 抽 gen 與
   real probe（1000/class，同 `--fid-per-class` 慣例）的 256 維特徵，
   `fid.feature_stats` + `fid.frechet_distance` 算 MNIST-FID。
3. 對每 seed 取 FID-argmin，對照 `selector_signal` 的 per-seed TSTR 算 `fidmin_regret`，
   輸出格式對齊 `run_c6_fidmin_duel.py` 的 per_seed 欄位（oracle、fidmin、regret、sep_step）。
4. **同時補算 clean-fid 版**（`fid_clean.clean_fid_two_sets`，gen vs real probe 的 Inception
   空間）作第二讀數，兩空間並列，避免「自製 FID 恰好有利」的質疑。

**驗收**：JSON 含 per-seed 兩空間的 argmin/regret 與 metadata；若 FID-argmin 落在 g1（oracle），
必須把「FID-min 在 MNIST 選錯」相關句（摘要、§5.1、§6.1、表 5.3、圖 5.2）全部改寫（見 T7 清單）；
若確實選錯，把表 5.3 的括註從「precision-argmax 代打」換成實測值。**兩種結果都要如實落地。**

### T1b. MNIST DINOv2 堆疊版 CaF【對應 A1(b)；GPU 低；免授權】

**目的**：把 MNIST 的選擇器搬到與 CIFAR 相同的 DINOv2 堆疊，拆開「資料集」與「特徵空間」。

**步驟**：新增 `src/experiments/run_mnist_dinov2_stack.py`。對 T1a 的同一批生成影像：
`(x+1)/2` 後餵 `metrics_features.dinov2_features`（灰階自動擴 3 通道，已支援）；
`compute_prdc_per_class`（k=5、num_classes=10）；`real_ref_precision` 沿
`run_cifar_cfg_multiseed.py:191-199` 的對半切法（並同時輸出 T4 的配平版）；分別以
signal_key="coverage"（CaF）與 "recall"（CaF-v2）跑 `select_and_report`。輸出 per-seed
選擇、regret、tau_robustness。

**驗收**：能回答「MNIST 上換成 DINOv2 特徵後，CaF 是否仍選中 oracle？」並把答案寫進
§6.1 的反轉歸因段。

### T1c. CIFAR-10/100 judge 特徵堆疊版 CaF【對應 A1(b)；GPU 中；需作者授權】

**目的**：反向交叉——CIFAR 上用 MNIST 式的任務對齊（judge penultimate）特徵算 PRDC 與選擇器。

**步驟**：
1. 新增 `src/experiments/run_cifar_judgefeat_stack.py`。特徵函式用
   `chamfer.cifar_penultimate_feature_fn`（512 維，eval 模式，輸入 [-1,1]），包 `no_grad`。
2. 影像來源優先序：(i) 檢查 `results/p1_assets*/seed*_w*/img_uint8.pt` 是否齊備（CIFAR-100
   seed10 的 w1/w1.5/w2.5 已知有；其餘與 CIFAR-10 需盤點 `run_p1_streaming.py` 與
   `regen_cifar100_cells.py` 的落盤內容）；(ii) 缺的 cell 以對應 regen 腳本決定性重生成
   （CIFAR-100 用 hash gseed；CIFAR-10 為重現凍結資料可用舊公式並在 metadata 註明）。
3. 先做單 seed（CIFAR-100 seed10、CIFAR-10 seed10）版本呈作者；作者授權後擴 8/3 seeds。
4. 輸出與 T1b 同構；最後產出一張 2×3「堆疊 × 資料集」regret 對照表（新增
   `docs/stack_ablation.md` 或直接進論文 §6.1）。

**驗收**：表格能明確回答「反轉跟著資料集走、還是跟著特徵空間走」；dry-run 模式不碰 GPU 可過。

### T2. 固定 w 與隨機可行點 baseline【對應 A7；GPU 無；免授權】

**目的**：補齊預註冊 D5 明列而論文未報的 baseline。純衍生，不碰 GPU。

**步驟**：
1. 新增 `src/experiments/run_baseline_fixed_random.py`，讀凍結的
   `results/cifar10_cfg_confirmatory.json`、`results/cifar100_cfg_confirmatory.json`、
   `results/selector_signal*.json`。
2. **固定 w 列**：對網格上**每一個** w 輸出 per-seed regret（整欄都報，避免事後挑點的
   HARKing 質疑）；在表中特別標注三行：網格最低 w（w1／g1）、w1.5、w2（文獻慣例值，出處
   Sariyildiz 2212.08420 與 Fan 2312.04567 的 SD scale 2.0，註明本專案 w 即 s-convention）。
3. **隨機可行點列**：對每 seed，用該 seed report 內的 `tau` 與各 config `precision` 重建
   可行集，輸出「均勻抽一個可行組態」的期望 regret（解析平均，不用抽樣）。
4. 更新論文表 5.3 為多行版（CaF/CaF-v2、FID-min、fixed-w1、fixed-w2、random-feasible）。

**驗收**：JSON per-seed 數字可由凍結檔逐位重導；表 5.3 更新後
`tools/verify_thesis_numbers.py` 相應擴充並通過。

### T3. CIFAR-10 confirmatory v2（無碰撞重跑）【對應 A4；GPU 高；需作者授權】

**目的**：舊公式在 30 cells 中造成 10 群、26 cells 的 latent 共用，破壞 seed 獨立性。

**步驟**：
1. `run_cifar_cfg_multiseed.py::gen_seed` 增加 `--gseed-formula {legacy,hash}` 參數：
   `hash` 對 cifar10 也走 `gseed_hash`；預設維持 legacy 以保舊資料重現，metadata 記錄公式。
2. 撰寫 `docs/amendment_cifar10_v2.md`（pre-registration amendment）：唯一改動＝種子公式
   換 hash；grid/seeds{10,11,12}/per_class 1000/steps 50/eta 0 全部不動；建議 reps 由原
   1 提為 5（消 σ_cls，須在 amendment 中明記為變更並給理由）。commit 早於真跑。
3. dry-run：枚舉 30 cells 的 hash gseed 驗無碰撞、driver 參數解析、單 cell 計時探針。
4. 【作者授權後】真跑 → `results/cifar10_cfg_confirmatory_v2.json`；以之重導
   `cifar10_c6_fidmin_duel_v2.json`、C8 支配結構複核（w2.5 是否仍嚴格支配三 oracle）、
   C2 偏相關複算。**舊檔保留**，論文正文改引 v2、附錄註明 v1 與碰撞問題。
5. ETA 估算：30 cells ×（10k 張生成 + 5×TSTR15）；以單 cell 計時探針實測後寫入 CHANGELOG。

**驗收**：v2 無碰撞（枚舉驗證入 metadata）；三判決在 v2 上重述；若判決翻轉（例如 FID-min
與 CaF 勝負互換），如實更新正文並在 §6.2 記錄差異。

### T4. τ 靈敏度與樣本數配平校準【對應 A6；GPU 低；免授權】

**目的**：CIFAR-100 的「CaF-v2 與 FID-min 打平」座落在 τ 刀鋒上（τ∈(0.783, 0.824] 恰好排除
oracle w1），且 real-vs-real 校準（250v250）與量測（500v500）樣本數不一致、系統性推高 τ。

**步驟**：
1. 新增 `src/experiments/run_tau_sensitivity.py`（純衍生 + 少量特徵運算）：
   a. 從兩個 confirmatory JSON 的 per-seed `report.tau_robustness` 提出完整 τ-掃描表
      （程式已算，論文沒報）；
   b. 對 tau_fraction ∈ {0.80, 0.85, 0.90, 0.95} 與「無 floor（裸 argmax recall/coverage）」
      重跑 `select_and_report`（輸入即凍結 JSON 的 configs），輸出每設定的 per-seed 選擇與
      regret。預期結果之一：無 floor 的 argmax recall 在 CIFAR-100 八 seed 全中 w1、regret
      0.00——必須如實報告，並同時報告「即便如此仍未達 D4 的 1.5pp 勝幅門檻」這一點。
2. **配平校準**（先做 seed 10，特徵已快取於 `results/p1_assets_cifar100/`）：
   a. 新函式 `real_ref_precision_matched`：把每 config 的 gen-vs-real PRDC 以 real 側
      子抽樣 250/class 重算（gen 側同步抽 250 以維持 gen=real 匹配），使量測與 250v250 的
      real-vs-real 校準在同一有效流形尺度；重跑選擇。
   b. 其餘 7 個 seed 的 gen 特徵未快取：標【需作者授權】的 GPU 選配（沿 T1c 的 regen 路徑）。
3. 在論文 §5.3 與 §6.3 增補「τ 靈敏度」小節：明寫選擇對 tau_fraction 的翻轉點，以及
   校準偏誤的方向與大小。

**驗收**：JSON 覆蓋全部設定組合；論文不再以單一 τ 值的「打平」作結，改為條件敘述。

### T5a. C1 配對統計【對應 A8；GPU 無；免授權】

**步驟**：新增 `src/experiments/run_c1_paired_stats.py`：讀兩個 confirmatory JSON，對
「TSTR argmax 組態 vs FID argmin 組態」做 per-seed 配對差、配對 t 檢定、符號檢定
（CIFAR-100 預期：差 [0.79,1.15,0.45,0.87,0.67,0.73,0.91,0.52]、t≈9.77、p≈2.5e-5、8/8）。
輸出並在論文 §5.3/§5.4.1 以**雙口徑**呈現：凍結格步口徑（路由依據，不回改）與標準配對檢定
（post-hoc 補充），兩者的結論差異明寫成一段：「偏移系統性存在、方向符合原假說，但幅度
0.76pp 實務可忽略，且 w1 為網格邊界點」。metadata 標 `post-hoc`。

**驗收**：統計量與審查報告重算值一致；正文新增段落不改任何凍結判決，只補充解讀。

### T5b. w<1 網格延伸 scout【對應 A8；GPU 中；需作者授權】

**步驟**：`run_cifar_cfg_multiseed.py` 已支援任意 guidance 列表；以
`--off-protocol --guidance 0.5 0.75 --seeds 10 11 12 --reps 5` 跑 CIFAR-100 scout（6 cells，
輸出 `results/cifar100_subunity_scout.json`，metadata 自動帶 off_protocol=True）。注意
`predict_eps` 對 `guidance_scale > 1.0` 才走 CFG 分支——w<1 需要修改該條件為
`guidance_scale != 1.0`（`src/gen1_mnist/ddpm.py:297`），修改屬行為擴充，需在 CHANGELOG
明記且驗證 w=1 路徑逐位不變（w=1 仍走純條件分支）。MNIST 側同樣補 g∈{0.5,0.75}（成本極低，
可併 T1a）。判讀目標：TSTR 峰是否落在 w<1（即 w1 是否只是邊界效應）。

**驗收**：w=1 舊路徑迴歸測試通過（同 gseed 同輸出）；scout 結果進 §6.3 限制或新觀察段。

### T6a. 開放結果資產與重現包【對應 A9；GPU 無；需作者確認發佈範圍】

**步驟**：
1. `.gitignore` 修訂：把一刀切的 `results/` 改為細則——追蹤 `results/*.json`（幾 MB 級），
   續排除 `results/p1_assets*/`、`results/*.png`、`fid-tmp/`。首次 commit 全部凍結 JSON。
2. 新增 `tools/make_release_bundle.py`：打包 checkpoints（cifar10/100_cfg.pt、judge.pt、
   mnist ckpt）與 p1 資產清單、sha256 清單，輸出上傳說明（Zenodo/HF 由作者執行）。
3. `README.md` 增「重現指南」節：從權重到每張表的指令序列（引用各 driver docstring 的
   confirmatory 指令）。

**驗收**：clone 後不依賴作者本機即可核對論文每個 scalar 的來源檔；
`tools/verify_thesis_numbers.py` 在乾淨 clone 上可跑。

### T6b. TSTR 全面種子化【對應 A9；GPU 無；免授權】

**步驟**：
1. `cifar_classifier.py`：`train_classifier`/`run_tstr` 增 `seed=None` 參數（預設 None＝
   維持現行不種子化行為，保住凍結對帳語意）；seed 非 None 時：`torch.manual_seed` 控制
   模型初始化、`DataLoader(generator=...)` 控制 shuffle、`_AugmentedTensorDataset` 的增強
   改用傳入的 `torch.Generator`（`torch.rand/randint` 加 generator 參數）。
2. `run_cifar_cfg_multiseed.py::measure` 的 reps 迴圈給定衍生種子
   `int(sha256(f"tstr_{dataset}_{seed}_{w:g}_{rep}").hexdigest()[:15],16)`；由
   `--tstr-seeded` 旗標啟用，凍結 v1 路徑不受影響；T3 的 v2 confirmatory 啟用。
3. 自我測試：同參數兩次 `run_tstr(seed=k)` 結果逐位相同；`seed=None` 行為與現行一致。

**驗收**：v2 之後 TSTR 進入可對帳集；CHANGELOG 記錄。

### T7. 相關工作與主張重寫【對應 A3、B3、A1 文字面；GPU 無；免授權】

**改 `docs/thesis_draft.md`（及 `docs/paper_branch3_diagnostic.md` 同步）**：

1. **參考文獻新增**（§2.4 與第 6.4 節逐一定位，全部給 arXiv 編號）：
   Ravuri & Vinyals 2019（1905.10887；FID/IS 不預測 CAS——「FID≠效用」的完整先行論文，
   不得再僅以 DPDM 旁註帶過）；Sariyildiz et al. CVPR 2023（2212.08420）；Azizi et al.
   TMLR 2023（2304.08466；**FID 與 CAS 最優重合於 1.25**，須正面引用為「FID-argmin 是好
   選擇器」的先例）；Shipard et al. CVPR-W 2023（2302.03298；CIFAR-10/100 上的多樣性
   guidance 技巧）；Tian et al. StableRep（2306.00984；w 非單調內部峰）＋ SynCLR
   （2312.17742）；Astolfi et al. 2024（2406.10429；PRDC 式前沿選組態——與 CaF 操作點
   最近，必須正面差異化）；CADS（2310.17347）；Kynkäänniemi et al. 2024 guidance interval
   （2404.07724）；Karras et al. autoguidance（2406.02507）；Kynkäänniemi et al. 2019
   improved P&R（1904.06991；recall/precision 定義出處）；Alaa et al. 2022（2102.08921）；
   Bartlett et al. 2017 與 Sorscher et al. 2022（機制章引用而未列）。
2. **DPDM 刻畫修正**（§2.4）：改寫為「Dockhorn et al. 全文維持 FID 調參與 Acc 調參兩套
   sampler 設定並明示 diversity 機制解釋」，刪去「順帶」的弱化措辭。
3. **B&N 引用降範圍**（§2.1、§3.3）：明寫其定理僅覆蓋隨機 CFG_DDPM、且「朝 gamma-powered
   中間分佈銳化」≠「朝 class prototype 集中」；新增一段討論「本文全程 η=0 DDIM 與 B&N
   理論的張力、以及 MNIST 上 η-null 觀察與其預測的關係」。
4. **頭條句條件化**（等 T1a/T1b 結果落地後執行）：摘要與 §6.1 的
   「FID-min……在 MNIST 上選錯組態」「同一套 coverage 選擇器」兩句，依 T1 實測結果改寫；
   「反轉」一律限定為「在本文三個量測堆疊配置下」直到 T1c 完成歸因。
5. **H3 句降級**（若 T8 未執行）：摘要與 §6.4 中「Chamfer 增益不見於 coverage」改為
   「在本文的單向簡化重寫、單 seed 設定下未見於 coverage；官方雙向實作報告相反方向
   （0.603→0.912），故此讀數可能是簡化假影」，並把 §5.6 移附錄。
6. **§3.5 matched-probe 描述改為與實作一致**（對應 B5）：明寫 FID-min 用 cleanfid 全訓練集
   stats（CIFAR-10 與 CaF probe 不同份），引 C7 說明小 probe 穩定性為佐證而非實作。

**驗收**：`tools/verify_thesis_numbers.py` 通過；全文搜尋不再有「順帶」「同一套」等被點名
措辭的未修訂殘留。

### T8. H3 公平化 v2【對應 A5；GPU 中；需作者授權】（與 T7 第 5 點二選一，建議做）

**步驟**：
1. `src/core/chamfer.py`：`chamfer_guided_ddim_sample` 增 `guide_every=1` 參數（官方
   G_freq=5）；`term="chamfer"`（雙向，函式已存在）納入對決設定。
2. `run_cifar100_h3_duel.py` v2：weight 掃描 {0.05, 0.1, 0.3, 1.0}（單 seed scout 選定後
   跑 ≥3 seeds）、`--guide-every 5`、雙向項為主結果、單向項作 ablation；Chamfer 臂與
   vanilla 臂 rep 數對稱（各 5）；新增 FLOPs 記帳欄（生成 NFE + 反傳次數 × 特徵網路成本），
   在 metadata 給出兩臂總 FLOPs，「matched-budget」的口徑（張數 vs FLOPs）明寫。
3. 量測補充：Chamfer 臂的 coverage 除 DINOv2-224 外，加算「導引所用特徵空間本身」
   （DINOv2-112 或 judge）的 coverage——檢驗「增益不在 coverage」是否為量測空間錯位。

**驗收**：v2 結果落 `results/cifar100_h3_duel_v2_*.json`；論文 §5.6 依 v2 重寫；若雙向版
coverage 轉為上升（與官方一致），必須撤回「破 Chamfer 機制敘事」的攻擊句。

### T9. TSTR 協定強化與 real 上限線【對應 B2；GPU 中；需作者授權】

**步驟**：
1. real-data TSTR 上限（便宜、先做）：`run_tstr` 直接吃 `load_real_per_class("cifar100",
   500)` 的真實資料，epochs=15 與 50 各 5 reps → `results/tstr_real_ceiling.json`；
   CIFAR-10（1000/class 匹配口徑）同。論文表 5.2/5.4 加一行 real ceiling。
2. epochs 消融：選 CIFAR-100 seed10 的 {w1, w1.5, w2.5} 三個 cell（影像資產已有/可 regen），
   epochs ∈ {15, 50} × 5 reps，量 σ_cls 是否隨訓練加長顯著縮小、排序是否不變 →
   `results/tstr_protocol_ablation.json`；結果寫進 §5.4.3（「噪聲地板為協定內生」的回應）。
3. 第二架構（選配）：WideResNet-16-4 或 ViT-Tiny 在同三個 cell 各 3 reps，驗排序穩健性。

### T10. margin-pruning 介入【對應 B3；GPU 中；需作者授權】

**步驟**：新增 `src/experiments/run_cifar100_margin_intervention.py`，沿
`run_cifar100_d3_intervention.py` 骨架：資料同 seed10 w2.5 資產（`judge_out.pt` 已含
margins）；介入＝移除 margin 最低的 n 個樣本 vs 等計數隨機移除（n 取與 C3 相同的 13606，
另加 n/2 檔位），N=8 retrains（沿 n8 慣例、用 T6b 的種子化）。這才是機制鏈中介量
（near-boundary 供給）的直接介入；輸出與 n8 檔同構（差、SE、t、CI、MDE）。論文 §5.5 併述
兩種介入（coverage-matched 與 margin-pruning）的結果。

### T11. ViT-L/14 PRDC 複算【對應 B1；GPU 低；免授權】

**步驟**：`metrics_features.get_dinov2` 已可載任意 model_name；新增
`src/experiments/run_prdc_vitl14_seed10.py`：對 CIFAR-100 seed10 全部 10 config
（w1/w1.5/w2.5 用快取影像，其餘沿 fd_dinov2 的 regen 路徑或其已存 dino_feat 不可用——
ViT-L 需要影像，缺的 config 標 GPU regen【需授權】；最小版先做三個快取 cell + real）以
`dinov2_vitl14`（1024 維）重算 per-class PRDC 與 FD，對照 ViT-B 數字，檢驗 coverage 排序
與 which-FID 讀數是否 backbone 相依。結果進 §6.3 或附錄；引 Stein et al. 的 ViT-L 建議。

### T12. 文字／圖表／衛生批次【對應 B5、B7、B8、C1–C5；GPU 無；免授權】

一次 PR 完成下列小項：

1. **路徑洩漏**：刪 `docs/thesis_draft.md:1259` 的 `C:\Users\fartw\...` 本機路徑；全 repo
   `grep -r "C:\\\\Users"` 清殘留（`.claude/`、歷史文件）。
2. **圖 5.2**（`src/figures/make_thesis_figures.py`）：MNIST 橘柱改用 T1a 實測 FID-min
   regret；在圖說明確標注各資料集保真代理的定義；未完成 T1a 前先在圖上加「precision-argmax
   （非 FID）」浮水印級標注。
3. **圖 5.6**：TSTR 軸自 0 起（或用 broken-axis 標記）；標題 "moat duel" 改為
   "matched-budget comparison"；圖說補「單 seed、rep 不對稱」。
4. **EDM gate 段**（§附錄 E 量測錨點、`docs/results_analysis.md`）：明寫本專案 1.848 為
   單次評估、官方 1.79 為 min-of-3（NVlabs README），兩者口徑差；選配：跑三次取 min 補一個
   同口徑數字（GPU 低）。
5. **去代號化**：正文首次出現處以描述名為主、代號括註（多數已做，逐節掃殘留：「護城河」
   全部改「matched-budget 對照」；C0–C8/D0–D12 在摘要與章首不得裸用）。
6. **README 重寫**：頭條改為診斷結論（現況仍是被反證的 "Sampling for Utility, not
   Fidelity" 敘事）；過時的 coverage 版 CaF 定義加「歷史定義，現行見 results_analysis
   §selector plan-of-record」指標。
7. **`var/`、`src/var_mini/` 支線**：移入 `attic/`（或獨立 branch），README 註明與主線無關。
8. **參考文獻格式**：補 T7 清單後統一格式；「Bradley & Nakkiran NeurIPS 2024 M3L Workshop」
   核對正式 venue 再定稿。

---

## 3. 禁止事項（違反任一即視為任務失敗）

1. 不得修改或重算後覆寫任何凍結 `results/*.json`；不得修改 `docs/prereg_cifar100.md`、
   `docs/verdict_cifar100.md` 的任何字元。
2. 不得改動 `FROZEN_CIFAR100` 常數值；新設定一律走新常數 + 新 amendment。
3. 不得在無 amendment 的情況下執行任何 confirmatory 等級的 GPU 跑；不得自行執行
   標注【需作者授權】的任務。
4. 不得為了讓對帳通過而調整對帳腳本的容差；對帳不符必須回報並停止。
5. T6b 的種子化不得改變 `seed=None` 預設路徑的行為（凍結資料的「TSTR 不入對帳集」語意
   必須保留）。
6. 文字修訂不得刪除「不主張」「限制」章節的任何既有誠實聲明；只能增補，不能弱化。

---

## 4. P1 任務（骨架規格，全部【需作者授權】，實作前先與作者確認方案）

**P1-1 EDM 第二 backbone（對應規模問題）**：目標＝驗證三判決是否 backbone 相依。方案 A：
以 NVlabs 官方 EDM CIFAR-10 cond + uncond 兩個 checkpoint 組 CFG（兩模型混合
`eps_u + w(eps_c − eps_u)`，需在 `phase1_edm_repro.py` 的取樣器上加雙模型路徑），掃同
10 點 grid、3 hash seeds、per_class 1000、TSTR 同協定。方案 B：自訓第二個不同架構/增強的
backbone。先寫 dry-run 與 ETA，作者選方案。

**P1-2 更高解析度資料集**：候選 STL-10（96×96，10 類，主張「便宜代理」經濟性開始成立的
最小尺度）或 ImageNet-100@64。需新 backbone 訓練 + 新 judge + 新 prereg（D 包同構）。此為
「投主會」的規模門檻項，成本最高，最後做。

---

## 5. 驗收總表（代理完成後逐項自查）

| 檢查 | 通過條件 |
|---|---|
| 凍結完整性 | `run_c6_fidmin_duel.py --no-write` 兩資料集 ALL_MATCH；prereg/verdict diff 為空 |
| 對帳 | `tools/verify_thesis_numbers.py` 通過（含 T2/T5a 新表擴充） |
| CHANGELOG | 每個任務至少一行，含結論數字 |
| 自我測試 | 每個新模組 `uv run python <file>` self-check 通過；T6b 種子化決定性測試通過 |
| 文字一致 | 論文中每個數字都能指到一個存在於 repo 的 results JSON；A1/A5/A8 相關句已條件化或以新數據落地 |
| 衛生 | repo 內無本機絕對路徑；圖表無截斷軸未標注；無「moat/護城河」殘留 |
