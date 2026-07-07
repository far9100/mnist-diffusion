<!-- 用途：confirmatory 揭盲後修正計畫——H-C2 凍結義務執行、confirmatory record 三判決分立、零重生成 exploratory 批、CIFAR-100 預註冊包（v2 §9.3 牆例外）、文件同步、流程債。依據：2026-07-06 外部嚴審三輪與稽核 dossier（2026-07-06-02）。本檔為執行計畫，非治理或定位文件；不動任何已凍結項目、不預選分支。 -->

<!-- SUPERSEDED（2026-07-06 α 定號步）：本檔（計畫 v1）由 records/2026-07-06-05_plan_remediation-final.md 取代，僅保留為決策軌跡；過時編號引用不回改，以 -05 §0 譜系表為準。 -->

# 修正計畫：confirmatory 揭盲後的執行、登記與同步

## 0. 性質、依據、合法性

本計畫把 2026-07-06 外部嚴審的結論落為工作包 A–F。三個合法性聲明：

1. 本檔是執行計畫，不是治理或定位文件，不觸 v2（records/2026-07-05-12）§9.3 的治理凍結。定位 v3 不在本計畫內產生——它是 CIFAR-100 分支樹（工作包 D）落枝後的輸出。
2. 工作包 D 援引 §9.3 明文例外：「牆的執行本身強制觸發的再登記，且以最小篇幅為之」。CIFAR-100 是登記在案的科學承重牆（v2 §9.4、README Phase 1-4），其 confirmatory 執行所需的判準、門檻、baseline 集即該再登記。
3. 一切已凍結項目照舊生效：H-C2 裁決程序與準則（-13）、grid/seeds/steps/η（-11）、judge 與 near-boundary threshold（-08）、regret/top-k 定義（-02 §6）。對 CIFAR-10 confirmatory 的裁決永久凍結（v2 §9.5）；本計畫只新增**前向**的規則與登記。

全計畫通用判準（v2 §8 揭盲後升級版）：**任何調整或規則，若其理由必須引用 confirmatory 數字的「方向」才成立，則其產生的一切主張皆為假設，不得由同批資料裁決，只能由 CIFAR-100 確認。** 各工作包逐項過測，見 §11。

慣例：每個工作包依 claude.md 先立 record（Goal / Result / Follow-up）。本檔擬編號 2026-07-06-03；入庫前依 records/ 現況核對 NN（-01 為 README 校對、-02 應為稽核 dossier，未落檔則由 B2 補）。

## 1. 事實基準（供各 record 與文件引用；全量數據見 dossier 2026-07-06-02）

1. **FID/TSTR 重合**：CIFAR-10 confirmatory 上 clean-fid 最小與 TSTR 均值峰同在 w1.5（8.82 / 63.96）。README 頭條句「效用最優點偏離 FID 最優點」在本資料上被反證。
2. **排序**：ρ_Spearman(−char_clean_fid, TSTR) ≈ 0.96；ρ(coverage, TSTR) ≈ 0.64。此二值為稽核手算，B1 入檔前由腳本複核。
3. **FID-min baseline 勝訴**：選 clean-fid 最小（w1.5）的 per-seed regret ≈ 0.91pp（2.45 / 0.28 / 0.00）；CaF 3.69pp（0.54 / 5.03 / 5.49）。未登記的平凡 baseline 在 confirmatory 自身資料上擊敗貢獻本體。w1/w2 的 per-seed FID 由 C6 補實。
4. **Pareto 支配（結構性失明）**：w2.5（.873, .792）在 (precision, coverage) 平面同時嚴格支配三個 per-seed oracle——w2（.858, .777）、w1.5（.841, .751）、w1（.806, .645）。故任何 τ 之下 `argmax coverage s.t. precision ≥ τ` 皆不可能選中任一 oracle。tau_robustness.picks（11 點掃描三 seed 幾乎全選 w2.5）為實證。此為訊號對的結構問題，非校準問題。
5. **τ 混合結論**：w1.5 三 seed 皆可行 → 過衝主體為 coverage 排序；w1 於 seed 11/12 不可行（邊際 .0034 / .0092）、seed 10 可行 → w1 可行性由 seed 級 precision 噪聲決定，τ=0.9×ref 落點壓在低段 precision 高原上（knife-edge）。seed 11 的 oracle 即 w1，其 5.03 regret 部分歸 τ。
6. **雙段機制**：低中段（w1→w2.5）near-boundary .256→.046 先崩、coverage .645→.792 仍升（coverage 為滯後代理）；w1 的 ln_excess +.044 為全網格唯一正值（label-noise 競爭機制在 w1 實證獲勝，解釋 w1 < w1.5）。高段（w3→w8）coverage 與 TSTR 同崩、near-boundary 反升 .033→.059 且 precision 同降（指標疑受離類樣本污染）。凍結的全網格偏相關會把雙段抹成單一 ρ——照跑（凍結義務），但科學重心已不在該 ρ。
7. **MNIST 證據降級**：分離量僅一格步（grid 無 w1.5）、FID 側單 seed argmax、bespoke classifier-FID 不可比文獻、-03-06 已記方向不轉移 → 軼事級。thesis 目前在任何尺度均無可辯護的支持點，唯一未決在 per-config FD-DINOv2（C1）。
8. **治理洞**：其一，selector 無凍結數字門檻（-02 §6 僅定性「regret 低、top-k 命中」）→ CIFAR-10 selector 結果只能描述性報告，不作 confirmatory 過/敗判定。其二（待 D1 核驗），H2「內部最優」是否有統計口徑不明；左肢（w1→w1.5，+0.8pp）在 seed 級翻向、TSTR 為單次分類器訓練，目前無法判定上升肢是否成立。
9. **時序**：052492c（07-05 14:59:54，規格凍結）→ ec1f746（23:53:43，儀器實作 +295/−19）→ 推導起跑 ≈07-06 00:29（mtime − ELAPSED，非登記值）→ confirmatory.json mtime 08:45:11。ec1f746 判定合規（prose 先凍、實作後寫、資料前跑）但實作層自由度（per-class floor 公式、FID 對 train split、樣本數）在凍結後行使、僅 1-config 探針——記為流程債，由 D11 修正。
10. **存活項**：guidance 災難性非單調（w≥3 崩 11–30pp）鐵證；最優位置跨資料集移動（MNIST w1、CIFAR-10 w1.5–2 帶噪）→「需要選擇程序」的動機句存活；「無單一便宜代理普遍可靠」（MNIST 與 CIFAR-10 給出相反的選擇器判決：MNIST 上 CaF 0 / FID-min 1.30，CIFAR-10 上 FID-min 0.91 / CaF 3.69）為新的、更強的候選命題。

## 2. 凍結邊界與格殺清單

不可動項見 §0.3。以下動作在執行全程被明文禁止；任何人（含本 agent 自身的中間推理）提出即拒絕並呈報作者：

- 以 clean-fid 反證為由改用 FD-DINOv2 作主 FID（-08「不得事後挑支持者」原則類推）。
- 重劃分段、只取高段裁決 H-C2（v2 §4 廢分段的理由即「無可事後劃定的邊界」）。
- 把本次 confirmatory 降格改稱 pilot、宣稱「真正的 confirmatory 在 CIFAR-100」（HARK-by-demotion；預期為未來數週最大誘惑）。
- 改以 mean-curve regret 2.77 為主數字（已裁定取嚴：per-seed 3.69 為主、2.77 並列註明）。
- 事後補一個恰可通過的 regret 門檻（門檻缺失的後果是描述性報告，不是補門檻）。
- 掃 τ fraction 尋找會選中 w1.5 的值（§1.4 已證無解；且屬事後調參）。
- C4 變異分解加跑到顯著為止（optional stopping；N 事前定死）。
- 在 D 包或任何裁決文件中引用 CIFAR-100 的任何讀數（scout 讀數亦然）。
- H-C2a 即使顯著，寫出「coverage 驅動效用」等因果措辭（-07 §4：偏相關屬相關非因果、不足，須疊加介入式證據）。因果語言的資格由 C2/C3 介入實驗與 CIFAR-100 決定。
- 修改 results/cifar10_cfg_confirmatory.json（唯讀）；一切分析輸出寫入新檔。

## 3. 工作包 A：H-C2 裁決執行（凍結義務，最先跑）

- **A0 執行記錄（跑前落檔）**：一份小 record 凍結 permutation 次數 N=100,000、RNG seed=0（作者確認後生效）；同檔寫入兩則分支敘事草稿（H-C2a 顯著 / 不顯著各一），並明文註記：**均值表已揭盲、點估計方向半可見，本次盲性僅餘 p 值本身**。
- **A1** 跑 `run_c2_partial` 於 confirmatory JSON（DINOv2 主裁決），程序與準則依 -13，一字不動。
- **A2** Inception robustness（-08 §4 凍結義務）。同批加跑：Inception coverage 上重放 selector（**標 exploratory**，結果歸 C 批 record）——檢查 coverage 峰是否仍在 w2.5，判別過衝是否表徵特定。
- **A3 報告要求**：效果量 + bootstrap CI 照 -13 報，但必須並列註明 3-seed bootstrap 之退化（重抽組合僅 10 種、資訊量極低），補充 per-seed 點估計範圍與 config-level jackknife，**標 supplementary、不取代裁決**；H-C2b 準則不對稱聲明（n=10 下「不顯著」近乎保證，不得讀為 precision 無效的正面證據）；全網格 ρ 抹平雙段之 caveat（引 §1.6）；措辭依 §2 因果禁令。

## 4. 工作包 B：confirmatory record（三判決分立）

- **B1** 建 record（編號依現況），結構強制分為三個互不裁決的判決，禁止交叉混寫：
  - **判決一（thesis）**：FID-opt 與 TSTR-opt 重合（§1.1–1.2，Spearman 複核後入檔）；災難性非單調成立；「內部最優」的上升肢因 seed 級翻向與單次訓練標 **unresolved**，待 C4；「任何固定 guidance 值必然次優」全稱句自所有文件撤下（E2 執行）。
  - **判決二（H-C2）**：依 A 結果如實登載，附 A3 全部 caveat。
  - **判決三（selector，描述性）**：開頭明文「協定未凍結 regret/top-k 數字門檻，故本節不作 confirmatory 過/敗判定，僅描述」；內容含 Pareto 失明分析（§1.4）、τ 混合結論與 knife-edge（§1.5）、FID-min 對決（§1.3，C6 補實後）、modal_fraction 1.0 重讀為**低變異高偏差**（一致地錯向同點，偏差不可被平均消除）、可辯護措辭「可靠避崖器、糟糕平台優化器——且 FID-min 同樣避崖、成本結構相同」。regret 主數字 per-seed 3.69、mean-curve 2.77 並列。
  - **附錄**：時序證據鏈（§1.9，推導起跑明標推導值）；ec1f746 diff 稽核結論；MNIST 降級注記（§1.7）。
- **B2** 核驗 dossier 是否已落檔為 2026-07-06-02；未落檔則原文入庫（稽核軌跡不回改），其漏判（clean-fid/TSTR 未對讀致 FID 重合與 FID-min 勝訴漏抓）之修正寫入 B1，不改 dossier 原文。

## 5. 工作包 C：零重生成 exploratory 批

通則 **C0**：每項在各自 record 先寫判定規則與全部參數、再執行；全部標 exploratory；規則不因結果回改；C 批結果入獨立 record，B1 以指標引用、不混編。

- **C1 which-FID 口徑凍結 + per-config FD-DINOv2 補算**：先落規則——雙特徵空間皆分離＝強命題；僅一空間分離＝表徵依賴弱版本，如實標注；皆不分離＝CIFAR-10 尺度反證確立（-08 原則類推：不得事後挑支持者）。規則落檔後才計算（DINOv2 特徵已抽好，零重生成）。此規則同時被 D 引用為 CIFAR-100 的 which-FID 口徑，**故 C1 規則段須在 D commit 前完成**。
- **C2 boundary-targeted pruning**（-07 §4(a) 的字面實例）：取 w1.5 既有樣本集，等量剪除 near-boundary 樣本 vs 剪除非 boundary 樣本（coverage 與樣本數 matched），比較兩者 TSTR 損失。判定規則先寫（例：剪 near-boundary 之 TSTR 損失顯著大於對照 → §4(a) 介入支持成立）。此項可在 D 中引用為創始要求（-07 §4）的執行。
- **C3 coverage-matched pruning**：把 w1.5 樣本剪至 w2.5 的 coverage 水位、precision 持平，測 TSTR 是否下降。-07 §4 不直接涵蓋此設計，橋接論證（其為 §4(a)「控制 coverage」的實例化）寫入 record；掉＝coverage 因果性獲介入支持，不掉＝coverage 連高段相關都可能只是伴生。
- **C4 分類器訓練變異分解**：w1–w2.5 × 3 gen-seeds，每組固定 +2 次重訓（N=2 事前定死，禁加跑）；判定規則先寫：左肢 +0.8pp 相對分類器噪聲 SD 的判讀標準（成立／不成立／未決三態），據此更新 B1 判決一的 unresolved 標記。
- **C5 near-boundary 純度過濾**：near-boundary ∧ judge 標籤正確 ∧ 流形內（precision 意義），重看高段回升是否為離類污染（§1.6），為任何未來機制敘事或 CaF-v2 的訊號設計提供地基。
- **C6** 自 JSON 抽 w1、w2 的 per-seed char_clean_fid，補實 FID-min per-seed 對決（預期不改結論，做實它）。
- **C7 small-probe FID 排序穩定性**：把真實參考重抽至 CaF probe 規模（n_real=10k、對半 5k 口徑），多次重抽看 clean-fid 排序穩定性 → 直接餵給 D5 的 matched-probe FID-min baseline 設計。
- **C8 Pareto 失明引理成文**（一頁，入 docs）：陳述為一般命題——凡 oracle 落於 (precision, coverage) 被支配區，`argmax coverage s.t. precision ≥ τ` 類選擇器在任何 τ 下必然錯過——附本案數字為實例。

## 6. 工作包 D：CIFAR-100 預註冊包（§9.3 牆例外，單檔、最小篇幅）

**時限（hard gate）**：D 以單一 record commit，時間必須早於任何 CIFAR-100 合成樣本的產生（scout 含），git 可驗。CIFAR-100 backbone 訓練可依 v2 §9.4 先行（訓練非量測結果），但**取樣程式在 D commit 前不得執行**；驅動腳本加入該檢查或由執行順序保證。

- **D0 前置核驗**：調出 -02 之 H1/H2/H3 原文與各自檢定口徑，全文附入 D。若 H2（內部最優）無統計口徑 → 認定第二治理洞，處置同 selector：CIFAR-10 上描述性報告（回饋 B1 判決一措辭），CIFAR-100 口徑在本包補登。
- **D1 四分支決策樹（資料前凍結）**：分支一＝分離出現（依 C1 口徑）且原版 CaF 於 matched probe 勝 FID-min → thesis 帶邊界條件復活，CIFAR-10 如實報告為不分離 regime 的資料點；分支二＝分離出現但 CaF 敗 → thesis 活、selector 死，論文＝邊界條件＋診斷；分支三＝不分離但雙段機制複製（D3）→ 診斷論文；分支四＝皆否 → 負結果短文，不硬寫。
- **D2 分支一方向性預測登記**：白紙黑字——「以約一成的整體先驗，預測 CIFAR-100 上分離出現（先驗約 1/3；機制理由：per-class 容量吃緊使 FID-opt 右移、類鄰接更密使邊界抽空提早令 TSTR-opt 持平或左移）且 CaF 勝 FID-min（條件先驗約 1/3）」。明文標注：此為誠實的預測登記，非資源承諾；命中則主張因被預測而升檔，未中則零 HARK 殘留。
- **D3 分支三操作化（防不可證偽）**：三觀察量事前寫死——(i) 低中段 near-boundary 在 coverage 上升或持平區間內單調下降；(ii) 高段 coverage 與 TSTR 同崩；(iii) 高段 near-boundary 與 coverage 脫鉤。三中二即判「機制複製」。
- **D4 selector 數字門檻（作者定數，agent 不得代填）**：格式建議「regret@selected ≤ ⟨X⟩pp 且 top-3 命中 ≥ ⟨Y⟩/3 seeds」。以 CIFAR-10 觀測（FID-min 0.91、CaF 3.69）為錨屬合法——相對 CIFAR-100 而言 CIFAR-10 已是先驗資料。
- **D5 強制 baseline 集**：FID-min（與 CaF 同一真實參考預算、matched probe，設計引 C7）、固定 w 慣例值（登記具體值與出處，例如 Fan 配方）、隨機可行點。三者缺一不可入 confirmatory 報告。
- **D6 復活條件**：CaF 存活需 (a) 分離依 C1 口徑出現 **且** (b) CaF matched-probe 勝 FID-min；缺一即依樹落枝，論文重心不再回談 selector 貢獻。
- **D7 Chamfer 牆條件化**：分支一／二落地 → matched-budget 對決升回必跑之牆；分支三／四 → 降為選配。CPU 空窗的 Chamfer 適配照 v2 §9.4 原排程進行，作為便宜保險。
- **D8 CaF-v2（作者決定去留）**：若納入——規格（候選訊號含 near-boundary 供給，引 C5）＋門檻同批登記；明寫 judge 依賴使「免任務標籤」定位破功之代價；框架為 discovery/validation split：CIFAR-10＝discovery（v2 在此設計，永遠 exploratory）、CIFAR-100＝validation（登記後才有 confirmatory 地位）。若不納入，本節記錄該決定。
- **D9 量測程序凍結（凍程序、不凍數字）**：CIFAR-100 judge 品質 gate 的定法程序（準確率與逐類 floor 如何定）、near-boundary threshold 重校準程序、特徵空間與交叉裁決依 -08。數字由程序在 D commit 後產出、以最小儀器 record 登載。
- **D10 執行時序**：D commit → judge 訓練與 threshold 校準（依 D9 程序）→ scout（1 seed，**僅用於定網格**，其讀數不得回饋任何 D 內判準）→ 網格凍結 amendment → confirmatory。即 CIFAR-10 流程原樣照抄。
- **D11 凍結定義升級（適用本包及其後一切凍結）**：凍結＝prose ＋ 實作程式 ＋ 於已揭盲資料（CIFAR-10 confirmatory JSON）上的 dry-run ＋ 輸出雜湊，四者同 commit。修正 ec1f746 型缺口（§1.9）。
- **D12 兩份一頁論文骨架**：分支一版、分支三版，於 D commit 前入 docs/（反 HARK 保險）。分支三骨架寫作時直接對準其 so-what 弱點（建設性尾章：FID-min 實務建議、near-boundary 供給監測、exploratory CaF-v2 三選），使 C2/C3/C5 的介入設計對其瞄準。

## 7. 工作包 E：文件同步（兩段式）

- **E1（即刻，不等 A/B）**：README 進度節純事實更新——Phase 1-3 confirmatory 已完成（2026-07-06 08:45，results/cifar10_cfg_confirmatory.json，30 configs，8h16m），H-C2 裁決待跑。不含任何詮釋。
- **E2（B 之後，措辭呈作者核准）**：README 頭條句**不改寫**（定位 v3 為分支樹輸出，§9.3 禁止現在寫），改為主張句後加狀態注記：「此主張於 CIFAR-10 confirmatory 未獲支持（FID-opt 與 TSTR-opt 重合於 w1.5），最終定位待 CIFAR-100 分支裁決，見 records/⟨B⟩ 與 ⟨D⟩」。「必然次優」全稱句撤下，改為邊界條件的開放問題措辭。
- **E3（B 之後）**：docs/results_analysis.md 新增 confirmatory 節，結構鏡像 B1 三判決；附雙段機制圖（near-boundary 與 coverage 沿 w 的解耦）與 pilot 對照。
- **E4**：docs/paper_intro_draft.md **不重寫**（等分支落地），檔頭加 banner：thesis 句待 CIFAR-100 裁決，引 B。最小變更原則。
- **E5（作者核准後）**：claude.md 加兩行慣例——凍結定義依 D11；一切 driver 於 metadata 寫入 start_timestamp。

## 8. 工作包 F：流程債

- **F1**：為現有與未來 driver（run_cifar_cfg_multiseed.py、run_c2_partial.py、CIFAR-100 各 driver）加 metadata start_timestamp。一次性小 patch，不觸量測邏輯，diff 呈報。
- **F2**：ec1f746 稽核結論入 B1 附錄（§1.9），不另立檔。

## 9. 執行順序（依賴關係）

1. E1 ∥ F1（即刻，與一切無依賴）。
2. A0（作者簽核 permutation 參數）→ A1 → A2、A3。
3. C1 規則段、C0 各項規則落檔（可與 A 並行；規則不依賴任何結果）。C6 隨時可做。
4. B1、B2（需 A 結果與 C6）。
5. C2–C5、C7、C8 執行（規則已凍，與 B 起草並行；結果入 C 批 record，B1 留指標）。
6. D 起草（可與上並行）→ D0 核驗、C1 規則引用就緒 → **作者簽核 D2/D4/D8** → D commit。hard gate：CIFAR-100 取樣前 D 必須已入 git。
7. E2–E4（B 之後）、E5（作者核准）。
8. CIFAR-100 依 v2 §9.4 續行：backbone 訓練不受阻；取樣、scout、confirmatory 依 D10 時序。

## 10. 授權邊界

- **agent 自主**：A1/A2 執行、C1 計算（規則凍後）、C2–C7 執行（設計 record 呈報後）、C8、E1、E3、E4、F1、一切 record 起草。
- **作者裁決（阻塞點）**：A0 參數簽核；D 全包 commit，尤其 D2 先驗聲明、D4 門檻數字、D8 CaF-v2 去留；E2 頭條句注記措辭；E5 claude.md 變更。
- 呈報：每工作包完成即呈 record 草稿與關鍵數字，不批次積壓。

## 11. 自我測試與護欄

逐包過升級版 §8（§0）：A＝執行 -13 已凍程序，pass；B＝事實報告，pass；C＝每項規則先於計算、標 exploratory，逐項於各自 record 過測，pass；D＝判準先於一切 CIFAR-100 資料，以 CIFAR-10 已揭盲資料錨定 CIFAR-100 門檻屬跨資料集先驗使用，合法，且 D 內禁引任何 CIFAR-100 讀數，pass；E＝事實注記與狀態標示、非定位改寫，pass；F＝與結果無關之流程，pass。

兩道護欄，寫給執行期的每一天：

- **資源配置與裁決分離**：分支機率估計（P₁≈10–12%、P₂≈20–22%、P₃≈40–45%、P₄≈20%）僅為 C/D7 準備權重的依據，不得出現在任何裁決條文，不得影響 D1/D3/D4/D6 的判準內容。D2 登記的先驗是對預測的誠實聲明，非裁決輸入。登記前唯一允許修正先驗的來源是外部文獻，絕非自家 CIFAR-100 資料——一次 scout 偷看即令判準登記作廢。
- **希望測試**：CIFAR-100 結果落地當日，若任何人發現自己希望某個特定數字出現——無論偏向哪個分支——即重讀本檔與 D1。分支三的最大優勢從來不是機率高，是它不需要任何人希望任何事：其論文的六成已在 results 目錄裡。

執行完 A–F 後，本專案的狀態應為：CIFAR-10 的三判決如實入檔、零重生成證據補齊、CIFAR-100 的球門在開球前白紙黑字釘死——連輸的分支一起。之後的一切由資料落枝，不由敘事。
