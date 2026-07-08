<!-- 用途：作者裁決與執行指令（交 coding agent）。彙整三份建議書（P0 裁決建議書、D 簽核審查意見書、B 骨架一致性核對）之全部可執行內容為單一指令檔；本檔由作者簽發生效，簽發即構成 §0.1 所列 records/2026-07-06-05 §10 阻塞點之裁決。本檔自足執行，三份建議書為 backing、可附可不附。依 -06-05（唯一執行計畫）、-07-01、-06-06、-06-09、-06-10、-06-16、-06-17。編號依 -06-05 §0 慣例不自宣，落庫以 ls records/ 取 max+1。 -->

# 作者裁決與執行指令：P0 路一＋進 β、γ／δ 準備

## 0. 效力與裁決清單

### 0.1 本檔簽發即生效之裁決（對應 -06-05 §10 阻塞點）

1. **P0 溯源**：採 -07-01「後續」之**路一**，附增修一（k-sweep 分診，§2.3）與增修二（探針定稿條款，§2.1）。
2. **P0 範圍**：維持 -06-05 §4 原定義——單 cell（seed 10 × 首 grid 點 w=1），不擴；全 30 config 對帳由 P1 streaming 即時完成。
3. **排程**：同意「P 先行」（對 -12 §9 第 4 項之排程偏差，本檔即為明示同意入檔）；backbone 訓練授權見 §5.4。
4. **D2 先驗登記時序**：以獨立 mini-record 於 C1 揭盲前先行登記（§1.3）。
5. **D6 空間口徑**：採 -06-05 §7 建議映射——雙空間＝完全復活、僅 DINOv2＝表徵條件弱復活，D 包照載。
6. **D7 一行裁量**：候選文案核定（§5.2 第 7 條），隨 D 包終簽生效。
7. D 起草納入三補強與兩確認（§5.2）；B 骨架補兩保留槽（§1.2）；C8 一頁版納兩補強（§4.2）。

### 0.2 仍保留之作者欄位（agent 不得代填，到點 STOP 呈報）

- D4 全部數字門檻（-06-05 §7 明文 agent 不得代填）
- D8 CaF-v2 去留（及若納入之增益門檻 X）
- E5 三選一（claude.md 凍結定義與 start_timestamp 慣例）
- E2 措辭核准（B 定稿後）
- **P0 STOP 呈報後之 P1 greenlight**（-06-05 §9 β 之單一決策點，本檔不預授 P1）
- D 包終簽（δ）

## 1. 即刻批（純文件、無 GPU）

1.1 **落庫**：本檔（隨附之建議書作 backing，若有）依 ls max+1 連號入 records/。
1.2 **B 骨架補槽**：於 -06-17 附錄補列兩個保留槽——「P 對帳結果（β 後填）」「ec1f746 結論（F2，引 dossier 乙-2：3 檔 +295/−19 純實作、凍結後行使實作自由度之流程債）」，標注「依⟨本檔⟩補」。骨架屬 γ 定稿前之活文件，此為結構補全、非內容回改。
1.3 **D2 mini-record**：登記分支一先驗（整體約一成，照 -06-05 §7 D2 原文），明標（i）誠實聲明、非裁決輸入；（ii）登記於 C1 揭盲前。
1.4 **commit 工作樹既核修正**：-08-01 之 cifar_data.py（+3/−2）與 docs/results_analysis.md（+4/−2）。
1.5 **F1** 若未完成：driver start_timestamp patch 以 diff 呈報（照 -06-05 §8）。

## 2. β 批之一：探針定稿 → P0（本檔即為「進 β 跑 P0」之 greenlight）

2.1 **探針定稿**（run_p0_probe.py，定稿後轉入 git 追蹤並 commit）：
  a. 補 inception_crosscheck 之 `detector is not None` 守衛；
  b. 解除 `d["per_seed"][0]` 對存檔順序之隱含耦合；
  c. nearest_k 升顯式 CLI 參數；探針輸出 metadata 強制含 nearest_k、有效 k=min(k, n−1)、tau_fraction、完整 argv echo、start_timestamp、環境版本（torch／cuda／cudnn）；
  d. 內建 k-sweep 模式（讀落盤特徵離線重算，免重生成）；
  e. 決定性環境旗標（torch deterministic、cudnn benchmark off、版本登記），與 confirmatory 同機同環境執行。

2.2 **執行 P0**（單 cell）：重生成 → 落盤（uint8 影像＋DINOv2 per-sample 特徵）→ 對帳 scalar：precision、coverage（DINOv2 與 Inception 側同名值一併對帳；k=5 標「依文件推定、非儲存值」）、char_clean_fid、near_boundary_frac、label_noise_excess（judge 相關同 pass 做齊——路一之 k-free 控制臂需要）→ 計時與儲存實測。判準依 -06-05 §4 三態：逐位相等／相對 ≤1e-4 容忍內（記錄漂移原因）／超容忍。

2.3 **超容忍之預註冊分診**（固定第一步，先於任何環境修復）：
  a. 三個 k-free scalar（char_clean_fid、near_boundary_frac、label_noise_excess）是否容忍內？**否** → 生成級非決定性 → 進 §4 原環境修復循環（釘版本、deterministic algorithms、關 cudnn benchmark）復測；不可修 → 依 -06-05 §6 降級語義續行。
  b. **是** → k-sweep：k∈{1..15} 於同一落盤特徵重算 precision／coverage（DINOv2 與 Inception 兩側同步約束），與凍結 JSON 比對。唯一 k\* 匹配 → 以 k=k\* 復測全部 scalar，全過即 P0 PASS；k\* 登入 P1 對帳參數與後續文件，-14 之 k=5 文件推定以執行 record 訂正——**凍結 JSON 不動、數字不回改**。無任何 k 匹配 → 真 STOP（PRDC 路徑非決定性：距離計算、topk 平手、特徵抽取），回環境循環。
  c. 相容性邊界：不修改凍結 JSON；不以重生成新增量測充作 confirmatory（sweep 只重算已登記定義）；目標值凍結在先、判準事前寫死，屬稽核性參數找回、非結果選購。

2.4 **STOP 呈報**（呈報後暫停，待作者 P1 greenlight）欄位：計時實測、儲存實測、決定性三態逐 scalar、k 溯源結論（反證支持 k=5／找回 k\*／未決）、判決二／三與 C8 實例與 C 批之信任鏈狀態、P1 排程建議。

## 3. β 批之二：P1×C1 預備指令（P1 greenlight 後方生效）

照 -06-05 §4 執行，此處僅摘要與增列：seed-major（seed 10 全 10 configs → 11 → 12）；走 run_cifar_cfg_multiseed.py 之生成路徑與 gseed 碰撞公式**原樣**（cifar_cfg_sample.py 未證逐位等價前不得代用）；落盤 uint8 影像＋DINOv2 per-sample 特徵；即時對帳（以 P0 確立之 k）；C1 隨算 per-config FD-DINOv2；任一 config 超容忍即 STOP；seed-10 中途呈報**禁詮釋**；Inception 特徵與 judge per-sample 同 pass 或事後自落盤影像計；C2／C3／C5 一律取自落盤影像、**禁第二次重生成**。recall／density 為落盤特徵之免費副產品：本檔**不預授權計算**——若 D8 裁納入 v2，先落 C0 規則 record 再算並標 exploratory。

## 4. γ 批

4.1 C1 三 seed 裁決依凍結口徑（-06-11），對號入座二元活結果空間；反證版與弱版本筆墨等量（既寫雙分支敘事）。
4.2 C2–C5、C7、C8 計算與 record。C8 一頁版入 docs 時併兩補強：（i）空可行集之 fallback 行為與 coverage 平手處理明文化（引理三情形隱含「CaF 不選不可行點」，成文寫出）；（ii）增列單調 selector 推廣——任何對 (precision, coverage) 嚴格單調遞增之 selector 於嚴格支配情形必不選 oracle（一行證明：f 嚴格遞增則 f(c\*)>f(o)）——並供 D8 規格引用。
4.3 B1 落 §1.1 排序複核腳本並標明計算式。消歧待辦：核對者獨立重算 Spearman ρ(coverage, TSTR)=0.624 對 §1.1 之 ≈0.64（或為 Pearson vs Spearman 或修約）；ρ(−char_clean_fid, TSTR)=0.964 無歧義。
4.4 B 定稿：判決一補 C1 結果；附錄填 P 對帳與 ec1f746／F2 兩槽；判決二逐字引 -09 不重寫；C6 數字只進判決三；regret 主 3.69、並列 2.77。**STOP 呈報**。

## 5. δ 批（可與 β／γ 並行起草）

5.1 D0：逐字調出 -02 之 H1/H2/H3 原文與口徑附入，終審 §1.8 之 H2 讀取、確認 H3 身分。
5.2 D 起草照 -06-05 §7 D1–D12，並納入下列（起草即納，隨 D 包終簽定案）：
  1. **D9 增列（逐字）**：「confirmatory driver 輸出 metadata 強制含全部 analysis 參數——nearest_k、有效 k=min(k, n−1)、tau_fraction、batch、完整 argv echo、start_timestamp、環境雜湊（torch／cuda／cudnn 版本）。」另併列 CIFAR-100 base-model gate 之定法（凍程序不凍數字）。
  2. **D9 種子公式**：提案無碰撞方案（例：hash 派生），附全網格枚舉唯一性之程式驗證輸出，公式與驗證一併登記。
  3. **D3 增列介入臂**：預註冊一個 C3 型 coverage-matched pruning 於 CIFAR-100 confirmatory 已生成合成集上，定位為**分支三論文宣稱之必要證據、非分支路由輸入**（路由仍由三中二觀察量判）；剪枝條件數由 agent 提案、隨 D 包簽核定死。成本註記：無新取樣，僅剪枝＋重訓。
  4. **D3 表徵口徑**：每觀察量標注量測空間與跨表徵要求（依 -08 §4 型式：DINOv2 主、Inception robustness），agent 起草、隨 D 包簽核。
  5. **D8 規格**：必引 §4.2 之單調 selector 不可能性作 v2 提案篩選判準（v2 須引入第三訊號或放棄單調性）；judge 依賴代價明寫；價值判準形式＝matched-budget 下對 FID-min 之 regret 增益 ≥X pp（X 屬 0.2 作者欄）。
  6. **D5／D7 對決條款**：對決報告固定三臂同表——FID-min／CaF（或 v2）／Chamfer，matched-budget；缺 FID-min 臂之「勝 Chamfer」無證明力。
  7. **D7 一行裁量（本檔核定文案）**：「作者裁定：-12 §9 第 3 項之範圍例外延伸至 Chamfer 牆條件化規則之事前登記（分支 1/2 必跑、3/4 選配），僅此一項。」
  8. **D6** 照 §0.1 第 5 條載入。
  9. **D12 骨架**：分支三版以 §1.10「無單一便宜代理普遍可靠」為 so-what 骨；是否為分支四另立第三份骨架，由作者於終簽時一句話裁定。
5.3 **D commit 前置**：main 併回（-07-01 掛帳 8）＋ C7 落地 ＋ §0.2 作者欄位補齊 ＋ D 包終簽；commit 須早於任何 CIFAR-100 合成樣本（scout 含），git 可驗。
5.4 **backbone 訓練授權**：CIFAR-100 訓練不受 D gate 阻（-06-05 §7 明文），本檔授權開跑——單 GPU 情形於 P1 完成後接續；有獨立算力則即刻。取樣程式（scout 含）D commit 前一律不得執行。訓練以 train_cifar.py 沿 CIFAR-10 配方起跑、週期性 checkpoint 照舊；base-model gate 數字屬 D 範圍，訓練不因其未定而等待。

## 6. E 批與低優先掛帳

6.1 E2／E3／E4 於 B 定稿後依 -06-05 §8 執行；E2 措辭先呈作者核准（§0.2）後方套用；E4 僅檔頭 banner、intro 不重寫。
6.2 低優先（不佔關鍵路徑，GPU 空窗處理；全部提案呈報、不自行定案）：cifar_data.py 與 datasets/cifar.py 合併方向（僅 validate_metrics.py 依賴，提 diff 呈報）；scout verdict 對非單調曲線誤報之修復或檔頭註記；run_tstr 四重複實作與 VAR-mini 去留維持不動。

## 7. 格殺與 STOP（摘錄，全文依 -06-05 §2；衝突即拒絕並呈報）

不修改凍結 JSON；不因 P 對帳結果回改凍結數字；P 之重生成不新增量測充作 confirmatory；P1 不修 gseed 公式；seed-10 預覽禁詮釋；D 包與一切裁決文件不引任何 CIFAR-100 讀數；regret 主 3.69、並列 2.77；H-C2a 顯著亦不得寫因果措辭；［推定］未核實不作規則前提；D4 數字 agent 不得代填。

本檔下之 STOP 點：P0 呈報（§2.4，等 P1 greenlight）、γ 呈報（§4.4）、δ 作者欄位與 D 包終簽（§5.3）、E2 措辭（§6.1）。
