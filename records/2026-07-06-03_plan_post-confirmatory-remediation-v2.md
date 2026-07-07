<!-- 用途：confirmatory 揭盲後修正計畫 v2。修訂依據：coding agent 對 v1 之七項稽核與外部覆審（七判決、四項稽核者漏抓）。核心修訂：新增持久化 pass（工作包 P）取代不成立的「零重生成」前提；治理新規降為自我約束以解除內在矛盾；引用格式全修；H2 改標；D7 換載體；STOP-gated 階段化。v1 依 §0 定號後標 superseded 保留。本檔為執行計畫；工作包 D 為 records/2026-07-05-12 §9 第 3 項例外下之預註冊設計。 -->

<!-- SUPERSEDED（2026-07-06 α 定號步）：本檔（計畫 v2）由 records/2026-07-06-05_plan_remediation-final.md 取代，僅保留為決策軌跡；過時編號引用不回改，以 -05 §0 譜系表為準。 -->

# 修正計畫 v2：confirmatory 揭盲後的執行、登記與同步

## 0. 相對 v1 的修訂與定號

v1 經 coding agent 稽核（七項）與外部覆審（判決＋四項稽核者漏抓），逐項修訂：

1. **「零重生成」前提不成立**［已核：driver 對每 config `del gen, gen_dino, margins, preds`，JSON 僅存 scalar；僅 C6 真零重生成］→ 新增工作包 P（一次性決定性持久化 pass），C 批除 C6 外全部改掛 P 閘之後（含 C4——重訓需生成圖，v1 稽核清單亦漏此項）。
2. **C1 需新接線**［已核：fd_from_features 未被 multiseed driver 呼叫；現存 fd_dinov2 僅 w=1］→ P 之後 C1 回到便宜（接線＋真實參考 DINOv2 Fréchet 統計＋矩陣運算）。
3. **治理引用錯誤與內在矛盾**：-12 之 §9 為扁平清單，無 §9.3/9.4/9.5 標號；「裁決不因結果改」之正確出處為 -13 §6 與 -12 §1，非 §9 第 5 項；「永久」非文件用語。v1 一面自稱非治理文件、一面新立兩條全域治理（「升級版 §8」、D11 全域凍結定義）→ 本 v2 依 §1.3 全數改標為自我約束或作者裁決項。
4. **H2 錯標**［暫准，D0 逐字終審］：依 -02，H2 為選擇器假設；「內部最優」為明文不受協定約束之 exploratory 項 → §2.8、B、D0、E2 措辭全改。
5. **D 之篇幅與排程現實**：D 正文瘦身至裁決必需項；GPU 排程裁決入 §10。
6. **C4 無功效**（決定因素為 σ_gen≈3–5pp，非分類器噪聲；配對 SE≈1.9pp 對 0.8pp 效應）→ C4 改目的為變異分解、餵 D 功效規劃；A1 揭盲時間線全揭露，但其預註冊檢定地位成立（理由見 §4 A1）。
7. **撞號** → 定號規則：依實際產出時序，dossier 取 2026-07-06-02、v1 改名 -03 標 superseded、本檔取 -04；已 commit 者不改名，衝突時後產出者讓號並於本節勘誤。

覆審另補四項（v1 與稽核皆漏）：D7 超出牆例外授權（§8 D7）；C4 受 P 閘（上列 1）；confirmatory JSON 是否含 Inception per-config PRDC 待查（§4 A-pre）；P 之完整性規格（§5）。

**通用新規**（源自修訂 1 之錯誤型態——推定寫成事實，與 v1 定位文件被稽之幽靈引用同型）：本計畫一切關於 repo 狀態的陳述標注［已核］或［推定］；［推定］項執行前必先核實。

## 1. 性質、依據、合法性

1. 本檔是執行計畫。工作包 D 為 -12 §9 第 3 項明文例外（「牆的執行本身強制觸發的再登記，且以最小篇幅為之」）下之預註冊設計。
2. 已凍結項照舊生效：H-C2 程序與準則（-13，含 §6「程序與準則不再依任何後續結果調整」）、grid/seeds/steps/η（-11）、judge 與 near-boundary threshold（-08）、regret/top-k 定義（-02 §6）、confirmatory pass/fail 準則（-12 §1）。定位不因結果回改（-12 §9 第 5 項）。
3. **本計畫之驗收判準（新立、僅約束本計畫工作項、非協定修訂）**：任何工作項的規則或調整，若其成立理由必須引用 confirmatory 數字的方向，則其產物為假設，只能由 CIFAR-100 裁決。此判準自身通過 -12 原版 §8 之測試（其理由與數字方向無關，資料反向時同樣成立）——此為其得以自我約束存在的理由，而非其成為協定的授權。任何協定層級的採納，僅得經 D（限 D 包範圍，見 D11）或作者對開發慣例之明示裁決（E5）。

## 2. 事實基準（v1 §1 修訂版；全量見 dossier；除註明外皆［已核］）

第 1–7、9–10 項同 v1 原文，其中第 6 項補充：per-seed TSTR 之 gen-seed 變異在低段極大——w1 SD≈3.8pp、w1.5 SD≈4.6pp；同 gen-seed 配對差（w1.5−w1）＝+0.80pp、配對 SD≈3.3pp、SE≈1.9pp［已核，由 per-seed 值直接計算］→ 上升肢在 3 gen seeds 下不可判定，與分類器重訓次數無關。此為 C4 改向與 D 功效規劃之依據。

第 8 項（改標）：治理洞為**一**——H2（選擇器假設）無凍結數字門檻，故 CIFAR-10 selector 結果僅得描述性報告。「內部最優／上升肢」依 -02 為明文不受協定約束之 exploratory 項——**不是治理洞，是未登記主張被 README 升為頭條**。B 判決一與 E2 均照此措辭：exploratory 觀察、附 seed 級不確定性、明文其從未受 confirmatory 保護。H3 身分未明，D0 一併逐字調出。

第 11 項（新增）：持久化狀態［已核］——driver 對每 config 刪除生成圖、DINOv2 特徵、per-sample margins/preds，JSON 僅 scalar；fd_from_features 從未被 multiseed driver 呼叫。DDIM η=0＋凍結 seeds 之決定性［推定——per-seed clean-fid 穩定性為間接支持；P2 閘為正式核實］。

## 3. 凍結邊界與格殺清單

v1 §2 全文有效（含九條格殺與 JSON 唯讀），另加三條：

- 不得以 P 之重生成「順便」新增未登記量測並充作 confirmatory 產出；P 的一切新量測輸出僅供 exploratory。
- 不得因 P2 閘結果回改任何凍結 JSON 數字；閘只驗證重現，不修訂記錄。
- ［推定］標注項在核實前不得作為任何規則或裁決之前提。

## 4. 工作包 A：H-C2 裁決（凍結義務）

- **A-pre（新）**：查 confirmatory JSON 是否含 Inception 表徵之 per-config coverage/precision（-08 要求 confirmatory 同時輸出兩表徵）。有 → A2 直跑；無 → A2 掛 P 之後，且此為對 -08 之執行偏差，記入 B 附錄。
- **A0**：小 record 凍結 permutation N=100,000、RNG seed=0（作者簽核）；附 H-C2a 顯著／不顯著兩則分支敘事草稿；明文揭盲時間線——均值表已於裁決前揭盲、分支敘事寫於揭盲後、殘餘自由度僅 permutation N 與 seed，且其對 p 之影響實質惰性（N=10⁵ 下 p≈.05 之 Monte-Carlo SE≈.0007）。
- **A1**：跑 run_c2_partial（DINOv2 主裁決），程序與準則依 -13 一字不動。證據地位標注：**檢定之預註冊地位成立**（程序於資料前凍結；殘餘自由度惰性且於運行前封存），受損者為詮釋層盲性；報告不得標「非 confirmatory」，但必須全文揭露時間線。
- **A2**：Inception robustness（-08 §4）。同批 exploratory：於 Inception coverage 上重放 selector，結果歸 C 批 record。
- **A3**：同 v1——效果量與 bootstrap CI 照 -13 報並注記 3-seed 重抽僅 10 種組合之退化，補 per-seed 範圍與 config-level jackknife（標 supplementary、不取代裁決）；H-C2b 準則不對稱聲明；全網格 ρ 抹平雙段之 caveat；因果措辭禁令（§3）。

## 5. 工作包 P：一次性持久化 pass（新）

目的：以決定性重生成把 30 configs 的 per-sample 資產落盤，使 C 批成為便宜讀取；兼作 D11 型 dry-run 之實體基底；若 A-pre 發現 Inception 缺口，於此補齊（僅 exploratory 地位）。

- **P0 計時探針**：單 config 全流程（取樣＋雙表徵特徵＋judge 推論＋落盤）計時，呈報 30-config 總估與儲存估（預估：影像 uint8 ≈2.8GB、DINOv2 特徵 ≈0.9GB、Inception 特徵與 judge 輸出另計［推定］）。
- **P1 重生成與落盤**：對每 (config, seed) 依凍結 seeds 決定性重生成。先驗證 per-(config, seed) 種子獨立（單 config 可重生成而不需重放全序列）。落盤：生成圖（uint8）、DINOv2 與 Inception per-sample 特徵、judge per-sample preds／margins／per-class、真實參考集之雙表徵 per-sample 特徵。環境釘死：torch deterministic algorithms、記錄 device／driver／torch 版本指紋。
- **P2 正確性閘**：對每 (config, seed) 以持久化資產重算 precision、coverage、char_clean_fid、near-boundary frac、ln_excess，逐項對凍結 JSON——目標逐位相等；容忍上限相對 1e-4，超出零容忍、達容忍須記錄環境漂移原因。**任一項超容忍＝STOP 呈報**：意味決定性假設破產，屆時 C 批全部改標「新資料 exploratory」，P 資產不得宣稱等同 confirmatory 樣本。
- **P3**：record 落檔——閘表全量、環境指紋、儲存清單、種子獨立性驗證結果。

## 6. 工作包 B：confirmatory record（三判決分立）

v1 §4 全文有效，修訂三處：

- 判決一中「內部最優」依 §2.8 改為未登記 exploratory 觀察之措辭，含配對差 +0.80±3.3pp（SE 1.9pp）之不確定性，明文其從未受 confirmatory 保護。
- 附錄增：A-pre 之 Inception 偏差（若有）、P2 閘結果指標、揭盲時間線。
- 引用格式全修：-12 §9 以「第 N 項」引用；裁決凍結之依據引 -13 §6 與 -12 §1。

B2 之 dossier 落檔依 §0 定號規則先行，dossier 原文不回改，其漏判之修正僅入 B1。

## 7. 工作包 C：持久化資產上的 exploratory 批

C0 通則同 v1（規則先於計算、全標 exploratory、獨立 record、規則不因結果回改）。**除 C6（僅讀 JSON，隨時可做）外，全部掛 P2 通過之後——含 C4**。

- **C1**：先凍 which-FID 規則（同 v1：雙空間皆分離＝強命題；僅一＝表徵依賴弱版本；皆不分離＝CIFAR-10 尺度反證確立；不得事後挑支持者）。規則落檔後：接線 fd_from_features、以 P 資產計算真實參考 DINOv2 Fréchet 統計與 per-config FD-DINOv2。**規則段須在 D commit 前完成**（D 引用之）。
- **C2 boundary-targeted pruning**（-07 §4(a) 字面實例）、**C3 coverage-matched pruning**（橋接論證入 record）、**C5 near-boundary 純度過濾**、**C7 small-probe FID 排序穩定性**、**C8 Pareto 失明引理成文**：同 v1，資料來源改為 P 資產。
- **C4（改目的）**：交付**變異分解**，不再宣稱裁決上升肢。設計：w1–w2.5 × 3 gen-seeds，每 cell 固定 +2 重訓（合計 24 次，N 事前定死、禁加跑）；pooled σ_cls（12 cells × 2 df）；與既有 per-seed 值分解出 σ_gen。**預期結論事前寫死**：上升肢維持 unresolved（σ_gen 支配，重訓無法改變）。交付物：σ_cls、σ_gen 估計 → D 之 CIFAR-100 功效規劃——若 CIFAR-100 之 σ_gen 同量級，3 seeds 定不住峰位 ±1 格，故 CIFAR-100 之登記主張須設計為對峰位噪聲穩健（高原＋懸崖措辭，不押點峰位）。
- **C6**：自 JSON 抽 w1、w2 之 per-seed char_clean_fid，補實 FID-min per-seed 對決。階段 α 即可執行。

## 8. 工作包 D：CIFAR-100 預註冊包（-12 §9 第 3 項例外，單檔、最小篇幅）

時限 hard gate 同 v1：D commit 早於任何 CIFAR-100 合成樣本之產生（scout 含），git 可驗；backbone 訓練不受阻，但取樣程式於 D commit 前不得執行。D 正文以裁決必需項為限；論證與背景一律引 B／C records，不重述。

- **D0**：逐字調出 -02 之 H1／H2／H3 原文與各自口徑，全文附入 D；終審「H2＝選擇器、內部最優＝不受協定約束之 exploratory」之讀取；確認 H3 身分。
- **D1–D6、D9、D10**：同 v1——四分支決策樹；分支一方向性預測與先驗登記（整體先驗約一成，明標誠實聲明、非裁決輸入）；分支三操作化（三觀察量、三中二判複製）；H2 數字門檻（作者定數，agent 不得代填；以 CIFAR-10 觀測為錨屬合法之跨資料集先驗使用）；強制 baseline 集（matched-probe FID-min、固定 w 慣例值、隨機可行點）；復活條件（分離出現 且 CaF 勝 FID-min，缺一即依樹落枝）；量測程序凍結（凍程序不凍數字：CIFAR-100 judge 品質 gate 定法、near-boundary threshold 重校準程序、特徵空間依 -08）；時序＝D commit → judge 與校準 → scout（僅定網格，讀數不得回饋任何判準）→ 網格凍結 amendment → confirmatory。
- **D7（重寫，載體修正）**：Chamfer 牆之條件化（分支一／二 → 對決升回必跑之牆；分支三／四 → 降為選配）**超出本例外授權**——其非 CIFAR-100 牆執行所強制觸發；且 -12 §9 第 3 項之凍結持續至兩牆皆有資料，分支三落枝時將現自指死鎖（對決已無科學意義、解除卻需被凍結之定位文件）。處置：條件規則照樣**現在（事前）**寫好，載體改為**明示入檔的作者裁量**——作者對 §9 第 3 項範圍作一行例外裁定並簽核，不假裝既有例外涵蓋。此裁定作成於 CIFAR-100 資料之前，故仍屬事前。CPU 空窗之 Chamfer 適配照 -12 §9 第 4 項原排程進行，不受本項影響。
- **D8**：CaF-v2 去留（作者決定）。若納入：規格（候選訊號含 near-boundary 供給，引 C5）＋門檻同批登記；明寫 judge 依賴使「免任務標籤」定位破功之代價；框架為 discovery／validation split——CIFAR-10＝discovery（永遠 exploratory）、CIFAR-100＝validation。若不納入，記錄該決定。
- **D11（縮限）**：凍結定義升級（prose＋實作程式＋已揭盲資料 dry-run＋輸出雜湊；P 資產為 dry-run 基底）**僅適用 D 包自身**。全案慣例化僅得經 E5 作者裁決。
- **D12**：兩份一頁論文骨架（分支一版、分支三版）入 docs/（與 paper_intro_draft 同類之論文草稿，非治理文件，不佔登記篇幅），D commit 前完成；分支三骨架直接對準其 so-what 弱點，使 C2／C3／C5 之設計對其瞄準。

## 9. 工作包 E／F

- **E1**（即刻，純事實）、**E3**（B 之後，鏡像三判決＋雙段機制圖）、**E4**（intro 不重寫、檔頭 banner）、**F1**（driver start_timestamp patch，diff 呈報）、**F2**（ec1f746 結論入 B 附錄）：同 v1。
- **E2**（B 之後，作者核措辭）：狀態注記依 §2.8 改寫——頭條句後注明「FID-opt 與 TSTR-opt 於 CIFAR-10 confirmatory 重合於 w1.5；『內部最優』從未為登記假設，以 exploratory 觀察報告；最終定位待 CIFAR-100 分支裁決，見 records/⟨B⟩⟨D⟩」；「必然次優」全稱句撤下。定位 v3 不在本輪產生。
- **E5**（作者裁決，含載體問題）：claude.md 擬加兩行——凍結定義（D11 內容之慣例化）、driver start_timestamp 慣例。明文承認凍結定義之慣例化可能本身即治理變更；作者三選一並入檔：現在以開發慣例入 claude.md／等 §9 第 3 項凍結自然到期／併入 D7 之同一行例外裁定。

## 10. STOP-gated 階段與排程（格式由 agent 對齊 claude.md 慣例）

- **階段 α（即刻；僅讀 JSON／純文件）**：定號修正（§0.7）→ E1 ∥ F1 → A-pre → A0（作者簽核）→ A1 →（A-pre 通過則 A2）→ C6 → C 批全部規則段落檔（含 C1 which-FID）→ B 草稿骨架。**STOP 呈報**：A 裁決數字、A-pre 結果、P0 排程需求。
- **階段 β（GPU）**：P0 → 呈報總估 → **作者排程裁決**（見下）→ P1 → P2 閘 → P3。**閘任一失敗即 STOP**。
- **階段 γ（P 資產上）**：（A2 若被 gate 則此時跑）→ C1–C5、C7、C8 → C 批 record → B 定稿 → E2–E4。**STOP 呈報**。
- **階段 δ（設計與登記；可與 β／γ 並行起草）**：D 起草 → D0 核驗 → 作者簽核 D2 先驗／D4 門檻／D7 裁量／D8 去留／E5 載體 → D commit → CIFAR-100 訓練與後續依 D10。

排程裁決（作者，β 前）：-12 §9 第 4 項寫「confirmatory 完成即啟 CIFAR-100 訓練」；P 需先佔 GPU（P0 後有實測估計）。建議序：P 先行（兼為 D11 dry-run 基底、可補 Inception 缺口）→ CIFAR-100 訓練 → C 批於訓練空窗以 CPU／輕 GPU 進行。此為對 §9 第 4 項排程句之偏差——屬排程、非裁決——入檔記錄即可，但須作者明示同意。

## 11. 授權邊界

作者阻塞點（依先後）：α 之 A0 簽核；β 前之排程裁決；δ 之 D2 先驗聲明、D4 門檻數字、**D7 範圍裁量（一行例外之載體）**、D8 去留、E5 載體三選一；E2 措辭。其餘 agent 自主，逐階段 STOP 呈報，不批次積壓。

## 12. 自我測試與護欄

- §1.3 判準逐包過測：A＝執行既凍程序，pass；P＝重現凍結資料之儀器作業、新量測僅 exploratory，pass；B＝事實報告，pass；C＝規則先行、C4 預期結論事前寫死，pass；D＝判準先於一切 CIFAR-100 資料、D7 以事前作者裁量為載體，pass；E／F＝事實注記與流程，pass。
- repo 狀態標注規（§0）全程生效；本檔自身之［推定］項：DDIM 決定性（P2 終審）、H2 讀取（D0 終審）、P 儲存估算（P0 實測）。
- 資源配置與裁決分離、希望測試：v1 §11 兩護欄原文有效——分支機率僅供準備權重，不入任何裁決條文；CIFAR-100 落地當日若發現自己在希望特定數字，重讀本檔與 D1。
