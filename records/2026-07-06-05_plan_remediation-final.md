<!-- 用途：定稿執行計畫——合併修正計畫 v2、其 errata、第三輪覆審四項裁決與 gseed 碰撞發現，為唯一執行依據；v1/v2/errata 標 superseded 保留為決策軌跡。本檔不自宣編號（該錯誤三度復發之修正）：落庫者以 ls records/ 取 max+1 填入檔名 XX 並於 §0 譜系表登實號。 -->

# 定稿執行計畫：confirmatory 揭盲後之執行、登記與同步

## 0. 定稿性質與文件譜系

本檔取代 v2 與其 errata 成為唯一執行文件；衝突處以本檔為準。譜系（依 agent 回報之 repo 現況，α 定號步核實）：-01 README 校對；-02 計畫 v1（superseded）；-03 計畫 v2（本檔落庫時標 superseded）；-04 errata（同標 superseded）；本檔續為 -05（現況 ls 已在此序）、dossier 落 -06（max+1，檔名 2026-07-06-06_proofread_confirmatory-materials-handoff.md，內容時序先於 -02/-03/-04）。superseded 檔內之過時編號引用不回改，以本表為準。譜系表在 commit 前一律推定，以 ls 落庫實況為唯一真值——此為編號自宣錯誤第四次出現之根因，收束於此。

依據不變：工作包 D 為 records/2026-07-05-12 §9 第 3 項例外（牆之執行強制觸發之再登記，最小篇幅）；已凍結項照舊——H-C2 程序與準則（-13，含 §6）、grid/seeds/steps/η（-11）、judge 與 threshold（-08）、regret/top-k 定義（-02 §6）、confirmatory pass/fail 準則（-12 §1）、定位不因結果回改（-12 §9 第 5 項）。

**本計畫之驗收判準（新立、僅約束本計畫工作項、非協定修訂）**：任何工作項之規則，若其成立理由必須引用 confirmatory 數字之方向，其產物為假設，僅得由 CIFAR-100 裁決。此判準通過 -12 原版 §8 之測試（理由與數字方向無關），此為其自我約束存在之理由、非其入協定之授權；協定層採納僅經 D（限 D 包）或 E5 作者裁決。

**repo 狀態標注規**：一切 repo 狀態陳述標 ［已核］／［推定］；［推定］項執行前必先核實，其支持理由本身亦屬可攻擊面。本規適用於檔名與編號（本檔不自宣編號即其執行）。

**計畫凍結**：本檔為計畫終版。後續發現一律入執行 record，不再改版計畫，除非 STOP 閘觸發結構性改變並經作者同意。

## 1. 事實基準（全量見 dossier；除註明外皆［已核］）

1. FID-opt 與 TSTR-opt 於 CIFAR-10 confirmatory 重合於 w1.5（clean-fid 8.82／TSTR 63.96）；README 頭條句在本資料上被反證。排序：ρ(−char_clean_fid, TSTR)≈0.96、ρ(coverage, TSTR)≈0.64［B1 前腳本複核］。
2. FID-min baseline per-seed regret ≈0.91pp（2.45/0.28/0.00）對 CaF 3.69pp（0.54/5.03/5.49）；w1/w2 per-seed FID 由 C6 補實。
3. Pareto 支配：w2.5（.873,.792）同時嚴格支配三 oracle——w2（.858,.777）、w1.5（.841,.751）、w1（.806,.645）→ 任何 τ 下 argmax coverage s.t. precision≥τ 不可能選中 oracle；tau_robustness.picks 為實證。結構性失明，非校準問題。
4. τ 混合結論：w1.5 三 seed 皆可行（過衝主體為 coverage 排序）；w1 於 seed 11/12 不可行（邊際 .0034/.0092）、seed 10 可行——可行性由 seed 級噪聲決定，τ=0.9×ref 壓在低段 precision 高原（knife-edge）；seed 11 之 5.03 regret 部分歸 τ。
5. 雙段機制：低中段 near-boundary .256→.046 先崩、coverage .645→.792 仍升（滯後代理）；w1 之 ln_excess +.044 為唯一正值（label-noise 機制於 w1 實證獲勝）；高段 coverage 與 TSTR 同崩、near-boundary 反升 .033→.059 且 precision 同降（疑離類污染）。全網格 ρ 將雙段抹平——照跑（凍結義務），科學重心已不在該 ρ。
6. 變異：per-seed TSTR 低段 SD——w1≈3.8pp、w1.5≈4.6pp；配對差（w1.5−w1）+0.80pp、配對 SD 3.26、SE 1.88pp → 上升肢於 3 gen seeds 下不可判定，與重訓次數無關。
7. MNIST「分離」證據為軼事級（一格步、FID 側單 seed argmax、bespoke FID、方向不轉移）。
8. 治理洞為一：H2（選擇器）無凍結數字門檻 → CIFAR-10 selector 僅描述性報告。「內部最優／上升肢」依 -02 為明文不受協定約束之 exploratory 項——非治理洞，是**未登記主張被 README 升為頭條**；B 判決一與 E2 照此措辭。H3 身分未明，D0 逐字調出。［H2 讀取暫准，D0 終審］
9. 時序：052492c（07-05 14:59:54 規格凍結）→ ec1f746（23:53:43 儀器）→ 推導起跑 ≈07-06 00:29（非登記值）→ mtime 08:45:11。ec1f746 合規但實作自由度於凍結後行使、僅 1-config 探針——流程債，D11 修。
10. 存活項：災難性非單調（w≥3 崩 11–30pp）；最優位置跨資料集移動；「無單一便宜代理普遍可靠」（MNIST 與 CIFAR-10 選擇器判決相反）為更強候選命題。
11. 持久化狀態：driver 逐 config 刪除影像／特徵／per-sample 輸出，JSON 僅 scalar；fd_from_features 未被 multiseed driver 呼叫。confirmatory JSON 含 coverage_inception、precision_inception（aggregate.per_config 與 per_seed[].configs）。DDIM 決定性之唯一核實途徑＝P0 對帳［推定］。
12. **gseed 碰撞**［已核—基於 agent 逐字引文 `gseed = seed*10_000_000 + int(w*1000)*10_000` 加本輪算術；α 內複核引文與作用域，引文不確則本節作廢］：對本網格 w 值公式退化為 (seed+w)×10⁷，30 cells 僅 14 份初始噪聲，沿 seed+w 反對角線共享（例 (10,w2)=(11,w1)；(10,w3)=(11,w2)=(12,w1)；獨享僅 (10,w1)(10,w1.5)(12,w2.5)(12,w8)）。完好：config 內三 seed 互異（per-config 均值合法）、C4 配對設計不受影響、對 pilot（和值 1–10）無重疊。受傷：跨 config 誤差非獨立 → H-C2 permutation 可交換性 caveat 入 A3/B。〔α 撤回〕原斷言「上緣 scout 與 confirmatory 噪聲重疊、fresh seeds 精度降為噪聲層級部分重用」經 α 核實證偽，此為撤回（明文留痕、非靜默刪除）：上緣 scout 生成呼叫傳 flat seed=0（run_cifar_cfg_upper_scout.py:71，全 w 同值、未套 gseed 公式；wide-grid scout run_cifar_cfg_scout.py:115 亦同），其 generator 種子恆為 0，值域與 confirmatory 之 (seed+w)×10⁷ ≥ 1.1×10⁸ 不相交，故無噪聲重疊；upper_scout 檔頭並自記「seed 與 confirmatory fresh seeds 分離、資料不入 confirmatory 統計」（:8、:46）。原「和值 8–20」係誤設上緣 scout 套用 gseed 公式（0+w，w∈{8,10,12,16,20}）所致，實際為 flat 0、無「和值」。dossier（-06 甲-5）原「fresh、無重疊」對兩 scout 皆成立。〔訂正留痕〕本撤回於 commit b3c1133 初次落庫時誤引 run_cifar_cfg_scout.py:115（wide-grid scout）為據；§1.12 所指「上緣 scout」實為 run_cifar_cfg_upper_scout.py，經複核兩檔皆 flat seed=0，結論不變、引用已改正。此更正方向對本研究有利（強化 fresh 宣稱），正因有利更明文留痕；移除對己有利之錯誤陳述不留痕，與塞入對己有利之陳述同病。scout 內部各 w 共用 seed 0 一事入 B 附錄，不進正文事實基。行動：P1 重生成**原樣沿用碰撞公式**（重現凍結資料、非修 bug）；CIFAR-100 種子公式改無碰撞方案＋程式驗證唯一性（D9）。待核：gseed 作用域是否亦及分類器訓練。

## 2. 凍結邊界與格殺清單

執行全程禁止，違者拒絕並呈報：以 clean-fid 反證為由改 FD-DINOv2 作主 FID；重劃分段或僅取高段裁決 H-C2；將本次 confirmatory 降稱 pilot（HARK-by-demotion，預期最大誘惑）；改 mean-curve regret 2.77 為主數字（per-seed 3.69 為主、2.77 並列）；事後補恰可通過之 regret 門檻；掃 τ fraction 尋會選 w1.5 之值；C4 加跑至顯著（N 事前定死）；於 D 包或任何裁決文件引用 CIFAR-100 任何讀數（scout 含）；H-C2a 顯著亦不得寫「coverage 驅動效用」等因果措辭（-07 §4）；修改凍結 JSON（唯讀，輸出寫新檔）；以 P 之重生成順便新增未登記量測充作 confirmatory；因 P 對帳結果回改凍結數字；［推定］未核實即作規則前提；**seed-10 預覽階段輸出任何詮釋性文字**；**P1 修改 gseed 公式**。

## 3. 工作包 A：H-C2 裁決（凍結義務）

- **A0**（作者簽核）：小 record 凍 permutation N=100,000、RNG seed=0；附 H-C2a 顯著／不顯著兩則分支敘事；明文揭盲時間線——均值已揭盲、敘事寫於揭盲後、殘餘自由度僅 N 與 seed 且實質惰性（N=10⁵ 下 p≈.05 之 MC-SE≈.0007）。
- **A1**：跑 run_c2_partial（DINOv2 主裁決），依 -13 一字不動。地位標注：預註冊檢定地位成立（程序資料前凍、殘餘自由度惰性且運行前封存），受損者為詮釋層盲性；不得標「非 confirmatory」，必須全揭時間線。
- **A2**（α 直跑，de-gated）：Inception robustness（-08 §4）；同批 exploratory 於 Inception coverage 重放 selector（結果歸 C 批）；real-vs-real Inception τ 參考若 JSON 未存，由真實資料現算（免重生成）。
- **A3**：效果量＋bootstrap CI 照 -13 報並注記 3-seed 重抽僅 10 組合之退化，補 per-seed 範圍與 config-level jackknife（supplementary）；H-C2b 不對稱聲明；雙段抹平 caveat；**gseed 碰撞之可交換性 caveat（§1.12）**；因果措辭禁令。

## 4. 工作包 P×C1：streaming 持久化與 thesis 關鍵數字

- **P0 探針**（首個 cell：seed 10 之第一 grid 點）：重生成→落盤→逐項 scalar 對帳（precision、coverage、char_clean_fid；judge 相關同 pass 或事後補）→計時與儲存實測。雙交付（計時＋決定性）。**STOP 呈報**，作者排程裁決＋greenlight（單一決策點）。
- **決定性三態**：逐位相等／容忍內（相對 ≤1e-4，記錄漂移原因）／超容忍。超容忍→STOP：先修環境（釘版本、deterministic algorithms、關 cudnn benchmark）復測；不可修→依 §6 降級語義續行，不作廢。
- **P1×C1 streaming**：seed-major（seed 10 全 10 configs→11→12）；每 config 重生成（**走 run_cifar_cfg_multiseed.py 之生成路徑與種子邏輯——含碰撞公式原樣；cifar_cfg_sample.py 須先證逐位等價方可代用**［推定待核］）→落盤影像 uint8＋DINOv2 per-sample 特徵→即時對帳→C1 隨算該 config FD-DINOv2。**任一 config 超容忍即 STOP**。seed 10 完成呈 1-seed 初步 FD 曲線——僅故障偵測與進度用，**禁詮釋**；which-FID 裁決以三 seed 全量依已凍規則。
- Inception 特徵與 judge per-sample 輸出同 pass 附帶或事後自落盤影像計算；C2/C3/C5 一律自落盤影像取用，**禁第二次重生成**。
- 儲存：影像 ≈0.92GB（30×10k×3072B）、DINOv2 ≈0.92GB；P0 實測為準。
- ［已核］每 cell gseed 可局部計算，單 config 重生成免重放全序列；［推定，P0 核實］同 gseed 跨執行像素同一。

## 5. 工作包 B：confirmatory record（三判決分立）

三判決互不裁決、禁交叉混寫：

- **判決一（thesis）**：FID/TSTR 重合（§1.1，Spearman 複核後入檔）；災難性非單調成立；「內部最優」為未登記 exploratory 觀察，附 +0.80±3.3pp（SE 1.9）不確定性，明文從未受 confirmatory 保護；「必然次優」全稱句撤下（E2 執行）。
- **判決二（H-C2）**：依 A 如實登載，附 A3 全部 caveat。
- **判決三（selector，描述性）**：開頭明文「協定未凍門檻，本節不作過／敗判定」；Pareto 失明（§1.3）、τ knife-edge（§1.4）、FID-min 對決（§1.2，C6 補實）、modal_fraction 1.0 重讀為低變異高偏差、可辯護措辭「可靠避崖器、糟糕平台優化器——FID-min 同樣避崖且成本結構相同」。regret 主 3.69、並列 2.77。
- **附錄**：時序鏈（推導起跑標推導值）、ec1f746 結論、P 對帳結果、**gseed 碰撞與 scout 噪聲重疊揭露（§1.12）**、MNIST 降級、揭盲時間線。引用格式：-12 §9 以「第 N 項」；裁決凍結引 -13 §6 與 -12 §1。
- **B2**：dossier 落檔（α 定號步已處理），原文不回改，漏判修正僅入 B1。

## 6. 工作包 C：持久化資產上之 exploratory 批

**C0**：每項規則先於計算落檔、全標 exploratory、獨立 record、規則不因結果回改。除 C6 外全掛 P。

- **C1**：規則段（α 內、先於 β、D 引用之）——(i) 分離口徑：**均值曲線 argmax 相異且 >1 格步、或三 seed 方向一致之相異**；口徑於 DINOv2（唯一未見之臂）揭盲前凍結，且 Inception 側於任何合理口徑下結論相同（不分離），故口徑選擇無法對已見資料 HARK——此論證寫入規則段。(ii) 據此**強命題分支（雙空間皆分離）已被 Inception 側先行關閉**；活結果空間為二元：DINOv2 分離→表徵依賴弱版本；不分離→CIFAR-10 尺度反證確立。(iii) 比照 A0：兩則分支敘事於計算前寫好存檔，**反證版之筆墨與弱版本等量**。計算：接線 fd_from_features、真實參考 DINOv2 Fréchet 統計、per-config FD 隨 P1 streaming 產出。
- **C2** boundary-targeted pruning（-07 §4(a) 字面實例）；**C3** coverage-matched pruning（橋接論證入 record）；**C5** near-boundary 純度過濾；**C7** small-probe FID 排序穩定性（餵 D5）；**C8** Pareto 失明引理成文（一頁入 docs）。
- **C4（變異分解）**：w1–w2.5×3 seeds，每 cell 固定 +2 重訓（N 死）；pooled σ_cls；**混池規則**——逐位相等→frozen TSTR 計第 3 replicate（3/cell、24df）；容忍內或超容忍→不混（2/cell、12df），取嚴。預期結論事前寫死：上升肢維持 unresolved。交付 σ_cls/σ_gen → D 功效規劃：若 CIFAR-100 σ_gen 同量級，3 seeds 定不住峰位 ±1 格，登記主張須為峰位噪聲穩健形式（高原＋懸崖，不押點峰）。決定性失敗時 C1 配對改標「同組態**跨抽樣（cross-draw）**估計」並明文 caveat（非 cross-dataset）。
- **C6**（α 即可）：抽 w1/w2 per-seed char_clean_fid，補實 FID-min 對決。

## 7. 工作包 D：CIFAR-100 預註冊包（-12 §9 第 3 項例外，單檔、最小篇幅）

Hard gate：D commit 早於任何 CIFAR-100 合成樣本（scout 含），git 可驗；backbone 訓練不受阻、取樣程式 D 前不得執行。正文限裁決必需項，論證引 B/C。

- **D0**：逐字調出 -02 之 H1/H2/H3 原文與口徑附入；終審 §1.8 之讀取；確認 H3。
- **D1** 四分支樹（1 分離且 CaF 勝 FID-min→邊界條件復活；2 分離但 CaF 敗→thesis 活 selector 死；3 不分離但機制複製→診斷論文；4 皆否→負結果短文）。**D2** 分支一預測與先驗（整體約一成）登記，明標誠實聲明非裁決輸入。**D3** 分支三操作化：三觀察量（低中段 near-boundary 於 coverage 升平區單調降；高段 coverage 與 TSTR 同崩；高段 near-boundary 脫鉤），三中二判複製。**D4** H2 數字門檻（作者定數，agent 不得代填；以 CIFAR-10 觀測為錨屬合法先驗）。**D5** 強制 baseline 集：matched-probe FID-min（引 C7）、固定 w 慣例值（登記出處）、隨機可行點。**D6** 復活條件＝分離出現 且 CaF 勝 FID-min；**新增作者決策：「分離」之空間口徑**——建議映射：雙空間＝完全復活、僅 DINOv2＝表徵條件弱復活，事前寫死。
- **D7**（載體修正）：Chamfer 牆條件化（分支 1/2→必跑；3/4→選配）超出本例外授權且 §9 第 3 項凍結至兩牆有資料（分支三之自指死鎖）——條件規則現在（事前）寫好，載體為**明示入檔之作者對 §9 第 3 項範圍一行裁量**，不假裝既有例外涵蓋。CPU 空窗 Chamfer 適配照原排程。
- **D8** CaF-v2 去留（作者）：納入則規格＋門檻同批登記、judge 依賴代價明寫、CIFAR-10=discovery（永遠 exploratory）/CIFAR-100=validation；不納入則記錄決定。
- **D9** 量測程序凍結（凍程序不凍數字）：CIFAR-100 judge 品質 gate 定法、near-boundary 重校準程序、特徵空間依 -08；**新增：種子公式改無碰撞方案（例：唯一性以程式對全網格枚舉驗證後入登記），修正 §1.12**。
- **D10** 時序：D commit→judge 與校準→scout（僅定網格，讀數不回饋判準）→網格凍結 amendment→confirmatory。**D11** 凍結定義（prose＋程式＋已揭盲資料 dry-run＋輸出雜湊；P 資產為基底）僅適用 D 包。**D12** 兩份一頁骨架入 docs/（非治理文件），分支三版直接對準 so-what 弱點。

## 8. 工作包 E／F

**E1**（即刻，純事實：confirmatory 已完成、H-C2 待跑）；**E2**（B 後、作者核措辭）：頭條句後注記「FID-opt 與 TSTR-opt 於 CIFAR-10 confirmatory 重合於 w1.5；『內部最優』從未為登記假設，以 exploratory 觀察報告；最終定位待 CIFAR-100 分支裁決，見 records/⟨B⟩⟨D⟩」，「必然次優」撤下，定位 v3 不在本輪產生；**E3**（B 後）results_analysis 鏡像三判決＋雙段機制圖；**E4** intro 不重寫、檔頭 banner；**E5**（作者三選一並入檔）：claude.md 之凍結定義與 start_timestamp 慣例——現在入／等凍結到期／併 D7 裁量；**F1** driver start_timestamp patch（diff 呈報）；**F2** ec1f746 入 B 附錄。

## 9. STOP-gated 階段

- **α（即刻；純文件＋讀 JSON）**：定號原子步（ls 取 max+1：落 dossier、落本檔、v2/errata 標 superseded、譜系表登實號；一個 commit）→ E1∥F1 → **複核 §1.12 之 gseed 引文與作用域** → A0（作者）→ A1 → A2 → C6 → C 批全部規則段（含 C1 口徑與雙分支敘事）→ B 骨架。**STOP 呈報**（A 數字、gseed 複核、P0 需求）。
- **β**：P0 → STOP 呈報（計時／決定性／儲存）→ 作者排程裁決＋greenlight → P1×C1 streaming（seed-10 中途呈報，禁詮釋）。
- **γ**：C1 裁決（三 seed 全量、依凍結口徑）→ C2–C5、C7、C8 → C 批 record → B 定稿 → E2–E4。**STOP 呈報**。
- **δ**（可與 β/γ 並行起草）：D 起草 → D0 → 作者簽核（D2 先驗／D4 門檻／D6 空間口徑／D7 裁量／D8／E5）→ D commit → CIFAR-100 依 D10。排程建議：P 先行（thesis 關鍵數字於 pass 中途產出）；對 -12 §9 第 4 項「完成即啟訓練」之偏差屬排程非裁決，入檔＋作者明示同意。

## 10. 授權邊界

作者阻塞點：α 之 A0；β 之排程＋greenlight（單點）；δ 之 D2、D4、**D6 空間口徑**、D7 一行裁量、D8、E5；E2 措辭。其餘 agent 自主，逐階段 STOP 呈報，不積壓。

## 11. 自我測試與護欄

- §0 判準逐包過測：A=既凍程序；P=重現凍結資料、新量測僅 exploratory；B=事實報告；C=規則先行、C1 口徑於未見臂前凍結且對已見臂無鑑別力、C4 預期結論事前寫死；D=判準先於一切 CIFAR-100 資料、D7 以事前作者裁量為載體；E/F=事實與流程。全 pass。
- 本檔自身之［推定］清單：gseed 引文與作用域（α 複核）、DDIM 決定性（P0）、H2 讀取（D0）、sample.py 種子流等價性（用前證）、P 儲存（P0 實測）。
- 資源配置與裁決分離（分支機率僅供準備權重，不入裁決條文）；希望測試（CIFAR-100 落地日自檢）；預覽禁詮釋；**計畫凍結**（§0）——自本檔起，發現入執行 record，計畫不再改版。
