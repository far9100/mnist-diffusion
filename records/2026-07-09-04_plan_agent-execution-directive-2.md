<!-- 用途：作者裁決與執行指令（二）（交 coding agent）。承 -08-02（首份指令）與 2026-07-09-03（B 定稿），處理 §4.4 STOP 後之全部待決：追認 B 先於 C 批之次序偏差、核定 E2/E4 措辭並放行 E2–E4、裁定 GPU 佇列（C7→C4→C2/C3/C5→backbone 訓練）、E5 採「現在入」、授權 D4/D8 決策單準備。本檔由作者簽發生效；D4 數字與 D8 去留仍為作者欄位。依 -06-05、-08-02、-09-03。編號依 §0 慣例不自宣，落庫以 ls records/ 取 max+1。 -->

# 作者裁決與執行指令（二）：E2–E4 放行、GPU 佇列、δ 收斂

## 0. 效力與裁決清單

### 0.1 本檔簽發即生效之裁決

1. **追認次序偏差**：B 定稿（-09-03）先於 C2/C3/C5/C7 完成，偏離 -06-05 §9 γ 與 -06-17「後續」之順序。追認理由：C0 設計（規則先行、全標 exploratory、規則不因結果回改）使該批結果結構上不可能變更三判決；C 批結果以獨立 record 補遺附掛，-09-03 原文不回改。本條即該偏差之留痕。
2. **E2 措辭核定**（全文見 §1.1）。範圍較 -06-05 §8 預稿增列 C1 句與 selector 句——理由：README 頭條段同時承載 thesis 與 CaF 兩主張，僅注記其一為選擇性誠實；屬「作者核措辭」權限內之擴充，此為留痕。
3. **E4 banner 文案核定**（全文見 §1.2）。
4. **E3 放行**（規格見 §1.3）。
5. **GPU 佇列**：C7 → C4 → C2/C3/C5 → CIFAR-100 backbone 訓練起跑（§2）。理由：C7 為 D commit 前置（餵 D5）；C4 交付 σ_cls/σ_gen 餵 D4；C1 反證確立後分支期望偏向三／四，C2/C3/C5 之介入式證據為該兩支論文之正文而非補強。整批約 1–1.5 GPU 日。如作者欲改訓練先行，改本條即可。
6. **E5 裁決：採「現在入」**——claude.md 增列凍結定義與 start_timestamp 慣例（僅慣例層、不觸協定層；P0 溯源事故為充分理由）。
7. **D4/D8 決策單準備任務授權**（§3）；欄位本身仍歸 0.2。

### 0.2 仍保留之作者欄位（agent 不得代填，到點 STOP 呈報）

- D4 全部數字門檻（決策單備齊後一次填）
- D8 CaF-v2 去留（及若納入之增益門檻 X）
- D 包終簽（δ）
- D10 各閘（judge 校準、網格凍結 amendment、confirmatory 起跑）

## 1. 文件批（即刻、無 GPU、可與 §2 並行）

1.1 **E2 執行**（README）：
  a. 頭條段之後插入注記（照載，僅 ⟨D⟩ 留待回填）：

  > **2026-07-09 confirmatory 注記（E2）**：FID-opt 與 TSTR-opt 於 CIFAR-10 confirmatory 重合於 w1.5，且 which-FID 交叉裁決（C1）於 Inception 與 DINOv2 兩表徵空間皆不分離——本段頭條主張在 CIFAR-10 尺度被本專案自家資料反證。「內部最優」從未為登記假設，以 exploratory 觀察報告（上升肢 +0.80pp、SE 1.9，3 gen seeds 下不可判定）；「必然次優」全稱句已撤下。selector 層為描述性結果：更便宜的 FID-min baseline per-seed regret 0.91pp 對 CaF 3.69pp（per-seed 2 勝 1 負），且 CaF 於本網格存在結構性 Pareto 失明（見 docs/ C8 一頁版）。P0/P1 對帳：全 30 configs 之量測 scalar 逐位重現、k=5 獲探針反證支持。最終定位待 CIFAR-100 預註冊分支裁決；三判決全文見 records/2026-07-09-03（工作包 D commit 後補其編號於此）。

  b. 刪除頭條段之「因此任何固定的 guidance 值跨資料集與任務必然次優」子句，使句子通順、不新增主張。
  c. 「目前進度」段純事實刷新：Phase 1-3 裁決完成（B 定稿 -09-03）；P0/P1 量測 scalar 全逐位、k=5 反證；C1 兩空間不分離。
  d. E2 範圍限頭條段與進度段；前言等其餘章節照 §8 不擴。⟨D⟩ 編號回填為預授權之一行純事實編輯（§3.4）。

1.2 **E4 執行**（paper_intro_draft.md）：檔頭註解之下加一段可見 banner，本文其餘一字不動：

  > **⚠️ 2026-07-09 狀態（E4，本文依 -12 §9 第 5 項不回改）**：主張 2 之頭條版本（效用最優偏離保真最優）已於兩表徵空間被 CIFAR-10 confirmatory 反證（FID/TSTR 重合於 w1.5、C1 不分離）；主張 3 全稱句撤下；主張 5 之 CaF 於 confirmatory 敗於 FID-min（per-seed regret 3.69 對 0.91，2 勝 1 負）且具結構性 Pareto 失明；主張 1 在 CIFAR-10 上因 FID-min 近最優而失去操作區辨力。三判決見 records/2026-07-09-03；CIFAR-100 分支裁決前本文僅供決策軌跡參考。

1.3 **E3 執行**（docs/results_analysis.md）：
  a. 以 -09-03 三判決之鏡像取代「先不列結論」佔位——判決二逐字忠實；判決三含開頭「不作過／敗判定」句、per-seed 對稱句、regret 主 3.69 並列 2.77；判決一含 C1 兩空間結論。
  b. 雙段機制圖：自凍結 JSON 離線繪製（TSTR／coverage／precision／near-boundary／label-noise 對 w，低中段與高段標注依 -06-05 §1.5）；全程觀察性措辭，**禁因果句**。
  c. P 對帳一段：全 30 configs 量測 scalar 逐位重現；k=5 獲 P0 反證支持；明文 TSTR 依協定含未種子化 shuffle、不在對帳集。
  d. 「待確認」清單刷新：移除已完成項；C2/C3/C5/C7 標 pending exploratory 補遺（非判決輸入）；CIFAR-100 為主線。

1.4 **稽核回聲**（下次呈報附上）：
  a. git log --oneline 近 15 筆——確認清理事件後 §1 批（D2 mini-record、F1、B 槽、-08-01 修正）records 俱在 HEAD。
  b. C1 關鍵數字回顯：FD-DINOv2 per-config 均值、argmin 位置、TSTR argmax、格步距離、per-seed 方向向量。
  c. 核對 -09-03 之 P 對帳措辭：若寫作「整個 confirmatory 逐位可重現」屬過寬（TSTR 不在對帳集），以一行補充 record 訂正為「量測 scalar 逐位重現」；若已精確則回報無需。

## 2. GPU 批（依 0.1 第 5 條佇列；全程自落盤資產、禁任何重生成）

2.1 **C7**（≲1 小時級）：small-probe FID 排序穩定性，依 α 既凍規則計算；產出直餵 D5 之 matched-probe 規格。
2.2 **C4**（約半天級）：w1–w2.5 × 3 seeds、每 cell 固定 +2 重訓、N 死。P1 全逐位 → 依凍結混池規則 frozen TSTR 計第 3 replicate（3/cell、24df）。預期結論已事前寫死（上升肢維持 unresolved），record 不得因結果回改。交付 σ_cls/σ_gen 餵 D4 決策單。
2.3 **C2/C3/C5**（約一日級，合批）：依各自既凍規則段執行；剪枝／過濾一律自落盤影像；隨機對照照規則；全標 exploratory。任一規則段有未定參數 → STOP 呈報，不代填。
2.4 **C 批收尾**：C 批 record 落檔＋一行「B 補遺」record 連結之（-09-03 不回改）；results_analysis 補遺段（標 exploratory、禁因果句）。record 撰寫與 2.5 並行。
2.5 **backbone 訓練起跑**：2.3 之 GPU 工作完成即刻，train_cifar.py 沿 CIFAR-10 配方起訓 CIFAR-100、週期性 checkpoint 照舊。取樣程式（scout 含）D commit 前一律不得執行。base-model gate 數字屬 D 範圍，訓練不因其未定而等待。
2.6 低優先掛帳照 -08-02 §6.2，訓練空窗處理、提案呈報。

## 3. δ 收斂

3.1 **D4 決策單**（C7/C4 落地後備）：σ_cls/σ_gen；更新 MDE 表；H2 候選數字門檻各附成立理由並逐條通過 §0 驗收判準（理由與 confirmatory 數字方向無關；以 CIFAR-10 觀測為錨屬合法先驗）；D5 matched-probe 規格數字（引 C7）。
3.2 **D8 決策單**：單調 selector 不可能性回顧；第三訊號候選——near_bnd（judge 依賴代價明列）、recall/density（C0 規則 record 草稿隨單附上，作者點頭即自落盤特徵離線計算、標 exploratory、零 GPU）；價值判準 ≥X pp 對 FID-min 之 X 候選；去／留兩案後續各自寫明（留 → 規格＋門檻同批登記；去 → 記錄決定，對決三臂仍為 FID-min／CaF／Chamfer）。
3.3 **D 起草收斂**：納 -08-02 §5.2 全部條款；D12 兩骨架成稿（分支三版以 §1.10「無單一便宜代理普遍可靠」為 so-what 骨——C1 反證後其現實性升高；分支四是否另立第三份，隨終簽一句話裁定）。
3.4 **簽核與 commit 序**：作者一次過簽（D4／D8／終簽）→ main 併回（-07-01 掛帳 8）→ D commit（早於任何 CIFAR-100 合成樣本，git 可驗）→ E2 注記之 ⟨D⟩ 編號回填（1.1 預授權）→ D10 時序啟動（judge 與校準 → scout 僅定網格、讀數不回饋判準 → 網格凍結 amendment → confirmatory）。

## 4. 格殺與 STOP（摘錄，全文依 -06-05 §2；衝突即拒絕並呈報）

不修改凍結 JSON；不因任何對帳或補遺回改凍結數字與 -09-03；C 批禁二次重生成；D 包與一切裁決文件不引任何 CIFAR-100 讀數；D commit 前禁 CIFAR-100 取樣（scout 含）；E3 與機制相關文字禁因果措辭（H-C2a 顯著亦然）；regret 主 3.69、並列 2.77；D4 數字 agent 不得代填；［推定］未核實不作規則前提。

本檔下之 STOP 點：§1.4 稽核回聲隨下次呈報；§2.3 規則段參數缺漏；C 批完成呈報（含 §2.4 收尾）；§3.1/3.2 決策單呈報（等作者填 D4/D8）；D 包終簽；D10 各閘。
