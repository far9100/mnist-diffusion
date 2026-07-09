<!-- 用途：D8 決策單（δ）——CaF-v2 去留。單調 selector 不可能性回顧（C8）、第三訊號候選（near_bnd/recall/density）、價值判準 X 對 FID-min、去/留兩案、recall/density 之 C0 規則草稿（作者點頭才離線計算）。D8 去留與 X 為作者欄位（agent 不得代填，-09-04 §0.2）。STOP 呈報。依 -09-04 §3.2、-06-05 §7 D8。 -->

# D8 決策單：CaF-v2 去留（作者裁）

## Goal

依執行指令（二）`records/2026-07-09-04` §3.2 備 D8 決策單。**D8 去留與增益門檻 X 為作者欄位**
（`records/2026-07-09-04` §0.2）；本檔備結構、候選與兩案後果供作者裁。STOP 呈報。

## Result

### 1. 單調 selector 不可能性（C8 補強二，`docs/c8_pareto_blindness.md`）

任何「對 (precision, coverage) 嚴格單調遞增」之 selector，在某組態嚴格支配 oracle 時必選不到 oracle
（一行證明：f 嚴格遞增 ⇒ f(c\*)>f(o)）。CIFAR-10 上 w2.5 嚴格支配三 oracle，故 CaF（及任何此類單調
selector）結構性失明。**推論**：CaF-v2 要脫離此盲點，**必須引入第三訊號**（非 (prec,cov) 之單調組合）或
**放棄單調性**。此為 v2 提案之篩選判準（任何仍純 (prec,cov) 單調之 v2 提案不予採納）。

### 2. 第三訊號候選

- **near-boundary 供給量（near_bnd）**：以 judge margin 量之 near-boundary 樣本佔比作第三軸。**代價明列**：
  依賴一個用真實資料訓練的 judge，使 CaF 的「免任務分類器／免任務標籤」定位破功——這是 CaF 相對 Chamfer
  的核心操作點賣點之一，納入 judge 依賴須明文承認此賣點折損。且 C2/C3/C5（`records/2026-07-09-08`）之介入
  未對 near-boundary 機制提供強支持，near_bnd 作訊號之經驗基礎在 CIFAR-10 偏弱。
- **recall／density（PRDC 之另二量）**：為 P1 落盤 DINOv2 特徵之免費副產品（compute_prdc 已回傳，僅依
  `records/2026-07-09-04` §3 未授權而未記錄）。作者點頭即離線計算、零 GPU、標 exploratory（C0 規則草稿見 §4）。
  不引入 judge 依賴，保住免任務標籤定位。惟 recall/density 與 coverage/precision 高度相關，是否構成「非單調
  組合之第三軸」須驗（可能仍落在被支配面上）。

### 3. 價值判準（X 待作者填）

CaF-v2 之保留條件＝matched-budget 下對 FID-min 之 regret 增益 ≥ **X** pp（`records/2026-07-06-05` §7 D8：
X 屬 0.2 作者欄）。錨定（方向無關）：CIFAR-10 上 FID-min（0.91）已勝 CaF（3.69），v2 須實質勝過 FID-min
才有存在理由；X 為該勝幅門檻，**建議與 D4 候選 B 之 X_B 同一數**（一次定死）。**agent 不代填 X。**

### 4. 去／留兩案（後果各寫明）

- **留（納入 v2）**：v2 規格（第三訊號選定：near_bnd 或 recall/density）＋門檻 X 同批登記於 D 包；judge
  依賴代價（若選 near_bnd）明寫；框架為 discovery／validation split——CIFAR-10＝discovery（永遠 exploratory）、
  CIFAR-100＝validation。D5/D7 對決三臂＝FID-min／CaF-v2／Chamfer。
- **去（不納入）**：記錄該決定；對決三臂維持 FID-min／CaF（原版）／Chamfer（`records/2026-07-06-05` §7 D8）；
  CaF 於 CIFAR-100 仍作描述性 selector，Pareto 失明明文為其已知限制。

### 5. recall／density 之 C0 規則草稿（作者點頭才計算；零 GPU、離線、標 exploratory）

> **C0 規則（草稿，規則先於計算）**：若 D8 裁納入 recall/density 為 v2 候選訊號，則於 P1 落盤 DINOv2 特徵
> （`results/p1_assets`）離線重算 per-config recall、density（compute_prdc_per_class，k=5，同 confirmatory
> 口徑），全標 exploratory、不入任何裁決、不因結果回改；並先驗登記「recall/density 是否提供 (prec,cov) 以外
> 之非支配訊號」之判定（若 v2(含 recall/density) 於 CIFAR-10 仍選 w2.5＝仍失明 → 該訊號無效）。此草稿於
> 作者點頭後落獨立 C0 record 再計算。

## Follow-up

- **STOP：等作者裁 D8**——去／留；若留則第三訊號選定（near_bnd／recall/density）與 X 數字；recall/density
  之 C0 規則是否啟動。agent 不代填（`records/2026-07-09-04` §0.2）。
- D8 裁定後併入 D 包，與 D4、D 終簽一次過簽（§3.4）。不觸凍結 JSON、無數字回改。
