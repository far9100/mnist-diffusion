<!-- 用途：修正計畫 v2 之增補（errata）——裁決 coding agent 對 v2 的五項殘留修正，其中 fix 1/4 經反修合成為 streaming P1×C1，並新增 C4 混池規則。本檔僅局部增補 v2 之 §0.7、§2 第 11 項、§4 A-pre、§5、§7 C4、§10；衝突處以本檔為準，v2 其餘條文不變。編號 -05 於 §1 兩分支下皆成立。 -->

<!-- SUPERSEDED（2026-07-06 α 定號步）：本檔（v2 errata）由 records/2026-07-06-05_plan_remediation-final.md 取代，僅保留為決策軌跡；過時編號引用不回改，以 -05 §0 譜系表為準。 -->

# 修正計畫 v2 增補：五項殘留之裁決

## 1. 定號原子步（fix 3，α 之第一動，先於一切 commit 與 B2）

一次喬定三檔，二擇一：(a) v1／v2 尚未 commit → 依內容時序改名：dossier→2026-07-06-02、v1→-03（標 superseded）、v2→-04；(b) 已 commit → 依「已 commit 者不改名」：v1 留 -02、v2 留 -03、dossier 取 -04 並於 header 註記其內容時序先於 -02/-03。兩分支下本檔皆取 -05。執行後於 v2 §0.7 勘誤註記實際走向。

## 2. A-pre 收斂（fix 2）

［已核:present］confirmatory JSON 之 aggregate.per_config 與 per_seed[].configs 含 coverage_inception、precision_inception。故：A2（-08 Inception robustness）入階段 α 直跑，不掛 P；B 附錄之「-08 執行偏差」注記取消。補充：A2 之 exploratory selector replay 若需 real-vs-real Inception τ 參考而 JSON 未存，於 α 內由真實資料現算（免重生成）。確認：which-FID 之 Inception 側＝char_clean_fid（已在 JSON），C1 僅欠 DINOv2 側，v2 §7 C1 條文不變。

## 3. P 重構為 streaming P1×C1（fix 1＋fix 4 合成，取代 v2 §5 之 P0/P1/P2 分段與 §7 C1 之時序）

- **P0 探針（＝原 P0 與 agent 所提 P0.5 之合併）**：迴圈之首個 (config, seed)（seed 10 之第一個 grid 點）全流程——重生成→落盤→逐項 scalar 對帳（precision、coverage、char_clean_fid；judge 相關 scalar 同 pass 或事後補）→計時與儲存實測。**雙交付：計時估算＋決定性判定。STOP 呈報**，作者 greenlight 後續跑。
- **決定性三態**：逐位相等（理想）／容忍內（各 scalar 相對誤差 ≤1e-4，記錄環境漂移原因）／超容忍。超容忍 → STOP：先修環境（釘 torch／driver 版本、deterministic algorithms、關 cudnn benchmark），復測；不可修 → 依 §5 降級語義續行，不作廢。
- **P1×C1 streaming**：seed-major（seed 10 全 10 configs → seed 11 → seed 12）。每 config：重生成 → 落盤影像（uint8）＋DINOv2 per-sample 特徵 → 即時對帳 → C1 隨算該 config 之 FD-DINOv2。**任一 config 超容忍即 STOP**（故障暴露於第 N 個 config，非燒完 30 個之後）。seed 10 完成即呈 1-seed 初步 FD 曲線——**僅預覽**，which-FID 裁決仍依已凍規則以三 seed 全量為之。
- **實作要求［推定待核］**：重生成必須走 run_cifar_cfg_multiseed.py 之生成路徑與種子邏輯，或先證明 cifar_cfg_sample.py 之種子流與其逐位等價；不得未經證明直接沿用後者。
- Inception per-sample 特徵與 judge per-sample 輸出：同 pass 附帶或事後自落盤影像計算；**C2／C3／C5 一律自落盤影像取用，禁止第二次重生成**。
- **儲存修正（fix 5）**：影像 ≈0.92GB（30 cells × 10,000 張 × 3,072B）；v2 之 2.8GB 為三倍算術誤，撤。DINOv2 特徵 ≈0.92GB 不變。Inception 特徵若持久化另計。P0 實測為準。

## 4. v2 §2 第 11 項支持理由撤銷（fix 1b）

「per-seed clean-fid 穩定性為間接支持」撤——該三值為跨 seed 各單次量測，對同 seed 重現性零證據力。決定性之［推定］唯一核實途徑＝P0 對帳。溯源註記：該理由源自 dossier 之括號註記、經 v2 抄入；標注制度使其可被指名攻擊，本次修正即制度之運作，型態記錄：**［推定］之支持理由本身亦屬可攻擊面**。

## 5. 降級語義精確化（fix 1c ＋ 新增 C4 混池規則）

C 批本即 exploratory；超容忍之真實損失＝**與凍結 TSTR／JSON 之配對**，分項如下：

- **C1**：FD（重生成樣本）對 TSTR（凍結樣本）之配對失效，改標「同組態新抽樣估計」並明文 cross-dataset caveat；對 thesis 之判讀力下降、不歸零。
- **C4（新規，任何情境皆適用）**：frozen TSTR 值混入 σ_cls pooling 之資格與決定性掛鉤——**逐位相等 → 混池**（frozen 值計第 3 replicate，3/cell、24 df）；**僅容忍內或超容忍 → 不混**（每 cell 僅用 2 次新重訓，12 df pooled），否則樣本集差異灌入 σ_cls。依「取嚴」先例，非逐位一律不混。
- **C2／C3／C5**：介入比較均為重生成樣本集之內部自我對照，配對損失幾乎無感，照走。

## 6. 階段與排程更新（fix 2／4 之後果，取代 v2 §10 對應處）

- **α**：定號原子步（§1）→ E1 ∥ F1 → A0（作者簽核）→ A1 → **A2（de-gated）** → C6 → C 批全部規則段落檔（含 C1 which-FID，須先於 β）→ B 草稿骨架。STOP 呈報。
- **β**：P0 探針 → STOP 呈報（計時、決定性、儲存）→ 作者排程裁決 ＋ greenlight（併為單一決策點，不新增獨立阻塞）→ P1×C1 streaming（含 seed-10 中途預覽呈報）。
- **γ**：C2–C5、C7、C8 → C 批 record → B 定稿 → E2–E4。
- **δ**：不變（D 起草可與 β／γ 並行；D commit 早於 CIFAR-100 首個合成樣本之 hard gate 不變）。
- 排程建議不變且更強：P 先行——thesis 關鍵數字（C1）如今於 pass 中途即開始產出，先跑 P 的機會成本進一步下降。對 -12 §9 第 4 項排程句之偏差照 v2 §10 入檔、作者明示同意。

## 7. 下一步

green-light agent 依 v2＋本增補產出可執行階段清單（其自請項），衝突處以本增補為準；清單完成即入 α。
