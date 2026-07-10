<!-- 用途：記錄 A 類文件錯誤修正——claude.md 殘留訊號處理模板措辭、audit 慣例補列、results_analysis.md 事實同步。 -->

# 2026-07-11-01 proofread：claude.md 模板措辭與文件事實同步

## Goal

使用者要求盤點專案完成度並檢查過時／錯誤內容，選擇「只修 A 類文件錯誤」。A 類為真正的文件錯誤，
與受預註冊協定凍結（`records/2026-07-05-12 §9` 不回改）的 B 類敘述分開。本次只更正不受凍結、可安全
處理的文件錯誤，不觸碰任何凍結敘述本體。

要修的項目：

1. `claude.md §4` 殘留另一專案（訊號處理／編碼／通道模型）的模板措辭，與本擴散取樣研究專案不符。
2. `claude.md §1.2` 允許 action 清單缺 `audit`，但 `records/2026-07-07-01_audit_repo-health.md` 已使用。
3. `docs/results_analysis.md` 兩處事實同步：預覽段舊語未標「已被 confirmatory 取代」；selector 之
   plan-of-record 已改為 CaF-v2（recall 版）但活文件未記。

## Result

已完成下列改動：

1. `claude.md §4` 三處措辭更正，四條原則語意不變，只替換借自訊號處理的例子：
   - §4.1「one fewer encoding」→「one fewer sampling configuration」。
   - §4.3「a new encoding, a new channel model, a new visualization」→「a new sampler, a new metric,
     a new dataset scale, or a new visualization」。
   - §4.4「explain the signal-processing rationale」→「explain the modeling and measurement
     rationale」；讀者設定由「first-year undergraduate learning the topic」改為「first-year graduate
     student new to diffusion models」。
   - `claude.md` 全檔為英文，此三處維持英文以與檔案一致。
2. `claude.md §1.2` action 清單加入 `audit`，接受倉庫健康稽核為正當且會重複出現的 action，
   使 `records/2026-07-07-01` 的用法合規；不改名已提交記錄。
3. `docs/results_analysis.md`：
   - 「CIFAR-10 預覽（初步…尚未定案）」段開頭加範圍指標，說明本段已被下方 confirmatory 定稿段取代，
     段內「目前不主張 CIFAR 結果」以 confirmatory 段為準。保留歷史、未刪除、未動 confirmatory 段數字。
   - 新增「selector plan-of-record（CaF-v2）」段，記錄因 C8 Pareto 失明、C0 recall 探針
     （`records/2026-07-09-12`），D 包（`records/2026-07-09-13`）已將 plan-of-record selector 改為
     CaF-v2 = `argmax recall s.t. precision ≥ τ`，並註明 README／intro 的 coverage 版屬凍結敘述、
     以本段為現行定義來源。

明確未觸碰（尊重凍結與作者權責）：README 頭條主張、`docs/paper_intro_draft.md` 正文；未改名任何已提交
記錄；未動程式碼（`cifar_data.py` 與 `datasets/cifar.py` 重複整併屬待作者決定）；未刪 gitignore 內
`data/cifar-100-python/file.txt~`。

## Follow-up

- 若作者要把 CaF-v2 定義與機制現況推進 README／intro 的敘述本體，需另走一道協定步驟（status note 或
  作者主導的回改授權），不在本次範圍。
- `cifar_data.py` 與 `datasets/cifar.py` 功能重複的整併仍待作者決定（`records/2026-07-07-01`、
  `records/2026-07-08-01` 已列）。
- 研究承重牆項目（CIFAR-100 judge／scout／confirmatory、CaF vs Chamfer 對決）與本次文件修正無關，
  維持 `records/2026-07-09-13` D 包規劃。
