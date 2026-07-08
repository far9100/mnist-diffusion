<!-- 用途：記錄一次全專案審查後的結果無關事實層修正（cifar_data.py 檔頭矛盾、results_analysis.md confirmatory 過時陳述）。 -->

# 校對：結果無關的過時陳述與檔頭矛盾修正

## Goal

全專案審查後，只修「與研究結果無關、且不受定稿計畫 `records/2026-07-06-05` 凍結治理」的兩項事實層問題，其餘受凍結或需作者裁決者一律不動：

1. `cifar_data.py` 檔頭自相矛盾：稱命名為 `cifar_data.py`「以避免遮蔽 HuggingFace `datasets`」，但 repo 內確實有 `datasets/` 套件，理由被實際結構打臉。
2. `docs/results_analysis.md` 頂部仍寫「CIFAR confirmatory 結果尚未產生／主結果待跑」，與 `README.md` 及實際檔案 `results/cifar10_cfg_confirmatory.json` 矛盾。

修正嚴格限 E1 純事實層級：更正假陳述並指向正確流程，不寫任何 confirmatory 數字、不寫否證敘事，不觸 E3、不越 `2026-07-06-05` §2 格殺清單。

## Result

1. `cifar_data.py:3-5` 檔頭改寫：如實描述本檔為 Phase 1 早期量測驗證（`validate_metrics.py`）使用、回傳 `[0,1]` 值域，主線改用 `datasets/cifar.py`（回傳 `[-1,1]`），並明點兩者值域差異以防無聲縮放。函式行為未動；`python -m py_compile cifar_data.py` 通過。

2. `docs/results_analysis.md` 兩處事實層更新：
   - 頂部：把「CIFAR confirmatory 結果尚未產生」改為「CIFAR-10 confirmatory 已於 2026-07-06 完成（`results/cifar10_cfg_confirmatory.json`），結果分析與三判決依 `records/2026-07-06-05` 工作包 B 定稿後補入（E3），此處先不列結論」。
   - 範圍註記段：把「confirmatory 主結果待跑」改為「已完成（見上），其分析待工作包 B 定稿後補入」。
   - body 表格與所有分析未動；未加入任何 confirmatory 數字或詮釋。

`git diff --stat`：`cifar_data.py`（+3/-2）、`docs/results_analysis.md`（+4/-2）。

## Follow-up

以下項目本輪刻意不動，路由給作者或等凍結到期：

- E2/E3/E4（README 措辭、results_analysis 鏡像三判決、intro banner）：`2026-07-06-05` §8 規定等工作包 B 定稿 + 作者核措辭。
- `run_p0_probe.py`（未追蹤）：屬 §9 β 階段 P0 探針產物，commit 時機屬作者排程裁決。
- `cifar_data.py` 與 `datasets/cifar.py` 重複、值域衝突（`[0,1]` vs `[-1,1]`）：本次僅修註解點出風險；是否合併/刪除屬 spec 外結構改動，需作者確認（CLAUDE.md §4.3）。
- `run_tstr` 邏輯重複四份（`run_selector_signal.py`、`run_guidance_study.py`、`run_comparison.py`、`cifar_classifier.run_tstr`）：DRY smell、非 bug；抽共用函式需作者確認，且牽動已凍結驅動程式。
- VAR-mini 預設 checkpoint（`var_*.pt`）不存在、磁碟僅 `var_*_smoke.pt`：VAR 為未定側支，待去留裁決後再處理。
- Confirmatory metadata 未存 `nearest_k`/`batch`：已由 P0 gate（`records/2026-07-07-01`、`run_p0_probe.py`）追蹤。
