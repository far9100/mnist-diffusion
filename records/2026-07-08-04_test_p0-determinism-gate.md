<!-- 用途：P0 決定性 gate 之執行與 STOP 呈報。單 cell (seed10, w1) 生成決定性隔離＋整鏈 scalar 重現之實測結果、k 溯源結論、信任鏈狀態與 P1 排程建議。依 records/2026-07-08-02 §2.2/§2.4、records/2026-07-06-05 §4。本檔為 β 之 STOP 呈報，等作者 P1 greenlight。 -->

# P0 決定性 gate：執行與 STOP 呈報

## Goal

依執行指令 `records/2026-07-08-02` §2.2 執行 P0 單 cell（seed10 × 首 grid 點 w1，gseed=110000000），
以定稿探針 `run_p0_probe.py`（commit 0d14c05）驗證兩事：

1. 路 A：同 gseed 生成兩次是否逐位相同（＝本環境生成決定性，P1 全量有效性唯一所繫）。
2. 整鏈：7 個 scalar（precision、coverage 之 DINOv2 與 Inception 兩側、char_clean_fid、
   near_boundary_frac、label_noise_excess_mean）對凍結 confirmatory JSON 是否重現，
   依 `records/2026-07-06-05` §4 決定性三態（逐位相等／相對 ≤1e-4 容忍內／超容忍）判定。

本檔為 §2.4 STOP 呈報，呈報後暫停，等作者 P1 greenlight（`records/2026-07-08-02` §0.2 之 β 單一決策點）。
凍結 JSON 未觸碰、無任何數字回改。

## Result

輸出 `results/cifar10_p0_probe.json`。verdict = **ALL_BITEXACT**（三態中最強者）。

### 路 A（生成決定性）

`bitexact = true`、`max_abs_diff = 0.0`。同 gseed 兩次生成之影像與標籤逐位相同。本環境
（RTX 5070、torch 2.11.0+cu128、cuda 12.8、cudnn 91900）生成決定性成立。

### 決定性三態（逐 scalar，全部逐位相等）

| scalar | k 依賴 | got | ref（凍結 JSON） | 三態 |
|---|---|---|---|---|
| precision（DINOv2） | 是 | 0.8126000344753266 | 0.8126000344753266 | 逐位相等 |
| coverage（DINOv2） | 是 | 0.6409000307321548 | 0.6409000307321548 | 逐位相等 |
| coverage_inception | 是 | 0.8959000468254089 | 0.8959000468254089 | 逐位相等 |
| precision_inception | 是 | 0.7362000405788421 | 0.7362000405788421 | 逐位相等 |
| char_clean_fid | 否 | 10.93039776801254 | 10.93039776801254 | 逐位相等 |
| near_boundary_frac | 否 | 0.2531999945640564 | 0.2531999945640564 | 逐位相等 |
| label_noise_excess_mean | 否 | 0.04239999471604823 | 0.04239999471604823 | 逐位相等 |

7 項全部 abs_delta=0、rel_delta=0。k-free 三臂（char_clean_fid、near_boundary_frac、
label_noise_excess_mean）與 k-dependent 四項同為逐位相等，無分診歧異，k_sweep 未觸發（`null`）。

補記：探針以決定性旗標（`use_deterministic_algorithms(warn_only)`、cudnn benchmark off、
cublas workspace `:4096:8`）執行，而 confirmatory 原跑未設此旗標；在此差異下仍逐位相等，說明本
生成/度量路徑於本硬體上本即決定性，非靠旗標湊出。

### k 溯源結論：反證支持 k=5

nearest_k 在 confirmatory metadata 未存，探針先驗標為 [NEW]「對帳失效候選」（`records/2026-07-07-01`
Route 一之文件推定 k=5）。本次以 k=5 執行，四個 k-dependent scalar（precision/coverage 兩側）
逐位重現凍結 JSON。依 `records/2026-07-08-02` §2.3「若 scalar 全對上則反向佐證 k=5」，此為 k=5 之
正向反證：不需 k-sweep 找回，`records/2026-07-05-14` 之 k=5 文件推定由 bitexact 重現實證確立，
nearest_k 之 [NEW] 對帳失效候選標記解除。凍結 JSON 不動、數字不回改。

### 計時實測（單 cell，秒）

| 步驟 | 秒 |
|---|---|
| gen_1（路 A 第一次生成） | 541.3 |
| gen_2（路 A 第二次生成） | 517.3 |
| dino_feat（DINOv2 抽取，gen+real） | 37.7 |
| judge（margin/preds） | 0.8 |
| clean_fid（char clean-fid） | 104.4 |
| incep_feat（Inception 兩側 coverage/precision） | 23.3 |

單 cell 總計約 1225 秒（20.4 分）。其中路 A 之兩次生成佔約 1059 秒；P1 每 config 僅需一次生成
（路 A 之雙生成僅為決定性測試），故 P1 每 config 生成約 530 秒，量測（除 TSTR）約 166 秒。
探針未跑 TSTR（driver measure() 含 `run_tstr`，15 epoch，另計），故 P1 每 config 實際 > 696 秒。
以每 config 約 12–15 分估，30 config 約 6–7.5 小時，與 `records/2026-07-05-14` 之 6–8 小時估一致。

### 儲存實測

單 cell 落盤 256,291,641 bytes（約 244 MiB），含 uint8 影像、DINOv2 與 Inception per-sample
特徵（gen 與 real 兩側）、judge 輸出、labels。探針之 30-cell 粗估 7.689 GB 係 7 檔 ×30 之高估
（把 real 參考特徵重算 30 次）；real 參考特徵（DINOv2 30.7MB＋Inception 81.9MB）僅需存一次，
校正後：

- 只存影像＋DINOv2 特徵（Inception 事後由影像重算，`records/2026-07-08-02` §3 允許）：
  每 config 約 61.7 MB，30 config＋real 參考約 1.9 GB。
- 影像＋DINOv2＋Inception 皆存：每 config 約 143.6 MB，30 config＋real 參考約 4.4 GB。

### 信任鏈狀態（P0 為單 cell spot-check，全網格由 P1 完成）

- **判決二（H-C2，`records/2026-07-06-09`）**：偏相關裁決之資料基底為逐 config precision/coverage
  （DINOv2 主、Inception 交叉）。P0 於 (seed10, w1) 證此四值逐位重現，判決二之資料基底在本 cell
  可重現；A1/A2 之全網格重現待 P1。
- **判決三（selector 描述性，`records/2026-07-06-05` §5.3）**：regret 用 precision/coverage/TSTR，
  C6 FID-min 用 char_clean_fid。P0 證 precision/coverage/char_clean_fid 於 w1 逐位重現；TSTR 探針
  未含，待 P1 對帳。selector 之 fidelity/coverage/FID 基底在本 cell 可重現。
- **C8 Pareto 失明實例（`records/2026-07-06-16`）**：w2.5 支配三 oracle 之判斷用逐 config
  (precision, coverage)。w1 之逐位重現支持整體可重現性；C8 所引之 w2.5 與 oracle cell 由 P1 驗。
- **C 批（C1–C7）**：全部取自 P 持久化資產。P0 證持久化之 DINOv2/Inception per-sample 特徵與
  confirmatory 逐位一致（生成 bitexact 且特徵抽取決定性），故 C1（which-FID DINOv2 揭盲）與
  C2/C3/C5/C7 將在已驗證基底上計算；資產由 P1 產出。

整體：P0 把「生成/度量決定性假設」由假設轉為單 cell 實測事實（ALL_BITEXACT），達 `records/2026-07-06-05`
§4 之 pass 條件。無 STOP 觸發（三個 k-free 控制臂與 k-dependent 皆逐位相等，非生成非決定性、非跨環境
漂移）。單 cell 為 spot-check，逐位對帳之全網格延伸至 P1。

## Follow-up

- **STOP：等作者 P1 greenlight**（`records/2026-07-08-02` §0.2、§2.4）。本檔不推進 P1；P1×C1 之
  預備指令見 `records/2026-07-08-02` §3，greenlight 後方生效。
- **P1 排程建議**：seed-major（seed10 全 10 config → 11 → 12）；走 `run_cifar_cfg_multiseed.py`
  之生成路徑與 gseed 公式原樣（`cifar_cfg_sample.py` 未證逐位等價前不代用）；每 config 落盤 uint8
  影像＋DINOv2 per-sample 特徵，即時以 k=5（本檔實證）對帳凍結 JSON，任一 config 超容忍即 STOP；
  seed-10 中途對帳禁詮釋（`records/2026-07-08-02` §3、§7）。單機約 6–7.5 小時、儲存約 1.9–4.4 GB，
  建議單一長時背景 session 或跨夜執行。Inception 特徵可事後由落盤影像重算以省儲存（§3 允許）。
- **k=5 定案之下游**：P1 對帳與後續 C 批一律用 k=5，`records/2026-07-05-14` 之 k=5 文件推定已由本檔
  bitexact 重現實證；探針 metadata 已強制記錄 nearest_k／有效 k／環境版本（`records/2026-07-08-02`
  §2.1c），F1 流程債於生成側 driver 亦已補（commit 2b68f8d §1.5）。
- P0 落盤資產在 `results/p0_probe_artifacts/`（不入 git，`.gitignore:25`），供 P1 對帳與 k-sweep 稽核
  複用；本檔之 `results/cifar10_p0_probe.json` 為完整實測留痕。
