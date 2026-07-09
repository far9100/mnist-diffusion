<!-- 用途：P1×C1 streaming 執行與對帳結果。全 30 config 逐位對帳凍結 JSON 之結果（P2 正確性 gate）、per-config FD-DINOv2 原始數字（C1 揭盲之數字產出，裁決留 γ）、計時與儲存實測。依 records/2026-07-08-02 §3、records/2026-07-06-05 §4。 -->

# P1×C1：streaming 持久化與逐 config 對帳

## Goal

依執行指令 `records/2026-07-08-02` §3（P1 greenlight 後生效）與定稿計畫 `records/2026-07-06-05` §4，
以 `run_p1_streaming.py`（commit 034cfed）執行 P1×C1：

1. seed-major 重生成全 30 config（seed 10 全 10 → 11 → 12），生成路徑與 gseed 公式原樣，落盤 uint8
   影像＋DINOv2 per-sample 特徵。
2. 每 config 即時以 k=5（P0 `records/2026-07-08-04` 實證）對帳凍結 confirmatory JSON 之 7 個確定性
   scalar；任一超容忍即 STOP。此即 P2 正確性 gate（`records/2026-07-06-05` §4）之全網格延伸。
3. 隨算 per-config FD-DINOv2（C1，接 `metrics_features.fd_from_features`）——數字產出，which-FID
   裁決留 γ（`records/2026-07-08-02` §4.1）。

凍結 JSON 唯讀、無數字回改。不對帳 TSTR（分類器訓練非決定性、不在 P2 gate），不計算/不記錄 recall、
density（`records/2026-07-08-02` §3 不預授權）。

## Result

`status = complete`，30/30 config 完成，無 STOP。輸出 `results/cifar10_p1_streaming.json`。

### 對帳結果（P2 正確性 gate）：全 30 config ALL_BITEXACT

30 個 config 之 7 個對帳 scalar（precision、coverage 之 DINOv2 與 Inception 兩側、char_clean_fid、
near_boundary_frac、label_noise_excess_mean）全部逐位相等。worst rel_delta = 0.00e+00；over_tol
config 數 = 0；within_tol-but-not-bitexact config 數 = 0。逐 scalar 非逐位數皆為 0：

| scalar | 非逐位 config 數（共 30） |
|---|---|
| precision（DINOv2） | 0 |
| coverage（DINOv2） | 0 |
| coverage_inception | 0 |
| precision_inception | 0 |
| char_clean_fid | 0 |
| near_boundary_frac | 0 |
| label_noise_excess_mean | 0 |

決定性三態：全 30 config 逐位相等（三態中最強者）。P0 於單 cell 之 ALL_BITEXACT 由此延伸至全網格：
整個 CIFAR-10 confirmatory 結果集在本環境逐位可重現，P2 正確性 gate 完全通過，無降級語義觸發。

### per-config FD-DINOv2（C1 揭盲之數字產出；裁決留 γ，本 record 不判讀）

grid = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8]

| seed | w1 | w1.5 | w2 | w2.5 | w3 | w4 | w5 | w6 | w7 | w8 |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 283.23 | 194.75 | 175.28 | 178.11 | 186.39 | 222.81 | 262.18 | 301.98 | 338.95 | 379.35 |
| 11 | 280.41 | 193.61 | 174.63 | 175.28 | 190.28 | 225.04 | 260.88 | 301.89 | 342.74 | 382.83 |
| 12 | 283.59 | 197.19 | 176.40 | 176.70 | 189.56 | 223.78 | 260.15 | 303.91 | 344.01 | 375.89 |
| mean | 282.41 | 195.19 | 175.43 | 176.70 | 188.74 | 223.88 | 261.07 | 302.59 | 341.90 | 379.36 |

which-FID 裁決（FD-DINOv2 之 argmin 與 TSTR-opt 是否分離）依凍結口徑 `records/2026-07-06-11`（分離口徑：
均值曲線 argmin 相異且 >1 格步、或三 seed 方向一致之相異）為 γ 工作（`records/2026-07-08-02` §4.1），
於獨立 C1 裁決 record 進行，反證版與弱版本筆墨等量。本 record 僅產出數字，不作分離判定、不與 TSTR 或
clean-fid argmin 比較。以上三 seed 數字之方向一致性亦為 γ 裁決之輸入，此處不預讀。

### 計時與儲存實測

總時 30079 秒（約 8.35 小時），每 config 平均約 1003 秒。高於 P0 推估之 6–7.5 小時：P0 以單次生成
約 530 秒外推，實測每 config 生成約 400–1000 秒（含 DINOv2 抽取約 38s、clean-fid 約 104s、Inception
兩側約 23s、judge 與 FD 可忽略）。單機單 session 一次跑完。

儲存實測：`results/p1_assets/` 共 1.8 GB。每 config 目錄約 61.6 MB（img_uint8 30.7MB＋dino_feat
30.7MB＋judge_out 0.12MB＋labels 0.08MB），30 config 約 1.85 GB，加 real 參考 DINOv2 特徵存一次。
Inception per-sample 特徵未落盤（對帳時由影像即算即棄，`records/2026-07-08-02` §3 允許），故較 P0 之
「含 Inception 4.4 GB」估值小。

### 信任鏈狀態（P0 單 cell → P1 全網格）

- **判決二（H-C2，`records/2026-07-06-09`）**：偏相關之資料基底（逐 config precision/coverage 兩側）
  全 30 config 逐位重現，A1/A2 之全網格資料基底確立可重現。
- **判決三（selector 描述性，`records/2026-07-06-05` §5.3）**：precision/coverage/char_clean_fid 全
  30 config 逐位重現；TSTR 不在對帳集（非決定性），凍結 JSON 之 TSTR 續為判決三依據。
- **C8 Pareto 失明（`records/2026-07-06-16`）**：所引 w2.5 與 oracle cell 之 (precision, coverage)
  在全網格逐位重現，引理實例之資料基底確立。
- **C 批（C1–C7）**：全部取自 `results/p1_assets/` 之落盤影像與 DINOv2 特徵；本 gate 證此資產與
  confirmatory 逐位一致，C1（本 record 之 FD-DINOv2）與 C2/C3/C5/C7 於已驗證基底上計算。

## Follow-up

- **which-FID（C1）裁決 = γ**（`records/2026-07-08-02` §4.1）：依凍結口徑 `records/2026-07-06-11`
  對號入座二元活結果空間（分離→表徵依賴弱版本；不分離→CIFAR-10 尺度反證確立），反證版與弱版本筆墨
  等量，另落 C1 裁決 record。本 record 之 FD-DINOv2 數字為其輸入。
- **C2/C3/C5/C7 計算 = γ**（`records/2026-07-08-02` §4.2）：一律取自 `results/p1_assets/` 落盤影像，
  禁第二次重生成；C8 一頁版入 docs 併兩補強（空可行集 fallback、單調 selector 不可能性推廣）。
- **B 定稿 = γ**（§4.4，STOP 呈報）：判決一補 C1 結果；附錄之「P 對帳結果」保留槽以本 record 填
  （30/30 ALL_BITEXACT）；判決二逐字引 -09；C6 數字只進判決三；regret 主 3.69、並列 2.77。
- **δ（D 起草）可與 γ 並行**（`records/2026-07-08-02` §5）：D4 門檻、D8 CaF-v2 去留等為作者保留欄位，
  到點 STOP 呈報，agent 不代填。
- P1 落盤資產在 `results/p1_assets/`（不入 git，`.gitignore:25`）；`results/cifar10_p1_streaming.json`
  為完整對帳與 FD-DINOv2 留痕。
