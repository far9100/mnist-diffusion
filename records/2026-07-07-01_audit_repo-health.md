<!-- 用途：全專案健康掃描的清單、分類與逐項處置（現修／掛帳／不管），並記 P0 探針參數溯源之 Route 一 讀碼結果。與 α/β 執行序並行，不混入判決序。 -->

# 2026-07-07-01 全專案健康掃描：清單、分類、處置

## 目標

使用者要求掃描全專案，回報完成進度、未完成項、過時內容與錯誤。本 record 記錄該掃描的健康清單、
分類與逐項處置，以及 P0 探針參數溯源（Route 一）的讀碼結果，供作者裁 P0 探針定稿或降級。本檔與
主線 α/β 執行序並行，不混入任何判決，避免下次重掃。

## 正面驗證（採納）

- 39 個 .py 全部有檔頭註解；無 TODO/FIXME/stub；AST 全數 parse 通過。
- 生成種子公式 `gseed = seed*10_000_000 + int(w*1000)*10_000` 四處一致（run_cifar_selector.py:146、
  run_cifar_cfg_multiseed.py:67、run_flip_earlywarning.py:122、run_p0_probe.py:77）。
- run_c2_partial.py 的 per_seed 索引正確（`s` 取自 `range(len(per_seed))`＝0/1/2 清單索引、非 seed 值，
  不會越界）；其讀取的 JSON key 與 confirmatory 輸出 schema 一致（程式驗證）。
- 結論「無阻斷 bug、代碼健康」成立。

## 清單與處置

### 現修（本輪 commit，純文件、非 GPU、與 confirmatory 結果無關）

1. README L17／L98 事實矛盾（bug，非刻意延後）：L17「confirmatory 已完成」與 L98「已凍結待跑」同一
   文件兩處事實打架，係 -01 半同步遺留。改 L98 使其與 L17（E1 已公開之純事實）一致，不加任何結果數字。
2. pyproject.toml L4 描述停留在「從零實作 DDPM 生成 MNIST」，未反映 CIFAR／CaF／EDM／VAR 實際範圍；
   另 scipy（run_c2_partial.py:22、phase1_edm_repro.py:92、fid_clean.py:24）與 Pillow
   （phase1_edm_repro.py:104）為直接 import 但未宣告依賴，現靠 clean-fid／torchvision 傳遞解析，脆弱。
   → 更新描述並補 scipy、pillow 進 dependencies。
3. metrics_features.py 檔頭 L14-17 宣稱提供 `backbone` 參數可在 CLIP／Inception 空間重算 coverage——
   實作無此參數、無 CLIP 路徑。docstring 與實作不符。→ 刪除虛假聲稱，改述實際：模組提供表徵無關的
   fd_from_features／prdc_from_features，交叉檢查以另一表徵（Inception）特徵餵入同一組函式達成
   （Inception 抽取見 run_cifar_cfg_scout.inception_crosscheck、run_cifar_selector.inception_feats）。
4. run_p0_probe.py docstring 行號 :68-107 → :65-111（measure() 實際位置）。工作樹修正；探針檔維持
   未追蹤（不定稿，見下）。

### P0 gate 前提（本次最重要；Route 一 已讀，待作者裁決；探針不定稿、P0 不跑）

5. run_p0_probe.py 參數溯源：nearest_k 未存於 confirmatory metadata、confirmatory.log、prereg 記錄、
   或 -14 invocation spec；唯一來源為 driver argparse 預設 5。confirmatory 明證用了非預設 CLI
   （per_class 1000≠driver 預設 500、10 點 grid、seeds 10/11/12、自訂 output），故「預設即 5」不構成
   證據。關鍵切分：只有 precision／coverage 依賴 k；另三個對帳 scalar 不依賴 k——char_clean_fid（clean-fid，
   無 k）、near_boundary_frac（judge margin＋threshold 0.9525，已存）、label_noise_excess_mean
   （judge preds＋judge_floor，已存）。處置見「後續」Route 一 結果與二選一。

### 掛帳（需判斷或涉順序，本輪不動）

6. cifar_data.py 與 datasets/cifar.py 重複同概念、值域矛盾（cifar_data.py 回 [0,1]、datasets/cifar.py
   回 [-1,1]），且 cifar_data.py 檔頭稱「命名為 cifar_data.py 以避免遮蔽 HuggingFace datasets」但 repo
   本就有 datasets/ 套件，理由自我矛盾。潛在地雷、違反慣例 §4。先查依賴圖（cifar_data 僅
   validate_metrics.py:21 使用）再定合併方向，不無腦刪。
7. E2／E3／E4 文件同步（README 頭條、docs/results_analysis.md、docs/paper_intro_draft.md 與 CIFAR-10
   confirmatory 反證不同步）：屬帶結果治理項，照計畫等 B 定稿，本輪不動。
8. 分支未併回 main：研究主線在 research/caf-main，main 與 origin/main 停在 fb51a78。掛到 D 前處理
   （D commit 時序依賴乾淨 git）。

### 不管（設計取捨或旁支）

9. clone 可重現性：third_party/（EDM，被 .gitignore:43 排除）、*.pt、checkpoints/、results/ 皆不入
   git，新 clone 跑不動 EDM／FID 流程。屬大檔／vendored 之設計取捨，不改。
10. VAR 全量 checkpoint（var_vqvae.pt、var_transformer.pt）為 inference_var.py／train_var.py 預設但
    不存在，只有 *_smoke.pt。旁支、去留待定，README 已標，不動。
11. run_tstr 四份各自實作（run_selector_signal.py:88、run_guidance_study.py:73、run_comparison.py:68、
    cifar_classifier.run_tstr）、mechanism.py CLI 預設指 MNIST：命名／重複 smell（非 bug），慣例 §4，暫不動。

## 後續

Route 一 讀碼結果（呈報作者裁 P0）：

- nearest_k = args.nearest_k，driver argparse 預設 5（run_cifar_cfg_multiseed.py:125）；PRDC（:73）、
  inception_crosscheck（:99）、real_ref_precision（:169）三處都用它。metadata／log／prereg／-14 均未記 k。
  -14 §47-48 之 invocation spec 列出所有 override（grid、seeds、per_class=1000、real_per_class=1000、
  FID 開、output），k 不在其中 → 文件推定 k=5（driver 預設，且 override 清單未含 k）。此為文件推定、
  非儲存值。
- 參考側可完全重現：load_real_per_class("cifar10", real_per_class=1000, seed=0)（seed 硬編碼:160）、
  DINOv2 抽取確定；char_clean_fid 走 clean_fid_vs_dataset train split res=32（:93，無 k）。
- tau_fraction、batch 亦未存：tau 不影響 5 個對帳 scalar（只進 selector）；batch 只影響生成批次，
  Path A（同 gseed 重生成逐位比對）直接測，非參考側 scalar 參數。

裁決二選一（GPU 不進場、探針不定稿，待作者裁 + greenlight）：

- 路一：以文件推定續行，探針 precision／coverage 對帳標「k=5 依文件推定、非儲存值」caveat；若 5 個
  scalar 全對上則反向佐證 k=5；若僅 precision／coverage 對不上而 3 個 k-free scalar 對上，歸因 k 未對齊
  而非生成非決定性（避免假 STOP）。
- 路二：k 不納 P0，探針只對 char_clean_fid／near_boundary／label_noise_excess 三個 k-free scalar；
  precision／coverage 決定性標「內部不可驗、留 caveat」（比照 A3）。

處置順序（承使用者裁定）：本 record 落檔＋現修 commit（本輪完成）→ Route 一 已讀畢並記於上 → 呈報
作者 → 作者裁 P0 探針定稿／降級並 greenlight，方進 β。
