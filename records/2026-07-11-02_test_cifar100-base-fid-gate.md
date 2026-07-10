<!-- 用途：記錄 D10 第一閘——CIFAR-100 base-model FID gate（50k、w=1、clean-fid ≤20）的目標、結果與後續。 -->

# 2026-07-11-02 CIFAR-100 base-model FID gate（D10 第一閘）

## Goal

依 D 包 `records/2026-07-09-13` D9／D10：CIFAR-100 backbone（`checkpoints/cifar100_cfg.pt`，ep1000）
訓練已完成，執行 D10 時序的第一閘——base-model FID gate。

- 量測設定（沿用 CIFAR-10 base gate `records/2026-07-05-09` 之口徑）：w=1（純條件）、steps=50、
  eta=0、seed=0，50000 張平衡樣本（每類 500，共 100 類）。
- 門檻（D9，作者授權 agent 填之委派 gate）：clean-fid ≤ 20 @ 50k、w=1。錨定 CIFAR-10 base 8.95
  （gate ≤10），CIFAR-100 更難、約 2× headroom。
- clean-fid 不內建 cifar100 參考統計（實測 get_reference_statistics 對 cifar100 回 404），故先以真實
  CIFAR-100 train 全 50k 影像建 clean-mode 自訂參考（make_custom_stats），再對生成樣本量 clean-fid。
- 依 `claude.md §5.2`，driver metadata 記 start_timestamp、完整 argv、參數、torch/cuda/cudnn 版本。

本閘為作者 STOP 點：跑完只回報數字與判定，不自行進入下一閘（judge 訓練與校準）。

## Result

通過。50k clean-fid = 11.226，落在事前門檻 ≤20 內，base-model gate PASS。

- 量測口徑：w=1、steps=50、eta=0、seed=0，50000 張（100 類 × 500），對真實 CIFAR-100 train 全 50k
  建的 clean-mode 自訂參考（`cifar100_train_clean32`）比對。
- 錨定比較：CIFAR-10 base 8.95（gate ≤10）；CIFAR-100 更難（100 類、500/類），11.226 落在 ≤20 內、
  且與 CIFAR-10 同量級，backbone 堪用。
- 時序：run 2026-07-10 16:43:13Z 起、17:32:21Z 止（約 49 分鐘），落盤 `results/cifar100_base_fid.json`。
- clean-fid 讀「Found 100000 images」為 Windows 大小寫不敏感 glob 對 50k 唯一檔的良性重複計數
  （`records/2026-07-09-06` 已載）；均勻重複不改 mean/cov，FID 值不受影響（N=50k 下偏差約 1e-5）。
- metadata（依 `claude.md §5.2`）：start/end timestamp、完整 argv、seed/steps/eta/guidance/batch、
  ref 來源張數、torch 2.11.0+cu128 / cuda 12.8 / cudnn 91900 皆入 JSON。
- 委派 gate 數字留痕：門檻 ≤20 為 D9 授權 agent 填之委派值；本次量得 11.226 未接近門檻，無「凍程序不
  凍數字」爭議。

## Follow-up

- 若通過（≤20）：STOP 等作者確認後才進 D10 第二閘（CIFAR-100 judge 訓練與 near-boundary 校準，D9 程序）。
- 若接近或超門檻：依 D9「凍程序不凍數字」例外之明文留痕，屬 backbone 品質之實質發現，不得移動門檻，
  回報並待作者裁示。
- 新增 driver `cifar100_base_gate.py`（重用 `cifar_cfg_sample.load_cfg_model`／`generate_balanced` 與
  `fid_clean` 之 scipy 相容修補與 PNG dump），屬 D9 CIFAR-100 base gate 之量測程序，不改既有 CIFAR-10 路徑。
