<!-- 用途：記錄 D10 第二閘——CIFAR-100 judge 訓練與 near-boundary threshold 校準的目標、結果與後續。 -->

# 2026-07-11-03 CIFAR-100 judge 訓練與 near-boundary 校準（D10 第二閘）

## Goal

依 D 包 `records/2026-07-09-13` D9／D10 第二閘：在真實 CIFAR-100 上訓練 ResNet-18 judge，作為 mechanism
標籤噪音與 near-boundary 診斷的裁判，並依 D9「凍程序不凍數字」以真實測試 margin 的 p20 相對分位校準
CIFAR-100 專用 near-boundary threshold（程序沿 CIFAR-10 judge `records/2026-07-05-05`，門檻數字隨資料）。

判定重點：judge 須夠準，near-boundary/label-noise 診斷才有意義；CIFAR-100 為機制的「未飽和主戰場」，
需驗證 judge 空間下 margin 分布不像 CIFAR-10 那樣飽和。

本閘為作者 STOP 點：跑完只回報，不自行進 scout。

## Result

judge：真實 CIFAR-100 全訓練集 50000 張，ResNet-18 訓 25 epoch（SGD + cosine + 增強），真實測試準確度
74.25%（train 96.78%）。存 `checkpoints/cifar100_judge.pt`。相對 100 類隨機 1%，74.25% 為堪用裁判。

near-boundary 校準：真實測試 margin（p_top1 − p_top2）分位數 p05=0.067、p10=0.149、p20=0.362、p25=0.472、
p50=0.913。取 p20 為 CIFAR-100 near-boundary threshold = 0.3622（真實 near-boundary 比例 0.200）。

飽和度對比（關鍵）：CIFAR-10 judge 的 margin 中位數高達 0.999（近飽和，`records/2026-07-05-05`）；CIFAR-100
judge 中位數 0.913、且低分位有實質展開（p20=0.362 遠低於 CIFAR-10 的 0.953）。代表 CIFAR-100 在 judge 空間
明顯較不可分、near-boundary 訊號未飽和，符合 D0 對 CIFAR-100 作為機制主戰場的預期。

judge 品質 gate：D9 凍程序但未凍 judge 準確度數字，本閘 74.25%（對 1% 隨機）以此作作者接受點；未設硬門檻
數字，避免在預註冊研究上自行凍結未授權數字。

metadata（依 `claude.md §5.2`）：start/end timestamp、完整 argv、epochs、torch 2.11.0+cu128 / cuda 12.8 /
cudnn 91900 皆入 `results/cifar100_judge.json`。時序：2026-07-10 17:50:34Z 起、17:55:56Z 止（約 5 分鐘）。

工具變更：`cifar_judge.py` 一般化為 `--dataset cifar10|cifar100`（自 `datasets.cifar.NUM_CLASSES` 取類數、
預設存檔路徑隨 dataset），CIFAR-10 預設行為不變；並補 §5.2 metadata 至輸出。

## Follow-up

- STOP 等作者確認後才進 D10 第三閘：CIFAR-100 寬 grid scout（僅定 confirmatory 網格，讀數不回饋判準，
  D10）。scout 後為網格凍結 amendment，再 confirmatory。
- judge `checkpoints/cifar100_judge.pt` 與 threshold 0.3622 為 confirmatory 之凍結量測儀器（near-boundary、
  label-noise 用），對應 CIFAR-10 的 0.9525。
- 解讀時仍註明：CIFAR-100 未飽和是相對 CIFAR-10 而言；確切機制複製由 confirmatory 三觀察量（D3）裁決。
