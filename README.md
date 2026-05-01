# DDPM MNIST 手寫數字生成

從零實作 Denoising Diffusion Probabilistic Model (DDPM)，用於生成 MNIST 手寫數字圖片。

## 環境設定

需要 [uv](https://docs.astral.sh/uv/) 管理 Python 環境。

```bash
uv sync
```

## 訓練

```bash
# 使用預設參數訓練
uv run python train.py

# 自訂訓練參數
uv run python train.py --epochs 50 --lr 1e-4 --batch-size 256

# 從已有的 checkpoint 繼續訓練
uv run python train.py --resume ddpm_mnist.pt --epochs 10
```

訓練過程中會在 `samples/` 資料夾定期儲存：
- `epoch_{N}.png` — 10×8 數字網格圖（每個數字 8 張）
- `denoise_epoch_{N}.png` — 去噪過程視覺化（每個數字一列，從純噪音到最終結果）

訓練結束後模型權重儲存為 `ddpm_mnist.pt`。

### 訓練參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--epochs` | `20` | 訓練輪數 |
| `--batch-size` | `128` | 訓練批次大小 |
| `--lr` | `2e-4` | 學習率 |
| `--timesteps` | `1000` | 擴散時間步數 |
| `--base-channels` | `64` | UNet 基礎通道數 |
| `--sample-interval` | `5` | 每 N 個 epoch 儲存一次樣本 |
| `--guidance-scale` | `3.0` | 取樣時的 classifier-free guidance 強度 |
| `--num-workers` | `2` | DataLoader 工作程序數 |
| `--output-dir` | `samples/` | 樣本圖片輸出資料夾 |
| `--save-path` | `ddpm_mnist.pt` | 模型權重儲存路徑 |
| `--resume` | — | 從指定 checkpoint 繼續訓練 |

## 推論

使用訓練好的模型生成數字。輸出只有 `generated/dataset.pt`（給 `evaluate.py` 用），生成過程的視覺化由 `train.py` 在訓練時記錄到 `samples/`，推論時不再重複輸出。

```bash
# 生成所有數字（0-9），每個數字 100 張
uv run python inference.py

# 只生成特定數字
uv run python inference.py --digits 3 7

# 調整每個數字的生成數量（建議 TSTR 標準流程跑 1000）
uv run python inference.py --per-digit 1000
```

### 推論參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--checkpoint` | `ddpm_mnist.pt` | 模型權重路徑 |
| `--digits` | `0-9` | 要生成的數字 |
| `--per-digit` | `100` | 每個數字的生成數量 |
| `--batch-size` | `64` | 生成批次大小 |
| `--guidance-scale` | `3.0` | Classifier-free guidance 強度 |
| `--output-dir` | `generated/` | 輸出資料夾 |

## 評估（TSTR：Train on Synthetic, Test on Real）

把整個專案串成一個端到端的故事：**訓練 DDPM → 用它生成手寫數字 → 拿這些生成圖（不碰真實 MNIST）訓練一個 CNN 分類器 → 看看這個 CNN 能不能認得真實人類寫的 MNIST 數字**。CNN 在真實 MNIST 測試集上的準確率，就是「生成資料的實用價值」——分數越高，代表生成圖在保真度與多樣性兩個面向上都越接近真實分佈。學界把這個量法叫做 **TSTR（Train on Synthetic, Test on Real）** 或 Classification Accuracy Score，比起單純檢查每張圖好不好看，更能抓出 mode collapse。

**訓練資料量建議**：標準流程用 10000 張（每類 1000）。MNIST 在這個量級即使用真實資料訓練也已經接近天花板（~98%），所以資料量不再是混淆變項，分數差異可乾淨歸因到生成器的 coverage / mode collapse。

```bash
# 1. 生成 10K 張（每類 1000）作為 TSTR 訓練集（建議規模）
uv run python inference.py --per-digit 1000 --output-dir generated

# 2. 訓練 CNN（純粹用生成圖），並在真實 MNIST 測試集上評估
uv run python evaluate.py --confusion-matrix \
    --report report.txt --report-json report.json

# 3. 之後快速重跑（用已訓練好的 CNN，跳過訓練）
uv run python evaluate.py --checkpoint mnist_cnn_gen.pt --report report.txt

# 4. CI 整合：accuracy 低於警告門檻則 exit code 非 0
uv run python evaluate.py --checkpoint mnist_cnn_gen.pt \
    --threshold 95 --threshold-warn 90 --strict
```

### 評估參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--generated` | `generated/dataset.pt` | 用來訓練 CNN 的生成資料集路徑 |
| `--checkpoint` | — | 已訓練好的 CNN 權重路徑；提供時跳過訓練 |
| `--save-cnn` | `mnist_cnn_gen.pt` | 訓練完將 CNN 權重存到此路徑 |
| `--epochs` | `20` | CNN 訓練輪數 |
| `--batch-size` | `64` | 訓練與評估批次大小 |
| `--lr` | `1e-3` | 學習率 |
| `--data-dir` | `./data` | MNIST 資料夾（自動下載真實 MNIST 測試集） |
| `--num-workers` | `2` | DataLoader 工作程序數（用於真實 MNIST 測試集載入） |
| `--confusion-matrix` | 關 | 列印混淆矩陣（在真實 MNIST 測試集上） |
| `--report` | — | 將純文字報告寫入此路徑 |
| `--report-json` | — | 將機器可讀 JSON 報告寫入此路徑 |
| `--threshold` | `95.0` | 真實 MNIST 評估準確率達標門檻（%）|
| `--threshold-warn` | `90.0` | 可接受門檻（%），低於此值視為未達標 |
| `--strict` | 關 | 未達標時以非 0 exit code 結束（CI 用）|

評估輸出包含：CNN 在生成訓練集上的準確率（sanity row，正常應接近 100%）、CNN 在真實 MNIST 測試集上的整體準確率（**主要指標**）、每個數字 0–9 的 per-class 準確率、混淆矩陣、TSTR 品質達標判定（PASS / FAIL）。

報告格式：
- **`.txt`**：給人閱讀，含 metadata（GPU、PyTorch 版本、git commit、checkpoint mtime、執行時間）、per-class 表格、混淆矩陣、達標結論、generalization gap（訓練集準確率減去真實準確率，正值越大代表 overfit 越嚴重）
- **`.json`**：給機器解析，方便比較不同 checkpoint 或在 CI 中追蹤指標

## 最新評估結果

以預設超參數（DDPM 訓練 20 epochs、`inference.py --per-digit 1000` 生成 10000 張）跑完整 TSTR pipeline，CNN 評估器在真實 MNIST 測試集上的判定如下：

| 指標 | 值 |
|---|---|
| CNN 訓練集準確率（生成圖 sanity row）| 100.00% (10000 / 10000) |
| **真實 MNIST 測試集準確率** | **95.30% (9530 / 10000)** |
| Generalization gap | +4.70 pp |
| 達標判定（門檻 95%）| **PASS（高品質）** |

per-class 準確率最強：`1`（98.85%）、`0`（98.37%）、`5`（97.31%）；最弱：`8`（89.73%）、`7`（92.90%）、`9`（92.96%）。混淆矩陣顯示真實的 8 常被誤判為 0（33 筆）或 9（18 筆）——通常代表生成器對數字 8 的多樣性覆蓋仍有限，是改進 DDPM 的主要方向。

要重新產生上述報告：

```bash
uv run python inference.py --per-digit 1000 --output-dir generated
uv run python evaluate.py --confusion-matrix --report report.txt --report-json report.json
```

完整數字以執行後產出的 `report.txt` / `report.json` 為準。

## 專案結構

```
ddpm.py       — 模型架構（UNet）與擴散排程（DiffusionSchedule）
train.py      — 訓練迴圈、取樣邏輯，定期把網格與去噪過程寫到 samples/
inference.py  — 推論腳本，輸出 dataset.pt（給 evaluate.py 用）
evaluate.py   — TSTR 流程：用生成圖訓練 CNN，然後在真實 MNIST 測試集上評估
samples/      — train.py 訓練過程的網格圖與去噪視覺化（不在 git 內）
generated/    — inference.py 預設輸出位置（自動建立、不在 git 內，可隨意覆蓋）
```

## 架構概覽

- **UNet**：編碼器-瓶頸-解碼器結構，含跳躍連接
  - 通道數：64 → 128 → 256
  - 在 14×14 解析度加入自注意力機制
  - 透過正弦位置編碼注入時間步資訊
- **擴散排程**：線性 β 排程（1e-4 至 0.02），共 1000 步
  - 前向過程：逐步加噪
  - 反向過程：逐步去噪生成圖片

## 預設超參數

| 參數 | 預設值 |
|------|--------|
| 時間步數 T | 1000 |
| β 範圍 | [1e-4, 0.02] |
| 基礎通道數 | 64 |
| 批次大小 | 128 |
| 學習率 | 2e-4 |
| 訓練輪數 | 20 |

所有超參數均可透過命令列參數覆寫，詳見上方訓練參數表。
