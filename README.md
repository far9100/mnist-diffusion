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

使用訓練好的模型生成數字：

```bash
# 生成所有數字（0-9），每個數字 100 張
uv run python inference.py

# 只生成特定數字
uv run python inference.py --digits 3 7

# 調整每個數字的生成數量
uv run python inference.py --per-digit 500

# 儲存預覽網格圖
uv run python inference.py --save-grid

# 儲存去噪過程視覺化
uv run python inference.py --save-denoising

# 調整去噪過程的快照數量（預設 9 個中間步驟）
uv run python inference.py --save-denoising --denoising-steps 15
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
| `--save-grid` | 關 | 儲存每個數字的預覽網格圖 |
| `--save-denoising` | 關 | 儲存去噪過程視覺化 |
| `--denoising-steps` | `9` | 去噪視覺化的中間快照數量 |
| `--save-pt` | 開 | 儲存為 `.pt` 張量資料集 |

## 評估

訓練一個 CNN 分類器在真實 MNIST 上學習辨識手寫數字（baseline ~99% 準確率），再用它評估擴散模型生成的圖片是否能被正確分類，作為品質指標。整體準確率越接近 baseline，代表生成圖越像真實的對應數字。

```bash
# 1. 先生成資料集（若尚未生成）
uv run python inference.py --per-digit 100 --output-dir generated

# 2. 訓練 CNN 並評估生成圖（含混淆矩陣）
uv run python evaluate.py --save-cnn mnist_cnn.pt --confusion-matrix

# 3. 之後快速重複評估（用快取的 CNN）
uv run python evaluate.py --checkpoint mnist_cnn.pt
```

### 評估參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--generated` | `generated/dataset.pt` | 生成資料集路徑 |
| `--checkpoint` | — | CNN 權重路徑；提供時跳過訓練 |
| `--save-cnn` | — | 訓練完將 CNN 權重存到此路徑 |
| `--epochs` | `10` | CNN 訓練輪數 |
| `--batch-size` | `256` | 訓練與評估批次大小 |
| `--lr` | `1e-3` | 學習率 |
| `--data-dir` | `./data` | MNIST 資料夾 |
| `--num-workers` | `2` | DataLoader 工作程序數 |
| `--confusion-matrix` | 關 | 列印混淆矩陣 |

評估輸出包含：CNN 在真實測試集的準確率（baseline）、在生成圖上的整體準確率、每個數字 0–9 的 per-class 準確率，以及（選用）混淆矩陣。

## 專案結構

```
ddpm.py       — 模型架構（UNet）與擴散排程（DiffusionSchedule）
train.py      — 訓練迴圈與取樣邏輯
inference.py  — 推論腳本，生成數字與去噪過程視覺化
evaluate.py   — CNN 分類器，用於評估生成圖品質
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
