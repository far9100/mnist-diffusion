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

# 儲存預覽網格圖（每個數字一張，square layout 包含全部生成樣本）
uv run python inference.py --save-grid

# 將每張圖各自存成 PNG（全部攤平在 generated/images/，檔名 digit_X_sample_NNN.png）
uv run python inference.py --save-images

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
| `--save-grid` | 關 | 儲存每個數字的預覽網格圖（square layout，包含全部生成樣本）|
| `--save-images` | 關 | 將每張圖個別存成 PNG，置於 `generated/images/digit_X_sample_NNN.png`（全部攤平在單一資料夾）|
| `--save-denoising` | 關 | 儲存去噪過程視覺化 |
| `--denoising-steps` | `9` | 去噪視覺化的中間快照數量 |
| `--save-pt` | 開 | 儲存為 `.pt` 張量資料集 |

## 評估

訓練一個 CNN 分類器在真實 MNIST 上學習辨識手寫數字（baseline ~99% 準確率），再用它評估擴散模型生成的圖片是否能被正確分類，作為品質指標。整體準確率越接近 baseline，代表生成圖越像真實的對應數字。

```bash
# 1. 先生成資料集（若尚未生成）
uv run python inference.py --per-digit 100 --output-dir generated

# 2. 訓練 CNN 並評估生成圖（含混淆矩陣，並產出 .txt 與 .json 報告）
uv run python evaluate.py --save-cnn mnist_cnn.pt --confusion-matrix \
    --report report.txt --report-json report.json

# 3. 之後快速重複評估（用快取的 CNN）
uv run python evaluate.py --checkpoint mnist_cnn.pt --report report.txt

# 4. CI 整合：accuracy 低於警告門檻則 exit code 非 0
uv run python evaluate.py --checkpoint mnist_cnn.pt \
    --threshold 95 --threshold-warn 90 --strict
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
| `--report` | — | 將純文字報告寫入此路徑 |
| `--report-json` | — | 將機器可讀 JSON 報告寫入此路徑 |
| `--threshold` | `95.0` | 整體準確率達標門檻（%）|
| `--threshold-warn` | `90.0` | 可接受門檻（%），低於此值視為未達標 |
| `--strict` | 關 | 未達標時以非 0 exit code 結束（CI 用）|

評估輸出包含：CNN 在真實測試集的準確率（baseline）、在生成圖上的整體準確率、每個數字 0–9 的 per-class 準確率、混淆矩陣、品質達標判定（PASS / FAIL）。

報告格式：
- **`.txt`**：給人閱讀，含 metadata（GPU、PyTorch 版本、git commit、checkpoint mtime、執行時間）、per-class 表格、混淆矩陣、達標結論
- **`.json`**：給機器解析，方便比較不同 checkpoint 或在 CI 中追蹤指標

## 最新評估結果

以預設超參數訓練 20 epochs、生成每數字 100 張圖（共 1000 張）後，CNN 評估器的判定如下：

| 指標 | 值 |
|---|---|
| CNN baseline（真實 MNIST 測試集） | 99.28% (9928 / 10000) |
| 生成圖整體準確率 | **100.00% (1000 / 1000)** |
| 每個數字 per-class 準確率 | 全部 100% |
| 混淆矩陣 | 完美對角，無誤認 |
| 達標判定（門檻 95%）| **PASS（高品質）** |

可預覽的視覺化已隨 repo 提交，可直接在 GitHub 上瀏覽：

- 每個數字 100 張的網格圖：[`generated/grid_digit_0.png`](generated/grid_digit_0.png) … [`generated/grid_digit_9.png`](generated/grid_digit_9.png)
- 去噪過程：[`generated/denoising_process.png`](generated/denoising_process.png)
- 個別樣本（全部 1000 張攤平於同一資料夾）：[`generated/images/`](generated/images/)，檔名格式為 `digit_X_sample_NNN.png`
- 1000 張的 tensor 資料集：[`generated/dataset.pt`](generated/dataset.pt)（直接餵給 `evaluate.py`）

要重新產生上面的數字，可執行：
```bash
uv run python evaluate.py --checkpoint mnist_cnn.pt \
    --confusion-matrix --report report.txt --report-json report.json
```

## 專案結構

```
ddpm.py       — 模型架構（UNet）與擴散排程（DiffusionSchedule）
train.py      — 訓練迴圈與取樣邏輯
inference.py  — 推論腳本，生成數字與去噪過程視覺化
evaluate.py   — CNN 分類器，用於評估生成圖品質
generated/    — 已生成的 1000 張範例圖（dataset.pt + grids + 個別 PNG）
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
