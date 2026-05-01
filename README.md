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

訓練過程每 `--sample-interval` 個 epoch 將以下檔案寫入 `samples/`：

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

使用訓練好的模型生成數字。推論僅輸出 `generated/dataset.pt`，供 `evaluate.py` 載入；生成過程的視覺化由訓練腳本於訓練期間寫入 `samples/`，推論階段不另行產生圖片。

```bash
# 生成所有數字（0-9），每個數字 100 張
uv run python inference.py

# 只生成特定數字
uv run python inference.py --digits 3 7

# 調整每個數字的生成數量（TSTR 標準流程建議 1000）
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

評估流程涵蓋完整 pipeline：以 DDPM 生成手寫數字、僅用合成圖訓練 CNN 分類器，再於真實 MNIST 測試集量測該分類器準確率。此準確率反映合成資料的實用價值——準確率越高，代表合成分佈在保真度與多樣性兩個面向上越接近真實分佈。此方法稱為 **TSTR（Train on Synthetic, Test on Real）** 或 Classification Accuracy Score，相較於僅以視覺品質檢視單張圖片，更能偵測 mode collapse。

**訓練資料量建議**：標準流程使用 10000 張（每類 1000）。在此量級下，即便以真實資料訓練 CNN 亦已接近準確率上限（約 98%），故資料量不再構成混淆變項，分數差異可直接歸因於生成器的 coverage 與 mode collapse 表現。

```bash
# 1. 生成 10K 張（每類 1000）作為 TSTR 訓練集
uv run python inference.py --per-digit 1000 --output-dir generated

# 2. 訓練 CNN（僅用合成圖），並在真實 MNIST 測試集上評估
uv run python evaluate.py --confusion-matrix \
    --report report.txt --report-json report.json

# 3. 後續快速重跑（沿用已訓練 CNN，跳過訓練階段）
uv run python evaluate.py --checkpoint mnist_cnn_gen.pt --report report.txt

# 4. CI 整合：accuracy 低於警告門檻時以非 0 exit code 結束
uv run python evaluate.py --checkpoint mnist_cnn_gen.pt \
    --threshold 95 --threshold-warn 90 --strict
```

### 評估參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--generated` | `generated/dataset.pt` | 用以訓練 CNN 的合成資料集路徑 |
| `--checkpoint` | — | 已訓練 CNN 權重路徑；提供時跳過訓練 |
| `--save-cnn` | `mnist_cnn_gen.pt` | 訓練完成後 CNN 權重儲存路徑 |
| `--epochs` | `20` | CNN 訓練輪數 |
| `--batch-size` | `64` | 訓練與評估批次大小 |
| `--lr` | `1e-3` | 學習率 |
| `--data-dir` | `./data` | MNIST 資料夾（自動下載真實 MNIST 測試集） |
| `--num-workers` | `2` | DataLoader 工作程序數 |
| `--confusion-matrix` | 關 | 列印混淆矩陣（針對真實 MNIST 測試集） |
| `--report` | — | 將純文字報告寫入此路徑 |
| `--report-json` | — | 將機器可解析 JSON 報告寫入此路徑 |
| `--threshold` | `95.0` | 真實 MNIST 評估準確率達標門檻（%） |
| `--threshold-warn` | `90.0` | 可接受門檻（%），低於此值視為未達標 |
| `--strict` | 關 | 未達標時以非 0 exit code 結束（CI 用） |

評估輸出包含：CNN 在合成訓練集上的準確率（sanity row，正常應接近 100%）、CNN 在真實 MNIST 測試集上的整體準確率（**主要指標**）、每個數字 0–9 的 per-class 準確率、混淆矩陣，以及 TSTR 品質達標判定（PASS / FAIL）。

報告格式：

- **`.txt`**：人類可讀，包含 metadata（GPU、PyTorch 版本、git commit、checkpoint mtime、執行時間）、per-class 表格、混淆矩陣、達標結論，以及 generalization gap（訓練集準確率減去真實準確率，正值越大代表 overfit 越嚴重）。
- **`.json`**：機器可解析，便於比較不同 checkpoint 或於 CI 中追蹤指標。

## 診斷工具

兩個輔助腳本協助驗證評估流程的可信度，以及在 TSTR 結果未達預期時定位失敗原因。兩者皆為非必要步驟，但建議於正式評估前後搭配執行。

### `test_classifier.py` — CNN 評估器健全性檢查

於執行 TSTR 之前，驗證評估用 CNN 在真實 MNIST 測試集上達到合格門檻（預設整體 ≥99%、每類 ≥97%）。CNN 必須足夠準確，TSTR 對「合成資料品質」的判定才具意義；若 CNN 本身在真實資料上即無法達標，TSTR 分數將無法區分「CNN 過弱」與「合成圖品質不足」兩種失敗模式。

```bash
# 驗證在真實 MNIST 上訓練的參考 CNN
uv run python test_classifier.py --checkpoint mnist_cnn.pt

# 自訂門檻
uv run python test_classifier.py --threshold-overall 99.0 --threshold-per-class 97.0
```

通過時 exit code 為 0、未通過時為 1，可整合至 CI 作為前置條件。

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--checkpoint` | `mnist_cnn.pt` | 待驗證 CNN 權重路徑 |
| `--data-dir` | `./data` | MNIST 資料夾 |
| `--batch-size` | `256` | 評估批次大小 |
| `--num-workers` | `2` | DataLoader 工作程序數 |
| `--threshold-overall` | `99.0` | 整體準確率門檻（%） |
| `--threshold-per-class` | `97.0` | 每類準確率門檻（%） |

### `analyze_distribution.py` — 合成分佈診斷

當 TSTR 出現未達標或某類別表現偏低時，比對合成圖與真實圖在像素及 CNN 特徵空間的分佈差異，協助辨識三類失敗模式：

- **Mode collapse**：類別內多樣性不足，gen/real diversity ratio 顯著低於 1。
- **Canonical-only bias**：CNN softmax confidence 接近 1.0 且變異極小，代表生成器僅產出「教科書式」標準樣本。
- **Off-manifold drift**：合成樣本在特徵空間的 centroid 偏離真實類別中心。

腳本逐類別輸出五項指標：像素空間 intra-class L2 多樣性、CNN 特徵空間 intra-class L2 多樣性、像素 std、softmax confidence 分佈、特徵空間 centroid 距離。

```bash
# 使用預設路徑（generated/dataset.pt 與 mnist_cnn.pt）
uv run python analyze_distribution.py

# 指定自訂合成資料集
uv run python analyze_distribution.py --generated generated/dataset.pt

# 調整真實樣本抽樣量與隨機種子
uv run python analyze_distribution.py --per-class 100 --seed 0
```

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--generated` | `generated/dataset.pt` | 合成資料集路徑 |
| `--checkpoint` | `mnist_cnn.pt` | 作為特徵抽取器的 CNN 權重 |
| `--data-dir` | `./data` | MNIST 資料夾 |
| `--per-class` | `100` | 每類別抽樣的真實樣本數 |
| `--seed` | `0` | 真實 MNIST 抽樣隨機種子 |
| `--diversity-ratio-warn` | `0.6` | gen/real 多樣性比例警告門檻 |

本工具依賴一份在真實 MNIST 上訓練的 CNN（`mnist_cnn.pt`）作為特徵抽取器。可透過 `evaluate.py` 以真實資料集訓練取得，或另行訓練後置於專案根目錄。

## 最新評估結果

以預設超參數（DDPM 訓練 20 epochs、`inference.py --per-digit 1000` 生成 10000 張）執行完整 TSTR pipeline，CNN 評估器在真實 MNIST 測試集上的判定如下：

| 指標 | 值 |
|---|---|
| CNN 訓練集準確率（合成圖 sanity row） | 100.00% (10000 / 10000) |
| **真實 MNIST 測試集準確率** | **95.30% (9530 / 10000)** |
| Generalization gap | +4.70 pp |
| 達標判定（門檻 95%） | **PASS（高品質）** |

per-class 準確率最高：`1`（98.85%）、`0`（98.37%）、`5`（97.31%）；最低：`8`（89.73%）、`7`（92.90%）、`9`（92.96%）。混淆矩陣顯示真實的 8 常被誤判為 0（33 筆）或 9（18 筆），反映生成器對數字 8 的多樣性覆蓋不足，為 DDPM 後續改進的主要方向。

重新產生上述報告：

```bash
uv run python inference.py --per-digit 1000 --output-dir generated
uv run python evaluate.py --confusion-matrix --report report.txt --report-json report.json
```

完整數值以執行後產出的 `report.txt` / `report.json` 為準。

## 專案結構

```
ddpm.py                   — 模型架構（UNet）與擴散排程（DiffusionSchedule）
train.py                  — 訓練迴圈、取樣邏輯，定期將網格與去噪過程寫入 samples/
inference.py              — 推論腳本，輸出 dataset.pt（供 evaluate.py 載入）
evaluate.py               — TSTR 流程：以合成圖訓練 CNN，於真實 MNIST 測試集評估
test_classifier.py        — CNN 評估器健全性檢查（TSTR 前置驗證）
analyze_distribution.py   — 合成分佈診斷（mode collapse / canonical bias / drift）
samples/                  — train.py 訓練過程的網格圖與去噪視覺化（不在 git 內）
generated/                — inference.py 預設輸出位置（自動建立、不在 git 內，可隨意覆蓋）
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
