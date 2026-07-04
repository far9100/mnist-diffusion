<!-- 用途：專案總覽——前言、背景知識、方法、實驗設計、程式碼分代、專案結構與重現指令。 -->

# Sampling for Utility, not Fidelity

擴散模型生成的合成影像越來越常被當成下游分類器的訓練資料。本專案研究擴散取樣的組態（取樣步數
steps、DDIM 隨機性 η、guidance 強度）應為**下游訓練效用（TSTR）**而非視覺保真度（FID）最佳化，
並提出一個免訓練、可遷移、有機制解釋的組態選擇器 **CaF（Coverage-at-Fidelity）**。

實驗結果與數據分析見 `docs/results_analysis.md`；研究計畫與進度記錄見 `records/`。

## 前言

控制擴散 sampler 的旋鈕（去噪步數、隨機性、guidance 強度）幾乎總是為了最佳化影像保真度（FID）
而調整，背後隱含「越逼真的影像就是越好的訓練資料」的假設。本專案指出這個假設在一個具體且可操作
的意義上是錯的：**讓保真度最高的取樣組態，並不是讓下游效用（Train on Synthetic, Test on Real,
TSTR）最高的組態**，而且兩者的落差是由**多樣性 / coverage**驅動，不是由保真度驅動。

這個觀察本身不新——差分隱私擴散的文獻曾順帶提及，近期的 guidance 方法也隱含此點。本專案的貢獻
不是這個觀察，而是它的**可用結果**：一個**免訓練的選擇器**，在只有一小份真實參考集、且沒有下游
分類器的情況下，從候選組態中挑出效用最佳的取樣組態。我們稱之為 **CaF（Coverage-at-Fidelity）**：
在 precision 不低於門檻 τ 的前提下，選 manifold coverage 最大的組態；τ 由真實對真實（real-vs-real）
的參考自動決定，不需要 TSTR、也不需要任務標籤。

## 背景知識

- **擴散取樣的旋鈕**：steps 是反向去噪的步數；η 是 DDIM 的隨機性（η=0 為 deterministic 的
  probability-flow ODE，η=1 且滿步時退回 DDPM ancestral sampling）；guidance 是 classifier-free
  guidance（CFG）的強度，控制取樣往條件類別集中的程度。
- **FID 與 TSTR**：FID 以特徵空間的 Fréchet distance 量單張影像的視覺保真度；TSTR 只用合成圖訓練
  一個分類器、再於真實測試集量準確率，反映合成資料的下游訓練效用。兩者量的是不同東西。
- **precision 與 coverage**：manifold-based 的生成指標（PRDC）。precision 近似保真度（生成樣本
  落在真實流形內的比例），coverage 近似多樣性涵蓋（真實流形被生成樣本涵蓋的比例）。
- **guidance 會銳化分佈**：提高 guidance 會把取樣分佈往 class prototype 集中（Ho & Salimans 2022；
  Bradley & Nakkiran 2024 的「denoise + sharpen」措辭），因而抽走低 margin、靠近決策邊界的樣本。

## 方法：CaF 與機制

**CaF 選擇器**：在候選 (steps × η × guidance) 組態上，用一小份真實 probe set 於特徵空間計算每個
組態的 precision 與 coverage，選 `argmax coverage s.t. precision ≥ τ`。門檻 τ 由真實 probe 的
real-vs-real precision 乘上一個比例自動決定，因此不依賴 TSTR、不需要在任務上訓練分類器，屬免訓練。
選擇品質以 `regret@selected`（選中組態的 TSTR 與網格最佳 TSTR 之差）與 top-k 命中衡量，取代容易被
爛組態撐高的全域相關係數。

**機制**：提高 guidance 使取樣往 class prototype 集中，移除下游分類器用來擺放決策邊界所需的低
margin、near-boundary 樣本。因此有一條因果鏈：**guidance 上升 → coverage 下降 → near-boundary
訓練樣本變少 → 下游 margin 變弱 → TSTR 下降**。near-boundary 以一個用真實資料訓練的分類器量測
（機率 margin，即 p(top1) − p(top2)），並對「低 guidance 引入離類 / 模糊樣本」這條 label-noise
競爭機制做對照。

## 與 Chamfer 的定位

另一條並行研究（Chamfer Guidance, NeurIPS 2025；Feedback-guided Synthesis, TMLR 2024；Deliberate
Practice, 2025）以**改變生成過程**（在每一步加入 guidance 項、讓梯度穿過特徵抽取器與解碼器，或把
下游任務分類器放進迴圈）讓合成資料更有用。CaF 佔據互補的操作點：

- **它選擇、不修改**：CaF 在既有的組態 / 輸出中挑選，sampler 不動。因此 CaF 可組合，能疊在任何
  生成器的輸出上，包括某個 guidance 方法的輸出。guidance 方法無法這樣宣稱，因為它必須擁有 sampler。
- **免任務分類器、無 guidance 強度超參**：CaF 對真實參考算一次 precision / coverage，沒有逐樣本
  的梯度導引，也沒有需要隨 backbone 調整的 guidance 強度。
- **附機制**：CaF 解釋了為什麼驅動效用的是 coverage 而非 fidelity，這是那些 guidance 方法沒有給
  的因果說法。

明確不主張為新穎的部分：coverage / precision 指標本身、「少量真實範例 + 特徵抽取器」的設定、以及
比 CFG 便宜（Chamfer 已宣稱）。差異化押在**操作點**（免訓練、任務無關、不修改、可組合）加上**為
選擇證明的機制**。

## 程式碼分代

- **Gen-1（MNIST sandbox，已完成）**：從零實作的條件式 DDPM 與 DDIM(η) 取樣器、MNIST-FID、TSTR
  評估、分佈診斷。用於最低成本驗證機制方向與選擇器可行性。
- **Gen-2（Phase 1，CIFAR，進行中）**：自訓 CFG-capable CIFAR 模型、以預訓練 EDM 作量測錨點、PRDC
  與 FD-DINOv2 量測堆疊、CaF 選擇器與機制分析在 CIFAR-10/100 上的複製與對決。
- **VAR-mini（旁支探索）**：MNIST 上的兩階段 VAR-mini（多尺度殘差 VQ-VAE + scale-wise transformer）。
  目前與研究主線無關，僅有 smoke 測試，去留待定。

## 環境設定

需要 [uv](https://docs.astral.sh/uv/) 管理 Python 環境。

```bash
uv sync
```

## 目前進度

- Phase 0（MNIST sandbox）：完成，機制方向正確、CaF 可行。
- Phase 1-1（量測堆疊正確性 gate）：完成，EDM CIFAR-10 FID 重現通過。
- Phase 1-2（CFG backbone）：CIFAR-10 訓練進行中；CIFAR-100 模型尚未開始。
- Phase 1-3（CIFAR-10 複製 C1）：僅有快速預覽，尚未定案。
- Phase 1-4（CIFAR-100 機制）：未開始，是全案科學承重牆。
- Phase 1-5（CaF vs Chamfer 對決）：未開始。

各階段的實際數據與分析見 `docs/results_analysis.md`。

## 實驗設計：完整實驗要確立的事

- CIFAR-10 全品質（18 步）：guidance 的內部最優是否持續，CaF 是否跨 3 個以上 seed 近最優選中
  （regret 小）。
- CIFAR-100（更難、非可分）：coverage 主導是否複製，near-boundary 機制是否可量測（不飽和）。這是
  真正的 go/no-go 硬門檻。
- 第二特徵表徵（CLIP / Inception）交叉驗證 coverage，破除 DINOv2 的雙重使用循環。
- matched-budget 對決：CaF 與重實作的 Chamfer 基線比下游準確率與成本，並展示可組合性（CaF 疊在
  Chamfer 的輸出之上）。
- τ 穩健性與 TSTR-free 的 τ 自動決定，誠實報告其敏感度。

## 專案結構

研究主線（Gen-2, CaF / CIFAR / 機制）：

```
selector.py              — CaF：argmax coverage s.t. precision ≥ τ，含 auto-τ、τ 穩健性、regret@selected
mechanism.py             — 機制分析：guidance 對 near-boundary 樣本支持度的影響、label-noise 對照
metrics_prdc.py          — PRDC（Precision/Recall/Density/Coverage），純 torch
metrics_features.py      — 表徵式生成指標（DINOv2 特徵、FD-DINOv2）
fid_clean.py             — 標準 Inception-FID（clean-fid），正確性錨點
train_cifar.py           — CFG-capable CIFAR-10/100 擴散模型訓練（EMA、週期性 checkpoint）
cifar_data.py            — CIFAR-10/100 載入
cifar_classifier.py      — 從零實作 CIFAR 分類器與 TSTR 測試框架
datasets/                — CIFAR-10/100 資料集載入器
phase1_edm_repro.py      — 正確性 gate：重現 EDM CIFAR-10 FID
run_comparison.py        — (steps × η × guidance) 聯合掃描，產出效用曲面與選擇器輸入
run_selector_signal.py   — MNIST 上 CaF 的多 seed go/no-go 訊號
run_cifar_selector.py    — CIFAR-10 上 CaF 選擇器訊號
run_guidance_study.py    — guidance 對 FID/TSTR/多樣性的取捨研究
validate_metrics.py      — 量測堆疊在真實 CIFAR-10 上的數值驗證
```

Gen-1（MNIST sandbox）：

```
ddpm.py                  — UNet、擴散排程與 DDPM/DDIM 取樣器
train.py                 — MNIST DDPM 訓練與取樣
inference.py             — 推論，輸出供 evaluate.py 使用的 dataset.pt
evaluate.py              — TSTR：以合成圖訓練 CNN，於真實 MNIST 測試集評估
fid.py                   — MNIST-FID（classifier-Fréchet distance），免 scipy
analyze_distribution.py  — 合成分佈診斷（mode collapse / canonical bias / drift）
test_classifier.py       — CNN 評估器健全性檢查
```

VAR-mini（旁支探索）：

```
var/                     — VAR-mini 套件（vqvae、transformer、sample）
train_vqvae.py           — Stage 1：多尺度殘差 VQ-VAE
train_var.py             — Stage 2：scale-wise transformer
inference_var.py         — VAR-mini 推論
```

輸出與記錄：

```
records/                 — 研究計畫與進度記錄（檔名 YYYY-MM-DD-NN_action_content）
docs/                    — 實驗結果的分析
results/                 — 各實驗的 json/csv/txt/log 輸出（不在 git 內）
checkpoints/             — 模型權重與參考統計（不在 git 內）
samples/、samples_cifar/ — 訓練過程的樣本網格（不在 git 內）
generated/               — 推論輸出的合成資料集（不在 git 內）
```

## 指標說明

- **MNIST-FID**（`fid.py`）：以 `mnist_cnn.pt` 的 penultimate 特徵計算 Fréchet distance，免 scipy。
  僅為 MNIST sandbox 內 samplers / guidance 之間的相對指標，不可與文獻的 Inception-FID 直接比較。
- **Inception-FID / clean-fid**（`fid_clean.py`）：標準 FID，用來重現公開模型數字作為量測正確性錨點。
- **FD-DINOv2 與 PRDC**（`metrics_features.py`、`metrics_prdc.py`）：Phase 1 的保真度與多樣性量測；
  PRDC 的 coverage 是 CaF 選擇器的核心訊號。

## 重現指令

Gen-1 MNIST sandbox 的 CaF 訊號：

```bash
uv run python run_selector_signal.py --seeds 0 1 2
```

EDM CIFAR-10 FID 量測錨點：

```bash
uv run python phase1_edm_repro.py --num 50000 --batch 256
```

CIFAR CFG 模型訓練：

```bash
uv run python train_cifar.py --epochs 1000 --batch-size 128
```

各腳本的完整參數以 `--help` 為準。

## 記錄與慣例

計畫與進度記錄於 `records/`（檔名 `YYYY-MM-DD-NN_action_content`），實驗結果分析於 `docs/`，開發
慣例（記錄格式、檔頭註解、語言與最小變更原則）定義於 `claude.md`。每一項工作都應先在 `records/`
建立記錄，內容涵蓋 Goal、Result、Follow-up。
