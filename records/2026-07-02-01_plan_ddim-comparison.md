# 2026-07-02 — DDIM 取樣器比較 + Guidance/多樣性/效用解耦研究（計畫）

## 背景

本專案目前是從零手刻的條件式 DDPM（Ho et al. 2020）：UNet + 自注意力 + Classifier-Free
Guidance（CFG），線性 β schedule、T=1000、ε-prediction。評估採 TSTR（Train on Synthetic,
Test on Real）分類器準確率，另有 `analyze_distribution.py` 的 mode-collapse / 多樣性診斷。
取樣器只有一種：`p_sample_loop`，固定跑滿 1000 步的 DDPM ancestral sampling。

## 核心論文

**DDIM — Song, Meng, Ermon, "Denoising Diffusion Implicit Models", ICLR 2021。**
選它的理由：

1. 擴散加速取樣的奠基之作（非最新研究），適合當研究核心。
2. 完美契合現況——DDIM **不是另一個要重訓的模型，而是一個取樣器**。它與 DDPM 有相同訓練
   目標，可直接重用現有 `ddpm_mnist.pt`，只換取樣公式。
3. 因此「DDPM vs DDIM」是**同一組權重、兩種 sampler、不同步數**的乾淨對照，無 confound、
   無需重訓。

## 統一研究問題

> 解耦「視覺保真度 (FID)」與「下游效用 (TSTR)」——取樣步數、η、以及 guidance scale 各自
> 如何以不同方式移動這兩者？

**核心假設**：FID 最佳 ≠ TSTR 最佳。guidance 調高會提升單張清晰度（FID 改善）卻壓縮類內
多樣性、走向 mode collapse，反而拉低 TSTR。

## 實驗設計

### A. DDIM 取樣器 + DDPM/DDIM 比較（`run_comparison.py`）
- DDPM ancestral：steps=1000（baseline）
- DDIM η=0：steps ∈ {1000, 100, 50, 20, 10, 5}
- η sweep（steps=50）：η ∈ {0.0, 0.5, 1.0}
- 每格量 TSTR 準確率、FID、wall-clock / imgs·s⁻¹。

### B. Guidance 強度 vs 多樣性 vs 效用（`run_guidance_study.py`，核心貢獻）
- 固定 DDIM η=0、steps=50，掃描 guidance_scale ∈ {1.0, 2.0, 3.0, 5.0, 7.0, 10.0}。
- 每個 scale 量：FID、TSTR、類內多樣性（像素/特徵 pairwise L2、pixel std）、
  max-softmax confidence、gen/real diversity ratio。
- 核心分析：FID-最佳 guidance 是否 ≠ TSTR-最佳 guidance。

## 新增評估指標：MNIST-FID

無 scipy 環境下手刻 FID（`fid.py`）。用 `mnist_cnn.pt`（真實資料訓練 judge CNN）的
penultimate 256-dim 特徵當 Inception 替代，誠實標示為 "MNIST-FID / classifier-Fréchet
distance"（samplers/guidance 間的相對指標，非標準 Inception-v3 FID）。
數學要點：FID 只需 Tr((Σ_r Σ_g)^½) = Σ√λᵢ(Σ_r Σ_g)，用 torch.linalg 特徵值求和，免 sqrtm。

## 產出檔案

- `ddpm.py`：`predict_eps`（重構）、`alphas_bar`、`ddim_timesteps`、`ddim_sample_loop`
- `inference.py`：`--sampler / --steps / --eta`
- `fid.py`（新）
- `run_comparison.py`（新）
- `run_guidance_study.py`（新）
- `records/`（本目錄）

## 狀態

- [x] DDIM 取樣器（`ddpm.py`：`predict_eps`、`alphas_bar`、`ddim_timesteps`、`ddim_sample_loop`）
- [x] inference CLI（`--sampler / --steps / --eta`）
- [x] fid.py（MNIST-FID，免 scipy）
- [x] run_comparison.py（DDPM/DDIM × 步數 × η）
- [x] run_guidance_study.py（guidance 取捨研究）
- [x] README + `.gitignore`（results/）

## 實作驗證（2026-07-02，煙霧測試）

以 GPU（CUDA）搭配既有 `ddpm_mnist.pt` / `mnist_cnn.pt` 驗證，全數通過：

- **DDIM 正確性**：η=0 同 seed 位元一致；η=1 + 完整 1000 步的 mean/std 與 DDPM ancestral
  差異 0.0000（證明數學與 ᾱ 索引正確，且 η=1 退回 DDPM）。
- **重構無副作用**：`p_sample` 抽出 `predict_eps` 後行為不變。
- **MNIST-FID 健全性**：real-vs-real（互斥切分）≈1.9；noise-vs-real≈5302；相同 stats=0.0000。
- **比較網格（quick，每類 50、TSTR 3 epochs）**：
  DDPM 1000 步 1.7 imgs/s、FID 15.9；DDIM 50 步 36 imgs/s（21×）、FID 23.7；
  DDIM 10 步 156 imgs/s（92×）、FID 24.3 → 速度·品質取捨清晰，wall-clock 隨步數近線性。
- **Guidance 研究（quick）**：guidance 1→3→7 時 div_feat 15.0→10.0→8.5、confidence
  0.984→1.0000→1.0000、FID 13.4→21.9→53.7；多樣性隨 guidance 單調崩塌（假設的機制成立）。
  此小樣本設定下 FID 與 TSTR 最佳點皆落在 guidance=1；正式全量（每類 1000 + 20 epochs）
  才能定論最佳點是否分離。

## 下一步（正式全量執行）

```bash
uv run python run_comparison.py --per-digit 1000        # 完整比較網格
uv run python run_guidance_study.py --per-digit 1000    # 完整 guidance 掃描
```

注意：1000 步的 DDPM/DDIM baseline 於 10K 張時最耗時（約 1–1.5 小時/格），只需跑一次。
之後以 `results/*.csv` 繪製「品質 vs 速度」與「guidance vs FID/TSTR/多樣性」曲線，撰寫報告。
