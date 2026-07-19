<!-- 用途：程式地圖。列出 src/ 下各類別與每個 .py 的一行用途，作為導覽索引。 -->

# 程式地圖

所有 Python 腳本依類別收於 `src/`。扁平 import（`import metrics_features` 等）由 `src/_pathfix.py`
墊片維持——它在被 import 時把專案根與各 `src/` 子資料夾補回 `sys.path`，故 import 語句不變。
腳本一律**從專案根目錄**以 `python src/<類別>/<檔>.py` 執行（相對路徑參數如 `--output results/...`
以專案根為 cwd）。既有套件 `datasets/`、`var/` 與工具 `tools/` 留在根目錄。

## src/core/ — 被 import 的共用函式庫（Gen-2 研究主線）

- `selector.py` — CaF（Coverage-at-Fidelity）：免訓練的取樣組態選擇器（argmax coverage s.t. precision ≥ τ）。
- `mechanism.py` — 機制分析（C2）：guidance 是否抽乾決策邊界附近的樣本支持度；label-noise 對照。
- `metrics_prdc.py` — PRDC（Precision/Recall/Density/Coverage），純 torch 實作。
- `metrics_features.py` — 表徵式生成指標（DINOv2 特徵、FD-DINOv2）。
- `fid_clean.py` — 標準 Inception-FID（clean-fid），量測正確性錨點。
- `chamfer.py` — 簡化版 Chamfer guidance 基線，供 matched-budget 對決（非官方實作）。
- `cifar_classifier.py` — 從零實作的 CIFAR 分類器與 TSTR 測試框架。
- `cifar_judge.py` — 真實 CIFAR-10/100 上訓練 ResNet-18 judge，校準 near-boundary threshold。
- `cifar_cfg_sample.py` — 自訓 CFG CIFAR 模型的平衡生成與 FID gate。

## src/experiments/ — Gen-2 執行腳本（driver／gate／重生成／訓練）

driver（confirmatory／scout／裁決）：

- `run_comparison.py` — (steps × η × guidance) 聯合掃描，效用曲面與選擇器輸入（MNIST 尺度）。
- `run_selector_signal.py` — MNIST 上 CaF 的多 seed go/no-go 訊號。
- `run_cifar_selector.py` — CIFAR-10 上 CaF 選擇器訊號（Phase 1 主體）。
- `run_cifar_cfg_scout.py` — 1-seed 寬 grid scout，定位 coverage 崩點以凍結 confirmatory grid。
- `run_cifar_cfg_upper_scout.py` — 上緣 coverage-only scout。
- `run_cifar_cfg_multiseed.py` — confirmatory 主 driver（多 seed 全量主結果）。
- `run_c0_recall_density.py` — C0：recall/density 能否打破 Pareto 支配（CaF-v2 第三訊號）。
- `run_c2_partial.py` — C2 全網格偏相關裁決。
- `run_c2c3c5_intervention.py` — C2/C3/C5：near-boundary 與 coverage 的介入式證據。
- `run_c4_variance.py` — C4：變異分解（σ_cls 對 σ_gen）。
- `run_c6_fidmin_duel.py` — C6：matched-budget FID-min 對決 CaF。
- `run_c7_probe_fid_stability.py` — C7：small-probe FID 排序穩定性。
- `run_flip_earlywarning.py` — CIFAR-10 難子集 coverage 主導鬆動的早期預警。
- `run_guidance_study.py` — guidance 對 FID/TSTR/多樣性的取捨研究。
- `run_p0_probe.py` — P0 探針：單 cell 生成決定性隔離＋整鏈 scalar 重現 gate。
- `run_p1_streaming.py` — P1×C1：streaming 持久化與逐 config 對帳。
- `run_d3_observables.py` — D3：CIFAR-100 機制複製的三觀察量。
- `run_cifar100_d3_intervention.py` — D3 介入臂：CIFAR-100 coverage-matched pruning。
- `run_cifar100_fd_dinov2.py` — CIFAR-100 per-config FD-DINOv2（補 which-FID 的 DINOv2 空間）。
- `run_cifar100_h3_duel.py` — H3 三臂 matched-budget 對決（FID-min／CaF-v2／Chamfer）。

gate／重生成／訓練：

- `phase1_edm_repro.py` — 正確性 gate：重現 EDM CIFAR-10 條件式 FID（約 1.79）。
- `validate_metrics.py` — 量測堆疊在真實 CIFAR-10 上的數值驗證。
- `cifar100_base_gate.py` — CIFAR-100 base-model FID gate（D 包第一閘）。
- `regen_cifar100_cells.py` — 重生成 CIFAR-100 confirmatory 的兩個 cell，供 D3 介入臂。
- `train_cifar.py` — 在 CIFAR-10/100 上訓練類別條件、支援 CFG 的擴散模型。

## src/gen1_mnist/ — Gen-1 MNIST sandbox（已完成）

- `ddpm.py` — UNet 噪音預測網路與擴散排程（DDPM/DDIM 取樣器）。
- `train.py` — MNIST DDPM 的訓練與取樣。
- `inference.py` — 以訓練好的 DDPM 生成 MNIST，輸出 evaluate.py 用的 dataset.pt。
- `evaluate.py` — TSTR：以合成圖訓 CNN，於真實 MNIST 測試集評估。
- `fid.py` — MNIST-FID（classifier-Fréchet distance），免 scipy。
- `analyze_distribution.py` — 合成分佈診斷（mode collapse / drift）。
- `test_classifier.py` — CNN 評估器健全性檢查。

## src/var_mini/ — VAR-mini 旁支（停放，僅 smoke，不列入論文）

- `train_vqvae.py` — Stage 1：多尺度殘差 VQ-VAE。
- `train_var.py` — Stage 2：scale-wise transformer。
- `inference_var.py` — VAR-mini 推論（輸出與 TSTR pipeline 同格式）。

## src/figures/ — 論文製圖

- `make_thesis_figures.py` — 由 `results/*.json` 產生碩論所需的 6 張發表級圖（純畫既有數字）。

## src/_pathfix.py

- sys.path 墊片，維持 src/ 下腳本的扁平 import，並輸出專案根路徑 `ROOT`。
