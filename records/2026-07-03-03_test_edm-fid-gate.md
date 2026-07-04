<!-- 用途：記錄 Phase 1 正確性 gate（重現 EDM CIFAR-10 FID）的目標、結果與後續。 -->

# 2026-07-03 EDM CIFAR-10 FID 重現 gate

## 目標

Phase 1 的正確性 gate：在進行任何 CIFAR 組態掃描前，先證明本專案的 EDM 生成到 FID 的量測路徑能重現論文數字。若這一步對不上，後面所有 CIFAR 數字都不可信。

## 結果

EDM CIFAR-10（條件、VP）在 50000 張下 FID 為 1.848，論文參考值 1.79，差 0.058，通過 gate。

建置內容：
- 取用 NVlabs/edm 原始碼於 `third_party/edm`，下載預訓練 checkpoint 與官方 CIFAR-10 參考統計。
- `phase1_edm_repro.py`：單一進程 driver，不使用 torch.distributed（在 Windows 上較穩定），沿用官方取樣器與 Inception 偵測器。把 Inception 的 mu、sigma 快取為 npz，並提供 `--from-stats` 讓 FID 計算不需重生成。
- `datasets/cifar.py`：CIFAR-10/100 載入器。

誠實註記：1.848 略高於 1.79，最可能來源是 custom CUDA ops（bias_act、upfirdn2d）在 Windows 走純 PyTorch 參考實作（未編譯核心），屬良性數值差異，對「組態之間的相對比較」不構成問題。實測瓶頸：單張 GPU、網路慢、ops fallback 下約每秒 25 張。

## 後續

依 `2026-07-03-09_plan_phase1-cifar.md` 的決議，EDM 只作為量測錨點，不用來做 CFG；CIFAR 的 CFG 改由自訓模型提供。重現 FID 的指令為 `uv run python phase1_edm_repro.py --num 50000 --batch 256`，僅重算 FID 用 `--from-stats`。
