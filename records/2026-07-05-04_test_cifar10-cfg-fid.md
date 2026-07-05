<!-- 用途：記錄 Stage 1（自訓 CFG CIFAR-10 的 FID gate）的目標、結果與後續。 -->

# 2026-07-05-04 自訓 CFG CIFAR-10 FID gate

## 目標

在任何組態掃描前，確認自訓 CFG CIFAR-10 模型（checkpoints/cifar10_cfg.pt 的 EMA 權重）產出的
影像落在 repo 內自證的堪用 FID 帶，並固定量測用的 sampler (steps, eta)。自訓小模型不追 SOTA、也
不可能是 EDM 的 1.79，只需落在合理堪用帶（約 5–15）以支撐「相對」組態比較、避免 home-grown
artifact 質疑。

## 結果

以 5000 張（每類 500）、steps=50、eta=0、guidance=1.0 量測：

- clean-fid（Inception 錨點）：13.952，落在堪用帶 [5, 15] 內（近上緣）。
- FD-DINOv2：324.262（相對指標，供後續跨組態交叉比較，無絕對帶）。

5000 張相對 50k 會略高估，真 50k 應更低，故 13.95 對相對組態研究綽綽可用。量測用 sampler 固定為
(steps=50, eta=0)，掃描沿用同一組。

附帶修復：新版 scipy 移除 linalg.sqrtm 的 disp 參數，導致 vendored 的 cleanfid 崩潰；於 fid_clean.py
加相容修補（disp=False 時回傳 (sqrtm, 0.0)）。生成與 FID 流程集中於新模組 cifar_cfg_sample.py，供
Stage 1 gate 與 Stage 3 scout 共用。

判定：通過。自訓 CFG 模型 FID 合理，可作為自訓主軸的 backbone。

## 後續

- Stage 2：在真實 CIFAR-10 訓練集訓一個 ResNet-18 judge，並以真實測試集驗其準確度；為 CIFAR 重新
  校準 mechanism 的 near-boundary threshold（現預設 0.5 為 MNIST 調）。
- 掃描（scout 與多 seed）沿用固定的 (steps=50, eta=0)，guidance 為變動軸。
