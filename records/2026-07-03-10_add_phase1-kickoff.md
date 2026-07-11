# 2026-07-03 — Phase 1 Kickoff（依 phase1-plan 執行）

依 `records/2026-07-03-09_plan_phase1-cifar.md` 啟動 Phase 1。本則記錄本階段已完成的建置、決策與
GPU-serialized 的後續啟動順序。

## 本次啟動的背景工作
- **multi-seed Gate A** 執行中（`run_selector_signal.py --seeds 0 1 2 --per-digit 1000`，
  ~3h）：替 sandbox 三主張補多 seed CI。完成後 GPU 才空出來跑後續。

## Backbone 子決策（plan 已定「自訓 CFG-capable CIFAR」，此處定實作路徑）
評估 `FutureXiang/Diffusion`（plan 優先評估對象）：**無可下載 checkpoint**、4×GPU Linux repo、
~14h/4×3080ti、FID 2–3.5、支援 CFG+DDIM+EDM。→ 無論如何都要自訓。

**決定：擴充我們自己的 `ddpm.py` 到 CIFAR**，而非叉 FutureXiang。理由：
1. 我們的 UNet 與 DiffusionSchedule **本就 resolution/channel-agnostic**（僅 docstring 寫 28×28，
   運算是通用的），CIFAR 只需 `UNet(in_channels=3, base_channels=128, channel_mults=(1,2,4))`
   於 32→16→8。
2. **CFG + DDIM(η) 旋鈕已在 sandbox 驗證**，與整套分析 stack（`metrics_prdc`/`selector`/
   `mechanism`/`run_comparison` 3D）**無縫相接**——叉 FutureXiang 反而要把整套 re-plumb。
3. Windows / 單卡友善。FutureXiang 的多 GPU DDP + 其 config 系統整合成本高。
4. plan 接受相對 FID ~5–10，並以預訓練 EDM 的 1.79 當量測錨點防「home-grown artifact」質疑。

## 已建置（本 session）
- `train_cifar.py`：CFG-capable CIFAR 訓練器。reuse UNet+DiffusionSchedule+CFG+DDIM(η)；
  新增 3-channel、flip 增強、**EMA(0.9999)**（CIFAR FID 關鍵）、**週期性 checkpoint**
  （早期 checkpoint 兼作 autoguidance 弱模型＝C3b 第二 guidance 方法）。
  **CPU shape-smoke 通過**：47.9M params、train step、DDIM 取樣 shape 皆正確。
- `cifar_data.py`：CIFAR-10/100 loaders（避免命名 `datasets/` 以免遮蔽 HF `datasets`）。
- `metrics_features.py`：DINOv2 特徵 + FD（reuse `fid.frechet_distance`）+ PRDC；預留第二表徵
  交叉驗證（破 DINOv2 雙重使用循環）。
- `fid_clean.py`：clean-fid 標準 Inception-FID 錨點（重現公開 FID 用）。
- 依賴確認：`clean-fid` 已裝；DINOv2 `vitb14` 權重+repo 已下載；CIFAR-10/100 已下載。

## GPU-serialized 後續啟動順序（Gate A 結束、GPU 空出後）
1. **量測堆疊數值驗證**（最便宜 gate）：real-vs-real clean-fid ≈ 小、FD-DINOv2 ≈ 小、PRDC ≈ 高；
   跑一個公開模型（預訓練 EDM 或退而求其次 diffusers `google/ddpm-cifar10-32`）重現其公開 FID
   作 correctness gate。
2. **啟動 `train_cifar.py` ~2 天背景訓練**（cifar10 先行；存早期 checkpoint 供 autoguidance）。
3. 訓練期間並行準備：`run_comparison.py` 接 CIFAR 生成 + CIFAR ResNet TSTR + clean-fid/FD-DINOv2/
   PRDC；`mechanism.py` 的 coverage 受控 margin 條件分析（介入式）。
4. 模型堪用後：**CIFAR-10 複製 C1** → **CIFAR-100 拿 C2 機制證據（最重要）** → **CaF vs Chamfer 對決**。

## 執行狀態（session 末）
- **Gate A 多 seed：決定性通過**（regret 0×3 seeds、100% oracle-hit、機制單調）——見 phase0 log。
- **量測堆疊數值驗證：核心 PASS**。FD-DINOv2 real-vs-real 84.2 vs noise 3891（46×）；
  PRDC real-vs-real precision/recall/coverage = 0.90/0.90/0.97。clean-fid（Inception 錨點）
  遇 **Windows multiprocessing pickling bug**（`make_resizer.<locals>.func`）→ 待用
  `num_workers=0` 修，非 CaF 關鍵路徑，延到 model-anchor 前處理。`validate_metrics.py` 已落地。
- **CFG CIFAR-10 訓練：已啟動、健康**。`train_cifar.py --epochs 1000 --batch-size 128`，
  47.9M params、9.4/12.2GB、3.70 it/s（~105s/epoch → ~1.2 天跑完 1000 epoch）；epoch1 loss
  1.05→0.085。每 10 epoch 存 sample grid、每 25 epoch 存 numbered checkpoint（供 autoguidance）。
- **待辦（訓練期間/之後）**：修 clean-fid Windows（num_workers=0）跑 model-anchor FID；建 CIFAR
  ResNet 下游（TSTR）；FID 堪用後 → CIFAR-10 C1 掃描 → CIFAR-100 C2 → Chamfer 對決。

## 誠實現況
Phase 1 本質是**多天、GPU 序列化**的工程：關鍵路徑已 de-risk（backbone 可跑、量測依賴齊備、
harness 就緒），但「可發表證據」仍在 2 天訓練 + CIFAR 掃描之後。本 session 完成的是 Phase 1 的
**建置與 de-risk**，非完整結果。

## Watch-list（延續 plan §5）
η-null 是否複製到 CIFAR（要驗不要假設）；難度上升時 CaF regret 是否擴大；標籤噪音在難集是否使
coverage 主導翻轉（CIFAR-100 硬門檻）；τ 循環 / DINOv2 雙重使用（實作要求不變）。
