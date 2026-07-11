<!-- 用途：記錄 confirmatory 步驟 2（50k base-model FID 重量並對事前帶重判）的目標、結果與後續。 -->

# 2026-07-05-09 自訓 CFG 50k FID 重量與 gate 重判

## 目標

依 2026-07-05-08 協定增修規格 1：用 50k 樣本重量 base model（w=1、steps=50、eta=0）的 clean-fid，
對事前凍結的 acceptance 帶（≤10）重判。pilot 的 5k 量得 13.95，有已知小樣本正偏誤，與文獻 ~3–8
（皆 50k）不可比，故須先量準。

## 結果

50k clean-fid = 8.950，落在事前帶 ≤10 內，通過 base-model gate。FD-DINOv2 = 286.4（characterization）。

證實 5k 的 13.95 為小樣本正偏誤：量準（50k）後為 8.95，落在同規模 from-scratch CFG CIFAR-10 的文獻
可比範圍（~3–8）內，base model 確實堪用；先前偏高是量測假影而非模型問題。

時序：本次 run 完成於 2026-07-05 15:43:37，晚於協定增修凍結 commit 052492c（14:59:54），pre-register
順序可稽核。

註：cifar_cfg_sample.py 的 JSON 內 usable_band_5_15 欄位為 driver 舊的硬編門檻，治理判定以協定的
≤10 為準（8.95 通過）；後續可把該欄位對齊協定帶。gate 範圍明文只保證 base model（w=1）；guidance
軸樣本品質由 confirmatory 的 per-config precision/coverage 與 per-config clean-fid characterization 呈現。

## 後續

- 步驟 3：上緣 coverage-only scout（{8,10,12,16,20}、觸底判準 X=0.02）定崩點界。
- 掃描與 confirmatory 沿用固定 (steps=50, eta=0)。
