# 2026-07-03 — 完整測試結果與分析

完整跑 `run_comparison.py --per-digit 1000` 與 `run_guidance_study.py --per-digit 1000`
（10K 張/組、TSTR 20 epochs、judge=`mnist_cnn.pt`、checkpoint=`ddpm_mnist.pt` @ 05f0d3d，
RTX 5070）。比較網格約 4.7 小時、guidance 掃描約 30 分鐘。

## A. 取樣器比較（DDPM vs DDIM，guidance=3.0）

| config | steps | η | imgs/s | TSTR% | FID |
|---|---|---|---|---|---|
| ddpm_s1000 | 1000 | 0 | 1.3 | 95.15 | **14.45** |
| ddim_s1000 | 1000 | 0 | 1.4 | 95.06 | 18.92 |
| ddim_s100 | 100 | 0 | 13.0 | 95.40 | 19.55 |
| ddim_s50 | 50 | 0 | 25.9 | **95.64** | 21.00 |
| ddim_s20 | 20 | 0 | 65.1 | 95.11 | 21.24 |
| ddim_s10 | 10 | 0 | 129.8 | 94.71 | 23.00 |
| ddim_s5 | 5 | 0 | 261.9 | 91.61 | 23.58 |
| ddim_s50 | 50 | 0.5 | 26.5 | 94.92 | 19.07 |
| ddim_s50 | 50 | 1.0 | 37.1 | 94.89 | 17.56 |

**發現**
1. **DDIM 幾乎零代價加速**：50 步比 DDPM 快 ~20×，TSTR 反而更高（95.64 ≥ 95.15）；
   20 步快 ~50×，仍 95.11%。品質懸崖在 5 步才出現（91.61%）。
2. **FID ≠ TSTR（取樣器層級）**：DDPM 1000 步 FID 最佳（14.45）但 TSTR 非最佳；
   DDIM 50 步 FID 較差（21.0）卻 TSTR 最佳（95.64）。
3. **η（隨機性）改善 FID 但不改善 TSTR**：50 步下 η 0→0.5→1 使 FID 21.0→19.1→17.6
   （逼近 DDPM），但 TSTR 微降。→ 下游效用偏好 deterministic（η=0）。

## B. Guidance 取捨研究（DDIM η=0, steps=50，核心貢獻）

| guidance | FID | TSTR% | div_feat | div_ratio | confidence |
|---|---|---|---|---|---|
| 1 | 10.19 | **97.22** | 14.86 | 1.047 | 0.982 |
| 2 | **8.88** | 95.92 | 11.37 | 0.801 | 0.9995 |
| 3 | 20.52 | 95.33 | 10.30 | 0.726 | 0.9999 |
| 5 | 38.74 | 91.86 | 9.19 | 0.648 | 1.0000 |
| 7 | 52.18 | 87.68 | 8.50 | 0.599 | 1.0000 |
| 10 | 69.17 | 79.08 | 7.78 | 0.548 | 0.9999 |

（真實參考：div_feat=14.18、confidence=0.9931）

**發現（核心假設成立）**
1. **FID-最佳 guidance = 2.0 ≠ TSTR-最佳 guidance = 1.0**（工具自動判定 differ=true）。
   視覺保真度與下游效用的最佳點確實分離。
2. **多樣性隨 guidance 單調崩塌**：div_feat 14.86→7.78、div_ratio 1.05→0.55
   （g≥7 跌破 0.6 mode-collapse 警戒線），confidence 於 g=2 起飽和到 ~1.0。
3. **g>2 後 FID 與 TSTR 雙雙惡化**；g=10 時 TSTR 崩到 79.08%。
4. **可行動結論**：專案預設 guidance=3.0 對兩個指標皆非最優。**降到 g=1 使 TSTR
   達 97.22%**（原 README 在 DDPM+g3.0 為 95.30%），逼近真實資料上限（~98–99%）。

## C. 跨面向洞見

預設組態（DDPM 1000 步、guidance 3.0）在「下游效用」上被 Pareto 支配：
**DDIM 50 步 + guidance 1.0 同時快 ~20× 且 TSTR 更高（97.22%）**。只有純視覺保真度
（FID）偏好 DDPM / 低 η / g=2。機制：guidance 銳化單張（低 g 時小幅助 FID）但壓縮類內
多樣性，而 TSTR 依賴多樣性 → 兩者最佳點分離。

## 下一步（見 2026-07-03 規劃）
- guidance × steps 二維聯合掃描，畫出完整 Pareto 前緣。
- 加入 precision/recall 或 density/coverage 指標，把「保真 vs 多樣」拆開，
  嚴謹解釋 FID/TSTR 為何分離。
- per-class TSTR + 混淆矩陣（低 guidance 是否特別救回弱類如數字 8）。
- 資料增強角度：TS+TR 混訓與 low-data regime（g=1 已逼近上限，適合測增益）。
