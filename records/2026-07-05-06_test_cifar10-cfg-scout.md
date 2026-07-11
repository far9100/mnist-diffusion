<!-- 用途：記錄 Stage 3（自訓 CFG CIFAR-10 寬 grid scout）的目標、結果與後續。 -->

# 2026-07-05-06 自訓 CFG CIFAR-10 寬 grid scout

## 目標

在自訓 CFG 主軸上，以 1-seed 寬 grid（w∈{1,1.5,2,3,5,8}、固定 steps=50 eta=0）量 precision +
coverage（DINOv2 主、Inception 交叉）+ TSTR + 標籤噪音/near-boundary（Stage 2 judge、threshold
0.9525），以（a）確認 EDM proxy 的 w≈1.5 甜蜜點是否遷移、（b）定位 coverage 崩點、（c）驗
judge/threshold，並據此定死最終 grid。

## 結果

| w | precision | cov(DINOv2) | cov(Incep) | TSTR% | label_noise | near_bnd |
|---|---|---|---|---|---|---|
| 1.0 | 0.830 | 0.713 | 0.904 | 41.07 | 0.120 | 0.267 |
| 1.5 | 0.878 | 0.808 | 0.947 | 41.30 | 0.030 | 0.117 |
| 2.0 | 0.886 | 0.831 | 0.948 | 41.23 | 0.015 | 0.069 |
| 3.0 | 0.902 | 0.836 | 0.923 | 42.94 | 0.008 | 0.040 |
| 5.0 | 0.905 | 0.756 | 0.868 | 34.86 | 0.005 | 0.035 |
| 8.0 | 0.880 | 0.607 | 0.753 | 29.50 | 0.009 | 0.056 |

判讀：

- 甜蜜點遷移但存在：TSTR 峰在 w≈3（非 proxy 的 1.5）。內部最優在主軸確認，位置移高，與真 CFG
  （label-dropout）機制不同於 EDM 代理一致。
- 掃寬 grid 成功：coverage 先升（w1→3）再崩（w3→8，DINOv2 0.836→0.607、Inception 0.923→0.753）。
  早期預警 w∈[1,3] 的平坦是範圍太低所致；延伸到 5、8 才見崩。
- 雙力拉鋸被量到（precision 為關鍵）：低段（w1→3）precision↑、coverage↑、label_noise↓（0.12→0.008），
  TSTR 平/微升，效用受 fidelity + 標籤噪音限制而非 coverage（支持 C2 低段修訂，定位 exploratory）；
  高段（w3→8）coverage 崩、precision 高原、TSTR 急跌，為 coverage 驅動效用（支持 C2 在 coverage
  真正變動的高段成立）。
- 標籤噪音是真力：w=1 有 12% 離類樣本，真實競爭機制，印證標籤噪音為必要儀器。
- 交叉檢查一致：DINOv2 與 Inception coverage 同形（先升後崩），結論不依賴特徵空間。
- judge/threshold 可用：near_boundary 隨 guidance 枯竭（0.267→0.035）合理。

更正：腳本 verdict 自動報「coverage 崩點=1.0」，是偵測邏輯對非單調曲線的誤判（它取第一個低於峰值
90% 的點，落在峰前的 w=1）。真正的崩在 w=5、8；後續版本應改為「峰值後首個跌破 90% 峰值的點」。

## 後續

- 定死最終 grid：動作在甜蜜點 w≈3 至崩點 w5,8。提議 {1, 2, 3, 4, 5, 8}（丟冗餘的 1.5、加 w=4 解析
  甜蜜點→崩的轉折），待作者確認後回填協定增修（2026-07-05-03）。
- Stage 4：於定死 grid 上跑 ≥3 seed，precision/coverage/TSTR/標籤噪音齊備、帶 CI、完整網格不剪枝。
- C2 裁決：低段（fidelity + 標籤噪音）標 exploratory；高段（coverage 驅動）以多 seed 全量為 confirmatory
  依據。同步更新 docs/results_analysis.md 與 README 的機制定位。
