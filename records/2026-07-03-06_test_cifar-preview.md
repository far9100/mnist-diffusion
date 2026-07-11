<!-- 用途：記錄 CIFAR-10 CaF 快速預覽的初步發現（未定案）與後續。 -->

# 2026-07-03 CIFAR-10 CaF 預覽發現（初步）

## 目標

快速預覽 CaF 是否把 MNIST 的結果複製到 CIFAR-10。此為初步，設定刻意求快：1 seed、每類 500 張、生成步數 8、Inception 特徵做 PRDC。

## 結果

CIFAR-10 的樣貌比 MNIST 更複雜，未定案。

CFG guidance 軸 w 屬於 {1,1.5,2,3}：

| w | precision | coverage | recall | TSTR% |
|---|---|---|---|---|
| 1.0 | 0.767 | 0.825 | 0.720 | 47.50 |
| 1.5 | 0.807 | 0.842 | 0.689 | 49.74（oracle 最佳） |
| 2.0 | 0.819 | 0.844 | 0.656 | 49.23 |
| 3.0 | 0.803 | 0.807 | 0.571 | 48.06 |

CaF（auto-τ 約 0.697）選中 w2（precision 過門檻中 coverage 最大），oracle TSTR 最佳為 w1.5；regret 0.51pp、rank 2/4、top-3 命中。TSTR 偏低（約 48 到 50%）是因為只有 5000 張合成、且用 8 步，屬弱訊號預覽。

與 MNIST 的差別：MNIST 上 coverage 隨 guidance 單調下降、TSTR 在最低 guidance 最佳；CIFAR-10 上 coverage 非單調（在 w2 達峰）、TSTR 在中間 guidance（w1.5）達峰，即效用有內部最優，而非「越低越好」。

兩種解讀，需全品質才能分辨：
1. 真實 CIFAR 行為（可能，且對論文是更好的故事）。CIFAR 不像 MNIST 近乎可分，適度 guidance 先提升保真與代表性使 TSTR 上升，只有過高 guidance 才崩多樣性使 TSTR 下降。內部最優更貼近文獻，選擇器的任務變成找甜蜜點，CaF 落在前二名（regret 0.5pp）而非精確命中，是誠實而細緻的結果。
2. 8 步 artifact。為求快把生成步數由 18 降到 8，低步數對 w=1（無 CFG）端傷害最大，可能人為壓低低 guidance 的 TSTR、製造出內部最優。必須用 18 步排除後才能下結論。

對脊椎的意涵（若內部最優在全品質下成立）：MNIST 的「coverage 單調驅動效用」標題會弱化，CIFAR 故事變成「效用對 guidance 非單調、需找甜蜜點」，CaF 的價值是免逐組態 TSTR 就到達，且 regret 小。這是更難也更有說服力的選擇器故事。「precision 不追蹤效用」需在 CIFAR 重驗。

## 後續

全品質 1-seed（18 步、每類 1000）已排入，但因 GPU 被平行的 CFG 訓練佔用而中止，待 GPU 空出再驗。若內部最優成立，擴到 3 個以上 seed 與 CIFAR-100，並重跑機制分析（near-boundary 在 CIFAR-100 應不飽和）。相關輸出：`results/cifar_selector_preview.json`。
