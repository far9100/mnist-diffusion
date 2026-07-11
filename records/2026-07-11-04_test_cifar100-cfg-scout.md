<!-- 用途：記錄 D10 第三閘——CIFAR-100 1-seed 寬 grid scout 的目標、結果與後續（含 confirmatory grid 提案）。 -->

# 2026-07-11-04 CIFAR-100 寬 grid scout（D10 第三閘）

## Goal

依 D 包 `records/2026-07-09-13` D10 第三閘：在自訓 CFG CIFAR-100 主軸上，以 1-seed 寬 grid 定位 coverage
崩點與甜蜜點，用來一次定死 confirmatory grid。依 D10，scout 僅定網格，讀數不回饋判準（不作裁決輸入）。

設定：w∈{1,1.5,2,3,5,8}、steps=50、eta=0、seed=0、per_class=200（scout 成本口徑，較 confirmatory 粗）、
DINOv2 主特徵 + Inception 交叉檢查、judge `checkpoints/cifar100_judge.pt` + threshold 0.3622。

本閘為作者 STOP 點：跑完提報網格提案，不自行凍結網格、不自行進 confirmatory。

## Result

| w | prec(DINO) | cov(DINO) | cov(Incep) | TSTR% | label_noise | near_bnd |
|---|---|---|---|---|---|---|
| 1.0 | 0.819 | 0.554 | 0.857 | 49.45 | 0.342 | 0.258 |
| 1.5 | 0.860 | 0.726 | 0.938 | 48.07 | 0.134 | 0.115 |
| 2.0 | 0.880 | 0.781 | 0.936 | 41.38 | 0.069 | 0.065 |
| 3.0 | 0.890 | 0.770 | 0.890 | 34.69 | 0.032 | 0.037 |
| 5.0 | 0.877 | 0.663 | 0.778 | 20.42 | 0.030 | 0.035 |
| 8.0 | 0.851 | 0.542 | 0.662 | 12.60 | 0.068 | 0.076 |

觀察（scout 敘事，非裁決）：

- coverage 先升後崩：DINOv2 峰在 w2（0.781），Inception 峰在 w1.5–2（0.938），崩點（峰後首個跌破 90%
  峰值）為 w5。兩表徵同形，grid 崩點定位不依賴特徵空間。
- TSTR 於 w1 最高（49.45），其後單調下降至 w8 的 12.60，本 1-seed scout 未見內部最優；災難性非單調明顯。
- near-boundary 隨 guidance 先枯竭（w1 0.258 → w5 0.035）後於 w8 回升（0.076）；label_noise 於 w1 高達
  0.342（大量離類樣本）、中段降至 ~0.03、高段回升，與 CIFAR-10 雙段結構同性質。
- judge/threshold 運作正常：near-boundary 隨 guidance 的變化合理。

時序：2026-07-10 18:25:31Z 起、21:02:51Z 止（約 2 小時 37 分）。落盤 `results/cifar100_cfg_scout.json`
（含 §5.2 metadata）。

## confirmatory grid 提案（待作者於網格凍結 amendment 裁定）

動作集中在 w∈[1,5]：低段 coverage 上升、TSTR 高；中段 coverage 峰（w2）；w5 起 coverage 與 TSTR 同崩；
w6–8 為深崩尾。提議沿用 CIFAR-10 confirmatory 之凍結 grid **{1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8}**（10 點、
封頂 w8，依 `records/2026-07-05-11` 之 CFG 實用範圍封頂），理由：(a) 覆蓋低段密（1,1.5,2,2.5,3）解析甜蜜點與
coverage 峰、涵蓋崩尾（4–8）；(b) 與 CIFAR-10 同 grid 便於跨資料集直接對照。

備註：TSTR 峰落在下界 w1，若作者要探 w<1（低於純條件、往無條件）需另議；D 包封頂於 CFG 實用範圍，
本提案維持 [1,8]。

## Follow-up

- STOP 等作者裁定 confirmatory grid，再寫網格凍結 amendment（D10 第四閘，pre-registration 步驟）。
- 網格凍結後才進 confirmatory（D10 末閘：8 seed × 5 rep、per_class 待定、量 precision/coverage/recall/
  TSTR/near-boundary/label-noise、FD-DINOv2；judge 與 threshold 0.3622 為凍結儀器）。
- scout per_class=200 較 confirmatory 粗；grid 形狀與交叉檢查一致，網格定位足夠，數字不作裁決。
- 工具：`run_cifar_cfg_scout.py` 一般化為 `--dataset`、自 judge json 讀 threshold、修正崩點偵測為峰後判定
  （`records/2026-07-05-06` 更正之採納）。
