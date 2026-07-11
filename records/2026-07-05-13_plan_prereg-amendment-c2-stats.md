<!-- 用途:C2 裁決統計規格協定增修。將定位 v2 §4 的全網格偏相關程序落為協定的事前統計規格。須與封頂 amendment、v2 同批 commit,時間早於任何 confirmatory 資料。 -->

# Amendment:C2 裁決統計規格(全網格偏相關)

## 1. 緣由

定位調整 v2(2026-07-05-12)§4 廢除 v1 的分段裁決,改回創始 roadmap 對 C2 的原始配方(偏相關
utility ~ coverage | precision)。本增修把該裁決程序落為協定 2026-07-05-02 的事前統計規格,於任何
confirmatory 資料產生前凍結。偏相關為 CIFAR 資料存在前既有的形式,且相對分段納入更多而非更少的資料、
無可供事後劃定的分段邊界。

## 2. C2 表述(不變)

coverage 驅動下游效用,precision 不驅動。

## 3. 裁決程序(事前定死)

- 觀測單位:config。confirmatory grid 為 10 點 {1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8}(封頂 amendment
  2026-07-05-11)。每 config 的 TSTR、coverage、precision、超額 label-noise 取 fresh seeds {10, 11, 12}
  的均值,故 n=10。
- H-C2a:partial Spearman ρ(TSTR, coverage | precision, 超額 label-noise) > 0。以 permutation test
  計 p(對殘差做排列),α = 0.05。
- H-C2b:partial Spearman ρ(TSTR, precision | coverage, 超額 label-noise) 不顯著或 ≤ 0,同法。
- 通過準則:C2a 顯著為正 且 C2b 不顯著。兩者皆滿足,C2 於 CIFAR-10 尺度獲確認;未達則 C2 於
  CIFAR-10 尺度未獲確認。準則不因結果改寫。

## 4. 報告要求

- 一律報效果量(偏相關係數點估計)、信賴區間(bootstrap over seeds)與 n=10 的功效限制,不得只報
  顯著性。「C2b 不顯著」屬 absence of evidence,在 n=10 下不能當「precision 不驅動」的強證據,須明文
  標注此限制。
- 明文承認:納入低段混淆點(w1→w2,coverage 與 precision、label-noise 同向)可能降低 C2a 顯著性。
  共線性降低功效、不造成偏誤;識別由高段 coverage 與 precision 的分離提供。此為不裁剪資料的代價,接受之。

## 5. 特徵空間

coverage、precision 以 DINOv2 為準;Inception 依協定 2026-07-05-02 §4 與增修 2026-07-05-08 的交叉
裁決規則作 robustness。兩表徵對 C2 結論定性矛盾即視為結果不穩、需更多 seed,不得事後挑支持者。

## 6. 事前性與凍結

偏相關為創始 roadmap(見 2026-07-03-07 §4)對 C2 的原始裁決配方,早於一切 CIFAR 資料。本增修回到
最早登記的形式。commit 後凍結,程序與準則不再依任何後續結果調整;commit 時間須早於任何 confirmatory
資料,git 可驗。confirmatory 主假設 H1/H2/H3 仍以 2026-07-05-02 為準,本檔僅落實 C2(H1)的統計裁決形式。
