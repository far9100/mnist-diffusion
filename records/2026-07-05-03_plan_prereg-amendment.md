<!-- 用途：pre-register 協定增修，於自訓 CFG 首掃前一次寫死（標籤噪音必要、precision 第一等、C2 低段 exploratory、grid 定死規則）。 -->

# 2026-07-05-03 預先登記協定增修

## 目標

早期預警（EDM-CFG 代理）顯示 CIFAR-10 上 coverage 在 w∈[1,3] 平坦、TSTR 有內部最優 w≈1.5。為在
自訓 CFG 主軸上正確檢驗此現象、且不落入 HARKing，於自訓模型首次掃描前，對
records/2026-07-05-02_plan_prereg-protocol.md 一次寫死以下增修（pre-data）。這些調整基於 proxy 的
exploratory 觀察，但登記時點在自訓模型任何掃描之前，故對自訓軸仍為 pre-data。

## 增修內容（鎖定）

1. 加寬 guidance grid：早期預警證實 w∈[1,3] 動不了 coverage，需掃到足以讓 coverage 崩的高段。最終
   grid 於 1-seed scout（見 2026-07-05-06）後一次定死並回填本檔，之後不得依結果微調。

   **鎖定的最終 grid（2026-07-05，據 scout 結果）：w ∈ {1, 2, 3, 4, 5, 8}。** 理由：scout 顯示 TSTR
   甜蜜點在 w≈3、coverage 在 w5,8 崩；丟冗餘的 1.5、加 w=4 以解析甜蜜點→崩的轉折，涵蓋低段
   fidelity 故事與高段 coverage 崩。Stage 4 多 seed 全量固定用此 grid，不再更動。

2. 標籤噪音由「後續」升為**必要儀器（第一等量測）**：低段 TSTR 可能受離類/模糊樣本限制，雙力拉鋸
   的解釋需要它。每個 config 皆量 label_noise_frac 與 near-boundary。

3. **precision 補為第一等量測**：撐「雙力拉鋸」的 fidelity 臂——需示範 precision 於低段（w=1→1.5）
   上升以解釋 TSTR 上升，coverage 於高段崩以解釋 TSTR 下降。precision 不得再於 driver 中丟棄。

4. C2 低段修訂（「低段 coverage 飽和、驅動效用的是 fidelity 而非 coverage」）定位為 **exploratory**：
   不得用促成它的 proxy 或同一批 scout 資料反過來確認；C2 的 confirmatory 裁決只在 coverage 真正
   變動的高 guidance 段下，並以獨立資料（多 seed 全量）為準。

5. 量測用 sampler 固定：自訓模型以 DDIM(η) 取樣，FID 量測與組態掃描使用同一組 (steps, η)，於 Stage 1
   固定並記錄。

## 後續

- Stage 1 固定 (steps, η) 並確立自訓 FID 堪用帶。
- Stage 3 scout 後，最終加寬 grid 回填本檔第 1 點並鎖定。
- confirmatory 主張（H1/H2/H3）仍以 records/2026-07-05-02 為準，本檔僅新增量測與定位，不放寬其門檻。
