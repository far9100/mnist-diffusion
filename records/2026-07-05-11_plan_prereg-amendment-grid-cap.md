<!-- 用途:confirmatory grid 上緣封頂 amendment。解除上緣 scout bottomed:false 觸發的再登記分支。須先於定位文件 v2 或與其同批 commit,時間早於任何 confirmatory 資料。落庫時指派 records 編號。 -->

# Amendment:confirmatory grid 上緣封頂

## 1. 緣由

上緣 coverage-only scout(1 seed,w ∈ {8, 10, 12, 16, 20})結果:coverage 0.533 → 0.259,所有相鄰 |Δcoverage| ≥ 0.02(最小 0.037),單調下降,依事前判準 X=0.02 未觸底。依前次增修的分支規定,未觸底不得臨場續掃,須以新的 pre-data record 裁決後續。本 amendment 即該 record。

## 2. 裁決

confirmatory grid 上緣封頂於 w = 8。不續掃、不追 coverage 觸底點。

grid 依既定中段規則與本封頂具體化並凍結:
- Δw=0.5 於 [1,3]:{1, 1.5, 2, 2.5, 3}
- Δw=1 於 [3,8]:{4, 5, 6, 7, 8}
- 完整點集:{1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8},共 10 點。
- 乘 fresh seeds {10, 11, 12},共 30 configs。開跑前依協定先呈報 GPU 時間估計。

## 3. 封頂理由(先驗,與 scout 曲線無關)

- CFG 在影像生成文獻的常用工作範圍為 w ∈ [1, 8] 量級;Stable Diffusion 預設 7.5;條件生成的 guidance 掃描多止於此量級。
- 合成資料訓練的應用偏好低 guidance(Fan 等固定 low-CFG 配方即在此範圍)。
- w > 8 的樣本無資料生成的實用情境;本研究主張的範圍是實用 guidance 軸(見定位文件調整一)。
- 高 w 樣本品質的劣化由既定的 per-config characterization FID 透明呈現,不需以 gate 或掃描覆蓋病理區。

明文排除的理由:續掃的算力成本、coverage 降幅趨緩。此二者不得作為封頂依據。

## 4. 原分支的處置

前次增修「未觸底則提高 w_max 再掃」分支的目的,是避免崩點落在 grid 外造成漏看。scout 結果顯示崩勢非點狀事件,而是貫穿並超出實用範圍的單調流失;追觸底會把掃描推入 w ≈ 30–40 的區域,該區樣本塌向類原型、無實用意義,與主張範圍無關。本 amendment 以先驗封頂取代該分支。崩勢本身以 scout 讀數作描述性報告(見定位文件調整四),confirmatory 不重跑 w > 8 區段。

## 5. 自我測試

封頂理由在看到 scout 曲線之前是否成立:成立。CFG 實用範圍與資料生成應用的低 guidance 偏好,均為文獻先驗,不依賴本次 scout 的任何讀數。

## 6. 凍結與時序

本 amendment commit 後凍結,grid 點集不再依任何後續結果調整。commit 時間須早於任何 confirmatory 資料,git 可驗。定位文件 v2 的引用以本 amendment 的實際 records 編號回填。

## 7. confirmatory 取樣設定凍結(steps、η)

confirmatory 生成沿用 base-model FID gate 與 pilot 的取樣設定:steps=50、η=0(DDIM deterministic),
對 grid 全 10 點與 fresh seeds {10, 11, 12} 一致套用。thesis 收緊至 CFG guidance 軸後,(steps, η) 為
背景變數,於此與 grid、seeds 同批事前釘死,之後不依結果調整。η 在 CIFAR 的行為屬 exploratory
(CIFAR-100 另做 spot-check,見 2026-07-05-01 §2.5),不在本 confirmatory 宣稱。此設定與 records
2026-07-05-09(50k FID gate,w=1、steps=50、eta=0)一致,沿用不另訂。
