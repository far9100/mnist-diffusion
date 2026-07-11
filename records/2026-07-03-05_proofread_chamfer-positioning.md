<!-- 用途：記錄對必打贏對手 Chamfer Guidance 的拆解與相對定位（Gate A 準則 a）。 -->

# 2026-07-03 Chamfer Guidance 拆解與定位

## 目標

在投入 EDM/CIFAR 前，確認相對最接近的對手 Chamfer Guidance，本方法的差異空間是否還在。這是投入前的硬前置：若差異太薄，現在轉向遠比後期才發現便宜。

## 結果

差異空間存在，但賣點必須轉向。

Chamfer Guidance（arXiv 2508.10631，NeurIPS 2025，作者群 Dall'Asen 等）要點：
- 機制：修改取樣。在分數上加一個導引項，用 DINOv2 特徵空間中對少量（k=2 到 32）真實範例的對稱 Chamfer 距離之梯度導引每條軌跡；每 5 步導引一次，需對 DINOv2 與解碼器做反向傳播。
- 目標函數：對稱 Chamfer，等於一個近似 coverage 項加一個近似 precision 項，是代理指標而非直接下游準確率。
- 算力：因丟掉無條件分支，總 FLOPs 比 CFG 低（LDM1.5 少 15%、LDM3.5M 少 31%），但比純條件取樣貴，且是逐樣本、需梯度。
- backbone 與資料：LDM1.5、LDM3.5M；ImageNet-1k 加 GeoDE、DollarStreet。
- 標題數字「最高 ID +15% / OOD +16%」是跨設定的最佳情況，非對 CFG 的一致增益；逐格增益較小。
- 機制解釋：沒有。只顯示 coverage 與下游增益相關，未解釋為何樣本更利於訓練。
- 程式碼：查無公開釋出（截至 2026-07-03）。Feedback-guided Synthesis 有 code，Chamfer 沒有。

差異對照（Chamfer 是導引，CaF 是選擇器）：

| 面向 | Chamfer Guidance | CaF 選擇器（本方法） |
|---|---|---|
| 機制 | 修改取樣（每 5 步梯度導引） | 在既有組態、輸出中選擇；不動取樣器 |
| 推論時所需 | k 個真實範例、DINOv2、對解碼器反傳 | 特徵抽取器、真實參考集，單次評分，無反傳 |
| 目標 | 對稱 Chamfer（coverage 加 precision 代理） | 在 precision 大於等於 τ 下取 coverage 最大（同項的受限形式） |
| 額外成本 | 逐樣本、梯度式（但相對 CFG 少 15 到 31% FLOPs） | 一次性選擇；每保留樣本等於一般條件取樣 |
| 需調的旋鈕 | γ、ω（ω 隨 backbone 變）、導引頻率 | 只有 τ，且由真實對真實參考自動得出 |
| 是否解釋機制 | 無 | coverage 而非 fidelity 驅動效用，加 near-boundary margin 枯竭 |
| 可組合 | 否，必須擁有取樣器 | 是，可套在任何產生器輸出上，含 Chamfer 的輸出 |

可辯護的差異（依強度排序）：
1. 可組合性，最強。Chamfer 是產生端、不做選擇；選擇器是事後、可疊在任何產生器（含 Chamfer）的輸出上。是消費對手而非硬碰。這點 Chamfer 結構上做不到。
2. 操作點：免任務分類器、無導引強度、成本一次性。不是「比 CFG 便宜」（那點 Chamfer 已佔），而是「無逐樣本梯度導引、無 γ 與 ω、選擇成本可攤提、可套在便宜的一般取樣上」。
3. 為選擇證明的機制，真缺口。Chamfer 無因果解釋；本方法的 near-boundary margin 枯竭故事可補上，但必須為選擇器證明、且不靠下游分類器回饋。

不可當新意（避免被質疑）：
- coverage 與 precision 度量本身（CaF 是同一兩項目標的受限重參數化，同一 DINOv2 空間）。
- 「少量真實範例加特徵抽取器」的框架（與 Chamfer 精神相同）。
- 「跨資料集同超參可遷移」（Chamfer 已宣稱跨資料集穩定；本方法的優勢是旋鈕更少，非遷移本身）。
- 純粹「coverage 驅動效用」與 Chamfer 論點重疊；「near-boundary、難樣本」與同團隊的 Deliberate Practice（2502.15588）重疊。

戰略現實：Chamfer Guidance、Feedback-guided Synthesis（2310.00158）、Deliberate Practice（2502.15588）大致出自同一個 Meta/Mila 團隊，集體擁有「導引生成朝有用、多樣、近邊界樣本」。後兩者都把下游分類器放進迴圈並修改生成。本方法唯一可辯護的楔子是操作點：免訓練、免任務分類器、不改取樣、可組合、零逐樣本成本，加上為選擇而非導引證明的機制。

## 後續

脊椎轉向：以可組合性與免任務分類器操作點為主打，機制為第二支柱；不再宣稱「發現 FID 不等於效用」「新度量」「比 CFG 便宜」。對決需自行重寫簡化 Chamfer 基線（無公開 code）。判定為準則 a 通過但需依此轉向。
