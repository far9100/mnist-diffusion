<!-- 用途：P1-2 更高解析度資料集（STL-10 96×96）的預先登記草稿（DRAFT，D 包同構於 CIFAR-100 prereg）。
     本檔為草稿，須經作者定稿並在任何 STL-10 合成取樣/訓練之前 commit，方滿足凍結四要件 (a)。 -->

# STL-10（96×96）規模門檻 預先登記（草稿 DRAFT）

狀態：**草稿**，待作者定稿。承 `docs/prereg_cifar100.md` 之 D 包結構與 `claude.md` §5.1 凍結四要件。
本檔定稿並 commit 前，不得執行任何 STL-10 backbone 訓練或 confirmatory 取樣。

## 動機與定位

CIFAR（32×32）已完成三判決。STL-10（96×96、10 類）是「便宜代理（CaF）之經濟性開始成立」的最小
尺度——解析度夠高使「為每個組態各訓一個下游分類器」的成本明顯，免訓練選擇器的價值才浮現。本規模項
驗三判決是否延伸到 32×32 以上（投主會的規模門檻）。

## D 包（與 CIFAR-100 同構，逐項待定稿）

- **D0 資料集**：STL-10。labeled train 500/class（共 5000）、test 800/class（共 8000）、unlabeled 100k
  （backbone 訓練可用）。影像 96×96 RGB。
- **D1 揭盲決策樹**：沿 CIFAR-100 之四分支（分離且機制成立→ selector 主張；不分離但機制複製→診斷；
  不分離且機制不成立→否證；分離但機制不成立→純選擇器）。
- **D2 backbone**：自訓 CFG-capable 擴散模型於 96×96（pixel-space UNet 或 latent diffusion，二選一於
  定稿時凍結）。unlabeled 100k 可併入無條件訓練。
- **D3 介入臂**：coverage-matched pruning 與 margin-pruning（同 CIFAR-100 §5.5），N≥8 retrain。
- **D4 主張門檻**：selector 勝幅 X=1.5pp（同 CIFAR-100 候選 B）；功效 8 seed × 5 rep，MDE 於定稿時
  以 σ_cls/σ_gen 探針表回填。
- **D5 baseline**：fixed-w（整欄）、random-feasible（解析平均），同 T2。
- **D8 選擇器**：CaF-v2＝`argmax recall s.t. precision ≥ τ`，τ 由 real-vs-real 自動決定，tau_fraction
  0.9。coverage/precision 於 DINOv2 特徵（96×96 resize 224，原生支援）。
- **D10 base gate**：base-model clean-fid ≤ 門檻 @50k（STL-10 無內建 cleanfid stats，須自建 96×96 或
  64×64 參考；門檻於定稿時依 backbone 能力設定，暫定 ≤30）。judge 測試準確度與 near-boundary margin
  校準（p20）於揭盲前登記。
- **D 生成種子**：§0.5 hash 派生（`sha256[:15]`），禁用舊公式。
- **grid/seeds/per_class**：guidance grid 沿 CIFAR 之 10 點 {1,1.5,2,2.5,3,4,5,6,7,8}（定稿時得依 96×96
  coverage 幾何微調並明記）；seeds {10..17}（8 seed）；per_class = real_per_class = 500（匹配口徑，
  同 CIFAR-100，STL-10 labeled 每類上限 500）。steps/eta 於 backbone 定稿時凍結。

## 凍結四要件對應（§5.1）

- (a) 本草稿定稿後 commit，早於任何 STL-10 訓練/取樣。
- (b) 計算以 committed code 表達：`src/experiments/run_stl10_pipeline.py`（骨架，dry-run 先行）。
- (c) dry-run 於資料就緒後先過（枚舉 cell、資料可用性、單階段計時探針→ETA）。
- (d) 每 driver 輸出 §5.2 完整 metadata；新結果一律寫新檔、標 exploratory/confirmatory。

## 成本與時序（重要）

此為 fix_tasks P1-2、成本最高、任務書列「最後做」。需序列完成：(1) backbone 訓練（96×96，數日級
GPU）、(2) judge 訓練、(3) base FID gate、(4) confirmatory sweep + TSTR + 介入。骨架與各階段 ETA 見
`run_stl10_pipeline.py --dry-run`。任一真跑須待本 prereg 定稿 commit 且作者授權。
