<!-- 用途：D12 分支三（診斷論文）一頁骨架。so-what＝「無單一便宜代理普遍可靠」（R-2026-07-06-05 §1.10）。非治理文件、草稿；C1 反證後現實性升高。依 -09-04 §5.2.9、-09-13 D12。CIFAR-100 分支裁決前僅供決策軌跡。 -->

# 一頁骨架：分支三（診斷論文）

分支三＝CIFAR-100 上 which-FID 不分離、但機制觀察量複製（三中二）。此時 thesis 之「FID≠效用」頭條不成立，
論文轉為診斷性貢獻。

## So-what（承 [CHANGELOG 2026-07-06-05](../CHANGELOG.md#2026-07-06) §1.10）

**沒有單一便宜代理（FID、coverage、precision）能跨資料集普遍可靠地選出下游效用最優的取樣組態。**
證據：同一套選擇器機制在 MNIST 與 CIFAR-10 給出相反判決——MNIST 上 coverage 單調驅動效用、CaF 選中 oracle
（regret 0）；CIFAR-10 上 FID/TSTR 重合、便宜的 FID-min 勝 CaF、且 CaF 有結構性 Pareto 失明。代理的可靠性
本身是資料集相依的。C1 反證（兩表徵空間 FID 皆不誤導）使此診斷更尖銳：連「換個表徵空間的 FID」都不救。

## 主張順序

1. 合成資料訓練分類器時，取樣組態選擇應為下游效用服務——這是既有動機（不新）。
2. 但「哪個便宜代理可靠」無普適答案：MNIST（coverage 驅動）與 CIFAR-10（FID-min 近最優、coverage 非單調）
   之選擇器判決相反。
3. 診斷來源：(a) which-FID 兩表徵不分離（C1）；(b) Pareto 失明使單調 (precision,coverage) selector 結構性
   選不到 oracle（C8 引理）；(c) 變異分解顯示 σ_cls 主導、峰位在少 seed 下不可解析（C4）。
4. 機制觀察（三中二複製，若 CIFAR-100 成立）：coverage 與 near-boundary 之雙段行為在更難資料集仍可量測，
   但不足以使任一便宜代理跨資料集可靠。
5. 誠實負面：CaF 作為「普適免訓練選擇器」的原始賣點在 CIFAR-10 不成立；貢獻降為診斷——代理可靠性之
   資料集相依性、及其量測方法（PRDC、FD-DINOv2、變異分解、matched-probe FID-min 對決）。

## 與 related work 差異化

- 不再宣稱「發現 FID≠效用」或「新度量」（Chamfer 等已隱含）。
- 差異化押在**診斷方法論**：跨資料集之選擇器判決反轉、Pareto 失明引理、σ 分解之功效意涵——為「何時哪個
  代理可靠」提供可操作的量測與反例，而非又一個宣稱普適的選擇器。

## 狀態

**現行路由（2026-07-17 更新）**：D1 已確認落分支三——CIFAR-100 which-FID 不分離（Inception 側 0/8）、
D3 三觀察量 3/3 機制複製、CaF-v2 平 FID-min。本骨架為現行；正文草稿見 `docs/paper_branch3_diagnostic.md`、
揭盲裁決見 `docs/verdict_cifar100.md`。

（原草稿註）CIFAR-100 分支裁決前不定稿；若落分支一/二/四則本骨架作廢或改寫。三判決見 [CHANGELOG 2026-07-09-03](../CHANGELOG.md#2026-07-09)。
