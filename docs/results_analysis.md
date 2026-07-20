<!-- 用途：彙整並分析各階段實驗結果（量測錨點、MNIST sandbox、CIFAR-10 confirmatory、CIFAR-100 confirmatory）。 -->

# 實驗結果分析

本文件彙整目前已產出的實驗數據與其解讀。逐次執行的結論摘要見 [CHANGELOG.md](../CHANGELOG.md)；此處只保留可作結論或
待確認的結果分析。CIFAR-10 confirmatory 已於 2026-07-06 完成（`results/cifar10_cfg_confirmatory.json`），
其三判決分析見下方「CIFAR-10 confirmatory（三判決定稿，E3）」段，鏡像 B 定稿 [CHANGELOG 2026-07-09-03](../CHANGELOG.md#2026-07-09)。

範圍註記（依 [CHANGELOG 2026-07-05-12](../CHANGELOG.md#2026-07-05) 定位 v2）：CIFAR 尺度的主張收緊到 CFG guidance 軸。取樣步數 steps
與 DDIM 隨機性 η 的效用結論只保留在 MNIST sandbox 尺度，不在 CIFAR 尺度宣稱聯合曲面。下方 MNIST 表
為 sandbox 動機證據，CIFAR 自訓 CFG 的 pilot 與 scout 為 exploratory，confirmatory 三判決已定稿（見下方
E3 段）。

## 量測堆疊正確性（backbone 已驗證）

在做任何 CIFAR 組態掃描前，先證明本專案的 EDM 生成到 FID 的量測路徑能重現論文數字。EDM CIFAR-10
（條件、VP）在 50000 張下 FID 為 1.848，論文參考值 1.79，差 0.058，通過正確性 gate。口徑註：本專案 1.848 為單次評估，
官方 1.79 為 min-of-3（NVlabs EDM README），兩者評估口徑不同、非同口徑比較。略高於 1.79
最可能來自 Windows 上以純 PyTorch 參考實作取代 custom CUDA ops 的良性數值差異，對組態之間的相對
比較不構成問題。詳見 [CHANGELOG 2026-07-03-03](../CHANGELOG.md#2026-07-03)。

## MNIST sandbox（乾淨案例，Phase 0）

設定：DDIM、η=0、steps=50，guidance g 屬於 {1,2,3,5,7,10}，每類 1000 張，seeds {0,1,2}，auto-τ
為 real-vs-real 參考 precision 的 0.9 倍（約 0.857）。

| g | precision | coverage | TSTR% | near-boundary 比例 |
|---|---|---|---|---|
| 1 | 0.916±0.003 | 0.875±0.006 | 97.30±0.15 | 0.0269 |
| 2 | 0.961±0.001 | 0.851±0.003 | 96.28±0.06 | 0.0008 |
| 3 | 0.966±0.001 | 0.732±0.004 | 95.01±0.20 | 0.0001 |
| 5 | 0.964±0.001 | 0.556±0.005 | 92.31±0.06 | 0.0 |
| 7 | 0.951±0.003 | 0.436±0.004 | 88.85±0.16 | 0.0 |
| 10 | 0.919±0.001 | 0.319±0.005 | 78.51±1.00 | 0.0 |

分析：

- **coverage 驅動效用**：coverage 隨 guidance 單調下降（0.875→0.319），TSTR 也單調下降
  （97.30→78.51），兩者近乎完美同向。
- **precision 不追蹤效用**：precision 在 g3 達峰（0.966），TSTR 卻在 g1 達峰。保真最佳與效用最佳
  落在不同組態，這是相對 Chamfer 在機制層的乾淨差異。
- **CaF 可行**：免訓練的 CaF 在三個 seed 全部選中 g1，與 oracle TSTR 最佳一致；regret 為
  0.00±0.00 pp、rank 1/6、top-3 命中 100%。信賴區間很緊、排序跨 seed 從不翻轉，代表「效用最佳點
  不等於保真最佳點」不是單 seed 雜訊。
- **機制在 MNIST 飽和**：near-boundary 比例隨 guidance 單調枯竭（0.027→0），方向正確；但 MNIST
  近乎可分，真實資料的 near-boundary 比例僅約 0.009，能被枯竭的邊界質量很少，機制的強證據需要更難、
  非可分的 CIFAR-100。
- **τ 誠實註記**：auto-τ 選中 g1（regret 0），但 τ 掃描的眾數選擇是 g2；g1 與 g2 對 τ 敏感，惟
  兩者皆在前二名、robust 區間內 regret 不超過約 1.3pp。

原始記錄見 [CHANGELOG 2026-07-03-02](../CHANGELOG.md#2026-07-03) 與
[CHANGELOG 2026-07-03-04](../CHANGELOG.md#2026-07-03)。

## CIFAR-10 預覽（初步，較複雜，尚未定案）

範圍指標：本段為 2026-07-03 的低步數快篩預覽，已被下方「CIFAR-10 confirmatory（三判決定稿，E3）」
段取代。保留作歷史脈絡，段內結論（含下方「目前不主張 CIFAR 結果」）以 confirmatory 段為準。

設定刻意求快：1 seed、每類 500 張、生成步數 8、Inception 特徵做 PRDC。CFG guidance 軸 w 屬於
{1,1.5,2,3}。

| w | precision | coverage | recall | TSTR% |
|---|---|---|---|---|
| 1.0 | 0.767 | 0.825 | 0.720 | 47.50 |
| 1.5 | 0.807 | 0.842 | 0.689 | 49.74（oracle 最佳） |
| 2.0 | 0.819 | 0.844 | 0.656 | 49.23 |
| 3.0 | 0.803 | 0.807 | 0.571 | 48.06 |

分析：

- 與 MNIST 不同，CIFAR-10 的 TSTR 似乎有**內部最優**（在中間 guidance w≈1.5–2 達峰，而非 w=1），
  coverage 也非單調。CaF（auto-τ 約 0.697）選中 w2，oracle 最佳為 w1.5；regret 0.51pp、rank 2/4、
  top-3 命中。
- 兩種解讀，需全品質才能分辨：（a）真實 CIFAR 行為——CIFAR 不像 MNIST 近乎可分，適度 guidance 先
  提升保真與代表性使 TSTR 上升，只有過高 guidance 才崩多樣性使 TSTR 下降，選擇器的任務變成找甜蜜點；
  （b）8 步的低步數假影——為求快把生成步數由 18 降到 8，對 w=1（無 CFG）端傷害最大，可能人為壓低
  低 guidance 的 TSTR、製造出內部最優。必須用 18 步全品質重跑後才能下結論。
- TSTR 偏低（約 48–50%）是因為只有 5000 張合成且用 8 步，屬弱訊號預覽。**目前不主張 CIFAR 結果。**

原始記錄見 [CHANGELOG 2026-07-03-06](../CHANGELOG.md#2026-07-03)。

## 自訓 CFG CIFAR-10（exploratory pilot 與上緣 scout）

以下為自訓 CFG 主軸的 exploratory 結果（pilot 用 seeds 0/1/2，grid 於看過 scout 後才鎖定，故只能定位為
exploratory；confirmatory 以 fresh seeds 10/11/12、10 點 grid 另跑）。

Stage 4 多 seed pilot（steps=50、η=0，grid {1,2,3,4,5,8}，3 seed 均值 ± std）：

| w | precision | coverage(DINOv2) | TSTR% | label_noise | near_bnd |
|---|---|---|---|---|---|
| 1 | 0.827±0.009 | 0.710±0.006 | 41.18±1.15 | 0.112±0.004 | 0.252±0.008 |
| 2 | 0.884±0.004 | 0.834±0.009 | 46.01±1.85 | 0.013±0.003 | 0.063±0.005 |
| 3 | 0.903±0.004 | 0.834±0.002 | 43.88±1.49 | 0.005±0.002 | 0.037±0.004 |
| 4 | 0.903±0.003 | 0.806±0.005 | 40.59±0.91 | 0.003±0.002 | 0.032±0.001 |
| 5 | 0.897±0.001 | 0.758±0.008 | 37.43±2.39 | 0.003±0.001 | 0.031±0.002 |
| 8 | 0.871±0.003 | 0.622±0.012 | 28.43±0.70 | 0.008±0.002 | 0.055±0.004 |

pilot 判讀（exploratory，敘事描述，不承擔裁決功能）：TSTR 峰在 w=2、三 seed oracle 一致，內部最優在
自訓主軸成立；CaF 每 seed 選 [w2,w3,w2]、regret 0.28pp、top-3 100%。三段結構——低段（w1→2）precision/
coverage 同升而 label-noise 降（三力同向、混淆）、中段（w2→3）coverage 平坦而 TSTR 微降、高段（w3→8）
coverage 崩且 precision 高原（兩者分離）——只作敘事，不作 C2 裁決。C2 的 confirmatory 裁決改以全網格偏
相關進行，納入全部 10 點含混淆段，不分段（見 [CHANGELOG 2026-07-05-13](../CHANGELOG.md#2026-07-05)）。

上緣 coverage-only scout（1 seed、coverage-only、DINOv2 特徵）：coverage 於 w∈[8,20] 由 0.533 單調降至
0.259，依判準 X=0.02 未觸底。此為描述性觀察，緊貼其條件標注；依封頂 amendment（[CHANGELOG 2026-07-05-11](../CHANGELOG.md#2026-07-05)）
以先驗 CFG 實用範圍封頂 w=8，confirmatory 不重跑 w>8 區段。併入雙力敘事：低段 fidelity 上升、高段
coverage 單調下降，兩力交會產生內部最優。

## CIFAR-10 confirmatory（三判決定稿，E3）

鏡像 B 定稿 [CHANGELOG 2026-07-09-03](../CHANGELOG.md#2026-07-09)。三判決互不裁決；本段為觀察性描述，禁因果措辭。
設定：steps=50、η=0、grid {1,1.5,2,2.5,3,4,5,6,7,8}、fresh seeds {10,11,12}、per_class 1000。

### 判決一（thesis）

FID-opt 與 TSTR-opt 重合於 w1.5（clean-fid 均值 8.82／TSTR 均值 63.96）。排序（n=10 均值曲線 Spearman）：
ρ(−char_clean_fid, TSTR)=0.964、ρ(coverage, TSTR)=0.624。README 頭條「FID-opt 偏離 TSTR-opt」在本資料上被
反證。which-FID 交叉裁決（C1，[CHANGELOG 2026-07-09-02](../CHANGELOG.md#2026-07-09)）：FD-DINOv2 均值 argmin=w2、TSTR argmax=w1.5，格步 1、
依凍結口徑（>1 才算分離）判不分離；疊加 Inception／clean-fid 側 0 格步不分離——兩表徵空間皆不分離。「內部
最優」為未登記 exploratory 觀察（上升肢 w1→w1.5 +0.80pp、SE 1.88，3 gen seeds 下不可判定）；「必然次優」
全稱句撤下。災難性非單調（w≥3 TSTR 崩 11–30pp）為存活觀察。

### 判決二（H-C2）

A 批（[CHANGELOG 2026-07-06-09](../CHANGELOG.md#2026-07-06)）：DINOv2 側 C2a ρ=+0.658 p=0.0188、Inception 側 ρ=+0.859 p=0.0008（跨表徵
一致），機械通過；但穩健性沿種子軸（bootstrap CI 跨零）與可交換性軸（gseed 碰撞）無法內部驗證，最終須
CIFAR-100 獨立複製回答。H-C2a 顯著亦不寫「coverage 驅動效用」等因果措辭。

### 判決三（selector，描述性）

協定未凍門檻，不作過／敗判定。FID-min per-seed regret 0.91（2.45/0.28/0.00）對 CaF 3.69（0.54/5.03/5.49）；
FID-min 為 3 seed 2 勝 1 負（seed10 上 CaF 0.54 反勝 FID-min 2.45）。regret 主 3.69（per-seed 均值）、並列
2.77（mean-curve 口徑：均值 oracle w1.5 之 63.96 減均值 CaF w2.5 之 61.19）。CaF 於本網格結構性 Pareto 失明
（w2.5 (.873,.792) 嚴格支配三 oracle，C8 引理 `docs/c8_pareto_blindness.md`）。可辯護措辭：CaF 為可靠避崖器、
糟糕平台優化器——FID-min 同樣避崖且成本結構相同。

### 雙段機制（觀察性，禁因果；均值曲線）

| w | 1 | 1.5 | 2 | 2.5 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| TSTR | 63.16 | 63.96 | 61.16 | 61.19 | 53.08 | 47.20 | 44.10 | 39.10 | 36.06 | 33.46 |
| coverage(DINOv2) | .645 | .751 | .777 | .792 | .778 | .745 | .698 | .648 | .604 | .559 |
| precision | .806 | .841 | .858 | .873 | .876 | .879 | .869 | .859 | .848 | .836 |
| near-boundary | .256 | .114 | .063 | .046 | .037 | .032 | .037 | .039 | .048 | .059 |
| ln_excess | +.044 | −.038 | −.057 | −.062 | −.065 | −.065 | −.065 | −.064 | −.063 | −.062 |
| FD-DINOv2 | 282.4 | 195.2 | 175.4 | 176.7 | 188.7 | 223.9 | 261.1 | 302.6 | 341.9 | 379.4 |

觀察（禁因果）：低中段（w1→w2.5）near-boundary 比例 .256→.046 下降、coverage .645→.792 上升（coverage 峰
落 w2.5）；w1 之 ln_excess +.044 為全網格唯一正值。高段（w2.5→w8）coverage 與 TSTR 同降、near-boundary 由
.032 回升至 .059、precision 同降。全網格 ρ 將雙段抹平（C2 裁決照跑之凍結義務，見判決二）。

### P 對帳

P0（單 cell）與 P1（全 30 config）對帳（[CHANGELOG 2026-07-08-04](../CHANGELOG.md#2026-07-08)、[CHANGELOG 2026-07-09-01](../CHANGELOG.md#2026-07-09)）：全 30 config 之
量測對帳 scalar（precision、coverage 之 DINOv2 與 Inception 兩側、char_clean_fid、near_boundary_frac、
label_noise_excess_mean）逐位重現凍結 JSON；k=5 獲 P0 探針反證支持。TSTR 依協定含未種子化 shuffle、非決定性、
不在對帳集、不宣稱逐位重現。

## C 批補遺（exploratory，非判決輸入，禁因果）

以 P1 落盤資產計算，標 exploratory；不改三判決（[CHANGELOG 2026-07-09-03](../CHANGELOG.md#2026-07-09) 不回改）。觀察性描述、禁因果句。
噪聲脈絡：C4 測得 σ_cls=2.963pp，下列單 seed／少重訓之差異多在 1–2 σ_cls 內。

- **C4 變異分解**（[CHANGELOG 2026-07-09-07](../CHANGELOG.md#2026-07-09)）：σ_cls=2.963pp（分類器訓練變異，主導）對 σ_gen=1.182pp（生成
  變異）；上升肢 w1→w1.5（+2.52pp）小於 σ_cls、維持 unresolved。餵 D4 功效規劃。
- **C7 small-probe FID 穩定性**（[CHANGELOG 2026-07-09-06](../CHANGELOG.md#2026-07-09)）：Kendall τ 最小 0.911（100/class）→ 1.000
  （500/class），FID-argmin 於 15/15 個 probe draw 全落 w1.5。依凍結判定 FID-min baseline 於小 probe 可靠，
  餵 D5。
- **C2/C3/C5 介入式證據**（[CHANGELOG 2026-07-09-08](../CHANGELOG.md#2026-07-09)）：C2 near-boundary 移除之 TSTR 掉幅未大於等數隨機移除；
  C5 TSTR 未隨 near-boundary 純度單調上升；C3 降 w2.5 coverage 至 w1 水準之 TSTR 代價略大於隨機剪枝，但
  w2.5≈w1 之等 TSTR 使橋接複雜。三項未對 near-boundary 機制提供強支持，與 C1 反證方向一致；機制之
  confirmatory 回答移 CIFAR-100（D3）。
- **C8 Pareto 失明引理**（`docs/c8_pareto_blindness.md`）：w2.5 嚴格支配三 oracle → CaF 結構性選不到 oracle。

## selector plan-of-record（CaF-v2）

因 C8 Pareto 失明，原 coverage 版 CaF（`argmax coverage s.t. precision ≥ τ`）在本網格結構性選不到
oracle。C0 recall/density 探針（[CHANGELOG 2026-07-09-12](../CHANGELOG.md#2026-07-09)）測得 recall 可打破 w2.5 的支配、density 不能。
故 D 包（[CIFAR-100 預註冊](prereg_cifar100.md)，作者終簽）將 plan-of-record selector 改為 **CaF-v2 =
`argmax recall s.t. precision ≥ τ`**，待 CIFAR-100 分支驗證。README 與早期 intro 草稿（已下架，
見 git 歷史）內定義 CaF 為 coverage 版，屬凍結敘述（依協定不回改），以本段為現行定義來源。

## CIFAR-100 confirmatory（機制複製，分支三）

CIFAR-100 為 CIFAR-10 結論的 validation（[CIFAR-100 預註冊](prereg_cifar100.md) D0）。設定：steps=50、
η=0、grid {1,1.5,2,2.5,3,4,5,6,7,8}、seeds {10..17}（8 seed）、reps 5、per_class=real_per_class=500
（per-class 凍結 amendment：CIFAR-100 訓練集每類上限 500）。揭盲裁決全文見 `docs/verdict_cifar100.md`。
本段為觀察性描述，禁因果措辭。

### 判決一（thesis／C1）：不分離

Inception clean-fid（char_clean_fid）argmin=w1.5、TSTR argmax=w1，相隔 1 格；逐 seed 分離格步 >1 的
數目 0/8（[CHANGELOG 2026-07-16-01](../CHANGELOG.md#2026-07-16)）。依凍結口徑（>1 才算分離）判不分離——
此為凍結 D1 路由之依據。事後補算之 seed-10 FD-DINOv2（`cifar100_fd_dinov2.json`，
[CHANGELOG 2026-07-19-01](../CHANGELOG.md#2026-07-19)）argmin=w2.5、離 TSTR-argmax w1 達 3 格、**判分離**：
兩個 FID 空間給出相反判決——Inception-FID 近最優（作為選擇器 regret 約 0.8pp）、DINOv2-FID 爛（選中
w2.5、約 8.8pp）。which-FID 可靠性表徵相依，強化診斷。單 seed、事後、Inception-only 凍結範圍之外、
不回改路由。

### 判決三（selector）：CaF-v2 與 FID-min 打平

matched-budget FID-min 對決：CaF-v2（recall）與更便宜的 FID-min 逐 seed regret 完全相同（per-seed
[0.79,1.15,0.45,0.87,0.67,0.73,0.91,0.52]、兩者均值各 0.76pp、同選 w1.5、oracle w1）。D4 門檻要求
CaF-v2 低於 FID-min ≥1.5pp；實得差 0.00pp，selector 主張不成立。此為 MNIST／CIFAR-10 之後第三個資料
點，CIFAR-100 與 CIFAR-10 同型（FID-min 近最優），非 MNIST 那型（coverage 主導、CaF 選中 oracle）。

### 雙段機制（觀察性，禁因果；均值曲線）

| w | 1 | 1.5 | 2 | 2.5 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| TSTR | 59.66 | 58.90 | 55.39 | 50.72 | 45.26 | 35.78 | 28.00 | 22.85 | 18.77 | 15.88 |
| coverage(DINOv2) | .481 | .643 | .697 | .698 | .679 | .617 | .560 | .508 | .468 | .433 |
| precision | .783 | .824 | .844 | .851 | .854 | .845 | .832 | .822 | .812 | .804 |
| recall | .476 | .446 | .394 | .338 | .288 | .216 | .167 | .133 | .108 | .092 |
| near-boundary | .258 | .117 | .067 | .047 | .038 | .033 | .036 | .044 | .056 | .072 |
| char_clean_fid | 11.17 | 7.17 | 8.40 | 10.77 | 13.17 | 17.11 | 20.10 | 22.53 | 24.64 | 26.60 |

D3 三觀察量（`results/cifar100_d3_observables.json`，純衍生）：(i) 升段 w1→w2.5 near-boundary
.258→.047 單調降；(ii) 高段 w2.5→w8 coverage .698→.433 與 TSTR 50.72→15.88 同崩；(iii) 高段
near-boundary 於 w4 谷 .033 後回升至 w8 .072、同段 coverage 續降（脫鉤）。三項全成立（3/3），過三中二
門檻，機制複製。與 CIFAR-10 §E3 的雙段結構同型。coverage 峰於 CIFAR-100 落 w2.5（絕對值 .698，跨
資料集不可與 CIFAR-10 直接比，per-class 樣本數不同，per-class 凍結 amendment 已登記此限制）。

### D1 路由：分支三

客觀觀察量（不分離＋機制複製 3/3＋CaF-v2 平 FID-min）唯一相容分支三（診斷論文）；branch 1/2 需分離、
branch 4 需機制不複製，皆被資料排除。作者 2026-07-17 簽核（[CHANGELOG 2026-07-17-03](../CHANGELOG.md#2026-07-17)）。
診斷論文正文見 `docs/paper_branch3_diagnostic.md`。

## 機制介入與matched-budget 對照（已補）

- **D3 介入臂**（C3 coverage-matched pruning，seed 10）：剪 w2.5 至 w1 之 coverage 水準（移 13606
  樣本）後重訓。N=2（`cifar100_d3_intervention.json`）得 cov-matched 45.75 對 random 45.86、差 −0.11pp
  但功效不足；N=8 follow-up（`cifar100_d3_intervention_n8.json`）得 46.30 對 46.63、差 −0.33pp（SE 0.66、
  t=−0.50、MDE≈1.85pp、CI 跨零）。有功效下 cov-matched 與 random 無法區分，介入臂未對 coverage 因果
  角色提供支持（由 underpowered 升為有功效之 null，仍限單 seed／單一介入型式）。重生成對帳
  `cifar100_regen_reconcile.json` rel 全 0（[CHANGELOG 2026-07-18-11](../CHANGELOG.md#2026-07-18)）。
- **H3 matched-budget 對照**（matched-budget 三臂，seed 10）：vanilla w1.5（＝FID-min＝CaF-v2）TSTR 58.65；
  Chamfer 任務無關 DINOv2 特徵 61.18（+2.54pp）、任務對齊 judge 特徵 61.75（+3.11pp），皆勝 vanilla
  全網格 oracle w1（59.66），但 DINOv2 coverage 反低（0.44–0.48 < 0.643）。此 coverage 反低讀數因本文
  Chamfer 為單向簡化、單 seed 而存疑（官方雙向實作報告 coverage 上升 0.603→0.912、方向相反），待雙向公平
  化後方可定論（`cifar100_h3_duel_dinov2.json`、`_judge.json`，[CHANGELOG 2026-07-18-01](../CHANGELOG.md#2026-07-18)）。

## 未來工作（本論文範圍外）

以下為 `docs/thesis_draft.md` 第七章列的延伸實驗，不屬本論文交付範圍，皆需重新生成、較高 GPU 成本：
- 多 seed 複製：FD-DINOv2（seed-10 已補、判分離，見上 C1 段）與 D3 介入（seed-10 N=8）之跨 seed 複製。
- H3 matched-budget 對照之多 seed／weight 掃描。
