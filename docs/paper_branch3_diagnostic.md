<!-- 用途：分支三診斷論文正文（表格版草稿）。依 paper_skeleton_branch3.md 五主張、CIFAR-100 揭盲裁決（verdict_cifar100.md）撰寫；數據引 results/*.json，圖後補。D1 於 2026-07-17 確認為現行路由。 -->

# 便宜代理何時可靠？合成訓練資料之取樣組態選擇的診斷研究

（暫定英文題名：*When Is a Cheap Proxy Reliable? A Diagnostic Study of Sampling-Configuration
Selection for Synthetic Training Data*）

本文為分支三（診斷論文）正文草稿。揭盲路由裁決見 `docs/verdict_cifar100.md`；一頁骨架見
`docs/paper_skeleton_branch3.md`；所有數據引 `results/*.json`（本機、gitignore），對照見附錄。
本稿為表格版，發表級圖後補。

## 摘要

擴散模型生成的合成影像越來越常被當成下游分類器的訓練資料。選擇取樣組態（此處為 classifier-free
guidance 強度 w）時，一個自然的問題是：能否用一個**便宜的、免下游分類器的代理**（保真度 FID、或
多樣性 coverage/precision）挑出下游訓練效用（Train-on-Synthetic-Test-on-Real, TSTR）最優的組態？
我們在三個尺度（MNIST、CIFAR-10、CIFAR-100）以預先登記的協定檢驗此問題，得到一個否定但可操作的
結論：**沒有單一便宜代理能跨資料集普遍可靠**。同一套 coverage 選擇器在 MNIST 選中效用最優組態
（regret 0），在 CIFAR-10 與 CIFAR-100 卻敗給或打平更便宜的 FID-min baseline；而 FID-min 在 CIFAR
近最優、在 MNIST 卻會選錯。我們給出此不可靠性的三個診斷來源——which-FID 不分離、選擇器的 Pareto
失明引理、變異分解的功效意涵——並報告 coverage 與 near-boundary 的雙段機制在最難的 CIFAR-100 仍
可量測、仍複製。貢獻由「一個普適的免訓練選擇器」誠實地降為「代理可靠性之資料集相依性及其量測
方法學」。

## 1. 動機

合成資料訓練下游分類器時，取樣旋鈕（去噪步數、DDIM 隨機性、guidance 強度）幾乎總是為最佳化影像
保真度（FID）而調。本文聚焦 CFG guidance 軸，問一個操作性問題：在只有一小份真實參考集、沒有下游
分類器的情況下，能不能用一個便宜代理選出 TSTR 最優的 w？這是「合成資料當訓練集」情境下的實際需求：
若某個便宜代理可靠，就不必為每個資料集/任務重訓分類器掃 w。本文檢驗此需求能否被滿足。

## 2. 無通用便宜代理：選擇器判決跨資料集反轉（主結果）

我們在三個尺度比較兩個便宜代理的選擇品質，評分用 `regret@selected`＝oracle（該 seed TSTR 最高組態）
之 TSTR 減被選中組態之 TSTR，越低越好：

- **coverage 選擇器 CaF**：`argmax coverage s.t. precision ≥ τ`（MNIST/CIFAR-10）；CIFAR-100 用
  plan-of-record 的 **CaF-v2**＝`argmax recall s.t. precision ≥ τ`（見 §3b 為何改 recall）。
- **FID-min**：直接挑 characterization clean-fid 最小的組態，更便宜、同樣不看 TSTR。

| 資料集 | TSTR-oracle | coverage 選擇器選中 | coverage 選擇器 regret | FID-min regret | 可靠的便宜代理 |
|---|---|---|---|---|---|
| MNIST | g1 | g1 | **0.00** | （見下）不可靠 | **coverage**（CaF 選中 oracle） |
| CIFAR-10 | 依 seed | w2.5 | 3.69 | **0.91**（2 勝 1 負） | **FID-min** |
| CIFAR-100 | w1 | w1.5 | 0.76 | **0.76**（打平） | **FID-min**（更便宜） |

（MNIST：`results/selector_signal.json`；CIFAR-10：`results/cifar10_c6_fidmin_duel.json`；CIFAR-100：
`results/cifar100_c6_fidmin_duel.json`。）

判讀——**哪個便宜代理可靠，隨資料集反轉**：

- **MNIST（coverage 可靠、fidelity 不可靠）**：coverage 隨 guidance 單調降（0.875→0.319）、TSTR 也單調
  降（97.30→78.51），兩者同向，coverage 選擇器 CaF 三 seed 全選中 oracle g1（regret 0.00±0.00、rank 1/6，
  §MNIST sandbox 表 `docs/results_analysis.md`）。反之 precision（保真）在 g3 達峰（.966）而 TSTR 在 g1
  達峰——一個 fidelity/precision 驅動的代理會選到 g3 一帶、選錯。
- **CIFAR-10（fidelity 可靠、coverage 不可靠）**：FID-min per-seed regret 0.91 勝 CaF 3.69（3 seed 2 勝
  1 負）。CaF 因 Pareto 失明（§3b）結構性選不到 oracle，選到高 coverage 但非最優的 w2.5。
- **CIFAR-100（第三資料點，同 CIFAR-10 型）**：CaF-v2 與 FID-min 逐 seed regret 完全相同（均 0.76、同選
  w1.5），selector 不帶來任何優勢；更便宜的 FID-min 已足夠。

故「用哪個便宜代理」無普適答案：coverage 選擇器在近乎可分的 MNIST 可靠、在較難的 CIFAR 失效；FID-min
在 CIFAR 近最優、在 MNIST 會選錯。**代理的可靠性本身是資料集相依的。**

## 3. 診斷來源：為何沒有便宜代理普遍可靠

### 3a. which-FID 兩表徵不分離（C1）

若「保真最優偏離效用最優」成立，換一個 FID 的特徵空間或許能救。資料顯示不能：

- **CIFAR-10**：FD-DINOv2 argmin=w2、Inception clean-fid argmin=w1.5、TSTR argmax=w1.5；DINOv2 側格步 1、
  Inception 側格步 0，依凍結口徑（>1 才算分離）**兩表徵空間皆不分離**（[CHANGELOG 2026-07-09-02](../CHANGELOG.md#2026-07-09)）。
- **CIFAR-100**：Inception clean-fid argmin=w1.5、TSTR argmax=w1，格步 1、逐 seed 0/8，**不分離**。範圍
  限制：CIFAR-100 未算 per-config FD-DINOv2（無對應 P1 跑），which-FID 只在 Inception 空間評估
  （`docs/verdict_cifar100.md`）。

即：FID 的 argmin 與 TSTR 的 argmax 落在相鄰或同一組態，換表徵空間也不拉開。保真代理與效用最優並未
分離到能靠「挑另一種 FID」可靠選中的程度。

### 3b. Pareto 失明引理（C8）：單調 (precision, coverage) 選擇器結構性選不到 oracle

`docs/c8_pareto_blindness.md` 證明：若存在組態 c\* 在 (precision, coverage) 平面上嚴格支配每一個
TSTR-oracle（precision 與 coverage 都更高），則對任何門檻 τ，形如 `argmax g(precision, coverage) s.t.
precision ≥ τ`、g 對兩引數皆單調遞增的選擇器（含 CaF 的 `argmax coverage`）都選不到 oracle。這不是
校準問題（沒有 τ 能救），是選擇器形式本身的限制。

CIFAR-10 實例（`results/cifar10_cfg_confirmatory.json` 均值）：w2.5 (precision .873, coverage .792)
嚴格支配三個 per-seed oracle——w2 (.858,.777)、w1.5 (.841,.751)、w1 (.806,.645)。故 CaF 在此網格
結構性選中 w2.5、選不到 oracle。C0 探針（`results/cifar10_recall_density_c0.json`）測得 **recall 可
打破 w2.5 的支配、density 不能**，故 plan-of-record 的 CaF-v2 把第三訊號改為 recall——但這只解 CIFAR-10
的特定支配，不改「無普適代理」的結論（CIFAR-100 上 CaF-v2(recall) 仍與 FID-min 打平）。

### 3c. 變異分解的功效意涵（C4）

`results/cifar10_c4_variance.json`：分類器訓練變異 σ_cls=2.963pp 主導生成變異 σ_gen=1.182pp（約 2.5
倍）。CIFAR-10 上升肢 w1→w1.5 的效用差（+2.52pp）小於 σ_cls，在少 seed 下不可解析。這說明「內部
最優」等細峰主張在此雜訊水準下不可判定，也是為何 CIFAR-100 功效配置定為 8 seed × 5 rep（D4，MDE
2.49pp）——診斷的一部分是**指出何種主張在何種功效下不可證**。

## 4. 機制觀察：coverage 與 near-boundary 的雙段行為在 CIFAR-100 複製

診斷之外，我們報告一個正面的觀察性結果：coverage 與 near-boundary 樣本支持度的雙段行為，在最難的
CIFAR-100 仍可量測、仍複製（觀察性描述，禁因果）。均值曲線（`results/cifar100_cfg_confirmatory.json`）：

| w | 1 | 1.5 | 2 | 2.5 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| TSTR | 59.66 | 58.90 | 55.39 | 50.72 | 45.26 | 35.78 | 28.00 | 22.85 | 18.77 | 15.88 |
| coverage(DINOv2) | .481 | .643 | .697 | .698 | .679 | .617 | .560 | .508 | .468 | .433 |
| precision | .783 | .824 | .844 | .851 | .854 | .845 | .832 | .822 | .812 | .804 |
| near-boundary | .258 | .117 | .067 | .047 | .038 | .033 | .036 | .044 | .056 | .072 |
| char_clean_fid | 11.17 | 7.17 | 8.40 | 10.77 | 13.17 | 17.11 | 20.10 | 22.53 | 24.64 | 26.60 |

D3 三觀察量（`results/cifar100_d3_observables.json`，三中二判準）：(i) 升段 w1→w2.5 near-boundary
.258→.047 單調降；(ii) 高段 w2.5→w8 coverage .698→.433 與 TSTR 50.72→15.88 同崩；(iii) 高段
near-boundary 於 w4 谷 .033 後回升至 w8 .072、同段 coverage 續降（脫鉤）。**三項全成立（3/3）**，機制
複製，與 CIFAR-10 雙段結構同型。

但此機制**不足以**使任一便宜代理跨資料集可靠：機制可量測 ≠ 便宜代理可選中效用最優（CIFAR-100 上
coverage 選擇器仍敗/平 FID-min）。機制解釋「為何效用隨 guidance 這樣變」，不解「哪個代理能選中最優」。

（機制的**介入**證據——C3 coverage-matched pruning——為預先登記之補充，待重生成合成集後補上；本稿
先陳觀察量。CIFAR-10 的 C2/C3/C5 介入未對 near-boundary 因果角色提供強支持，故 CIFAR-100 介入臂屬
真正待驗。）

## 5. 誠實負面與貢獻定位

CaF 作為「普適免訓練選擇器」的原始賣點，在 CIFAR-10 敗於更便宜的 FID-min、在 CIFAR-100 與其打平，
不成立。我們據實把貢獻降為**診斷**：

- 代理可靠性之**資料集相依性**（§2 的判決反轉）；
- 其量測方法學：PRDC（DINOv2 空間）、FD-DINOv2、matched-probe FID-min 對決、變異分解、Pareto 失明
  引理——為「何時哪個代理可靠」提供可操作的量測與反例；
- 一個誠實記錄：便宜的 FID-min baseline 在 CIFAR 尺度常已近最優，任何更貴的選擇器須先勝過它。

## 6. 與 related work 差異化

- 不再宣稱「發現 FID≠效用」或提出「新度量」（Chamfer Guidance、Feedback-guided Synthesis 等已隱含
  保真非效用最優）。
- 差異化押在**診斷方法論**：跨資料集之選擇器判決反轉、Pareto 失明引理、σ 分解之功效意涵。相對於
  guidance 方法（改取樣過程、須擁有 sampler）、固定 low-CFG 配方（Fan et al.），本文的操作點是「先問
  哪個便宜代理在哪個資料集可靠」，並給出「沒有普適答案」的量測證據。

## 7. 明確不主張

- 不主張任何普適的免訓練選擇器。
- 不主張「FID≠效用」為新發現。
- 不主張 CaF/CaF-v2 之操作優勢（在 CIFAR 尺度它未勝更便宜的 FID-min）。

## 附錄：主張與數據來源對照

| 主張/表 | 數據來源（`results/`） | 狀態 |
|---|---|---|
| MNIST coverage 主導、CaF regret 0 | `selector_signal.json`（selected g1、regret 0、rank 1/6） | 已算 |
| CIFAR-10 FID-min 0.91 勝 CaF 3.69（2-1） | `cifar10_c6_fidmin_duel.json` | 已算 |
| CIFAR-100 CaF-v2 平 FID-min 0.76 | `cifar100_c6_fidmin_duel.json` | 已算 |
| C1 CIFAR-10 兩空間不分離 | `cifar10_cfg_confirmatory.json`、`cifar10_c6_fidmin_duel.json` | 已算 |
| C1 CIFAR-100 Inception 不分離（0/8） | `cifar100_c6_fidmin_duel.json` | 已算 |
| C8 Pareto 失明實例（w2.5 支配 oracle） | `cifar10_cfg_confirmatory.json`、`cifar10_recall_density_c0.json` | 已算 |
| C4 變異 σ_cls 2.963 / σ_gen 1.182 | `cifar10_c4_variance.json` | 已算 |
| CIFAR-100 機制雙段 + D3 三觀察量 3/3 | `cifar100_cfg_confirmatory.json`、`cifar100_d3_observables.json` | 已算 |
| D3 介入臂（C3 pruning）CIFAR-100 | `cifar100_d3_intervention.json` | 待跑（GPU） |
| H3 CaF-v2 vs Chamfer 三臂對決 | `cifar100_h3_duel.json` | 待跑（GPU） |

## 狀態

表格版草稿，D1 已於 2026-07-17 確認為現行路由（`docs/verdict_cifar100.md`）。待補：D3 介入臂與 H3
對決之 GPU 結果（§4、§附錄）、發表級圖（§4.3 新視覺化，另行確認）。三判決脈絡見
[CHANGELOG 2026-07-09-03](../CHANGELOG.md#2026-07-09)。
