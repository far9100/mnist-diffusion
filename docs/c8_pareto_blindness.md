<!-- 用途：C8 Pareto 失明引理之一頁版（非治理文件）。形式化 CaF 於 (precision, coverage) 平面被支配組態存在時結構性選不到 oracle，附證明、CIFAR-10 實例與兩補強（空可行集 fallback、單調 selector 不可能性推廣）。exploratory（C0）。依 records/2026-07-06-16、2026-07-08-02 §4.2。 -->

# C8：Pareto 失明引理

CaF 選擇器 `selected = argmax coverage s.t. precision ≥ τ` 有一個結構性盲點：當某組態在
(precision, coverage) 平面上嚴格支配所有 TSTR-oracle 時，CaF 在任何門檻 τ 下都選不到 oracle。
這不是校準問題（換 τ 無解），是選擇器形式本身的限制。

## 引理與證明

**引理**：設組態 c\* 嚴格支配每個 oracle o，即 precision(c\*) > precision(o) 且 coverage(c\*) > coverage(o)。
則對任何 τ，CaF 不選任何 o。

**證明**：任取一個 oracle o，分三種 τ：

- τ ≤ precision(o)：o 可行；c\* 也可行（precision(c\*) > precision(o) ≥ τ）；而 coverage(c\*) > coverage(o)，
  CaF 在可行集取 coverage 最大者，必不選 o。
- precision(o) < τ ≤ precision(c\*)：o 不可行、c\* 可行，CaF 不選 o。
- τ > precision(c\*)：c\* 與 precision 更低的 o 皆不可行，CaF 不選 o。

三種情形皆不選 o。故任何 τ 下 CaF 選不到任何 oracle。QED。

## CIFAR-10 實例

confirmatory（`results/cifar10_cfg_confirmatory.json`，均值）中：

- c\* = w2.5：precision .873、coverage .792。
- 三個 oracle（per-seed TSTR 最佳 [w2, w1, w1.5]）：w2 (.858, .777)、w1.5 (.841, .751)、w1 (.806, .645)。

w2.5 的 precision 與 coverage 皆高於三個 oracle，構成嚴格支配。故 CaF 結構性選不到 oracle：
`tau_robustness.picks` 實證此點——τ 全段皆選 w2.5 或更高（dossier `records/2026-07-06-06` 乙-5），
非 w1 刀鋒。這是「內部最優」在 selector 層失效的結構根因。

## 補強一：空可行集與 coverage 平手（fallback 行為明文化）

引理證明的第三種情形（τ > precision(c\*)）隱含一個未明說的假設：CaF 不選不可行點。把它成文：

- **空可行集**：若 τ 高到可行集 `{c : precision(c) ≥ τ}` 為空，CaF 無合法選擇。實作上須定義 fallback
  （例如：放寬 τ 至最高可行、或回報「無選擇」）。本引理的三情形分析假設「不可行點不被選」，此假設在空
  可行集時使 CaF 無輸出，屬校準失敗而非引理反例。
- **coverage 平手**：`argmax coverage` 遇平手時取決於 tie-break（實作為索引序）。平手不改引理——嚴格支配
  下 c\* 的 coverage 嚴格大於任何 oracle，不進入平手。

## 補強二：單調 selector 不可能性（推廣，供 D8 引用）

Pareto 失明不是 CaF 特有，而是任何「對 (precision, coverage) 嚴格單調遞增」之 selector 的通性：

**推廣**：設 selector 為 `argmax f(precision, coverage)`，其中 f 對兩個引數皆嚴格遞增。若 c\* 嚴格支配
oracle o，則 f(c\*) > f(o)（f 嚴格遞增，兩引數皆大），故該 selector 不選 o。

一行證明：precision(c\*) > precision(o) 且 coverage(c\*) > coverage(o)，f 嚴格遞增 ⇒ f(c\*) > f(o)。∎

**推論（供 D8 CaF-v2 規格）**：任何純粹在 (precision, coverage) 上單調的 selector，在嚴格支配情形下都
選不到 oracle。故 CaF-v2 若要脫離此盲點，**必須引入第三訊號**（例如 near-boundary 供給量）或**放棄
單調性**。單靠調 τ 或換 (precision, coverage) 的單調組合無法解決。
