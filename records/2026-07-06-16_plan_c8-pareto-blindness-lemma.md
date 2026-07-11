<!-- 用途：C8（Pareto 失明引理）——形式化為何 CaF 於 (precision, coverage) 平面被支配組態存在時結構性選不到 oracle。本引理由 confirmatory 之 (precision, coverage) 已知值證出、無需重生成，故本輪即可成文。一頁版入 docs（B 定稿時）。exploratory（C0）。依 records/2026-07-06-05 §1.3、§6 C8。 -->

# C8：Pareto 失明引理

## 目標

形式化 CaF 之結構性失明：當某組態於 (precision, coverage) 平面嚴格支配所有 TSTR-oracle 時，CaF 於任何 τ 皆選不到
oracle。此為引理（由已知值證出、無需重生成），成文一頁入 docs。exploratory（C0）。

## 結果（引理與證明）

- **CaF 定義**：selected = argmax coverage s.t. precision ≥ τ。
- **引理**：設組態 c* 嚴格支配每個 oracle o（precision(c*) > precision(o) 且 coverage(c*) > coverage(o)）。則對任何 τ，
  CaF 不選任何 o。
- **證明**：任取 oracle o。
  - 若 τ ≤ precision(o)：o 可行，c* 亦可行（precision(c*) > precision(o) ≥ τ）；coverage(c*) > coverage(o) → CaF 於
    可行集取 max coverage，必不選 o（c* 覆蓋更高）。
  - 若 precision(o) < τ ≤ precision(c*)：o 不可行、c* 可行 → CaF 不選 o。
  - 若 τ > precision(c*)：c* 與 o（precision 更低）皆不可行 → CaF 不選 o。
  三情形皆不選 o。故任何 τ 下 CaF 選不到任何 oracle。QED。
- **CIFAR-10 實例**：c* = w2.5 (.873, .792) 嚴格支配 w2 (.858, .777)、w1.5 (.841, .751)、w1 (.806, .645)——三 oracle
  之 precision 與 coverage 皆低於 w2.5。故 CaF 結構性選不到 oracle，非校準問題（tau_robustness.picks 為實證：τ 全段皆選
  w2.5 或更高，見 dossier 乙-5）。這是「內部最優」在 selector 層失效的結構根因。

## 後續

一頁版於 B 定稿時入 docs/（非治理文件）。引本引理於判決三（Pareto 失明）。與判決二獨立。引理不依賴任何未算數字。
