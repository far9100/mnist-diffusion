<!-- 用途：B 補遺（一行連結）——將 C 批 exploratory 結果連結入 B 定稿之閱讀脈絡，B 定稿 records/2026-07-09-03 原文不回改（-09-04 §4 凍結義務）。依 -09-04 §2.4。 -->

# B 補遺：C 批 exploratory 結果連結

## Goal

依執行指令（二）`records/2026-07-09-04` §2.4，將 C 批 exploratory 結果以一行連結入 B 定稿之閱讀脈絡；
B 定稿 `records/2026-07-09-03` 原文不回改（§4 凍結義務）。

## Result

B 定稿三判決之機制/穩健性補強（exploratory 補遺，非判決輸入，與三判決獨立並讀）：

- **C1 which-FID 裁決**（`records/2026-07-09-02`）：兩表徵空間皆不分離，CIFAR-10 頭條反證確立（已入判決一）。
- **C4 變異分解**（`records/2026-07-09-07`）：σ_cls=2.963 > σ_gen=1.182；上升肢 unresolved。餵 D4。
- **C7 small-probe FID 穩定性**（`records/2026-07-09-06`）：Kendall τ 高、FID-argmin 15/15 穩定於 w1.5；
  FID-min baseline 小 probe 可靠。餵 D5。
- **C2/C3/C5 介入式證據**（`records/2026-07-09-08`）：三項介入未對 near-boundary 因果機制提供強支持（單 seed、
  σ_cls 主導），與 C1 反證方向一致；機制最終回答移 CIFAR-100（D3）。
- **C8 Pareto 失明引理**（`docs/c8_pareto_blindness.md`）：已入判決三。

## Follow-up

- 全數 exploratory、不改三判決；`records/2026-07-09-03` 不回改。E3（`docs/results_analysis.md`）補遺段鏡像本連結。
- 機制之 confirmatory 回答於 CIFAR-100 工作包 D（D3 預註冊 coverage-matched 介入臂）。
