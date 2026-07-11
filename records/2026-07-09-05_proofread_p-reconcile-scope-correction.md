<!-- 用途：依 records/2026-07-09-04 §1.4c 之一行補充 record 訂正——P 對帳「逐位可重現」之精確範圍。P0/P1/B 定稿原文不回改（§4），本檔為訂正留痕：對帳限量測 scalar，TSTR 非決定性、不在對帳集。 -->

# 訂正：P 對帳「逐位可重現」之精確範圍

## Goal

依執行指令（二）`records/2026-07-09-04` §1.4c，訂正 P 對帳措辭之過寬處，不回改原記錄本文（`records/2026-07-06-05`
§4、`records/2026-07-09-04` §4 之凍結義務）。

## Result

`records/2026-07-08-04`（P0）、`records/2026-07-09-01`（P1）、`records/2026-07-09-03`（B 定稿附錄）皆有
「整個 confirmatory 結果集（在本環境）逐位可重現」之表述。此措辭**過寬**：對帳集只含量測對帳 scalar，
不含 TSTR。精確表述為：

- **全 30 config 之量測對帳 scalar 逐位重現凍結 JSON**——precision、coverage（DINOv2 與 Inception 兩側）、
  char_clean_fid、near_boundary_frac、label_noise_excess_mean，共 7 項，worst rel_delta = 0。
- **TSTR 不在對帳集**：TSTR 依協定（dossier `records/2026-07-06-06` 甲-8）之訓練含未種子化 DataLoader
  shuffle（吃全域 RNG），非決定性，故不重生成、不對帳、不宣稱逐位重現。凍結 JSON 之 TSTR 續為判決三依據。

原記錄本文不回改（凍結義務）；本檔為訂正留痕。`docs/results_analysis.md` 之 E3 段（`records/2026-07-09-04`
§1.3c）已採精確措辭（「量測對帳 scalar 逐位重現」＋TSTR 排除之明文），無過寬。

## Follow-up

- 後續一切引用 P 對帳結論者，以本訂正之精確範圍為準（量測 scalar 逐位、TSTR 排除）。
- 不涉任何凍結數字之變更；純措辭精確化。
