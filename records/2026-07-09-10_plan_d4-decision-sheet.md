<!-- 用途：D4 決策單（δ）——交付 σ_cls/σ_gen（C4）、MDE 表、H2 候選數字門檻與成立理由、D5 matched-probe 規格（引 C7）。D4 全部數字門檻為作者欄位（agent 不得代填，-09-04 §0.2）；本檔備齊候選與理由供作者一次填。STOP 呈報。依 -09-04 §3.1、-06-05 §7 D4。 -->

# D4 決策單：H2 數字門檻與功效規劃（數字待作者填）

## Goal

依執行指令（二）`records/2026-07-09-04` §3.1 備 D4 決策單：交付變異量、MDE 表、H2 候選門檻與成立理由、
D5 matched-probe 規格。**D4 全部數字門檻為作者欄位**（`records/2026-07-09-04` §0.2、`records/2026-07-06-05`
§7 D4 明文 agent 不得代填）；本檔僅備候選結構與方向無關之理由，數字由作者一次填。STOP 呈報。

## Result

### 1. 變異量（C4，`records/2026-07-09-07`）

σ_cls = 2.963pp（分類器訓練變異，主導）；σ_gen = 1.182pp（生成變異）。

### 2. MDE 表（80% power、α=.05 雙尾；兩組態 TSTR 差之最小可偵測效應）

單組態均值 SE = √(σ_gen²/n_seed + σ_cls²/(n_seed·n_rep))；MDE_diff = 2.80·√2·SE。

| n_seed | n_rep | SE_mean | MDE_diff (pp) |
|---|---|---|---|
| 3 | 1 | 1.842 | 7.29 |
| 3 | 2 | 1.389 | 5.50 |
| 3 | 3 | 1.200 | 4.75 |
| 5 | 3 | 0.930 | 3.68 |
| 8 | 3 | 0.735 | 2.91 |
| 8 | 5 | 0.628 | 2.49 |
| 10 | 3 | 0.658 | 2.60 |
| 10 | 5 | 0.562 | 2.22 |

**功效意涵**：CIFAR-10 之 confirmatory 配置（3 seed × 1 rep）MDE = 7.29pp，遠粗於上升肢觀測差 ≈2.52pp
（C4）；即現配置無法解析峰位。若 CIFAR-100 之 σ 同量級，欲偵測 ~2.5pp 峰位偏移需 MDE ≤ 2.5，對應
(n_seed, n_rep) ≈ (8,5) 或 (10,3) 以上。**建議登記主張採峰位噪聲穩健形式（高原＋懸崖，不押點峰）**，
與 `records/2026-07-06-05` §6 C4 事前寫死之交付一致。（此為功效判斷，非結果選購。）

### 3. H2 候選數字門檻（結構與理由；X 待作者填）

H2（selector 假設）在 CIFAR-10 無凍結數字門檻，故 CIFAR-10 selector 僅描述性（`records/2026-07-06-05`
§1.8）。CIFAR-100 前須凍結數字門檻。候選（理由與 confirmatory 數字方向無關、以 CIFAR-10 觀測為錨屬合法
先驗）：

- **候選 A（regret 上限）**：H2 通過 ⟺ CaF（或 v2）之 per-seed regret@selected ≤ **X_A** pp。
  錨定理由（方向無關）：X_A 應小於「災難性崖」量級（CIFAR-10 高段 w≥3 崩 11–30pp），使「通過」代表
  避崖有效；不得設為恰好通過或恰好否決 CIFAR-10 之 3.69。
- **候選 B（對 FID-min 之增益，與 D8 X 對齊）**：H2 通過 ⟺ CaF（或 v2）之 regret 至少比 matched-budget
  FID-min baseline **低 X_B** pp。錨定理由：CIFAR-10 上 FID-min（0.91）已勝 CaF（3.69），故 selector 之
  存在理由須為「勝過更便宜的 FID-min」；X_B 為該勝幅門檻。**與 D8 §價值判準之 X 為同一量**，建議一次定死。
- **候選 C（top-k 命中，輔助）**：top-3 命中率 ≥ **X_C**。輔助非主，因 Pareto 失明（C8）下 top-k 可被支配
  組態撐高，主門檻建議用 A 或 B。

作者填：擇 A／B（建議 B，直接對應「便宜代理」之 so-what）＋X 數字；C 是否列輔助。**agent 不代填 X。**

### 4. D5 matched-probe FID-min 規格（引 C7，`records/2026-07-09-06`）

C7：小 probe 下 FID 排序穩定（Kendall τ 最小 0.911＠100/class、FID-argmin 15/15 穩定於 w1.5）→ FID-min
baseline 於小 probe 可靠。**候選 matched-probe 尺寸**：≥100/class 即 τ≥0.91、≥250/class τ≥0.956；建議
matched-probe baseline 用與 CaF probe 同尺寸之真實參考（confirmatory 為 1000/class，遠在穩定區）。D5 三臂
對決（FID-min／CaF 或 v2／Chamfer）之 probe 尺寸由作者於 D 包定死；C7 證其在候選尺寸皆穩定。

## Follow-up

- **STOP：等作者填 D4**——H2 門檻（A／B／X 數字）、功效配置（n_seed × n_rep，參 MDE 表）、D5 probe 尺寸。
  agent 不代填數字（`records/2026-07-09-04` §0.2）。
- D4 數字定後併入 D 包（`records/2026-07-06-05` §7），與 D8、D 終簽一次過簽（§3.4）。
- 不觸凍結 JSON、無數字回改。σ 來源 `results/cifar10_c4_variance.json`、C7 來源 `results/cifar10_c7_probe_fid_stability.json`。
