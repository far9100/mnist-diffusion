<!-- 用途：CIFAR-100 confirmatory 的 D1 揭盲路由裁決（§1.3 對外證據）。載明客觀觀察量、四分支路由結果、作者簽核與涵義。揭盲後定稿，隨 repo 發布。 -->

# CIFAR-100 confirmatory 揭盲裁決（D1 路由）

本文是 CIFAR-100 confirmatory 的正式揭盲裁決。依預註冊 `docs/prereg_cifar100.md` 的四分支決策樹
（D1），由三項**客觀觀察量**路由；路由不由先驗（D2 登記之分支一約一成）決定。confirmatory 真跑
見 [CHANGELOG 2026-07-16-01](../CHANGELOG.md#2026-07-16) 與 [2026-07-11-10](../CHANGELOG.md#2026-07-11)。

## 客觀觀察量

三項讀數皆自凍結的 `results/cifar100_cfg_confirmatory.json`（8 seed 10–17、grid 10 點、reps 5、
per_class=real_per_class=500）純衍生，可逐值重現。

### C1（which-FID 分離）：不分離

口徑（D6）：FID-argmin 與 TSTR-argmax 在 grid 上相隔 > 1 格才算分離。

- Inception clean-fid（`char_clean_fid`）argmin = w1.5，TSTR argmax = w1，相隔 **1 格**；逐 seed
  分離格步 > 1 的數目 = **0/8**（`results/cifar100_c6_fidmin_duel.json`：`incep_sep_steps` 全 1、
  `incep_separated_seeds_gt1` = 0、`mean_sep_step` = 1）。判**不分離**。

口徑範圍註記（對 [CHANGELOG 2026-07-16-01](../CHANGELOG.md#2026-07-16) 「兩空間」措辭的精確化）：
CIFAR-100 confirmatory 只算了 **Inception clean-fid 一個 FID 空間**（另有 Inception 與 DINOv2 兩套
PRDC 的 precision/coverage，但 PRDC 非 Fréchet distance）。CIFAR-10 §E3 的 FD-DINOv2 來自 P1
streaming 的 per-config 特徵（[CHANGELOG 2026-07-09-01](../CHANGELOG.md#2026-07-09)）；CIFAR-100
無對應 P1 跑，**未算 per-config FD-DINOv2**。故 CIFAR-100 的 which-FID 分離只在 Inception 空間評估，
DINOv2 側的 which-FID 分離不可得。此為口徑範圍限制，不改路由：可得的 FID 空間判不分離，未算的空間
不能宣稱分離。DINOv2 PRDC 的 coverage 峰（w2.5）距 TSTR-opt（w1）亦遠，與不分離一致。

### H2（selector）：CaF-v2 未勝 FID-min（打平）

matched-budget FID-min 對決（`results/cifar100_c6_fidmin_duel.json`）：CaF-v2（recall selector）與
更便宜的 FID-min 逐 seed **regret 完全相同**（per-seed `[0.79, 1.15, 0.45, 0.87, 0.67, 0.73, 0.91,
0.52]`，兩者均值各 **0.76pp**），逐 seed 同選 w1.5、oracle 皆 w1。D4 門檻要求 CaF-v2 per-seed regret
至少低於 FID-min **1.5pp**；實得差 0.00pp。**selector 主張不成立**。

### H1（機制）：三觀察量複製

D3 三觀察量（`results/cifar100_d3_observables.json`，純衍生）：(i) 升段 [w1..w2.5] near-boundary
單調降、(ii) 高段 [w2.5..w8] coverage 與 TSTR 同崩、(iii) 高段 near-boundary 谷（w4）後回升脫鉤——
**三項全成立（3/3）**，過三中二門檻。**機制在 CIFAR-100 複製**。

## D1 路由：branch 3（診斷論文）

四分支樹（`docs/prereg_cifar100.md` D1）：

1. 分離 且 CaF-v2 勝 FID-min → 邊界條件復活。（**排除**：不分離、CaF-v2 未勝。）
2. 分離 但 CaF-v2 敗 → thesis 活、selector 死。（**排除**：不分離。）
3. **不分離 但機制複製 → 診斷論文。**（**符合**：不分離＋機制複製 3/3。）
4. 皆否 → 負結果短文。（**排除**：機制複製，非「不複製」。）

客觀觀察量唯一相容 **分支三**。branch 1/2 需分離、branch 4 需機制不複製，皆被資料排除。

## 作者簽核

作者於 **2026-07-17** 確認揭盲路由為 branch 3。此為預註冊揭盲之不可逆裁決（`claude.md §4.2`）。

## 涵義

- thesis 頭條「取樣效用最優偏離保真最優（FID≠效用）」在 CIFAR-100 尺度**無正面支持**（which-FID
  不分離），與 CIFAR-10 confirmatory 同向（[CHANGELOG 2026-07-09-03](../CHANGELOG.md#2026-07-09)）。
- 免訓練 selector（CaF-v2）的操作優勢在 CIFAR-100 **不成立**（與更便宜的 FID-min 打平），為 MNIST／
  CIFAR-10 之後的第三個資料點；CIFAR-100 與 CIFAR-10 同型（FID-min 近最優），非 MNIST 那型
  （coverage 主導、CaF 選中 oracle）。
- 機制觀察量（coverage 與 near-boundary 的雙段行為）在更難的 CIFAR-100 **仍可量測、仍複製**，但不足以
  使任一便宜代理跨資料集普遍可靠。
- 論文形式為 **分支三診斷論文**（`docs/paper_branch3_diagnostic.md`）：貢獻由「普適免訓練選擇器」降為
  「代理可靠性之資料集相依性及其量測方法學」。分支四（負結果短文）因機制複製而排除。

## 尚待補之機制與護城河證據（不改路由）

- D3 介入臂（C3 coverage-matched pruning）：預註冊 D3 宣稱之機制複製的**介入**證據（觀察量已成立），
  待重生成合成集後執行（GPU）。非路由輸入。
- H3 CaF-v2 vs 簡化 Chamfer matched-budget 對決：D7 於分支三列為選配；作者裁定納入，排 GPU 佇列。
