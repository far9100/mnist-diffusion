<!-- 用途：C1 which-FID 裁決——將凍結口徑（records/2026-07-06-11）機械套用於 P1 揭盲之 per-config FD-DINOv2 數字（records/2026-07-09-01），對號入座二元活結果空間。DINOv2 側不分離（1 格步落點），對應反證版敘事。exploratory（C0）。依 records/2026-07-08-02 §4.1、records/2026-07-06-11、records/2026-07-06-05 §6。 -->

# C1：which-FID 裁決（DINOv2 揭盲後對號入座）

## Goal

依執行指令 `records/2026-07-08-02` §4.1，將凍結於 DINOv2 揭盲前的 C1 口徑（`records/2026-07-06-11`）
機械套用於 P1 產出的 per-config FD-DINOv2 數字（`records/2026-07-09-01`），裁 DINOv2 特徵空間中
FID-optimal 與 TSTR-optimal 是否分離，對號入座 `records/2026-07-06-11` §3 之二元活結果空間與 §4 之
雙分支敘事。本裁決為 exploratory（C0）：口徑先於計算、不因結果回改；反證版與弱版本筆墨等量之敘事
已於揭盲前預寫，本 record 只對號、不臨時解釋。

## Result

### 套用凍結口徑（`records/2026-07-06-11` §1 主裁）

主裁：均值曲線之 FID-argmin 與 TSTR-argmax 相異且 **>1 格步** = 分離；否則不分離。口徑以格步距離判，
非 regret 大小。grid = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8]（格步＝格點索引距離）。

- **FD-DINOv2 均值 argmin = w2**（均值曲線：w1 282.4、w1.5 195.2、**w2 175.4（最小）**、w2.5 176.7、
  w3 188.7、w4 223.9、w5 261.1、w6 302.6、w7 341.9、w8 379.4；資料 `records/2026-07-09-01`）。
- **TSTR 均值 argmax = w1.5**（均值曲線：w1 63.16、**w1.5 63.96（最大）**、w2 61.16、w2.5 61.19、
  w3 53.08、w4 47.20、w5 44.10、w6 39.10、w7 36.06、w8 33.46；資料凍結 confirmatory JSON）。
- **格步距離：w1.5（索引 1）↔ w2（索引 2）= 1 格步。1 不 >1 → 不分離。**

### 裁決：DINOv2 側不分離（反證確立），1 格步落點

依 `records/2026-07-06-11` §3 活結果空間：DINOv2 =1 格步 → **不分離（反證確立）**，落於「1 格步」點——
DINOv2 一格偏移、仍判不分離，記該 1 格偏移為誠實邊註、與 Inception（clean-fid）0 格步對照。

疊加 §2 強命題 pre-close（Inception/clean-fid 側：均值 argmin w1.5 = TSTR argmax w1.5，0 格步，不分離；
per-seed 核對 C6 格步 [1,1,0] 皆 ≤1），**兩特徵空間皆不分離**。

### 對號入座 §4 反證版（揭盲前預寫，此處引用不改寫）

FD-DINOv2 之 argmin 與 TSTR-argmax 差 ≤1 格步，即 DINOv2 空間中 FID 亦不誤導。疊加 Inception 側不分離，
**兩特徵空間皆不分離 → CIFAR-10 尺度「FID≠效用」頭條命題反證確立**：FID（兩表徵）皆為近最優
selector，CaF 無附加價值。此與判決三（FID-min regret 0.91 勝 CaF 3.69，`records/2026-07-06-10`）一致。
thesis 於 CIFAR-10 尺度被資料反證，科學重心移 CIFAR-100（是否更難資料集上 FID 才誤導）。

落點區分（揭盲前預寫，零自由度）：本次為 **1 格步 = DINOv2 一格偏移、仍判不分離**，記該 1 格偏移為誠實
邊註、與 Inception 0 格步對照；「一格偏移」≠「全重合」，邊註不改判、只誠實記經驗差異。

### 誠實邊註（輔助：三 seed 方向一致性，`records/2026-07-06-11` §1 輔助——不單獨構成裁決）

- FD-DINOv2 per-seed argmin 三 seed 全為 w2（seed10 175.3、seed11 174.6、seed12 176.4）。
- TSTR per-seed argmax：seed10 w2、seed11 w1、seed12 w1.5。
- per-seed 格步（FD-argmin vs TSTR-argmax）= [0, 2, 1]。方向非三 seed 一致（seed11 達 2 格步、seed10
  重合 0 格步）。依 §1 輔助口徑，主裁判不分離時此為誠實穩健性邊註，**不單獨構成分離裁決、不覆蓋均值主裁**。
  記於此以誠實呈現 per-seed 離散度；均值主裁（1 格步、不分離）為裁決依據。

### 反 HARK（`records/2026-07-06-11` §5）

口徑於 DINOv2（唯一未見臂）揭盲前凍結（commit 21abd7f/b267d2b，早於本次 P1 之 FD-DINOv2 計算）；對已見
之 Inception 側無鑑別力（任何合理口徑皆判不分離）；雙分支敘事於揭盲前等量寫定。故口徑選擇無法對已見
資料 HARK，本裁決為凍結規則之機械對號，零設計自由度。

## Follow-up

- **判決一（thesis，B 定稿）引本 C1 結果**（`records/2026-07-06-11` §6、`records/2026-07-08-02` §4.4）：
  CIFAR-10 尺度頭條反證確立（兩表徵皆不分離）；「內部最優」仍為未登記 exploratory 觀察；「必然次優」
  全稱句撤下（E2）。C1 與判決三（selector）獨立。
- C1 為全案 thesis 於 CIFAR-10 尺度之唯一未決缺口，本裁決後該缺口關閉（結論：反證）；thesis 存活與否
  之最終回答移 CIFAR-100（工作包 D）。
- 後續 γ：C2/C3/C5/C7 計算（取 P 資產）、C8 一頁版補強、B 定稿（§4.4 STOP 呈報）。
- 本裁決不觸凍結 JSON、不回改任何數字；FD-DINOv2 原始數字留痕於 `records/2026-07-09-01` 與
  `results/cifar10_p1_streaming.json`。
