<!-- 用途：A 批（H-C2）收束 record——彙整 A1（DINOv2 C2）、A2（Inception C2 robustness）、A3（block-perm (c) 裁定）之結果，並逐字鎖定 B 判決二之內容（顶格 C2b 框架句、主體三重 caveat、A2 增列、A 批總結句）。B 骨架之判決二直接引本檔逐字，不重寫。依 records/2026-07-06-07（A0）、2026-07-06-08（A3）與 results/cifar10_c2_partial.json（A1）。 -->

# A 批（H-C2）收束：判決二之凍結內容

## 目標

收束 A 工作包（A0 凍結、A1 主檢驗、A2 Inception robustness、A3 block-perm (c) 裁定），
把 B 判決二之措辭逐字鎖定於本檔，使 B 骨架直接引用、不於揭盲後重寫。判決二（C2）與判決三
（selector）獨立，本檔不涉判決三。

## 結果

### A1（DINOv2 主檢驗，-13 一字不動，N=100,000/seed=0）

- C2a partial ρ(TSTR, coverage | precision, excess_ln) = +0.658；permutation p(單尾>0) = 0.0188；
  bootstrap-over-seeds 95% CI = [−0.601, +0.675]（跨零）。
- C2b partial ρ(TSTR, precision | coverage, excess_ln) = −0.124；p(雙尾) = 0.744；不顯著。
- 機械 verdict = pass（C2a 顯著正 且 C2b 不顯著）。原始輸出 results/cifar10_c2_partial.json。

### A2（Inception robustness，同方法、N=100,000/seed=0，純離線）

- C2a partial ρ(TSTR, cov_incep | prec_incep, excess) = +0.859；p = 0.0008。
- C2b partial ρ(TSTR, prec_incep | cov_incep, excess) = +0.163；p = 0.654；不顯著。
- coverage 峰：DINOv2 w2.5 / Inception w2.0（差一格步）。兩曲線皆單峰、mid-peak、高段崩。

### A3（block-perm 敏感度，(c) 裁定）

方案 b（per-w 中心化殘差）經 §7 評估不可行（低段 TSTR 3-seed SE 蓋過上升肢訊號）；cell/config 錯配
加塊內 w 異質，CIFAR-10 內部無乾淨 restricted 補救。裁 (c)：不硬跑、誠實承認內部驗不了。

### B 判決二之凍結內容（B 骨架逐字引用，不重寫）

**顶格**：C2b 不顯著（兩表徵皆是）為 absence of evidence 非 evidence of absence——理論上 n=10 ＋
-13 §4 不對稱準則下近乎自動；經驗上兩表徵 C2b 符號相反（DINOv2 −0.124 / Inception +0.163），坐實其
為噪聲、無穩定結構。不得讀成「precision 不驅動」的正面證據，亦不得使 verdict_pass 被理解為
coverage/precision 雙向坐實。

**主體**：機械 verdict = pass（主檢驗 -13 定義 p=0.0188，不翻案）＋三重 caveat：(i) bootstrap CI
[−0.601, +0.675] 跨零（種子軸脆弱）；(ii) gseed 碰撞損害可交換性、CIFAR-10 內部無乾淨 restricted
補救（A3 (c)）、穩健性不可內部驗證；(iii) 全網格偏 ρ=+0.658 抹平雙段機制。

**A2 增列**：C2a 跨表徵一致（Inception p=0.0008、同號）為表徵不變性支撐，但為表徵層非種子層——不解
bootstrap 種子脆弱、同受可交換性 caveat；coverage 峰差一格（w2.5/w2.0）在噪聲內非 -08 §4 定性矛盾；
C2b 號翻（DINOv2 −0.124 / Inception +0.163）證其為噪聲。

**A 批總結句**：C2a 在 CIFAR-10 獲跨表徵一致的機械通過（兩表徵皆顯著正、同號），但穩健性沿兩條獨立
軸均無法內部驗證——種子軸（bootstrap CI 跨零、3 seed）與可交換性軸（gseed 碰撞、內部無乾淨 restricted
補救）；A2 補的第三軸（表徵）通過，但表徵一致不替代種子穩健或可交換性。三軸中一軸過、兩軸內部不可驗。
淨：CIFAR-10 已盡其所能，C2a 最終裁決須 CIFAR-100 獨立複製回答。

**結論**：CIFAR-10 機械通過且跨表徵一致，但穩健性須 CIFAR-100 獨立複製回答。

## 後續

- B 骨架之判決二逐字引本檔「B 判決二之凍結內容」全段，不重寫。
- A2 之 selector 在 Inception 上重放（需 real-vs-real Inception τ、真實 Inception 特徵）歸 C 批，延後。
- 判決二（C2）與判決三（selector，C6 FID-min 對決）獨立，B 骨架分立、不混寫。
