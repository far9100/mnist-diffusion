<!-- 用途：confirmatory 揭盲後之稽核材料交付（dossier）。彙整外部嚴審兩輪索取之全部原始材料（協定門檻、per-seed×per-config 全量、τ、FID、pilot、時序、記錄檔、TSTR 協定，及第二輪七項補件），為定稿計畫 records/2026-07-06-05 §1 事實基之「全量」backing。全部唯讀自 repo 現有檔案讀出，不存在者明確標示，不臆造數字。內容時序先於 -02/-03/-04/-05，落庫序號依 CLAUDE.md（NN=當日建檔序）取 -06。 -->

# confirmatory 稽核材料交付（dossier）

## 目標

為 selector/裁決者角色提供對 confirmatory 結果作過／敗判定所需之全部凍結門檻、原始數據、
參考值與協定細節。本檔是 records/2026-07-06-05 §1 十二項事實基的原始 backing；§1 為蒸餾結論，
本檔為可逐項核對之來源。全部唯讀，凡不存在者明確標示。

## 結果

### 元資料

results/cifar10_cfg_confirmatory.json：grid 10 點 {1,1.5,2,2.5,3,4,5,6,7,8}、seeds {10,11,12}、
per_class 1000、real_per_class 1000、near_boundary 0.9525、steps=50、eta=0、
coverage_feature「DINOv2 primary + Inception cross-check」、tstr_epochs 15。

### 甲、第一輪八項（按優先序）

**1. 協定 2026-07-05-02 §6：regret／top-k 定義與凍結門檻**
- regret@selected = TSTR(CaF 選中) − TSTR(完整未剪枝網格最佳)，掃描期間不得先剪枝。
- top-k 命中 = 選中是否落在 TSTR 前 k 佳，實務 k=3。
- 基礎協定無數字門檻（go/no-go 為定性五條）；數字門檻在增修：clean-fid ≤10 @50k base w=1（-08 規格1）、
  judge 93.08% real-test 逐類 floor（-08 規格2）、near-boundary 0.9525（-08/-12）、coverage 觸底 X=0.02
  （-08 規格5）、grid 封頂 w=8（-11）、C2 偏 Spearman α=0.05 n=10（-13）。

**2. per-seed×per-config 全量（五指標齊全）**
- TSTR(mean；per-seed)：w1 63.16(60.49,67.52,61.47)／w1.5 63.96(58.71,67.24,65.94)／
  w2 61.16(61.16,62.78,59.54)／w2.5 61.19(60.62,62.49,60.45)／w3 53.08／w4 47.20／w5 44.10／
  w6 39.10／w7 36.06／w8 33.46。
- Precision(mean)：w1 .806／w1.5 .841／w2 .858／w2.5 .873／w3 .876／w4 .879／w5 .869／w6 .859／
  w7 .848／w8 .836。
- Coverage(DINOv2,mean)：w1 .645／w1.5 .751／w2 .777／w2.5 .792／w3 .778／w4 .745／w5 .698／
  w6 .648／w7 .604／w8 .559。
- Near-boundary_frac(mean)：w1 .256／w1.5 .114／w2 .063／w2.5 .046／w3 .037／w4 .033／w5 .037／
  w6 .039／w7 .048／w8 .059。
- Label-noise excess_mean：w1 +.044／w1.5 −.038／w2 −.057／w2.5 −.062／…／w8 −.062。
- selection：per_seed 全 w2.5、modal_fraction 1.0、regret@selected mean 3.687(0.54/5.03/5.49)、
  rank_per_seed [2,4,3]、topk_hit_rate 0.667、oracle_best_per_seed [w2,w1,w1.5]。
- 稽核旗標：三 seed 全選 w2.5，oracle-best 都在更低 w，系統性過度導引，regret 最大 5.49pp，top-3 僅 2/3；
  對照 pilot（regret 0.28、top-3 100%）為明顯退化。

**3. τ、real-vs-real 參考、可行集**
- seed10 ref_precision .88656 → tau .79791（stability .818），selected w2.5，oracle w2，regret 0.54，rank 2。
- seed11 ref_precision .90153 → tau .81137（stability .909），selected w2.5，oracle w1，regret 5.03，rank 4。
- seed12 ref_precision .89517 → tau .80565（stability .909），selected w2.5，oracle w1.5，regret 5.49，rank 3。
- tau = 0.9 × real_ref_precision（selector.py auto_tau）。real-vs-real 參考 = 真實 DINOv2 特徵對半切互量
  precision（run_cifar_cfg_multiseed.py:54、real_ref_precision，local Generator seed=plain seed）。
  核對 0.9×.88656=.79791。
- 可行集 = {c: precision(c) ≥ tau}（selector.py），JSON 無 feasible 欄位，可由 tau_robustness.picks 重建（見乙-5）。

**4. 每 config FID**
- per-config fd_dinov2 不存在；FD-DINOv2 僅 guidance=1.0：5k clean-fid 13.95／fd_dinov2 324.26，
  50k clean-fid 8.95／fd_dinov2 286.41。
- per-config 只有 char_clean_fid（reported not gated，對 cifar10 train）：w1 10.98／w1.5 8.82／w2 10.18／
  w2.5 12.20／w3 14.42／w4 18.87／w5 22.46／w6 25.91／w7 28.77／w8 31.33。
- 旗標：選中 w2.5 之 clean-fid 12.20 差於 w1.5 之 8.82；char_clean_fid argmin(w1.5) 與 TSTR argmax(w1.5) 重合。

**5. pilot 檔、seeds、無重疊、峰位移**
- pilot = results/cifar10_cfg_multiseed.json，seeds {0,1,2}，per_class 500，grid {1,2,3,4,5,8}。
- confirmatory fresh seeds {10,11,12}，與 pilot 及上緣 scout（seed 0）皆不重疊（-08 L53-54、-11、-12、-13 佐證）。
  （噪聲層級之非重疊經 α 進一步核實，見 -05 §1.12 及本輪 α 報告。）
- 峰位移（現象有記，詞未逐字）：proxy w≈1.5 → scout w≈3 → multiseed w=2。grid 間距刻意與 pilot 峰無關（反 HARK）。

**6. git log 與時序**
- 052492c @ 2026-07-05 14:59:54：規格凍結（寫死 fresh seeds、acceptance 帶、交叉裁決），早於一切 confirmatory 產出。
- ec1f746 @ 2026-07-05 23:53:43：儀器（見乙-2），早於全量開跑。
- confirmatory.json mtime 2026-07-06 08:45:11；timing 29763s（8h16m）。
- 起跑時間戳：無登記（JSON metadata 無 timestamp）；推導起跑 ≈2026-07-06 00:29（mtime−ELAPSED），為計算值非登記值。
- author date = commit date（全 35 commit），無 rebase/amend 分歧。

**7. 記錄 -11／-08／-07**
- -11：封頂 w=8；Δw=0.5 於[1,3]、Δw=1 於[3,8]；10 點×3 seeds=30 configs；steps=50 eta=0。
- -08 規格4 交叉裁決：DINOv2 為準、Inception 僅 robustness；兩表徵對「coverage 崩點 w」定性矛盾則視為不穩需更多 seed，
  不得事後挑支持者。
- -07 §4：偏相關「相關非因果」且不足、要求介入式證據疊加（原文見乙-7）；-12/-13 卻升為 C2 主裁決程序，屬重新定位。

**8. TSTR 評估協定**
- 每 (w,seed) 訓練 1 個全新 ResNet18（無重複、無平均）；confirmatory 共 30 次。
- epochs=15；無早停（固定 epoch + CosineAnnealingLR）；SGD lr0.1 mom0.9 nesterov wd5e-4，batch128，augment on
  （flip + reflect-pad-4 crop）；無 val 分割、無 patience、無 best-ckpt。DataLoader shuffle=True 無 seed（吃全域 RNG）。
- tau 不得在 TSTR 上調整。judge 另訓（cifar_judge.py 25 epoch，93.08%）供 label-noise／near-boundary。

### 乙、第二輪七項補件

**1. MNIST sandbox：fidelity-opt 與 TSTR-opt 之分離（records 2026-07-03-02）**
- CaF 選 g1 = oracle TSTR-best g1；regret 0.000pp；rank 1/6；top-3 hit True。
- per_config TSTR：g1 97.3／g2 96.28／g3 95.01／g5 92.31／g7 88.85／g10 78.51，與 coverage(0.875→0.319) 單調同向。
- precision 於 g3 達峰 0.966，TSTR 卻於 g1 達峰 → fidelity 側(precision-opt g3) 與 utility 側(TSTR-opt g1) 分離一格步。
- 定性：-05 §1.7 標為軼事級（一格步、FID 側單 seed argmax、bespoke FID、方向不轉移）；MNIST 上機制敘事改由 coverage 單調性承載。

**2. ec1f746 diff（confirmatory 儀器）**
- 3 檔、+295/−19：run_c2_partial.py 新增 157 行（C2a/C2b 偏 Spearman + permutation + bootstrap CI）；
  run_cifar_cfg_multiseed.py +107（per-class 超額 label-noise、per-config char clean-fid、per_class=1000、metadata）；
  records/2026-07-05-14 +50。commit message 明言「純實作，不動已凍結規格」。
- 計時探針：1-config ≈1000s（char-FID ~106s、setup ~300s）；30 configs 估 6–8h（實際 8h16m）。

**3. char_clean_fid 出處**
- run_cifar_cfg_multiseed.py:89-94，measure() 內以 clean_fid_vs_dataset((gen+1)/2, dataset_name="cifar10",
  dataset_split="train", dataset_res=32) 對 CIFAR-10 train split 算；重用生成集、reported not gated。

**4. CaF probe n_real**
- real_per_class=1000（confirmatory）；real-vs-real 參考以真實 DINOv2 特徵對半切（各半互為 query/ref）算 precision，
  供 TSTR-free tau。

**5. tau_robustness.picks（每 seed 11 點 τ sweep）**
- seed10：τ 0.8126→0.8682 皆 w2.5，0.87515 w3，0.8821 w4。
- seed11：τ 0.8080→0.87127 皆 w2.5，0.8783 w4。
- seed12：τ 0.7965→0.86841 皆 w2.5，0.8764 w3。
- 讀法：此 sweep 於實際 τ 上緣往高掃，全程 w2.5 直到高 τ 使可行集縮才被迫升 w3/w4；示 modal 穩定，非 w1 刀鋒。
  w1 之可行刀鋒（seed11/12 邊際 .0034/.0092）為另算，非此 sweep 涵蓋。

**6. per-seed w1／w1.5 precision 與 TSTR**
- seed10：w1 prec .81260 cov .64090 tstr 60.490；w1.5 prec .83840 cov .74950 tstr 58.710。
- seed11：w1 prec .80800 cov .64680 tstr 67.520；w1.5 prec .83970 cov .74870 tstr 67.240。
- seed12：w1 prec .79650 cov .64690 tstr 61.470；w1.5 prec .84400 cov .75500 tstr 65.940。
- 配對差 w1.5−w1：−1.78／−0.28／+4.47pp（mean +0.80、SD 3.26、SE 1.88）→ 上升肢於 3 gen seeds 下不可判定。

**7. -07 §4 介入式證據原文**
> ## 4. 機制（C2）：因果性要求
> - 偏相關 `utility ~ coverage | precision` 是相關非因果；coverage 與 utility 可能同被潛在因子驅動。
> - margin/near-boundary 故事需介入式證據：控制住 coverage 後 margin 效應是否消失；或直接證明高 guidance 丟掉的
>   near-boundary 佔比 vs guidance；(b) coverage 受控下的 margin 條件分析。
- 張力：-07 §4 明列偏相關不足、要求介入式疊加；-12/-13 升其為 C2 主裁決。C2a 顯著亦不得寫「coverage 驅動效用」等因果措辭。

## 後續

本檔為唯讀交付，供 -05 §1 逐項核對，不含裁決。α 階段對 §1.11/§1.12 之碼級複核（gseed 公式、DDIM 決定性、
scout 種子路徑）另見本輪 α 報告與 -05 §1.12 更新。裁決材料完整性由後續 A/B 工作包引用。
