<!-- 用途：CIFAR-10 confirmatory 重跑的 pre-register 增修（規格 1/2/4/5 與衛生界線），於任何 confirmatory 資料前凍結。 -->

# 2026-07-05-08 confirmatory 重跑預先登記增修

## 目標

pre-register 稽核判定：已完成的多 seed 主結果（Stage 4, commit 57a830c）其 grid 在看過自訓 scout 結果
後才鎖定、且 seed 0 與 scout 共用，故只能定位為 exploratory。為取得 clean confirmatory 主結果，本檔在
執行任何 confirmatory 資料生成之前，一次寫死並凍結以下所有規格。本檔 commit 時間須早於後續一切
confirmatory 產出（git 可驗）。凍結後不依任何後續結果調整。

pilot（Stage 1–4）保留為 exploratory 假設來源，引用時一律標 exploratory。

## 規格 1 — FID gate（嚴格帶、先量準、範圍明文界定）

- 先用 50k 樣本重量 base model（w=1）FID，不重訓。現有 13.95 是 5k 量、有已知小樣本正偏誤，與文獻
  ~3–8（皆 50k）不可比。
- 事前寫死 acceptance 帶 = clean-fid ≤ 10（依同規模 from-scratch CFG CIFAR-10 文獻，多落 ~3–8＠50k，
  ≤10 為堪用上限）。
- 用 50k 對此帶重判：≤10 過 gate；>10 再訓或調取樣（該付成本）。不接受「5k 高估」當放水理由。
- gate 涵蓋範圍明文界定：gate 只就 base model（w=1）保真度 pass/fail。guidance 軸樣本品質不以 FID 作
  gate（高 guidance 刻意以多樣性換保真、FID 混淆兩者），改由 per-config precision（fidelity）+ coverage
  （diversity）刻畫。另在 confirmatory 每個 per-config 生成集上算 clean-fid 作 characterization（重用生成、
  只報告不 gate、不需額外 50k），呈現 FID 隨 guidance 的軌跡供透明檢視。

## 規格 2 — judge（凍結 93.08%、逐類扣 floor）

- judge 凍結：已過事前 ≥93% 門檻，凍結為固定儀器不再訓（看過 pilot 才升級量測工具＝結果導向自由度）。
  checkpoint checkpoints/cifar10_judge.pt、真實測試準確度 93.08%、逐類誤判率（步驟 1/2 算出後附此檔）
  作固定儀器規格，之後不動。
- label-noise 扣 real floor，逐類算、逐類相減：不可用單一全域 ~7%。judge 誤判在難類（貓/狗、鹿/馬）
  遠高於易類，全域減會混淆「難類本就難判」與「guidance 製造污染」。以真實 CIFAR-10 各類的 judge
  誤判率為各類 floor，報「generated 各類 label-noise 減該類 floor」的超額值。
- 超額 label-noise 隨 confirmatory fresh multi-seed 報信賴區間，不以裸點值進 docs；須能看出 w1 超額離
  0 多遠（含 CI）。

## 規格 5 — confirmatory grid 上緣（先量崩點、不猜）

- 先做便宜的上緣 coverage-only scout：1 seed、少樣本、只量 coverage（不訓 TSTR、不跑 judge）。
- 事前 w 上限 = 20，掃描集 {8, 10, 12, 16, 20}。
- 量化觸底判準（事前定死 X = 0.02）：相鄰兩點 |Δcoverage| < 0.02 即視為觸底/回穩；取第一個滿足者之
  w 為崩點界。X 為絕對 coverage 變化，事前定死，不得看曲線後再挑。
- 到 w=20 仍未觸底的分支：若掃到上限仍 |Δcov| ≥ 0.02，不臨場續掃——視為新 pre-data 決策，新增
  records + commit（時間早於新資料）提高 w 上限後再掃。
- 依 scout 定死 grid 上緣 = 觸底界 + 1 點，凍結。

## 跨項 pre-register 衛生界線

- grid 上緣只由 coverage-only scout 決定：coverage 是幾何量、非主結果指標（TSTR/regret 才是），用它定
  範圍不算對主結果 HARKing；scout 只看 coverage，grid 一定死即不再依任何後續結果調整。
- 中段密度用與 pilot 峰無關的固定間隔規則：Δw=0.5 於 [1,3]（標準 CFG 低段、先驗關注區）、Δw=1 於
  [3, 上緣]。所產生的 1.5/2.5/… 來自均勻間隔，非來自 pilot 峰位置。
- seed 獨立：上緣 coverage scout 可用任意 seed，其資料不得進入 confirmatory 統計；confirmatory 只用未被
  pilot（seed 0/1/2）及上緣 scout 用過的 fresh seeds（10, 11, 12）。
- 規格 4（DINOv2/Inception 交叉裁決）：DINOv2 為準、Inception 僅作 robustness；兩者對「coverage 崩點
  w」定性矛盾 → 視為結果不穩、需更多 seed，不得事後挑支持者。confirmatory 彙總須同時輸出兩表徵的
  per-config coverage 並套此規則。
- 規格 3（near-boundary threshold 凍結）：已符合。threshold 0.9525 由真實資料定，以固定 --threshold 套用
  所有 guidance 的合成集（run_cifar_cfg_multiseed 對所有 w 用同一 threshold）。此凍結不隨合成/guidance 變。

## 後續（執行順序，每步先呈報再續）

1. 本檔 commit 凍結（本步）。
2. 50k FID 重量並對事前帶 ≤10 重判 base model gate。
3. 上緣 coverage-only scout（{8,10,12,16,20}，X=0.02）定崩點界；未觸底走再登記分支。
4. 依固定間隔規則 + 觸底界+1 定死 grid，回填本檔並凍結。
5. 開跑前先報 GPU 時間估計；經同意後跑 fresh-seed（10,11,12）confirmatory 多 seed，含 per-class 超額
   label-noise（帶 CI）與 per-config FID characterization。
6. 裁決：與 pilot 質性一致才升 confirmatory 寫入 README/docs。
