# 2026-07-03 — Phase 0：CaF 早期訊號（go/no-go 通過）

依修訂路線圖（`pure-cuddling-valiant.md`，C3-first）執行 Phase 0：在 MNIST sandbox 上建立
可重用基礎設施，並取得「CaF selector 是否可行」的關鍵早期訊號。

## 建立的基礎設施（皆含自我測試）
- `metrics_prdc.py` — 從零實作 PRDC（Naeem 2020 / Kynkäänniemi 2019 P&R），torch-only 免 sklearn；
  含 per-class（intra-class）變體。自我測試通過（matched dist coverage 0.97；collapsed 全 0）。
- `selector.py` — **CaF**：`argmax coverage s.t. precision ≥ τ`；**TSTR-free τ 自動決定**（真實 probe
  的 real-vs-real precision × fraction）；**τ 穩健性掃描**；**regret@selected / top-k 命中**（取代全域 Spearman）。
- `mechanism.py` — C2：以真實資料訓練分類器量 synthetic 樣本 margin 與 near-boundary 佔比；含 label-noise
  競爭機制檢查。
- `run_selector_signal.py` — 早期訊號驅動。
- `run_comparison.py` — 改寫為**三維聯合掃描 (steps×η×guidance)** + 每格 PRDC + per-class TSTR + 收尾 CaF。

## 關鍵早期訊號（guidance 軸，DDIM η=0 steps=50，per-digit 1000）

| guidance | precision | coverage | recall | TSTR% |
|---|---|---|---|---|
| 1 | 0.9106 | **0.8707** | 0.9108 | **97.22** |
| 2 | 0.9633 | 0.8491 | 0.7985 | 95.92 |
| 3 | 0.9659 | 0.7332 | 0.6498 | 95.33 |
| 5 | 0.9624 | 0.5534 | 0.4233 | 91.86 |
| 7 | 0.9487 | 0.4399 | 0.2692 | 87.68 |
| 10 | 0.9216 | 0.3225 | 0.1700 | 79.08 |

（real-vs-real 參考 precision = 0.9517 → auto-τ = 0.857）

**CaF 選中 g1 = oracle TSTR-best g1；regret@selected = 0.000 pp；rank 1/6；top-3 hit = True。**

### 三個要點
1. **coverage（與 recall）隨 guidance 單調下降、且與 TSTR 單調同向**——coverage 0.87→0.32 對 TSTR
   97.2→79.1，幾乎完美單調。**這就是 C2 機制的縮影**。
2. **precision 不追蹤 TSTR**（於 g3 達峰 0.966，TSTR 卻於 g1 達峰）→ 佐證「**驅動效用的是多樣性/coverage，
   而非保真度/precision**」。這是相對 Chamfer（無機制）的乾淨差異化。
3. **τ 穩健性誠實註記**：auto-τ 選 g1（regret 0），但 τ 掃描下 g2 為 modal（82%）——g1↔g2 對 τ 敏感；
   惟兩者皆近最優，robust 區間內 regret ≤ 1.3pp。CaF 穩定落在 top-2、auto-τ 命中確切最優。

### coverage vs recall
先前 per-digit-50 smoke 出現 coverage 非單調、mis-pick g3；per-digit-1000 平衡取樣後消失，證實為
**樣本數假影**（real 1000/class vs fake 50/class 不平衡）。平衡取樣下 coverage 與 recall 皆良好；
依 brief 保留 **coverage 為主**（Naeem，較穩健），recall 佐證。

## 誠實caveats（決定後續 go/no-go）
- 只驗證**單一 guidance 軸、單一 seed、η=0/steps=50**。完整 go/no-go 需**聯合 (steps×η×guidance) 網格
  + 多 seed（≥3）+ 難資料集**。
- **MNIST margin 飽和**：`mechanism.py` smoke 顯示 real near-boundary 佔比僅 ~0.006、gen ~0.000——MNIST 近乎
  可分，near-boundary 訊號微弱。機制的強證據需 CIFAR-100。MNIST 上改以 coverage 單調性承載機制敘事。

## Gate A 多 seed 硬化（seeds 0/1/2，per-digit 1000）— 決定性通過
`run_selector_signal.py --seeds 0 1 2`（PRDC+TSTR 同源樣本、含機制）彙總：
- **CaF 三 seed 全選 g1（modal 100%），regret mean/max = 0.0，top-3 命中率 100%**——
  sandbox 的 g1 選擇非單 seed 雜訊。
- per_config TSTR 均值：g1 97.3 / g2 96.28 / g3 95.01 / g5 92.31 / g7 88.85 / g10 78.51，
  與 coverage（0.875→0.319）單調同向。
- **機制跨 seed 成立**：near-boundary frac 隨 guidance 單調枯竭
  {g1 0.027, g2 0.001, g3+ ~0}，且 `nb_vs_tstr_aligned=True`。絕對值極小＝MNIST 飽和
  caveat，方向正確 → 強機制證據待 CIFAR-100。

## go/no-go 判定與下一步
**guidance 軸：綠燈且已多 seed 硬化**（regret 0×3、coverage-drives-utility、機制單調）。下一步：
1. 跑**三維聯合掃描**（`run_comparison.py`，per-digit 1000、多 seed）取得完整效用曲面 + 網格級 CaF regret。
2. 早 Phase 1：實作**簡化 Chamfer** 作 head-to-head baseline（現在值得投資，因 CaF 已驗證可行）。
3. 接 CIFAR-10/100（預訓練 EDM）——機制與難集驗證的主戰場。
