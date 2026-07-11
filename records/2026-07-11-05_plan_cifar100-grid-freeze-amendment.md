<!-- 用途：CIFAR-100 confirmatory grid 凍結 amendment（D10 第四閘）。scout 後定死網格，早於任何 confirmatory 取樣。 -->

# 2026-07-11-05 CIFAR-100 confirmatory grid 凍結 amendment（D10 第四閘）

## Goal

依 D 包 `records/2026-07-09-13` D10：scout（`records/2026-07-11-04`）後、confirmatory 取樣前，一次定死
CIFAR-100 confirmatory 的 guidance grid，並登記其餘量測規格。本 amendment 為 pre-registration 步驟，
早於任何 confirmatory 合成樣本；grid 選擇不得以 scout 之 TSTR 讀數為據（D10：scout 讀數不回饋判準）。

## Result

### 凍結 grid（作者裁定，本 session）

confirmatory guidance grid = **{1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8}**（10 點，封頂 w8）。沿用 CIFAR-10
confirmatory 之凍結 grid（`records/2026-07-05-11`）。

grid 選擇依據（不含 TSTR 讀數）：

- **coverage 崩點定位**：scout 之 DINOv2 coverage 峰在 w2、崩點（峰後首個跌破 90% 峰值）在 w5；Inception
  交叉檢查同形。低段密點 {1,1.5,2,2.5,3} 解析 coverage 峰與轉折，{4,5,6,7,8} 涵蓋崩尾。
- **跨資料集可比性**：與 CIFAR-10 同 grid，使 CIFAR-100 之 coverage/TSTR 曲線可與 CIFAR-10 直接對照
  （D0 之 CIFAR-100=validation 角色）。
- **CFG 實用範圍封頂**：w8 上限沿 `records/2026-07-05-11`。
- 明文排除：scout 之 TSTR 峰落 w1 為 exploratory 觀察，不作 grid 選擇依據；grid 選擇僅用 coverage 幾何
  與可比性。

### 一併登記（多數已於 D 包凍結，此處彙整並補 grid 相依項）

- 量測 sampler：steps=50、eta=0（固定）。
- 量測儀器：judge `checkpoints/cifar100_judge.pt`（真實測試 74.25%）、near-boundary threshold 0.3622
  （p20 相對分位，`records/2026-07-11-03`）；coverage 特徵 DINOv2 主、Inception robustness。
- selector：CaF-v2 = `argmax recall s.t. precision ≥ τ`，auto-τ = 0.9 × real-vs-real precision（D8）。
- 功效配置：8 seed × 5 rep（D4，MDE 2.49pp）。matched-probe FID-min baseline 1000/class（D5）。
- H2 門檻：CaF-v2 之 per-seed regret 至少低於 matched-budget FID-min 1.5pp（D4，X=1.5）。
- 種子公式：`gseed(seed,w)=int(sha256(f"{seed}_{w:g}").hexdigest()[:15],16)`，`datasets/cifar100_gseed.py`
  （D9，全網格無碰撞已驗）。此 10 點 grid × seeds 之無碰撞於 confirmatory driver dry-run 再驗。
- D3 介入臂剪枝目標：剪至最低 guidance（w1）之 coverage 水準，沿 CIFAR-10 C3 規則（`records/2026-07-09-08`）；
  於 confirmatory 生成集落盤後執行，非路由輸入。

### 凍結落實（依 `claude.md §5.1`）

本 amendment 完成 (a) prose 登記。(b) grid 將寫入 CIFAR-100 confirmatory driver 之預設 config（confirmatory
閘實作）；(c) dry-run（含 gseed 無碰撞、grid 列舉）於真跑前執行；(d) 輸出 hash / 逐位對帳於 confirmatory
之 P 資產。(b)-(d) 屬下一閘工作，本閘只鎖 grid 與規格 prose。

## Follow-up

- STOP 等作者確認後才進 D10 末閘：CIFAR-100 confirmatory。
- **規模警示（供作者決策）**：8 seed × 5 rep × 10 grid = 400 個 (seed,rep,w) cell，每 cell 需平衡生成一份
  CIFAR-100 合成集並訓一個 TSTR 分類器；單 GPU 下屬數日級重計算，且為預註冊揭盲之不可逆步。confirmatory
  的 per_class 與是否分批、續跑機制於該閘定案。
- 需先將 confirmatory driver（`run_cifar_cfg_multiseed.py`）一般化為 `--dataset cifar100`（沿 judge/scout
  的一般化方式），並接 CaF-v2 recall selector 與 gseed 公式。
- grid 一經此 amendment 凍結，confirmatory 不得改點；若真跑後發現 grid 不足，屬另一輪 pre-registration。
