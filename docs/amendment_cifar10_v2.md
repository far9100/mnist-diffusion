<!-- 用途：CIFAR-10 confirmatory v2 的預先登記修正案（pre-registration amendment）。唯一實質改動為
     生成種子公式由 legacy 換成無碰撞 hash，並將 TSTR 重訓次數 reps 由 1 提為 5；其餘凍結規格一律不動。
     本檔須於任何 v2 合成取樣之前 commit，作為凍結四要件 (a) 的版本控制證據。 -->

# CIFAR-10 confirmatory v2 修正案（無碰撞種子重跑）

登記日：2026-07-21。適用於 `results/cifar10_cfg_confirmatory_v2.json`（撰寫本檔時尚未執行）。
承 CIFAR-100 預註冊（`docs/prereg_cifar100.md`）與 `claude.md` §5.1 凍結四要件之慣例。

## 1. 動機

CIFAR-10 confirmatory v1（`results/cifar10_cfg_confirmatory.json`，已定稿、凍結、不動）使用舊生成
種子公式 `gseed = seed*1e7 + int(w*1000)*1e4`。此公式在本網格（seeds {10,11,12} × guidance
{1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0,7.0,8.0}）上退化：30 個 cell 只得 14 個相異 gseed，其中 26 個 cell
與其他 cell 共用同一 gseed（10 個碰撞群），即這些 cell 以相同 latent 噪聲生成，破壞 seed 間的獨立性。
碰撞結構（dry-run 枚舉，見 §4c）：

| gseed | 共用之 cell |
|---|---|
| 120000000 | s10w2.0, s11w1.0 |
| 125000000 | s10w2.5, s11w1.5 |
| 130000000 | s10w3.0, s11w2.0, s12w1.0 |
| 135000000 | s11w2.5, s12w1.5 |
| 140000000 | s10w4.0, s11w3.0, s12w2.0 |
| 150000000 | s10w5.0, s11w4.0, s12w3.0 |
| 160000000 | s10w6.0, s11w5.0, s12w4.0 |
| 170000000 | s10w7.0, s11w6.0, s12w5.0 |
| 180000000 | s10w8.0, s11w7.0, s12w6.0 |
| 190000000 | s11w8.0, s12w7.0 |

v2 以無碰撞的 hash 公式 `gseed = int(sha256(f"{seed}_{w}")[:15], 16)`（`datasets/cifar100_gseed.py`，
與 CIFAR-100 同一函式）重跑，恢復 seed 獨立性。

## 2. 與 v1 的差異（僅此三項）

1. **生成種子公式**：legacy → hash（`--gseed-formula hash`）。唯一為修 bug 的改動。
2. **TSTR 重訓次數 reps**：1 → 5。理由：v1 每 cell 僅一個 from-scratch 分類器，σ_cls（分類器訓練
   變異）未被平均，使 per-seed TSTR 帶入單次訓練噪聲；提為 5（與 CIFAR-100 confirmatory 同）以壓低
   σ_cls。此為對量測協定的變更，於此明記並給理由。
3. **TSTR 種子化**：啟用 `--tstr-seeded`（T6b）。每 rep 以 `sha256(tstr_cifar10_<seed>_<w>_<rep>)`
   衍生決定性種子，使 v2 的 TSTR 逐位可對帳（v1 未種子化、語意不變、不受影響）。

## 3. 一律不動（沿 v1 凍結值）

guidance grid [1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0,7.0,8.0]、seeds {10,11,12}、per_class 1000、
real_per_class 1000、steps 50、eta 0、nearest_k 5、tau_fraction 0.9、tstr_epochs 15、
near_boundary_threshold 0.9525（自凍結 `results/cifar10_judge.json` 載入）、selector CaF（coverage，
DINOv2 primary + Inception cross-check）、judge 與 base checkpoint 不變。

## 4. 凍結四要件對應（§5.1）

- (a) **版本控制之規則文件先於真跑**：本檔於任何 v2 取樣前 commit（此 commit 即是）。
- (b) **計算以 committed code 表達**：`run_cifar_cfg_multiseed.py::gen_seed` 之 `--gseed-formula`
  參數（同 commit）。
- (c) **dry-run 於已揭盲資料上先過**：枚舉 30 cell 的 hash gseed，得 30/30 相異（無碰撞）；legacy
  對照得 14/30、10 群、26 共用（重現 §1 表）；legacy 路徑 `gen_seed(10,1.0)=110000000` 逐位不變
  （回歸）。dry-run 輸出見 CHANGELOG `2026-07-21-05`。
- (d) **輸出對帳**：v2 driver 於 metadata 記錄 `gseed_formula_choice=hash` 與完整 §5.2 欄位；供事後核。

## 5. 執行指令（凍結；真跑前不得更動）

```
uv run python src/experiments/run_cifar_cfg_multiseed.py \
    --dataset cifar10 \
    --guidance 1.0 1.5 2.0 2.5 3.0 4.0 5.0 6.0 7.0 8.0 \
    --seeds 10 11 12 --per-class 1000 --real-per-class 1000 \
    --reps 5 --gseed-formula hash --tstr-seeded \
    --output results/cifar10_cfg_confirmatory_v2.json
```

## 6. 分析計畫（v2 產出後）

1. 以 v2 重導 `cifar10_c6_fidmin_duel_v2.json`（FID-min vs CaF regret）。
2. C8 支配結構複核：w2.5 是否仍嚴格支配三 oracle。
3. C2 偏相關複算。
4. **v1 保留不動**；論文正文改引 v2、附錄註明 v1 之種子碰撞與其影響。若任一判決翻轉（例如 FID-min
   與 CaF 勝負互換），如實更新正文並於 §6.2 記錄 v1/v2 差異。

## 7. 登記時序

本修正案（含 §5 指令與 §2 差異）之 commit 早於任何 v2 合成樣本之生成，git 提交時間可驗。
