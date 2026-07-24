<!-- 用途：P1-1 EDM 第二 backbone（方案 A）的預先登記修正案（pre-registration amendment）。唯一實質改動
     為把生成 backbone 由自訓 CFG 換成 NVlabs 官方 EDM（cond+uncond 於取樣器組 CFG）；量測堆疊與判準
     一律不動。本檔須於任何 EDM sweep 真跑之前 commit，作為凍結四要件 (a) 的版本控制證據。 -->

# P1-1 EDM 第二 backbone 修正案（方案 A：官方 EDM cond+uncond 組 CFG）

登記日：2026-07-23。適用於 `results/edm_cfg_sweep.json`（撰寫本檔時尚未真跑）。承 CIFAR-100 預註冊
（`docs/prereg_cifar100.md`）、CIFAR-10 v2 修正案（`docs/amendment_cifar10_v2.md`）與 `claude.md`
§5.1 凍結四要件之慣例。

## 1. 動機（要驗的假設）

本專案三判決——(i) FID-min 近最優（which-FID 交叉裁決）、(ii) coverage 型 CaF 選擇器的可靠性、
(iii) TSTR 對 guidance 的內部峰——全在**自訓 CFG backbone** 上得出。審查（規模問題）問：換一個
**來源完全不同**的 backbone，三判決是否仍成立，即判決是否 backbone 相依。方案 A 用 NVlabs 官方預訓練
EDM 的 cond 與 uncond 兩個 checkpoint，在取樣器層組出 classifier-free guidance，掃與 CIFAR-10
confirmatory **相同的** grid/seeds/per_class，量測堆疊完全不變。如此「backbone 換人、尺一模一樣」，
可乾淨隔離 backbone 因素。

## 2. 與 CIFAR-10 confirmatory 的差異（僅 backbone 與其必要的取樣器）

1. **生成 backbone**：自訓 CFG UNet → NVlabs 官方 EDM CIFAR-10（`edm-cifar10-32x32-cond-vp.pkl`
   與 `edm-cifar10-32x32-uncond-vp.pkl`，已在 `checkpoints/`）。
2. **CFG 實作**：兩模型於 EDM denoiser 空間組導引——
   `D_w(x,σ,c) = D_uncond(x,σ) + w·(D_cond(x,σ,c) − D_uncond(x,σ))`。w 即本專案 s-convention
   的 guidance 強度（w=1 純 cond、w>1 往類別方向外插），與 confirmatory 同義。
3. **取樣器（必要的方法學差異）**：EDM 官方 18 步 Heun（`third_party/edm` 的 `edm_sampler`，
   deterministic、S_churn=0），取代 confirmatory 的 50 步 DDIM η=0。此差異為 backbone 內生（不可能用
   我方 DDIM 驅動 EDM 網路），於此明記為方案 A 的已知界限：本 sweep 測「EDM backbone + 其原生取樣器」
   這一整體，非單獨拆解 backbone 與取樣器。

## 3. 一律不動（沿 CIFAR-10 confirmatory 凍結值）

guidance grid [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]、seeds {10, 11, 12}、per_class
1000、TSTR 協定（ResNet-18、reps 5、epochs 15、`--tstr-seeded` 決定性種子）、coverage 量測
（DINOv2 primary、nearest_k 5、num_classes 10）、FID-min 讀數（clean-fid vs CIFAR-10 全訓練集
stats）、真實參考集（1000/class）。生成種子改用 §0.5 hash 派生（`sha256[:15]`，per (seed,w,class)），
禁用舊 `seed*1e7` 公式。

## 4. 判準（go/no-go；兩種結果都如實落地）

以 EDM sweep 的三結構對照 CIFAR-10 confirmatory（v2）：
- **FID-min**：FID-argmin 之 per-seed regret 是否仍近 0、是否仍近最優（confirmatory v2：regret 0.00）。
- **CaF/coverage**：coverage-argmax 是否仍非 TSTR-argmax（Pareto 失明是否複製）。
- **TSTR 峰**：TSTR-argmax 落點與 confirmatory（oracle w1.5）之異同。

若三結構複製→三判決 **backbone 穩健**（強化診斷普適性）；若任一翻轉→判決 **backbone 相依**，如實
更新正文並於 §6.x 記錄差異。無論何者，EDM sweep 為 exploratory、寫新檔、不改任何凍結判決。

## 5. 凍結四要件對應（§5.1）

- (a) **版本控制之規則文件先於真跑**：本檔於任何 EDM sweep 真跑前 commit（此 commit 即是）。
- (b) **計算以 committed code 表達**：`src/experiments/run_edm_cfg_sweep.py` 之 `CFGWrapper` 與
  `generate_cell`（同 commit）。
- (c) **dry-run 於已揭盲資料上先過**：已跑，枚舉 30 cell；CFG 正確性全通過（w=1==cond、w=0==uncond、
  w=3≠cond、輸出 shape (8,3,32,32)、finite）；計時探針 20 張生成 15.6s（779 ms/張，torch-fallback
  路徑），生成 ETA ≈ 65 GPU 小時（300k 張，不含 TSTR/DINOv2/clean-fid；若 custom CUDA op 能編譯則
  大幅縮短）。dry-run 輸出 `results/edm_cfg_sweep_dryrun.json`，見 CHANGELOG `2026-07-24-02`。
- (d) **輸出對帳**：driver 於 metadata 記錄 backbone、cfg_formula、hash 種子公式與完整 §5.2 欄位。

## 6. 執行指令（凍結；真跑前不得更動）

```
uv run python src/experiments/run_edm_cfg_sweep.py --run \
    --grid 1.0 1.5 2.0 2.5 3.0 4.0 5.0 6.0 7.0 8.0 \
    --seeds 10 11 12 --per-class 1000 --reps 5 --tstr-epochs 15 \
    --output results/edm_cfg_sweep.json
```

## 7. 登記時序

本修正案（含 §6 指令與 §2 差異）之 commit 早於任何 EDM 合成樣本之生成，git 提交時間可驗。真跑須待
作者授權（GPU 高）。
