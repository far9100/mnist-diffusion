<!-- 用途：記錄 CIFAR-100 confirmatory driver 一般化與 dry-run 驗證（D10 末閘前置，§5.1 dry-run），含真跑指令與待定參數。 -->

# 2026-07-11-06 CIFAR-100 confirmatory driver 一般化與 dry-run（D10 末閘前置）

## Goal

依作者裁示「先建 driver + dry-run 後再停」：把 confirmatory 主 driver `run_cifar_cfg_multiseed.py`
一般化為 CIFAR-100，接上 D 包規格（CaF-v2 recall selector、無碰撞 gseed、8×5 rep），並依 `claude.md §5.1`
在真跑前做 dry-run（gseed 無碰撞驗證 + 單元 quick 全鏈路驗證）。不進 confirmatory 真跑、不揭盲全 grid。

## Result

### 工具一般化

- `selector.py`：`select_caf`/`tau_robustness`/`select_and_report` 加 `signal_key` 參數，向後相容
  （預設 coverage＝CaF）。CIFAR-100 傳 `recall`＝CaF-v2（D8 `argmax recall s.t. precision≥τ`）。
  report 併記 `signal_key` 供追溯。self-check 通過（預設 coverage 行為不變）。
- `run_cifar_cfg_multiseed.py`：一般化為 `--dataset cifar10|cifar100`：
  - 生成種子 `gen_seed()` dataset 相依：CIFAR-100 用 `datasets/cifar100_gseed.gseed`（sha256[:15] 無碰撞）；
    CIFAR-10 沿用原公式以保凍結資料逐位重現（不改）。
  - 每 (seed,w) 做 `--reps` 次 from-scratch TSTR 重訓（D4 CIFAR-100=5，消 σ_cls），均值餵 selector；
    per-config 記 `tstr_reps`。
  - 記錄 recall/density（CaF-v2 需 recall）；char clean-fid 對 CIFAR-100 用 base gate 自建參考
    `cifar100_train_clean32`（clean-fid 無內建 cifar100 stats）。
  - num_classes、判 judge、real ref、label-noise 逐類、Inception 交叉檢查全部依 dataset 類數。
  - metadata（§5.2）補 dataset/num_classes/reps/selector/gen_seed_formula/env。

### dry-run 驗證（§5.1）

- **gseed 無碰撞**：凍結 grid {1,1.5,2,2.5,3,4,5,6,7,8} × 候選 8 seed {10..17} = 80 cell，80 distinct，
  collision_free=True。
- **selector self-check**：CaF（coverage）預設行為通過。
- **全鏈路 quick**（`--dataset cifar100 --quick`：2 seed × 2 w × 2 rep、per_class 32）：管線端到端無誤。
  CaF-v2 以 recall 選中 w1（coverage 版 CaF 會選 coverage 較高的 w3），確認 recall 訊號生效且與 coverage
  取向不同。輸出 `results/cifar100_cfg_multiseed_dryrun.json`。quick 數字為煙霧測試，不具科學意義。

### confirmatory 真跑指令（待作者授權才執行）

```
uv run python run_cifar_cfg_multiseed.py --dataset cifar100 \
  --seeds 10 11 12 13 14 15 16 17 --reps 5 \
  --guidance 1 1.5 2 2.5 3 4 5 6 7 8 \
  --per-class <待定> --real-per-class 1000 \
  --output results/cifar100_cfg_confirmatory.json
```

## Follow-up

- **STOP**：driver 已就緒並通過 dry-run。confirmatory 真跑（預註冊揭盲）需作者明確授權，未授權。
- **待定參數（confirmatory 真跑前須定）**：
  - `--per-class`（生成張數/類）：CIFAR-10 confirmatory 用 1000。CIFAR-100 有 100 類，per_class=1000 即
    每 cell 生成 100k 張、80 cell 共 8M 張生成 + 400 次 TSTR 重訓，單 GPU 屬**數週級**、可能不可行。
    需作者定一個可行的 per_class（如 200–500），此決定若偏離 CIFAR-10 口徑，宜以 pre-registration
    amendment 明記（凍程序）。
  - `--seeds`：本 dry-run 以 {10..17} 驗 gseed 無碰撞；實際 8 seed 由作者確認。
- char_fid 之 CIFAR-100 自訂參考路徑（`clean_fid_gen_vs_ref`）已於 base gate（`records/2026-07-11-02`）
  驗證可用；quick dry-run 因 `--no-fid` 未經 driver 再驗，首次真跑（或開 fid 的小測）時確認。
- confirmatory 之後才是三判決分析、C2 偏相關、D3 介入臂、D5 三臂對決（Chamfer 條件化），依 D1 分支樹。
