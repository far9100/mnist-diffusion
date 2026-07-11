<!-- 用途:記錄 confirmatory 前補齊三項凍結規格儀器（per-class 超額 label-noise、per-config characterization FID、C2 全網格偏相關）與 1-config 計時探針結果。 -->

# 2026-07-05-14 confirmatory 儀器補齊與計時探針

## 目標

confirmatory 多 seed 全量開跑前，把三項已凍結規格的量測儀器補進管線，並以 1-config 計時探針把
30 configs 的 GPU 時間釘成實數。純實作，不動任何已凍結的規格或門檻。

三項儀器對應凍結文件：
- per-class 超額 label-noise（增修 2026-07-05-08 規格2）：逐類合成 label-noise 減真實 judge floor。
- per-config characterization clean-fid（規格1）：重用生成集算 clean-fid，只報告不 gate。
- C2 全網格偏相關（2026-07-05-13）：生成後腳本，partial Spearman C2a/C2b + permutation + bootstrap CI。

## 結果

實作：
- run_cifar_cfg_multiseed.py：measure() 增 per-class 超額 label-noise（floor 自 cifar10_judge.json 的
  per_class_accuracy，floor_c = 1 - acc_c/100）與 per-config characterization clean-fid（--no-fid 可關，
  僅 smoke 用）；跨 seed 彙總逐類超額帶 CI，metadata 記 judge_floor 與 prereg 編號。confirmatory 用
  per_class=1000（協定 §3 消 coverage 樣本數假影）。
- run_c2_partial.py（新）：讀 confirmatory JSON，以 config 為觀測單位（n=10）計 partial Spearman
  ρ(TSTR, coverage | precision, 超額 label-noise) 與 ρ(TSTR, precision | coverage, 超額 label-noise)，
  permutation p、bootstrap-over-seeds CI，通過準則 C2a 顯著正且 C2b 不顯著，並強制報效果量與 n=10
  功效限制。

驗證：
- driver --quick smoke 通過：judge floor 正確載入（貓 idx3=0.151、狗 idx5=0.105）；per-class 超額
  label-noise 正常（w1 為正、w3 為負）。
- C2 腳本以合成 10-config×3-seed 資料通過：C2a ρ=+0.938（perm p=0.0003、CI[0.90,0.98]）顯著正、
  C2b ρ=+0.229（perm p=0.53）不顯著、裁決通過；統計路徑正確執行。
- char-FID 單獨計時：10000 張約 105.6 s/config。

1-config 計時探針（w=2、seed=0（pilot seed，不動 fresh 10/11/12）、per_class=1000、FID 開）：
- 單 config 全載約 1000 s（含一次性 setup：模型、DINOv2 hub、Inception detector、10000 張真實 DINOv2
  特徵）。分解：char-FID ~106 s、setup ~300 s、其餘為生成 + 特徵 + TSTR。
- 探針數值健全：precision 0.862、coverage(DINO) 0.776 / (Incep) 0.94、TSTR 64.3%、char-FID 10.03、
  超額 label-noise 均值 -0.058（w2 較 judge floor 乾淨，符合中段 guidance 預期）。TSTR 高於 pilot 的
  46%，因 per_class 由 500 升到 1000（10000 張），佐證協定堅持 1000 的必要。

30 configs（10 grid × 3 fresh seeds {10,11,12}）估計：setup 攤提後約 6-8 小時（overnight）。
可省點：inception_crosscheck 每 config 重算 10000 張真實 Inception 特徵（約佔全程 10%），可快取，
非必要。

## 後續

- 經作者同意後跑 30 configs（run_cifar_cfg_multiseed，10 點 grid、fresh seeds 10/11/12、per_class=1000、
  real_per_class=1000、FID 開），輸出 results/cifar10_cfg_confirmatory.json，完整網格不剪枝。
- 生成後跑 run_c2_partial 取 C2 裁決；依協定裁決 H1/H2，呈報 30 configs 數字（precision/coverage
  DINOv2+Inception/TSTR/per-class 超額 label-noise/characterization FID/regret@selected/top-k/C2 偏相關）。
