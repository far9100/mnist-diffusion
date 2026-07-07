<!-- 用途：記錄 README 過時處校對與同步（僅限與 confirmatory 結果無關的三項）。 -->

# 2026-07-06-01 README 結構清單同步（結果無關項）

## 目標

使用者要求說明專案目標/現狀/下一步，並檢查 README 是否需要更新。校對後確認 README 的研究定位已是
v2、方向正確，但結構清單與部分用語落後於實際檔案。依作者裁定，本輪只補與 confirmatory 結果無關的
過時處，不動進度數字、不跑任何實驗。

背景：confirmatory 實驗已於 2026-07-06 08:45 跑完（results/cifar10_cfg_confirmatory.json，約 8.3 小時、
30 configs），但 run_c2_partial 尚未對真實資料裁決、也還沒有對應 record。依定位 v2（2026-07-05-12
§7/§9）的治理順序，帶入 confirmatory 進度與數字的 README/docs/intro 同步應在 C2 裁決之後、與 docs、
intro 一起做，故本輪排除。

## 結果

只改 README.md，三處，皆與 confirmatory 結果無關：

- (A) 專案結構清單（Gen-2 主線區塊）補列 8 個實際存在但漏列的研究主線腳本：cifar_judge.py、
  cifar_cfg_sample.py、run_cifar_cfg_scout.py、run_cifar_cfg_upper_scout.py、run_cifar_cfg_multiseed.py、
  run_c2_partial.py、run_flip_earlywarning.py、chamfer.py。描述只寫腳本功能，不含任何結果數字。此區
  補齊後，README 進度節所列 Chamfer 對決與翻轉檢查在結構清單有對應實作檔可循。
- (D) 於 README 指向文件句補上 docs/paper_intro_draft.md。
- (E) run_comparison.py 描述加註「(MNIST sandbox 尺度)」，避免與 v2「CIFAR 尺度不宣稱聯合曲面」混淆。

驗證：對照根目錄 py 檔清單與 README 結構清單，補列後一致、無幽靈引用；README 進度節文字未動，仍不含
任何 confirmatory 結果數字；本 record 依命名與三段格式建立。

## 後續

- confirmatory 資料已落地（2026-07-06 08:45）但尚未裁決與記錄。C2 裁決（run_c2_partial）與帶
  confirmatory 進度與數字的 README/docs/paper_intro_draft 正式同步，依作者裁定延後至 C2 之後，作為
  一個獨立治理步驟。屆時另建 record 記錄 confirmatory 結果與裁決。
- 尚待處理的 README 帶結果項：進度節（L90 附近）confirmatory 狀態文字仍寫「待跑」；重現指令節缺
  confirmatory 與 C2 命令。兩者留待上述同步步驟一併更新。
