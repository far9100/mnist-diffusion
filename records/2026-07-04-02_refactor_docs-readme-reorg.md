<!-- 用途：記錄 Stage 4 輸出整理與 docs/README 重組（前言方法進 README、結果分析進 docs）的目標、結果與後續。 -->

# 2026-07-04-02 Stage 4 輸出整理與 docs/README 重組

## 目標

完成整理計畫 Stage 4（輸出安全整理，不刪大檔），並依作者新指示調整原 Stage 5：docs 資料夾的 .md
改放實驗結果的分析，前言、背景知識、方法、實驗設計則整理進 README。現階段不在此撰寫論文。

## 結果

Stage 4：.gitignore 補上 samples_cifar/（原本忽略 samples/ 卻漏掉 samples_cifar/，12MB 會被誤
commit），git check-ignore 確認已忽略。補上後檢視未追蹤且未忽略的項目，剩下的全是原始碼與文件
（研究主線 .py、datasets/、docs/、records/、claude.md），本就應進版控、非輸出，故無其他輸出目錄
會被誤 commit。未刪除任何大型檔案，也未更名 results/ 內的散落結果檔（避免破壞 records 對其路徑的
交叉引用）。

docs/README 重組：把原 docs/paper_intro_draft.md（英文手稿草稿）拆解——前言、機制、與 Chamfer 的
定位、完整實驗清單整理成繁體中文段落併入 README（新增前言、背景知識、方法：CaF 與機制、與 Chamfer
的定位、實驗設計等節，原檔的 [ ] 核取方塊改為純列表以符合不使用狀態符號的規範）；實驗結果的實際
數字與解讀（EDM 量測錨點、MNIST sandbox、CIFAR-10 預覽）移到新檔 docs/results_analysis.md。原
英文草稿已刪除。因手稿內容已全部落為繁中，先前保留英文的語言例外不再需要，未寫入 claude.md。

README 的詳細結果數字移出、改為高階進度並指向 docs/results_analysis.md。

## 後續

- 可清理但本次未動，待作者決定：checkpoints/ 約 22GB 的週期性 checkpoint（可只留最終加少數早期）、
  results/train_cifar10.log（約 47MB）、results/edm_cifar_stats.npz 與 checkpoints/cifar10-32x32.npz
  兩個同大小 npz 是否重複、根目錄散落的權重與 report 檔。
- results/ 內 cifar_selector 與 selector_signal 的多版命名散亂，因會破壞 records 引用而未更名，
  待作者決定是否要一併重整並更新引用。
- 另行討論：研究主線程式碼提交與 experiment/var-mini 分支收斂、專案進度不合理問題。
