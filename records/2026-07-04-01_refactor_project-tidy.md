<!-- 用途：記錄依 claude.md 規範整理專案（records 改名、檔頭與註解中譯、README 更新）的目標、結果與後續。 -->

# 2026-07-04-01 依 claude.md 規範整理專案

## 目標

新增版 claude.md（與舊 CLAUDE.md 為同一實體檔，Windows 不分大小寫，磁碟名為小寫 claude.md）
把 records 檔名格式改為 YYYY-MM-DD-NN。依此為最高規範整理專案：records 改名、程式檔頭與所有
註解中譯、README 更新，並盤點實驗結果存放位置。版控與分支收斂（研究主線程式碼 untracked、HEAD
停在 experiment/var-mini）另行討論，不在本次範圍。

## 結果

records 改名：11 份 records 全部改為 YYYY-MM-DD-NN 格式，07-02 一份、07-03 十份，序號依時間先後
排列。records 內部與 selector.py、run_cifar_selector.py 對舊檔名的交叉引用共 11 處同步更新，grep
確認無舊格式殘留，且更新後引用皆指向存在的檔案。

檔頭中譯：30 個 .py 檔的檔頭第一句 purpose 說明由英文改為繁體中文，技術術語維持英文。

註解全面中譯：以子代理分組並行處理 27 個 LF 檔，另自行處理 ddpm.py、train.py（CRLF）與訓練中的
train_cifar.py，把所有 # 註解與三引號 docstring 翻成繁體中文。argparse 的 help=、print、CLI 範例
等程式輸出字串維持原文。以 tokenize 驗證確認 30 檔的程式 token 與非 docstring 字串逐一相同（只有
註解與 docstring 改變），py_compile 全數通過，無簡體字殘留、無狀態符號。翻譯期間未中斷正在進行的
CIFAR 訓練。

README 更新：由舊的「DDPM MNIST 手寫數字生成」改寫為研究主線「Sampling for Utility, not Fidelity」。
補上研究定位（CaF/C2/C1 與對 Chamfer 的定位）、程式碼分代（Gen-1 MNIST sandbox 已完成、Gen-2 CIFAR
Phase 1 進行中、VAR-mini 旁支）、目前進度、Phase 0 已定案結果、完整專案結構與指標說明。CIFAR 全品質
結果尚未定案，未列為結論。

實驗結果存放位置盤點：checkpoints/ 約 22GB、results/ 79MB、samples/ 與 samples_cifar/、generated/、
data/ 776MB、docs/、records/。samples_cifar/ 目前未被 gitignore、會被誤 commit；results/ 內
cifar_selector 與 selector_signal 各有多版命名散亂。詳見計畫檔 Stage 4。

## 後續

- Stage 4 輸出安全整理：把 samples_cifar/ 等加入 .gitignore、統一 results/ 命名、根目錄散落物歸位；
  不刪除大型 checkpoint 與 data。動到 results/ 時需留意訓練仍在寫入。
- Stage 5：docs/paper_intro_draft.md 保留英文並在 claude.md 增列投稿英文手稿之語言例外。
- 另行討論：研究主線程式碼提交與 experiment/var-mini 分支收斂、專案進度不合理問題。
