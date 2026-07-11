<!-- 用途:研究定位調整 v2(範圍收緊、敘事重心、邊界條件)。修訂依據:coding agent 稽核(幽靈引用、C2 分段漏洞、護城河淡化、優先序)。須與封頂 amendment 同批 commit,時間早於任何 confirmatory 資料。落庫時指派 records 編號,並確認本檔脫離 untracked 狀態。 -->

# 研究定位調整 v2

## 0. 相對 v1 的修訂

v1 經本 agent 稽核判定四處需修:引用了不存在的封頂 amendment;調整二缺資料獨立的分段規則,且「收緊範圍不等於降低門檻」為修辭滑動;調整一淡化護城河變薄;本檔為 untracked 草稿卻自稱凍結。v2 逐項修正:封頂 amendment 已另立實檔(2026-07-05-11,同批 commit);C2 裁決改為全網格偏相關、廢除分段;護城河變薄明文陳述;本檔隨本批 commit 凍結。v1 經作者核准改名入庫、標 superseded 保留(2026-07-05-10),作決策軌跡。

## 1. 本文件的性質與界線

本文件調整三件事:主張的範圍、敘事重心、邊界條件的描述。本文件不調整任何已凍結項目。

已凍結、不因本文件而變:
- confirmatory 之 pass/fail 準則(regret@selected、top-k 等,依協定登記版本)。
- judge(93.08%、逐類 floor、checkpoint)與 near-boundary threshold 0.9525。
- 觸底判準 X=0.02。其於 w=20 未觸底的結果已依分支由封頂 amendment(2026-07-05-11,同批 commit)裁決;判準本身不回改。
- fresh seeds {10, 11, 12};grid 點集依封頂 amendment(2026-07-05-11)凍結為 {1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8};取樣設定 steps=50、η=0 依同 amendment §7 凍結。

原則:收緊改變主張的內容。防止收緊淪為方便的裁剪的,不是宣稱門檻不變,而是裁決程序須事前定死、且其形式與任何 pilot 數字無關——本文件採全網格偏相關,無任何分段邊界可供事後劃定。

## 2. 時間要求

本文件與封頂 amendment、C2 統計規格增修同批 commit。commit 時間須早於任何 confirmatory 資料的產生,git 時間戳可驗。若 confirmatory 已有任何資料落地,本文件不得再寫入,需呈報。本檔此前為 untracked 草稿;本批 commit 前,其任何內容不具凍結效力。

## 3. 調整一:thesis 範圍收緊到 CFG guidance 軸

原表述:取樣組態(steps × η × guidance)的效用最優點系統性偏離 FID 最優點。

新表述:在 CFG guidance 軸上,下游效用的最優點偏離 FID 最優點;效用對 guidance 非單調,存在內部最優;該最優點可被免訓練的方法定位。

理由:
- CIFAR 上實際執行的設計只掃 guidance 軸。主張範圍對齊實際設計。
- MNIST sandbox 顯示 η 對效用為 null、steps 為次要。此二軸結論保留在 sandbox 範圍,不在 CIFAR 尺度宣稱。

後果(明文,intro 須如實陳述,不得中性帶過):此調整使護城河變薄。η 與聯合曲面支點由 CIFAR 尺度退為 sandbox 證據;CIFAR 尺度的差異化剩三項——非單調性、CaF、機制。此三項相對 Chamfer、Fan、DP 擴散文獻是否充分,由 CIFAR-100 gate 與 Chamfer matched-budget 對決回答;在該兩者有資料前,充分性為未決。已排定的 CIFAR-100 η spot-check 提供 CIFAR 尺度的部分回補。若日後在 CIFAR 補跑 η 軸,可再以新 amendment 擴回。

## 4. 調整二:C2 的裁決程序(全網格偏相關,廢除分段)

C2 表述不變:coverage 驅動效用,precision 不驅動。

裁決程序(事前定死,寫入協定增修):
- 觀測單位為 config(grid 10 點),每 config 之 TSTR、coverage、precision、label-noise 超額取 3 個 fresh seeds 的均值。
- H-C2a:partial Spearman ρ(TSTR, coverage | precision, label-noise 超額) > 0,以 permutation test 計 p,α = 0.05。
- H-C2b:partial Spearman ρ(TSTR, precision | coverage, label-noise 超額) 不顯著或 ≤ 0,同法。
- 通過準則:C2a 顯著為正且 C2b 不顯著。未達則 C2 於 CIFAR-10 尺度未獲確認,據實報告效果量、信賴區間與 n=10 的功效限制;準則不因結果改寫。
- 特徵空間:coverage、precision 以 DINOv2 為準,Inception 依既定交叉裁決規則作 robustness。

分段的處置:pilot 呈現的三段結構(低段三力同向而混淆、中段 coverage 平坦、高段 coverage 與 precision 分離)降為論文的敘事描述,不承擔任何裁決功能。混淆段(w1→w2)納入偏相關、不排除:共線性降低統計功效,不造成偏誤;識別由高段的分離提供。明文承認:納入混淆段可能降低 C2a 的顯著性——此為不裁剪資料的代價,接受之。

低 guidance 段的效用上升由什麼驅動(候選:precision 上升、label-noise 下降),維持 exploratory 定位,不由本次 confirmatory 裁決,需獨立資料。

事前性:偏相關即創始 roadmap 對 C2 的原始裁決配方(utility ~ coverage | precision),早於一切 CIFAR 資料存在。本調整是回到最早登記的形式,並且相對分段方案納入更多而非更少的資料。

## 5. 調整三:敘事重心從甜蜜點移到 CaF

主張順序改為:效用對 guidance 非單調;因此任何固定 guidance 值跨資料集與任務必然次優;因此需要組態 selector;CaF 以免訓練、低成本定位效用最優組態。

甜蜜點的存在是「為什麼需要 selector」的動機。CaF 是貢獻本體。

與前人的區隔照此順序寫:Fan 給固定 low-CFG 配方,非單調性使任何固定值不可靠;Chamfer 是取樣時的 guidance 方法,需要真實參考影像;CaF 是組態層的 selector,與取樣時方法屬不同層次,可互補。

README、docs、論文 intro 草稿依此順序改寫。

## 6. 調整四:邊界條件描述

寫入定位:1-seed coverage-only scout 讀數顯示,coverage 於 w ∈ [8, 20] 由 0.533 單調降至 0.259,依 X=0.02 未觸底。條件標注(coverage-only、1 seed、DINOv2 特徵)緊貼主張,不得省略。

此為描述性觀察,無假設依附其上;confirmatory 不重跑 w > 8 區段。併入雙力敘事:低段 fidelity 上升,高段 coverage 單調下降,兩力交會產生內部最優;低段驅動力的定位依第 4 節,維持 exploratory。

## 7. 落地(檔案層面)

- 封頂 amendment(2026-07-05-11):另立實檔,同批 commit。
- 協定增修(2026-07-05-13):加入第 4 節的 C2 裁決統計規格(觀測單位、雙偏相關、permutation、α、通過準則、功效限制陳述)。
- README:thesis 一句話依第 3、5 節改寫。
- docs/results_analysis.md:MNIST 3D 曲面重新定位為 sandbox 證據;加入第 6 節觀察,含條件標注;pilot 三段結構標為敘事描述。
- docs/paper_intro_draft.md:主張順序依第 5 節;範圍聲明與護城河變薄陳述依第 3 節;related work 差異化段落依第 3 節後果改寫。
- 各檔修改處註明依據本文件。

## 8. 自我測試

每項調整須通過:此調整的理由,在看到 pilot 與 scout 結果之前是否成立。

- 調整一:成立。CIFAR 設計自始只掃 guidance;η-null 為 MNIST 既有結果。
- 調整二:成立。偏相關為創始 roadmap 的原始配方,早於 CIFAR 資料;且相對分段納入更多資料,無可供事後劃定的邊界。
- 調整三:成立。敘事定位選擇,不依賴任何數字。
- 調整四:描述性報告,條件標注緊貼主張,無假設依附。

任何未通過此測試的後續調整,不得寫入本文件或其增補。

## 9. 執行順序與治理凍結

1. 同批 commit:封頂 amendment(2026-07-05-11)、本檔 v2(2026-07-05-12)、C2 統計規格協定增修(2026-07-05-13),並含 v1 改名 superseded 入庫(2026-07-05-10)。驗證 git 時間戳早於任何 confirmatory 資料、本檔脫離 untracked。
2. 依第 7 節改 README、docs、intro 草稿,commit,呈報。
3. 治理凍結:自本批 commit 起,至 CIFAR-100 gate 與 Chamfer 對決取得資料前,不再新增任何治理或定位文件。例外僅限牆的執行本身強制觸發的再登記,且以最小篇幅為之。
4. GPU 排程即刻接上:confirmatory(10 點 × 3 seeds,開跑前呈報時間估計,約 1 天)→ 完成即啟 CIFAR-100 CFG 訓練(排序穩定停;訓練空窗 CPU 並行 Chamfer 之 CIFAR 適配與對決 driver)→ CIFAR-100 η spot-check、guidance 掃描、機制分析 → Chamfer matched-budget 對決。
5. confirmatory 結果出來後,裁決依協定。本文件的定位不因結果回改;若結果與定位衝突,據實報告衝突,不改寫本文件。
