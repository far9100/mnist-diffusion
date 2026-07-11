<!-- 用途:研究定位調整 v1(範圍收緊、敘事重心、邊界條件)。狀態:superseded,保留作決策軌跡。 -->

# 研究定位調整 v1(superseded)

## 狀態

本檔為定位調整的 v1 草稿,從未 commit。經本 agent 稽核發現四項缺陷:(1)引用了不存在的封頂
amendment(第 14 行「已另行裁決」為幽靈引用);(2)調整二(C2)缺資料獨立的分段規則,構成 HARKing
漏洞,且「收緊範圍不等於降低門檻」為修辭滑動;(3)調整一淡化了護城河變薄;(4)本檔為 untracked
草稿卻自稱凍結。上述缺陷由 2026-07-05-12(定位調整 v2)逐項修訂取代。

作者核准:v1 不刪、改名入庫、標 superseded 保留。理由:v2 §0 引用 v1 的四項缺陷,刪 v1 等於製造
第二個幽靈引用;且 v1 從未入 git,刪除即永久消失,稽核發現與 v2 修訂理由是答辯資產。現行定位以
2026-07-05-12 為準,本檔不再作為依據。

## 1. 本文件的性質與界線

本文件調整三件事:主張的範圍、敘事重心、邊界條件的描述。本文件不調整任何已凍結項目。

已凍結、不因本文件而變:
- confirmatory 主假設與 pass/fail 準則(依協定登記版本)。
- judge(93.08%、逐類 floor、checkpoint)與 near-boundary threshold 0.9525。
- 觸底判準 X=0.02。其在 w=20 未觸底的結果已觸發再登記分支;判準本身不回改。
- fresh seeds {10,11,12};grid 中段固定間隔規則。
- grid 上緣封頂依「實用 guidance 上限」的先驗理由另立 amendment(已另行裁決),不在本文件內處理。

原則:收緊主張的範圍,不等於降低檢驗的門檻。本文件每一項調整都使主張變小,不使 confirmatory 更容易通過。confirmatory 的通過條件一字不動。

## 2. 時間要求

本文件與協定增修同批或更早 commit。commit 時間須早於任何 confirmatory 資料的產生。git 時間戳可驗。若 confirmatory 已有任何資料落地,本文件不得再寫入,需呈報。

## 3. 調整一:thesis 範圍收緊到 CFG guidance 軸

原表述:取樣組態(steps × η × guidance)的效用最優點系統性偏離 FID 最優點。

新表述:在 CFG guidance 軸上,下游效用的最優點偏離 FID 最優點;效用對 guidance 非單調,存在內部最優;該最優點可被免訓練的方法定位。

理由:
- CIFAR 上實際執行的設計只掃 guidance 軸。主張範圍對齊實際設計。
- MNIST sandbox 顯示 η 對效用為 null、steps 為次要。這兩軸的結論保留在 sandbox 範圍,不在 CIFAR 尺度宣稱。
- 全立方體的主張會引來「η 在 CIFAR 上如何」的質問,而目前沒有 CIFAR 資料可答。

後果(需在 intro 與 related work 反映):相對前人的差異化由「η 軸與聯合曲面」移到「非單調性、免訓練 selector、機制解釋」三項。MNIST 的 3D 曲面降為動機與 sandbox 證據。若日後在 CIFAR 補跑 η 軸並取得資料,可再以新 amendment 擴回,現階段不宣稱。

## 4. 調整二:C2 的可檢驗範圍限定

C2 表述不變:coverage 驅動效用,precision 不驅動。

限定:C2 的裁決只在 coverage 有變異的區段進行,即 confirmatory grid 的高 guidance 段。coverage 平坦的區段不構成 C2 的檢驗,也不構成否證。

低 guidance 段的效用上升由什麼驅動(候選:precision 上升、label-noise 下降),維持 exploratory 定位,不由本次 confirmatory 裁決,需獨立資料。

理由:平坦變數無變異可解釋,不可檢驗。此為方法論的先驗陳述,不依賴任何已見結果。

## 5. 調整三:敘事重心從甜蜜點移到 CaF

主張順序改為:效用對 guidance 非單調;因此任何固定 guidance 值跨資料集與任務必然次優;因此需要組態 selector;CaF 以免訓練、低成本定位效用最優組態。

甜蜜點的存在是「為什麼需要 selector」的動機。CaF 是貢獻本體。

與前人的區隔照此順序寫:Fan 給固定 low-CFG 配方,非單調性使任何固定值不可靠;Chamfer 是取樣時的 guidance 方法,需要真實參考影像;CaF 是組態層的 selector,與取樣時方法屬不同層次,可互補。

README、docs、論文 intro 草稿依此順序改寫。

## 6. 調整四:新增邊界條件描述

scout(coverage-only,1 seed,w ∈ {8,10,12,16,20})結果:coverage 0.533 → 0.259,單調下降,依 X=0.02 未觸底。

寫入定位:在 CFG 的實用範圍內,coverage 隨 guidance 單調流失、不回穩。此為描述性觀察,引用 scout 讀數並標明其條件(coverage-only、1 seed),confirmatory 不重跑大於封頂值的區段。

此觀察併入雙力敘事:低段 fidelity 上升,高段 coverage 單調下降,兩力交會產生內部最優。低段驅動力的定位依第 4 節,維持 exploratory。

## 7. 落地(檔案層面)

- 協定增修:加入第 4 節的 C2 可檢驗範圍限定;附上第 3 節的 thesis 範圍表述。
- README:thesis 一句話依第 3、5 節改寫。
- docs/results_analysis.md:MNIST 3D 曲面重新定位為 sandbox 證據;加入第 6 節的 coverage 單調流失觀察,標明資料來源與條件。
- docs/paper_intro_draft.md:主張順序依第 5 節;範圍聲明依第 3 節;related work 差異化段落依第 3 節的後果改寫。
- 各檔修改處註明依據本文件。

## 8. 自我測試

每項調整須通過:此調整的理由,在看到 pilot 與 scout 結果之前是否成立。

- 調整一:成立。CIFAR 設計自始只掃 guidance;η-null 是 MNIST 既有結果。
- 調整二:成立。可檢驗性需要變異,為先驗方法論陳述。
- 調整三:成立。敘事定位選擇,不依賴任何數字。
- 調整四:描述性報告,標明來源與限制,無假設依附其上。

任何未通過此測試的後續調整,不得寫入本文件或其增補。

## 9. 執行順序

1. 本文件與協定增修同批 commit,時間早於 confirmatory 資料。
2. 依第 7 節改 README、docs、intro 草稿,commit,呈報。
3. 進 confirmatory:50k FID gate、封頂 amendment、定死 grid、fresh-seed 多 seed,依既定計畫。
4. confirmatory 結果出來後,裁決依協定。本文件的定位不因結果回改;若結果與定位衝突,據實報告衝突,不改寫本文件。
