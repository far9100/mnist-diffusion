<!-- 用途：工作包 A0——H-C2 permutation 檢驗之揭盲前凍結 record。凍死主檢驗參數（N、RNG seed）、揭盲時間線、兩則分支敘事，並依作者裁決新增 block-permutation 敏感度之分塊結構凍結（A3 exploratory）。本 record 於 A1 揭盲前落庫，為 freeze-before-data 之governance 憑證。依 records/2026-07-06-05 §3 A0 與 2026-07-05-13。 -->

# A0：H-C2 permutation 檢驗揭盲前凍結

## 目標

在 A1（run_c2_partial 揭盲 partial Spearman p 值）之前，凍死 H-C2 檢驗之全部殘餘自由度，
並事前寫定兩則分支敘事，杜絕揭盲後選擇檢驗設計（HARK）。本 record 落庫時間先於 A1 任何 p 值產出。

## 結果

### 主檢驗（依 2026-07-05-13，A1 一字不動、凍結義務）

- 統計量：partial Spearman `utility ~ coverage | precision`（C2a）；C2b 不對稱聲明依 -13。
- permutation：N = 100,000；RNG seed = 0；α = 0.05；n = 10 configs（全網格、不剪枝）。
- A1 執行 run_c2_partial（DINOv2 主裁決），程式與參數一字不動。
- 殘餘自由度僅 N 與 seed，且實質惰性：N = 10⁵ 下 p ≈ .05 之 Monte-Carlo SE ≈ .0007。

### 揭盲時間線（明文揭露）

- 各 config 之 coverage／precision／utility 均值已揭盲（confirmatory JSON 已讀）。
- 本凍結 record 寫於均值揭盲之後、p 值揭盲之前。
- 殘餘自由度（N、seed）於本 record 運行前封存；A1 之 p 值為唯一尚未揭盲項。
- A1 結果地位：預註冊檢定地位成立（程序資料前凍、run_c2_partial 決定性、殘餘自由度惰性且封存）；
  受損者為詮釋層盲性，不得標「非 confirmatory」，必須全揭時間線（依 -05 §3 A1）。

### 兩則分支敘事（事前寫定，揭盲前凍結，禁因果措辭）

- **H-C2a 顯著**：控制 precision 後，coverage 與 utility 仍有殘餘單調關聯，與機制假設 H1 之預測方向
  一致。此為相關非因果——不得寫「coverage 驅動效用」等因果句（-07 §4、-05 §2 格殺）。結論仍受雙段抹平
  caveat 與 gseed 可交換性 caveat（見下）約束。
- **H-C2a 不顯著**：控制 precision 後，coverage 與 utility 無顯著殘餘關聯；機制假設於 CIFAR-10 尺度
  未獲全網格偏相關支持。此與雙段機制觀察並存（全網格 ρ 將雙段抹平，-05 §1.5），科學重心移至 CIFAR-100
  機制複製（D3）。不得因不顯著而事後重劃分段或僅取高段裁決（-05 §2 格殺）。

### Block-permutation 敏感度之分塊結構凍結（新增，作者裁決；A3 exploratory）

兩句凍結：

1. **主檢驗依 -13 原樣不變**（A1 一字不動，凍結義務）；下列敏感度不改、不取代主檢驗。
2. **block-permutation 敏感度為 A3 exploratory 補充**，其分塊結構於 A1 結果揭盲前凍結：
   塊 = gseed 反對角線等價類 `{cells : seed+w = const}`（30 cells → 14 塊，因公式退化為 (seed+w)×10⁷、
   同 seed+w 之 cell 共享初始噪聲，-05 §1.12）。敏感度須尊重此共享噪聲之相關結構；其檢驗機制與偏誤方向
   量化於 A3 實作與報告，本 record 只凍分塊定義、不預設機制數字。

理由：分塊規則若留到看完 A1 再定，即為揭盲後選擇檢驗設計（HARK）。碰撞之正確處理為「主檢驗不動 ＋
敏感度事前定結構」，非偷換檢驗。此敏感度屬壞消息側（跨 config 誤差非獨立），B 定稿時與 scout 乾淨（好消息）
分開陳述，不得以好消息稀釋其能見度（-05 §5 附錄、作者裁決）。

## 後續

- A1 待作者 go 後執行（run_c2_partial，DINOv2，一字不動），全揭時間線。
- A3 依本 record 之凍結分塊結構實作 block-permutation 敏感度（exploratory），並補 per-seed 範圍、
  config-level jackknife、H-C2b 不對稱聲明、雙段抹平 caveat、因果措辭禁令（-05 §3 A3）。
- 因果措辭格殺持續至一切裁決文件。
