<!-- 用途：C5（near-boundary 純度過濾）規則段——以 near-boundary 純度為軸過濾生成集、量 TSTR 對純度之依賴，直接測 near-boundary 機制。計算前凍結。exploratory（C0），計算留 P1/β。與 flip_earlywarning 對賬（見後續）。依 records/2026-07-06-05 §6 C5。 -->

# C5：near-boundary 純度過濾規則段

## 目標

直接測 near-boundary 機制：以 near-boundary 純度（margin < 0.9525 之樣本佔比）為軸過濾固定組態之生成集，
量 TSTR 對純度之單調依賴。規則先於計算。exploratory（C0）。

## 結果（規則，凍結）

- **干預**：取固定組態（凍：w1.5），按 margin 排序、構造數個純度水準之子集（純度階梯凍：{原始, 高純度前 50%,
  低純度後 50%}），各重訓 TSTR。
- **判定（事前）**：TSTR 隨 near-boundary 純度單調上升 → 支持 near-boundary 承載效用；平坦或反向 → 不支持。
- **重複（凍死）**：每水準 2 次重訓，N 事前定。
- **計算**：留 P1/β 自落盤影像；禁第二次重生成。因果措辭禁；exploratory。

## 後續

P1/β 後計算。**掛帳對賬（作者裁決）**：C5 與 run_flip_earlywarning.py（coverage 主導方向性訊號）之輸出須對賬，
確認二者講同一機制或兩個；flip_earlywarning 若自飛且結論與 C 批不一致，B 定稿撞車。與 C2/C3 介入證據並讀。
與判決二獨立。
