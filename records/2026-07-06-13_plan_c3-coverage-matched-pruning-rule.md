<!-- 用途：C3（coverage-matched pruning）規則段——以覆蓋匹配之剪枝隔離「coverage 本身」是否驅動效用，橋接偏相關到介入。計算前凍結干預與橋接論證。exploratory（C0）。計算留 P1/β（需落盤影像）。依 records/2026-07-06-05 §6 C3。 -->

# C3：coverage-matched pruning 規則段

## 目標

橋接偏相關（C2a）到介入證據：把高 coverage 組態剪枝至與低 coverage 組態同 coverage，看 TSTR 差是否隨之消失，
以隔離 coverage 本身之因果貢獻（對 precision 等 w-共變項）。規則先於計算。exploratory（C0）。

## 結果（規則，凍結）

- **干預**：取一對組態（凍：w2.5 高 coverage、w1 低 coverage），剪 w2.5 之生成集使其 DINOv2 coverage 降至 w1 水準
  （剪枝規則凍：移除離真實流形最近之樣本以降 coverage、程序事前定），重訓 TSTR。
- **橋接論證（入 record）**：若 coverage 匹配後 TSTR 差顯著縮小 → coverage 本身承載效用差（支持機制）；若不縮小
  → 效用差來自 coverage 以外之 w-共變項（機制敘事須修）。
- **對照/重複（凍死）**：等計數隨機剪枝對照；每情境 2 次重訓，N 事前定。
- **計算**：留 P1/β 自落盤影像；禁第二次重生成。因果措辭禁；C3 exploratory、不入協定裁決。

## 後續

P1/β 後計算，對號入座、入 C 批 record。橋接結論與 C2 介入證據並讀。與判決二獨立。
