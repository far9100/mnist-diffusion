<!-- 用途：C2（boundary-targeted pruning）規則段——-07 §4(a) 介入式證據之字面實例，於計算前凍結干預程序與判定。exploratory（C0），規則先於計算、不因結果回改。計算需落盤 per-sample 影像與 margin，留 P1/β（driver 原已刪 per-sample）。依 records/2026-07-06-05 §6 C2、2026-07-03-07 §4(a)。 -->

# C2：boundary-targeted pruning 規則段

## 目標

以介入式證據補強 -07 §4「偏相關非因果」之缺口：直接移除 near-boundary 樣本、量下游 TSTR 掉幅，測 near-boundary
樣本是否因果驅動效用。規則於計算前凍結。exploratory（C0）。

## 結果（規則，凍結）

- **干預**：取固定組態（凍：w1.5 與 w2.5 各一，涵蓋 FID-min 與 CaF 選中）之落盤生成集，移除 margin < 0.9525
  之 near-boundary 樣本，重訓 TSTR、量準確率掉幅。
- **對照（凍）**：等數量之隨機移除（同 seed、同計數），分離「移除 near-boundary」與「單純減少樣本數」。
- **重複（凍死）**：每情境 2 次重訓，N 事前定，不加跑至顯著。
- **判定（事前）**：near-boundary 移除掉幅顯著大於隨機移除掉幅 → 支持 near-boundary 因果角色（介入式，超越偏相關）；
  否則 → 不支持，偏相關可能為潛在因子共驅。因果措辭仍禁（-07 §4、-05 §2 格殺）；C2 為 exploratory，不入協定裁決。
- **計算**：需 per-sample 影像與 margin（driver 原已刪），留 P1/β 自落盤影像取用，禁第二次重生成。

## 後續

P1/β 落盤後計算，結果對號入座事前判定、入 C 批 record。與判決二（H-C2）獨立、不混寫。
