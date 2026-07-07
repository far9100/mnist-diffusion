<!-- 用途：C1（which-FID）規則段——於 FD-DINOv2 計算（P1/β）之前凍結分離口径、強命題 pre-close、活結果空間（二元）與雙分支叙事。分離口径寫入正文（執行者於 DINOv2 揭盲時照判），非只 record 頭。exploratory（C0）。C1 只鎖規則不算，DINOv2 側 FD 計算留 P1/β。依 records/2026-07-06-05 §6 C1、2026-07-06-10（C6，Inception per-seed 輸入）。 -->

# C1：which-FID 規則段（DINOv2 揭盲前凍結）

## 目標

於 FD-DINOv2 計算（P1/β）前，凍結 which-FID 之分離口径、強命題 pre-close 判定、活結果空間與雙分支
叙事，杜絕揭盲後選設計。本段為 exploratory（C0）：規則先於計算、不因結果回改。C1 只鎖規則，DINOv2
側 FD-DINOv2 之實際分離裁決留 P1/β。

## 結果（規則正文，執行者於 DINOv2 揭盲時照此判）

### 1. 分離口径（凍結，正文執行用）

- **主裁**：均值曲線之 FID-argmin 與 TSTR-argmax 相異且 **>1 格步** = 分離；否則不分離。
- **輔助**：三 seed 方向一致性，僅作穩健性描述——主裁判分離時加強、判不分離時作誠實邊註，**不單獨構成
  分離裁決**。
- **口径以格步距離判，非 regret 大小**。示例（C6）：seed10 之 FID-argmin(w1.5) 與 TSTR-argmax(w2) 差 1 格、
  regret 達 2.45pp，但格步 1 → 依口径不分離。regret 大不改判。

### 2. 強命題（雙空間皆分離）pre-close 判定

- **Inception 側（clean-fid，已見臂）**：主裁均值 argmin w1.5 = TSTR argmax w1.5，0 格步 → 不分離。
- **per-seed 核對（C6，修正「均值當 per-seed」）**：FID-argmin vs TSTR-argmax 格步 = [1, 1, 0]，0 個 >1 →
  確認均值未藏 per-seed 分離。
- **區分句（正文寫死）**：強命題 pre-close **成立於均值主裁口径（0 格步）**；per-seed 核對為**加強、非必要**
  （確認均值未藏分離），**不得讀成 pre-close 需 per-seed 撐**。誠實邊註：2/3 seed 之 FID-argmin 與 TSTR-argmax
  相鄰一格（seed10 regret 達 2.45pp），但格步 ≤1、依口径不分離。
- **結論**：強命題 pre-closed。活結果空間為乾淨二元（見 §3）。

### 3. 活結果空間（二元，DINOv2 揭盲前鎖）

- **DINOv2 分離（>1 格步）** → 表徵依賴弱版本，標「非預期、削弱普適性」。
- **DINOv2 不分離（≤1 格步）** → CIFAR-10 尺度反證確立。

### 4. 雙分支叙事（事前寫定，DINOv2 揭盲前，等量、反證版筆墨不少於弱版本）

- **弱版本（DINOv2 分離）**：FD-DINOv2 之 argmin 與 TSTR-argmax 相異 >1 格步，即 DINOv2 特徵空間中 FID 誤導
  而 TSTR 不。但 Inception 側（clean-fid）已確認不分離（§2、C6），故此為**表徵依賴之弱版本**：FID 是否誤導
  取決於特徵空間。這削弱普適性——CaF 的價值隨表徵而變、非普適。作為 thesis 於 CIFAR-10 的存活形式，須明標
  「表徵依賴、非普適」；且 FID-min baseline 於 clean-fid 上仍勝 CaF（判決三 regret 0.91 vs 3.69），故即便
  DINOv2 分離，selector 層的 CaF 優勢未獲支持。
- **反證版（DINOv2 不分離）**：FD-DINOv2 之 argmin 與 TSTR-argmax 差 ≤1 格步，即 DINOv2 空間中 FID 亦不誤導。
  疊加 Inception 側不分離（§2、C6），**兩特徵空間皆不分離 → CIFAR-10 尺度「FID≠效用」頭條命題反證確立**：
  FID（兩表徵）皆為近最優 selector，CaF 無附加價值。此與判決三（FID-min regret 0.91 勝 CaF 3.69）一致。
  thesis 於 CIFAR-10 尺度被資料反證，科學重心移 CIFAR-100（是否更難資料集上 FID 才誤導）。

### 5. 反 HARK 論證（正文）

口径於 DINOv2（唯一未見臂）揭盲前凍結；對已見之 Inception 側無鑑別力（任何合理口径皆判不分離，C6 坐實
格步 [1,1,0] 皆 ≤1）；故口径選擇無法對已見資料 HARK。雙分支叙事於揭盲前等量寫定，反證版筆墨不少於弱版本。

### 6. 計算（留 P1/β，C1 不算）

接線 fd_from_features（metrics_features.py:63，現存未被 multiseed driver 呼叫）、真實 DINOv2 Fréchet 參考、
per-config FD-DINOv2（隨 P1 streaming 產出）。C1 只鎖規則。

## 後續

DINOv2 側 FD-DINOv2 於 P1/β 依 §1 口径裁分離/不分離 → 對應 §4 叙事、對號入座不臨時解釋。判決一（thesis）
引本 C1 結果。C1 與判決三（selector）獨立。
