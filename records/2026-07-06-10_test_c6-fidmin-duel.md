<!-- 用途：C6——per-seed FID-min 對決（判決三 selector 之核心數字）＋ Inception 側 per-seed 分離（C1 which-FID 之輸入，供強命題 pre-close 的 per-seed 誠實核對）。純讀 confirmatory JSON、無 GPU。原始輸出 results/cifar10_c6_fidmin_duel.json。判決三獨立於判決二（C2）。依 records/2026-07-06-05 §5 判決三、§6 C1。 -->

# C6：per-seed FID-min 對決 ＋ Inception 側 per-seed 分離

## 目標

抽 per-seed char_clean_fid 與 TSTR，定每 seed 之 FID-min 選中組態、算 per-seed FID-min regret vs
oracle，對照 CaF——供 B 判決三（selector，描述性）。同一計算給出 Inception(clean-fid)側 per-seed 之
FID-argmin 與 TSTR-argmax 格步距離，供 C1 強命題 pre-close 之 per-seed 誠實核對（修正「均值當 per-seed」）。

## 結果

### FID-min 對決（判決三核心，per-seed）

| seed | oracle(TSTRmax) | FID-min(argmin clean-fid) | FID-min regret | CaF(w2.5) regret |
|---|---|---|---|---|
| 10 | w2 (61.16) | w1.5 (fid 8.77, TSTR 58.71) | 2.45 | 0.54 |
| 11 | w1 (67.52) | w1.5 (fid 8.85, TSTR 67.24) | 0.28 | 5.03 |
| 12 | w1.5 (65.94) | w1.5 (fid 8.85, TSTR 65.94) | 0.00 | 5.49 |

- **FID-min regret mean 0.91（2.45/0.28/0.00）；CaF regret mean 3.69（0.54/5.03/5.49）。**
- FID-min（三 seed 皆選 w1.5）在 regret 上勝 CaF（三 seed 皆選 w2.5）約 2.78pp 均值。FID-min 為更便宜的
  baseline（只需 clean-fid、免 τ 免 coverage），卻近最優；CaF 系統性過度導引至 w2.5。此為判決三核心。

### Inception(clean-fid)側 per-seed 分離（C1 輸入，依 -05 §6 C1 凍結口径）

分離口径（凍）：主裁＝均值曲線 FID-argmin 與 TSTR-argmax 相異且 **>1 格步**；輔助＝三 seed 方向一致性，
僅穩健性描述、不單獨構成分離裁決。

- **主裁（均值）**：clean-fid argmin w1.5 與 TSTR argmax w1.5，格步 0 → 不分離 → 強命題 pre-close 成立於主裁口径。
- **輔助（per-seed，修正均值當 per-seed）**：FID-argmin vs TSTR-argmax 格步 = [1, 1, 0]，**0 個 seed >1 格步**。
  seed10（oracle w2 vs FID-min w1.5，1 格、regret 2.45）、seed11（w1 vs w1.5，1 格、regret 0.28）為相鄰一格；
  seed12 重合。**無 seed 達 >1 格步分離門檻。**
- **淨（誠實核對）**：Inception 側於主裁（均值）口径與 per-seed 輔助口径皆不分離；per-seed 未推翻主裁。
  強命題 pre-close 乾淨成立於兩口径。誠實邊註：2/3 seed 之 FID-argmin 與 TSTR-argmax 相鄰一格（其中 seed10
  之 regret 達 2.45pp）——但口径以格步距離（非 regret 大小）判分離，1 格不構成分離。

## 後續

- 判決三（selector）引本檔 FID-min 對決（0.91 vs 3.69）；與判決二（C2）獨立、不混寫。
- C1 規則段（下一 chunk）引本檔 Inception per-seed 分離：強命題 pre-close 於兩口径乾淨成立、附一格相鄰之誠實邊註；
  DINOv2 側 FD-DINOv2 之分離裁決留 P1/β 計算。
- 原始數字 results/cifar10_c6_fidmin_duel.json。
