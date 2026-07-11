<!-- 用途：B 骨架——confirmatory 揭盲後之 record 骨架。三判決分立、禁交叉混寫；判決二逐字引 A 批收束（2026-07-06-09），判決三引 C6，判決一待 C1（P1/β）。附錄含時序鏈、gseed 碰撞、scout 乾淨、MNIST 降級、fee419e 純事實時序、揭盲時間線。本檔為骨架（結構＋鎖定內容指標），B 定稿於 γ、待 C 批計算。依 records/2026-07-06-05 §5、2026-07-06-09、-10。 -->

# B 骨架：confirmatory 三判決分立 record

## 目標

搭 confirmatory record 骨架：三判決互不裁決、禁交叉混寫。判決二逐字引 A 批收束（-09），判決三引 C6（-10），
判決一待 C1（P1/β）。本檔為骨架，B 定稿於 γ、待 C 批計算補入。

## 判決一（thesis）

- **FID/TSTR 重合**：於 w1.5（clean-fid 8.82 / TSTR 63.96）；ρ(−char_clean_fid, TSTR) ≈ 0.96（B1 前腳本複核）。
  README 頭條「FID-opt 偏離 TSTR-opt」在本資料上被反證。
- **災難性非單調**：w≥3 崩 11–30pp；「最優位置跨資料集移動」；此為存活觀察。
- **「內部最優」**：為未登記 exploratory 觀察，附上升肢 +0.80±3.3pp（SE 1.9）不確定性，明文從未受 confirmatory 保護。
- **「必然次優」全稱句撤下**（E2 執行）。
- **C1 which-FID（待 P1/β）**：強命題 pre-closed（-11 §2）；活結果空間二元——DINOv2 分離→表徵依賴弱版本／不分離→
  反證確立（-11 §3、§4）。裁決於 DINOv2 揭盲後對號入座。

## 判決二（H-C2）

逐字引 records/2026-07-06-09「B 判決二之凍結內容」全段（顶格 C2b 號翻經驗證據 ＋ 主體三重 caveat ＋ A2 增列 ＋
三軸總結句 ＋ 結論），不重寫。淨：CIFAR-10 機械通過且跨表徵一致，但穩健性須 CIFAR-100 獨立複製回答。

## 判決三（selector，描述性）

- **開頭明文**：「協定未凍門檻，本節不作過／敗判定。」
- **Pareto 失明**：w2.5 (.873,.792) 嚴格支配三 oracle → CaF 結構性選不到 oracle，非校準問題（C8 引理、tau_robustness.picks 實證）。
- **τ knife-edge**：w1 於 seed11/12 邊際 .0034/.0092，可行性由 seed 級噪聲決定。
- **FID-min 對決（C6）**：FID-min regret 0.91（2.45/0.28/0.00）vs CaF 3.69（0.54/5.03/5.49），約 2.78pp；FID-min 更便宜
  （免 τ 免 coverage）卻近最優，打在 CaF 存在理由上。
  **per-seed 對稱句（必寫，防讀成全面碾壓）**：FID-min 為 3 seed 2 勝 1 負——seed10 上 CaF(0.54) 反勝 FID-min(2.45)，
  FID-min 負的那次輸得少、贏的兩次贏得多。壞消息不稀釋、CaF 那一格好消息不抹掉。
- **regret 數字口径**：主 3.69（per-seed regret 均值）；並列 **2.77（mean-curve oracle 口径：均值 oracle w1.5 之 63.96
  減均值 CaF w2.5 之 61.19）**。兩數差異來源即口径（per-seed vs mean-curve），明標。
- **可辯護措辭**：CaF 為可靠避崖器、糟糕平台優化器——FID-min 同樣避崖且成本結構相同。modal_fraction 1.0 重讀為低變異
  高偏差。

## 附錄

- **時序鏈**：052492c（規格凍結 07-05 14:59）→ ec1f746（儀器 23:53）→ 推導起跑 ≈07-06 00:29（標推導值、非登記）→ mtime 08:45。
- **gseed 碰撞（壞消息）**：反對角線共享噪聲、H-C2 可交換性受損、CIFAR-10 內部無乾淨 restricted 補救（A3 (c)、-08 spec）。
- **scout 乾淨（好消息，不稀釋壞消息能見度）**：兩 scout（upper_scout:71、scout:115）皆 flat seed=0，種子值域與 confirmatory
  之 (seed+w)×10⁷ 不相交、無噪聲重疊（§1.12 訂正）。與碰撞分述、不互稀釋。
- **MNIST 降級**：分離證據軼事級（一格步、單 seed argmax、bespoke FID、方向不轉移）。
- **fee419e 時序（純事實，去自我敘事）**：「fee419e commit 於 2026-07-05 22:37，早於 confirmatory 起跑約兩小時；定位重寫
  為 pre-data，與隨後揭盲的資料未經對賬。」（讀者自得結論，不代敘事。）
- **揭盲時間線**：均值揭盲 → A0 凍結（permutation N/seed、分支敘事、block 結構）→ A1 p 值揭盲。
- **保留槽：P 對帳結果（β 後填）**（依 records/2026-07-08-02 §1.2 補）：P0/P1 之持久化重算對帳結論（決定性三態、k 溯源、逐 config 逐位/容忍狀態）於 β 完成後填入，作判決一與 C 批信任鏈之基底。現為空槽，待 P 資產落地。
- **保留槽：ec1f746 結論（F2）**（依 records/2026-07-08-02 §1.2 補）：引 dossier（records/2026-07-06-06）乙-2——ec1f746 @ 2026-07-05 23:53:43 之 confirmatory 儀器 commit，3 檔 +295/−19（run_c2_partial.py 等，純實作），早於全量開跑；屬凍結後行使實作自由度之流程債，純事實入附錄、不代敘事。

## 後續

B 定稿於 γ：待 C 批（C2/C3/C5/C7/C8）計算補入、判決一補 C1（P1/β）結果。三判決分立、C6 數字只進判決三、判決二逐字引 -09。
E2/E3/E4 於 B 定稿後。dossier（-06）為事實基 backing。
