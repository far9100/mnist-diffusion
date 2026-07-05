<!-- 用途：記錄 Stage 2（真實 CIFAR-10 judge 訓練與 near-boundary threshold 校準）的目標、結果與後續。 -->

# 2026-07-05-05 真實 CIFAR-10 judge 與 near-boundary 校準

## 目標

標籤噪音由「後續」升為必要儀器（見 2026-07-05-03 協定增修），需要一個在真實 CIFAR-10 上訓練的
ResNet-18 judge 當裁判，量合成樣本的 margin 與 top1 是否等於給定標籤。judge 須夠準，診斷才有意義。
並為 CIFAR 重新校準 mechanism 的 near-boundary threshold（現預設 0.5 是 MNIST 調的）。

## 結果

judge：真實 CIFAR-10 全訓練集（50000 張）訓 ResNet-18 25 epoch（SGD + cosine + 增強），真實測試
準確度 93.08%（train 98.51%），足以作裁判。存 checkpoints/cifar10_judge.pt。

threshold 校準：真實測試 margin（p_top1 − p_top2）分位數為 p05=0.408、p10=0.725、p20=0.953、
p25=0.978、p50=0.999。取 p20 為 CIFAR 專用 near-boundary threshold = 0.9525（真實 near-boundary
比例 0.20）。

重要觀察：93% judge 下真實 CIFAR-10 的 margin 中位數高達 0.999，代表 CIFAR-10 在 judge 空間仍相當
可分、near-boundary 訊號偏飽和（與 MNIST 同性質，程度輕些）。因此機制的強證據仍須靠 CIFAR-100；
CIFAR-10 以難子集為次選。以「相對分位」（p20）校準 threshold 可強制 20% 解析度，繞過絕對飽和，
但此飽和 caveat 需在解讀時交代。

## 後續

- Stage 3 scout 以此 judge + threshold 0.9525 量各 guidance 的標籤噪音與 near-boundary，配合
  precision/coverage/TSTR。
- 解讀難子集/CIFAR-10 的 near-boundary 時，須註明 judge 空間下 CIFAR-10 仍偏可分；CIFAR-100 為
  未飽和主戰場。
