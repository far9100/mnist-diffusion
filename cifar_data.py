"""Phase 1 效用研究用的 CIFAR-10 / CIFAR-100 載入。

本檔為 Phase 1 早期量測驗證（見 validate_metrics.py）使用的載入器，
回傳 [0,1] 值域影像。主線 CIFAR 腳本改用 `datasets/cifar.py`，該檔回傳
[-1,1] 值域。兩者值域不同，勿混用同一路徑的資料，以免無聲縮放。

慣例：
  - 原始影像以 [0,1] float (N,3,32,32) 回傳；生成式評估的
    feature 抽取器（Inception/DINOv2）與 clean-fid 預期此範圍，並
    自行處理縮放/正規化。
  - 下游分類器在訓練時套用自己的逐 channel 正規化
    （見 CIFAR_MEAN/STD 常數），與標準 CIFAR 配方一致。
"""

import torch
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


def _dataset_cls(name):
    return {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}[name]


def load_cifar_tensors(name="cifar10", train=True, data_dir="./data"):
    """回傳 (images, labels)：images 為 [0,1] 範圍的 (N,3,32,32) float，labels 為 (N,) int64。"""
    cls = _dataset_cls(name)
    ds = cls(data_dir, train=train, download=True, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
    imgs, labels = [], []
    for x, y in loader:
        imgs.append(x)
        labels.append(y)
    return torch.cat(imgs), torch.cat(labels)


def load_cifar_per_class(name="cifar10", per_class=100, split="test",
                         data_dir="./data", seed=0):
    """每個 class 取樣 `per_class` 張影像（用於 PRDC / FID 參考探測）。"""
    images, labels = load_cifar_tensors(name, train=(split == "train"), data_dir=data_dir)
    n_classes = NUM_CLASSES[name]
    g = torch.Generator().manual_seed(seed)
    out_i, out_l = [], []
    for c in range(n_classes):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        k = min(per_class, idx.numel())
        pick = idx[torch.randperm(idx.numel(), generator=g)[:k]]
        out_i.append(images[pick])
        out_l.append(labels[pick])
    return torch.cat(out_i), torch.cat(out_l)


def normalize_for_classifier(images, name="cifar10"):
    """對 [0,1] 範圍的影像套用標準 CIFAR 逐 channel 正規化。"""
    mean = CIFAR10_MEAN if name == "cifar10" else CIFAR100_MEAN
    std = CIFAR10_STD if name == "cifar10" else CIFAR100_STD
    mean = torch.tensor(mean, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def train_transform(name="cifar10"):
    """標準 CIFAR 訓練資料增強 + 正規化（供下游 ResNet 使用）。"""
    mean = CIFAR10_MEAN if name == "cifar10" else CIFAR100_MEAN
    std = CIFAR10_STD if name == "cifar10" else CIFAR100_STD
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def eval_transform(name="cifar10"):
    mean = CIFAR10_MEAN if name == "cifar10" else CIFAR100_MEAN
    std = CIFAR10_STD if name == "cifar10" else CIFAR100_STD
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
