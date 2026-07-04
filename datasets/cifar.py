"""效用對保真度研究用的 CIFAR-10 / CIFAR-100 載入器。

對應 MNIST sandbox 的輔助函式（analyze_distribution.load_real_per_class /
evaluate.build_dataloaders），使 CaF selector、TSTR、PRDC 與 mechanism 程式碼
能以最小改動延用到 CIFAR。影像以 [-1, 1] 範圍的 float 張量回傳，
形狀 (N, 3, 32, 32)——與 EDM sampler 經重新縮放後輸出的
慣例相同，也是 from-scratch 分類器訓練時所用的格式。

CIFAR-100 是計畫所需的「難以線性分離」資料集，用來 (a) 在 CIFAR-10 之外
驗證 CaF<->TSTR，並 (b) 展示在 MNIST 上會飽和的
近邊界 margin 機制。

用法：
    from datasets.cifar import load_cifar, load_real_per_class, build_test_loader
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets as tvd
from torchvision import transforms

NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


def _to_tensor(dataset):
    """將 torchvision dataset 堆疊成 (images[-1,1] N3HW float, labels long)。"""
    tf = transforms.Compose([
        transforms.ToTensor(),                     # [0,1], CHW
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1,1]
    ])
    dataset.transform = tf
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
    imgs, labels = [], []
    for x, y in loader:
        imgs.append(x)
        labels.append(y)
    return torch.cat(imgs), torch.cat(labels)


def load_cifar(name="cifar10", data_dir="./data", train=True):
    """回傳該切分的 (images[-1,1] (N,3,32,32) float, labels long)。"""
    name = name.lower()
    if name == "cifar10":
        ds = tvd.CIFAR10(data_dir, train=train, download=True)
    elif name == "cifar100":
        ds = tvd.CIFAR100(data_dir, train=train, download=True)
    else:
        raise ValueError(f"unknown dataset {name}")
    return _to_tensor(ds)


def load_real_per_class(name, per_class, seed=0, data_dir="./data", train=True):
    """每個 class 取樣 `per_class` 張真實影像（平衡的真實探測/參考集）。

    平衡取樣很重要：real-vs-fake 數量不平衡會扭曲 PRDC
    coverage（MNIST 每數字 50 張的假影）。回傳 (images, labels)。
    """
    images, labels = load_cifar(name, data_dir, train)
    num_classes = NUM_CLASSES[name.lower()]
    g = torch.Generator().manual_seed(seed)
    sel_imgs, sel_labels = [], []
    for c in range(num_classes):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(idx.numel(), generator=g)[:per_class]]
        sel_imgs.append(images[perm])
        sel_labels.append(labels[perm])
    return torch.cat(sel_imgs), torch.cat(sel_labels)


def build_test_loader(name, batch_size=128, data_dir="./data", num_workers=2):
    """供 TSTR 評估用的真實測試集 loader（在合成資料上訓練、在真實資料上測試）。"""
    images, labels = load_cifar(name, data_dir, train=False)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size,
                      shuffle=False, num_workers=num_workers)


def _self_check():
    imgs, labels = load_real_per_class("cifar10", per_class=5, seed=0)
    assert imgs.shape == (50, 3, 32, 32), imgs.shape
    assert imgs.min() >= -1.001 and imgs.max() <= 1.001, (imgs.min(), imgs.max())
    assert sorted(labels.unique().tolist()) == list(range(10))
    print(f"CIFAR-10 probe OK: {tuple(imgs.shape)}, range [{imgs.min():.2f},{imgs.max():.2f}]")


if __name__ == "__main__":
    _self_check()
