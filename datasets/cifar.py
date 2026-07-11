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


def _build(name, data_dir, train):
    name = name.lower()
    if name == "cifar10":
        return tvd.CIFAR10(data_dir, train=train, download=True)
    if name == "cifar100":
        return tvd.CIFAR100(data_dir, train=train, download=True)
    raise ValueError(f"unknown dataset {name}")


def load_cifar(name="cifar10", data_dir="./data", train=True):
    """回傳該切分的 (images[-1,1] (N,3,32,32) float, labels long)。"""
    return _to_tensor(_build(name, data_dir, train))


def load_cifar_01(name="cifar10", data_dir="./data", train=True):
    """同 load_cifar，但影像為 [0,1] 值域——feature 抽取器（Inception / DINOv2 / clean-fid）
    預期的輸入範圍，它們自己處理後續的正規化。

    本專案只保留這一個 CIFAR 載入模組。先前另有一份 cifar_data.py 平行實作同樣的東西但回傳
    [0,1]，兩份值域不同、易生無聲縮放（R-2026-07-07-01 掛帳），已併入此處。此函式直接用
    ToTensor 取值，而非由 [-1,1] 換算回來，以保與舊路徑的數值逐位相同。
    """
    ds = _build(name, data_dir, train)
    ds.transform = transforms.ToTensor()          # [0,1], CHW
    loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
    imgs, labels = [], []
    for x, y in loader:
        imgs.append(x)
        labels.append(y)
    return torch.cat(imgs), torch.cat(labels)


def load_real_per_class(name, per_class, seed=0, data_dir="./data", train=True):
    """每個 class 取樣 `per_class` 張真實影像（平衡的真實探測/參考集）。

    平衡取樣很重要：real-vs-fake 數量不平衡會扭曲 PRDC
    coverage（MNIST 每數字 50 張的假影）。回傳 (images, labels)。

    樣本數不足時直接拋錯，不做靜默截斷：CIFAR-100 訓練集每類僅 500 張（CIFAR-10 為
    5000），若沿用 CIFAR-10 的 per_class=1000 口徑，torch 的切片會安靜地只回傳 500 張，
    使實際參考集大小與 metadata 記錄的 per_class 不符——這正是 claude.md §5.2 要防的
    「量不出參數來源的 scalar」。寧可讓呼叫端在開跑前就失敗。
    """
    images, labels = load_cifar(name, data_dir, train)
    num_classes = NUM_CLASSES[name.lower()]
    g = torch.Generator().manual_seed(seed)
    sel_imgs, sel_labels = [], []
    for c in range(num_classes):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        if idx.numel() < per_class:
            raise ValueError(
                f"{name} class {c} 僅有 {idx.numel()} 張（train={train}），不足 per_class="
                f"{per_class}。CIFAR-100 每類上限 500，請調低 per_class 或改用其他切分。")
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
