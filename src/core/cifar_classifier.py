"""從零實作的 CIFAR 分類器與 TSTR 測試框架（Phase 1）。

CIFAR 的下游效用指標是 TSTR：僅用 diffusion 生成的 CIFAR 影像
訓練此分類器，並在真實 CIFAR 測試集上測試。計畫
指定使用「from-scratch 的 ResNet 空間」（依 feature 空間不匹配的護欄，
與 DINOv2 PRDC 空間有別）。這是一個標準的 CIFAR ResNet-18
（3x3 stem、無 maxpool），以慣用的 SGD + cosine 配方訓練。

影像為 [-1, 1] 範圍的 float 張量，形狀 (N, 3, 32, 32)——與
datasets/cifar.py 及 EDM sampler 的輸出慣例一致。

用法：
    from cifar_classifier import ResNet18, train_classifier, evaluate, run_tstr
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    """CIFAR ResNet-18：3x3 stem、無起始 maxpool、4 個 stage [64,128,256,512]。"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return F.adaptive_avg_pool2d(out, 1).flatten(1)  # (N, 512)

    def forward(self, x):
        return self.linear(self.features(x))


class _AugmentedTensorDataset(torch.utils.data.Dataset):
    """帶有訓練時 CIFAR 資料增強（random crop + 水平翻轉）的 TensorDataset。"""

    def __init__(self, images, labels, augment=True, generator=None):
        self.images = images
        self.labels = labels
        self.augment = augment
        self.gen = generator   # 提供時增強用此 CPU Generator（決定性）；None 則用全域 RNG（現行行為）

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, i):
        img = self.images[i]
        if self.augment:
            g = self.gen
            if torch.rand(1, generator=g).item() < 0.5:
                img = torch.flip(img, dims=[2])
            pad = F.pad(img.unsqueeze(0), (4, 4, 4, 4), mode="reflect").squeeze(0)
            top = torch.randint(0, 9, (1,), generator=g).item()
            left = torch.randint(0, 9, (1,), generator=g).item()
            img = pad[:, top:top + 32, left:left + 32]
        return img, self.labels[i]


def train_classifier(model, images, labels, device, epochs=30, lr=0.1,
                     batch_size=128, augment=True, weight_decay=5e-4, verbose=False, seed=None):
    # seed 非 None：以 CPU Generator 控制 DataLoader shuffle 與增強，使 TSTR 可決定性重現（凍結 v1
    # 走 seed=None，行為與本次改動前一致，保住對帳語意）。CUDA 上殘留 cuDNN 非決定性另議、不由此保證。
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    ds = _AugmentedTensorDataset(images, labels, augment=augment, generator=gen)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0,
                        drop_last=False, generator=gen)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    model.train()
    for ep in range(1, epochs + 1):
        tot, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            tot += x.size(0)
        sched.step()
        if verbose:
            print(f"    ep {ep}/{epochs} loss={loss_sum/tot:.3f} train_acc={100*correct/tot:.2f}%")


@torch.no_grad()
def evaluate(model, loader, device, num_classes=10):
    model.eval()
    correct = torch.zeros(num_classes)
    total = torch.zeros(num_classes)
    for x, y in loader:
        preds = model(x.to(device)).argmax(1).cpu()
        for p, t in zip(preds, y):
            total[t] += 1
            if p == t:
                correct[t] += 1
    overall = 100.0 * correct.sum().item() / total.sum().item()
    per_class = {c: (100.0 * correct[c] / total[c]).item() if total[c] > 0 else float("nan")
                 for c in range(num_classes)}
    return overall, per_class


def run_tstr(images, labels, real_test_loader, device, num_classes=10,
             epochs=30, lr=0.1, batch_size=128, augment=True, seed=None):
    """在 generated 的 (images,labels) 上訓練一個全新 ResNet；在真實測試 loader 上評估。

    seed 非 None 時：`torch.manual_seed(seed)` 控制模型權重初始化，並把 seed 傳入 train_classifier
    控制 shuffle 與增強；seed=None 維持現行未種子化行為（凍結對帳語意不變）。
    """
    if seed is not None:
        torch.manual_seed(seed)
    model = ResNet18(num_classes=num_classes).to(device)
    train_classifier(model, images, labels, device, epochs=epochs, lr=lr,
                     batch_size=batch_size, augment=augment, seed=seed)
    overall, per_class = evaluate(model, real_test_loader, device, num_classes)
    return overall, per_class


def _self_check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = ResNet18(10).to(device)
    x = torch.randn(4, 3, 32, 32, device=device)
    assert m(x).shape == (4, 10)
    assert m.features(x).shape == (4, 512)
    print(f"ResNet18-CIFAR OK on {device}: logits {tuple(m(x).shape)}, feat {tuple(m.features(x).shape)}, "
          f"params={sum(p.numel() for p in m.parameters())/1e6:.1f}M")

    # T6b：seeded TSTR 決定性——CPU 上同 seed 兩次逐位相同（CUDA 因 cuDNN 非決定性不保證，故測 CPU）。
    cpu = torch.device("cpu")
    imgs, labs = torch.randn(64, 3, 32, 32), torch.randint(0, 10, (64,))
    tl = DataLoader(torch.utils.data.TensorDataset(torch.randn(20, 3, 32, 32),
                                                   torch.randint(0, 10, (20,))), batch_size=10)
    a, _ = run_tstr(imgs, labs, tl, cpu, epochs=2, seed=7)
    b, _ = run_tstr(imgs, labs, tl, cpu, epochs=2, seed=7)
    assert a == b, f"seeded run_tstr 非決定性：{a} vs {b}"
    print(f"T6b seeded-TSTR determinism OK (seed=7 兩次 → {a:.4f} == {b:.4f} on CPU)")


if __name__ == "__main__":
    _self_check()
