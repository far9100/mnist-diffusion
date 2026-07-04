"""以 clean-fid 計算標準 Inception-FID，作為正確性錨點。

與 metrics_features.py 分開，使回報的 FID 與正統的 clean-fid
實作（Parmar et al., CVPR 2022）及其預先計算好的 CIFAR
statistics 一致。這是我們用來重現已發表模型 FID（例如 EDM
CIFAR-10 約 1.79）的方法，並藉此在任何 sampler sweep 之前
驗證整條量測流程。

clean-fid 讀取影像資料夾，因此張量輸入會被寫成暫存目錄中的
PNG。影像預期為 [0,1]、(N,3,H,W)。
"""

import os
import tempfile

import torch
from torchvision.utils import save_image


def _dump_pngs(images, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(images.size(0)):
        save_image(images[i], os.path.join(out_dir, f"{i:06d}.png"))


def clean_fid_vs_dataset(images, dataset_name="cifar10", dataset_split="train",
                         dataset_res=32, mode="clean", tmp_dir=None):
    """generated `images`（[0,1] 張量）相對於 clean-fid 預先計算 stats 的 FID。"""
    from cleanfid import fid as cfid
    cleanup = tmp_dir is None
    tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="cleanfid_gen_")
    try:
        _dump_pngs(images, tmp_dir)
        # num_workers=0：clean-fid 的縮放器是區域 closure，無法被
        # Windows 的 'spawn' DataLoader worker pickle（否則會出 EOFError）。
        return cfid.compute_fid(tmp_dir, dataset_name=dataset_name,
                                dataset_res=dataset_res, dataset_split=dataset_split,
                                mode=mode, num_workers=0)
    finally:
        if cleanup:
            for f in os.listdir(tmp_dir):
                os.remove(os.path.join(tmp_dir, f))
            os.rmdir(tmp_dir)


def clean_fid_two_sets(images_a, images_b, mode="clean", tmp_root=None):
    """兩組張量影像之間的 FID（例如 real-vs-real 檢查應該很小）。"""
    from cleanfid import fid as cfid
    root = tmp_root or tempfile.mkdtemp(prefix="cleanfid_pair_")
    dir_a, dir_b = os.path.join(root, "a"), os.path.join(root, "b")
    try:
        _dump_pngs(images_a, dir_a)
        _dump_pngs(images_b, dir_b)
        return cfid.compute_fid(dir_a, dir_b, mode=mode, num_workers=0)
    finally:
        for d in (dir_a, dir_b):
            if os.path.isdir(d):
                for f in os.listdir(d):
                    os.remove(os.path.join(d, f))
                os.rmdir(d)
        if os.path.isdir(root):
            os.rmdir(root)


def has_dataset_stats(dataset_name="cifar10", dataset_split="train", dataset_res=32, mode="clean"):
    """clean-fid 是否已對此 dataset 設定備有預先計算的 stats。"""
    from cleanfid import fid as cfid
    return cfid.test_stats_exists(f"{dataset_name}", mode, dataset_res, dataset_split) \
        if hasattr(cfid, "test_stats_exists") else None
