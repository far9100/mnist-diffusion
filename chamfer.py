"""簡化版 Chamfer guidance 基線，用於 matched-budget 對決（非官方實作，忠實重寫核心）。

背景（給初次接觸的讀者）：Chamfer Guidance（arXiv 2508.10631, NeurIPS 2025）是一種
「免訓練」的 guidance——它不改模型、也不需要無條件模型，而是在取樣過程中，用少量（2 至 32 張）
真實 exemplar 去「推」生成軌跡，使整批生成樣本更能覆蓋真實範例的多樣性。這正好對應本專案主張
「驅動下游效用的是 coverage（多樣性）」的那條軸，因此拿它當對照最合適。

本檔是為 matched-budget 對決所寫的「簡化且忠實」的核心重寫，非官方程式碼。簡化之處明列於下，
使實驗誠實可拆：
  - 在單一特徵空間 f 上導引（官方可用多層/多特徵）。
  - 直接對預測的乾淨影像 x0 導引，不走解碼器梯度。
  - guidance 權重固定，不做逐步 schedule 調參。

核心量：對「生成批次特徵 G」與「真實 exemplar 特徵 E」定義對稱 Chamfer 距離
    CD(G, E) = mean_i min_j ||g_i - e_j||  +  mean_j min_i ||g_i - e_j||
其中第二項（exemplar -> 生成）是「覆蓋項」：要求每個真實 exemplar 附近都有一個生成樣本，
最小化它會鼓勵生成批次把 exemplar 的多樣性都涵蓋到。取樣時對 x0 取此量的梯度、往下降方向移動，
即把整批樣本推去覆蓋 exemplar。

Usage (self-check):
    uv run python chamfer.py
"""

import torch


def chamfer_distance(a, b):
    """集合 a (N,D) 與 b (M,D) 之間的對稱 Chamfer 距離（純量）。

    兩項：每個 a 到最近 b 的平均距離，加上每個 b 到最近 a 的平均距離。相同集合為 0。
    """
    d = torch.cdist(a, b)                      # (N, M) 兩兩歐氏距離
    return d.min(dim=1).values.mean() + d.min(dim=0).values.mean()


def chamfer_coverage_term(gen_feats, exemplar_feats):
    """覆蓋項：每個 exemplar 到最近生成樣本的平均距離（exemplar -> 生成）。

    最小化它 = 讓每個真實 exemplar 附近都有生成樣本 = 鼓勵覆蓋 exemplar 的多樣性。
    """
    d = torch.cdist(gen_feats, exemplar_feats)  # (N, M)
    return d.min(dim=0).values.mean()           # 對每個 exemplar 取最近生成樣本，再平均


def chamfer_guidance_grad(x0, feature_fn, exemplar_feats, weight=1.0, term="coverage"):
    """取樣時對預測乾淨影像 x0 的 Chamfer guidance 位移。

    參數：
        x0: (N, C, H, W) 目前步驟預測的乾淨影像批次。
        feature_fn: 把影像映到特徵空間的可微函式（例如分類器 penultimate 或 DINOv2）。
        exemplar_feats: (M, D) 真實 exemplar 在同一特徵空間的特徵。
        weight: guidance 強度。
        term: "coverage"（覆蓋項）或 "chamfer"（對稱 Chamfer 距離）。

    回傳與 x0 同形狀的位移，方向為「降低該距離」，即把整批樣本推去覆蓋 exemplar。
    取樣迴圈可把它加到 x0（或等價地折算進更新式）上。
    """
    x0 = x0.detach().requires_grad_(True)
    feats = feature_fn(x0)
    if term == "coverage":
        obj = chamfer_coverage_term(feats, exemplar_feats)
    elif term == "chamfer":
        obj = chamfer_distance(feats, exemplar_feats)
    else:
        raise ValueError(f"未知的 term：{term}")
    grad = torch.autograd.grad(obj, x0)[0]
    # 往下降方向：x0 沿 -d(obj)/dx0 移動可降低距離，故位移為 -weight * grad。
    return -weight * grad


def _self_check():
    """以合成資料與線性特徵驗證核心量與梯度方向正確（不需模型/資料檔）。"""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 基本性質：相同集合 Chamfer 距離為 0，互斥集合為正。
    a = torch.randn(64, 8, device=device)
    b = a.clone()
    far = torch.randn(64, 8, device=device) + 10.0
    cd_same = chamfer_distance(a, b).item()
    cd_far = chamfer_distance(a, far).item()
    print("chamfer self-check:")
    print(f"  CD(a, a)   = {cd_same:.6f}")
    print(f"  CD(a, far) = {cd_far:.4f}")
    assert cd_same < 1e-2, "相同集合的 Chamfer 距離應近 0（cdist 有微小浮點誤差）"
    assert cd_far > cd_same, "互斥集合的 Chamfer 距離應較大"

    # 梯度方向：對 x0 施加一步 guidance 位移後，覆蓋項應下降。
    # 用恆等特徵（feature_fn = 攤平）讓 x0 直接活在特徵空間，便於檢查。
    def feature_fn(x):
        return x.flatten(1)

    x0 = torch.randn(32, 1, 4, 4, device=device)               # 32 張「影像」
    exemplars = torch.randn(16, 16, device=device)             # 16 個 exemplar 特徵（4x4=16）
    before = chamfer_coverage_term(feature_fn(x0), exemplars).item()
    disp = chamfer_guidance_grad(x0, feature_fn, exemplars, weight=1.0, term="coverage")
    x0_moved = x0 + 0.5 * disp                                 # 沿位移方向走一步
    after = chamfer_coverage_term(feature_fn(x0_moved), exemplars).item()
    print(f"  coverage term  before={before:.4f}  after={after:.4f}")
    assert after < before, "沿 guidance 位移一步後覆蓋項應下降"
    print("  OK（僅驗證核心量與梯度方向；取樣整合與真實特徵抽取器待下一增量）")


if __name__ == "__main__":
    _self_check()
