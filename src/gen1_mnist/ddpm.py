"""
Denoising Diffusion Probabilistic Model (DDPM) 的核心元件。

實作 UNet 噪音預測網路與擴散排程，用於生成 MNIST 手寫數字。
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import _pathfix  # noqa: E402  路徑墊片，見 src/_pathfix.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(tensor, t, shape):
    """以時間步索引 `t` 從 `tensor` 取值，並 reshape 以利 broadcasting。"""
    batch_size = t.shape[0]
    out = tensor.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(shape) - 1)))


class SinusoidalPositionEmbedding(nn.Module):
    """以正弦編碼把整數時間步映射成 dense 向量。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """兩層卷積，含 GroupNorm + SiLU、時間 embedding 注入與 residual 連接。"""

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # 注入時間 embedding
        t = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """空間自注意力，含 multi-head QKV 投影。"""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W).unbind(1)
        # q, k, v: (B, heads, head_dim, H*W)
        q = q.permute(0, 1, 3, 2)  # (B, heads, H*W, head_dim)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        scale = (C // self.num_heads) ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)  # (B, heads, H*W, head_dim)

        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    """以 strided convolution 做空間下採樣。"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """以最近鄰插值 + 卷積做空間上採樣。"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """
    完整的噪音預測網路。

    Input (1, 28, 28) -> Conv -> 64ch
    Encoder: [64, 28x28] -> down -> [128, 14x14, +attn] -> down -> [256, 7x7]
    Bottleneck: [256, 7x7] ResBlock + Attn + ResBlock
    Decoder: [256+256, 7x7] -> up -> [128+128, 14x14, +attn] -> up -> [64+64, 28x28]
    Output: GroupNorm -> SiLU -> Conv -> (1, 28, 28)
    """

    def __init__(self, in_channels=1, base_channels=64, channel_mults=(1, 2, 4),
                 time_emb_dim=256, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        channels = [base_channels * m for m in channel_mults]  # [64, 128, 256]

        # 時間 embedding
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # 類別 embedding（索引 num_classes = 無條件 / null class）
        self.class_emb = nn.Embedding(num_classes + 1, time_emb_dim)

        # 初始 conv
        self.init_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # 編碼器
        self.enc_block1a = ResidualBlock(channels[0], channels[0], time_emb_dim)
        self.enc_block1b = ResidualBlock(channels[0], channels[0], time_emb_dim)
        self.down1 = Downsample(channels[0])

        self.enc_block2a = ResidualBlock(channels[0], channels[1], time_emb_dim)
        self.enc_block2b = ResidualBlock(channels[1], channels[1], time_emb_dim)
        self.enc_attn2 = AttentionBlock(channels[1])
        self.down2 = Downsample(channels[1])

        self.enc_block3a = ResidualBlock(channels[1], channels[2], time_emb_dim)
        self.enc_block3b = ResidualBlock(channels[2], channels[2], time_emb_dim)

        # 瓶頸層
        self.mid_block1 = ResidualBlock(channels[2], channels[2], time_emb_dim)
        self.mid_attn = AttentionBlock(channels[2])
        self.mid_block2 = ResidualBlock(channels[2], channels[2], time_emb_dim)

        # 解碼器
        self.dec_block3a = ResidualBlock(channels[2] + channels[2], channels[2], time_emb_dim)
        self.dec_block3b = ResidualBlock(channels[2] + channels[2], channels[2], time_emb_dim)
        self.up2 = Upsample(channels[2])

        self.dec_block2a = ResidualBlock(channels[2] + channels[1], channels[1], time_emb_dim)
        self.dec_block2b = ResidualBlock(channels[1] + channels[1], channels[1], time_emb_dim)
        self.dec_attn2 = AttentionBlock(channels[1])
        self.up1 = Upsample(channels[1])

        self.dec_block1a = ResidualBlock(channels[1] + channels[0], channels[0], time_emb_dim)
        self.dec_block1b = ResidualBlock(channels[0] + channels[0], channels[0], time_emb_dim)

        # 輸出層
        self.out_norm = nn.GroupNorm(8, channels[0])
        self.out_conv = nn.Conv2d(channels[0], in_channels, 3, padding=1)

    def forward(self, x, t, class_labels=None):
        t_emb = self.time_emb(t)
        if class_labels is None:
            class_labels = torch.full((x.shape[0],), self.num_classes,
                                      device=x.device, dtype=torch.long)
        t_emb = t_emb + self.class_emb(class_labels)

        # 初始 conv
        x0 = self.init_conv(x)  # (B, 64, 28, 28)

        # Encoder 第 1 層 — 28x28
        h1a = self.enc_block1a(x0, t_emb)
        h1b = self.enc_block1b(h1a, t_emb)
        d1 = self.down1(h1b)  # (B, 64, 14, 14)

        # Encoder 第 2 層 — 14x14，含 attention
        h2a = self.enc_block2a(d1, t_emb)
        h2b = self.enc_block2b(h2a, t_emb)
        h2b = self.enc_attn2(h2b)
        d2 = self.down2(h2b)  # (B, 128, 7, 7)

        # Encoder 第 3 層 — 7x7
        h3a = self.enc_block3a(d2, t_emb)
        h3b = self.enc_block3b(h3a, t_emb)

        # 瓶頸層
        mid = self.mid_block1(h3b, t_emb)
        mid = self.mid_attn(mid)
        mid = self.mid_block2(mid, t_emb)

        # Decoder 第 3 層 — 7x7
        x = self.dec_block3a(torch.cat([mid, h3b], dim=1), t_emb)
        x = self.dec_block3b(torch.cat([x, h3a], dim=1), t_emb)
        x = self.up2(x)  # (B, 256, 14, 14)

        # Decoder 第 2 層 — 14x14，含 attention
        x = self.dec_block2a(torch.cat([x, h2b], dim=1), t_emb)
        x = self.dec_block2b(torch.cat([x, h2a], dim=1), t_emb)
        x = self.dec_attn2(x)
        x = self.up1(x)  # (B, 128, 28, 28)

        # Decoder 第 1 層 — 28x28
        x = self.dec_block1a(torch.cat([x, h1b], dim=1), t_emb)
        x = self.dec_block1b(torch.cat([x, h1a], dim=1), t_emb)

        # 輸出層
        x = self.out_norm(x)
        x = F.silu(x)
        x = self.out_conv(x)
        return x


class DiffusionSchedule:
    """
    預先計算噪音排程張量，並實作前向/反向擴散。

    線性 beta 排程，出自 Ho et al. (2020)。
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas_bar = alphas_bar  # ᾱ_t，DDIM 在任意 sub-step 都需要
        self.sqrt_alphas = torch.sqrt(alphas)
        self.sqrt_alphas_bar = torch.sqrt(alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)
        # posterior 平均的係數
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_bar_prev) / (1.0 - alphas_bar)
        self.posterior_mean_coef2 = (1.0 - alphas_bar_prev) * torch.sqrt(alphas) / (1.0 - alphas_bar)
        self.posterior_variance = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)

    def to(self, device):
        self.device = device
        for attr in [
            "betas", "alphas_bar", "sqrt_alphas", "sqrt_alphas_bar",
            "sqrt_one_minus_alphas_bar", "posterior_mean_coef1",
            "posterior_mean_coef2", "posterior_variance",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def q_sample(self, x_0, t, noise=None):
        """前向過程：在時間步 t 對乾淨影像 x_0 加噪。"""
        if noise is None:
            noise = torch.randn_like(x_0)
        return (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0
            + extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )

    @torch.no_grad()
    def predict_eps(self, model, x_t, t, class_labels=None, guidance_scale=1.0):
        """含 classifier-free guidance 的噪音預測。

        DDPM（`p_sample`）與 DDIM（`ddim_sample_loop`）的反向過程共用此函式，
        讓 CFG 邏輯只存在於一個地方。
        """
        # T5b（行為擴充）：條件由 `> 1.0` 改為 `!= 1.0`，使 w<1 也走 CFG 分支（往無條件內插、降低類別
        # 銳化，供 w<1 網格 scout）；w==1.0 仍落下方純條件分支（逐位不變）、w>1 分支不變。見 CHANGELOG 2026-07-21-08。
        if class_labels is not None and guidance_scale != 1.0:
            # Classifier-free guidance：條件與無條件一起前傳
            null_labels = torch.full_like(class_labels, model.num_classes)
            x_double = torch.cat([x_t, x_t], dim=0)
            t_double = torch.cat([t, t], dim=0)
            labels_double = torch.cat([class_labels, null_labels], dim=0)
            pred_both = model(x_double, t_double, labels_double)
            pred_cond, pred_uncond = pred_both.chunk(2)
            return pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        return model(x_t, t, class_labels)

    @torch.no_grad()
    def p_sample(self, model, x_t, t, class_labels=None, guidance_scale=1.0):
        """單一反向步驟，可選用 classifier-free guidance。"""
        pred_noise = self.predict_eps(model, x_t, t, class_labels, guidance_scale)

        # 用 Ho et al. 較簡潔的式子計算平均
        alpha_t = extract(self.sqrt_alphas, t, x_t.shape)
        beta_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus_ab = extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape)

        mean = (x_t - beta_t / sqrt_one_minus_ab * pred_noise) / alpha_t

        # t > 0 時加入噪音
        if (t == 0).all():
            return mean
        variance = extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, class_labels=None, guidance_scale=1.0,
                      snapshot_steps=None):
        """完整反向過程：從純噪音生成影像。

        Args:
            snapshot_steps: 若設定，會在這麼多個等距時間步擷取中間狀態。
                回傳 (final_images, snapshots)，其中 snapshots 是張量的 list，
                每個擷取步驟一個。
        """
        device = self.device
        x = torch.randn(shape, device=device)

        # 決定要在哪些時間步做 snapshot
        capture = set()
        if snapshot_steps is not None and snapshot_steps > 0:
            indices = torch.linspace(self.timesteps - 1, 0, snapshot_steps).long().tolist()
            capture = set(indices)

        snapshots = []

        for i in reversed(range(self.timesteps)):
            if i in capture:
                snapshots.append(x.clone())
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, class_labels, guidance_scale)

        if snapshot_steps is not None:
            snapshots.append(x.clone())  # 最終結果
            return x, snapshots

        return x

    def ddim_timesteps(self, num_steps):
        """建立 DDIM 取樣用的遞減時間步子序列。

        回傳 (t_cur, t_prev) 整數配對的 list。最後一組的 `t_prev == -1` 表示
        前一個狀態就是乾淨影像，也就是 ᾱ_prev = 1.0。
        """
        num_steps = min(num_steps, self.timesteps)
        steps = torch.linspace(0, self.timesteps - 1, num_steps).round().long()
        steps = torch.unique(steps)                # 防止 num_steps 很小時出現重複
        seq = list(reversed(steps.tolist()))       # 例如 [999, ..., 0]
        seq_prev = seq[1:] + [-1]                   # t_prev；-1 => ᾱ_prev = 1.0
        return list(zip(seq, seq_prev))

    @torch.no_grad()
    def ddim_sample_loop(self, model, shape, num_steps=50, eta=0.0,
                         class_labels=None, guidance_scale=1.0, generator=None):
        """DDIM 反向過程（Song, Meng & Ermon, 2021）。

        原封不動重用訓練好的 ε-network——DDIM 是一個 sampler，不是另外
        訓練的模型。透過 `predict_eps` 使用與 DDPM 相同的 classifier-free
        guidance 路徑。

        Args:
            num_steps: DDIM 子序列的長度（越少越快）。
            eta: 隨機性。eta=0 -> deterministic（probability-flow ODE）；
                eta=1 且用完整 1000 步序列時會還原成 DDPM ancestral
                sampling（sigma 等於 DDPM posterior std）。
            generator: 可選的 torch.Generator，用於可重現的取樣。
        """
        device = self.device
        x = torch.randn(shape, device=device, generator=generator)

        for t_cur, t_prev in self.ddim_timesteps(num_steps):
            t = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)
            eps = self.predict_eps(model, x, t, class_labels, guidance_scale)

            ab_t = self.alphas_bar[t_cur]
            ab_s = (self.alphas_bar[t_prev] if t_prev >= 0
                    else torch.ones((), device=device))

            # 由 x_t 與噪音估計得到的乾淨影像 x_0。
            x0 = (x - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt()
            # 隨機性排程；eta=0 => sigma=0（deterministic）。
            sigma = (eta
                     * ((1.0 - ab_s) / (1.0 - ab_t)).sqrt()
                     * (1.0 - ab_t / ab_s).sqrt())
            # 指回 x_t 的方向。
            dir_xt = (1.0 - ab_s - sigma ** 2).clamp(min=0.0).sqrt() * eps
            x = ab_s.sqrt() * x0 + dir_xt
            if t_prev >= 0 and eta > 0:
                x = x + sigma * torch.randn(shape, device=device, generator=generator)

        return x
