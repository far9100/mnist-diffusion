"""
Denoising Diffusion Probabilistic Model (DDPM) — core components.

Implements the UNet noise-prediction network and diffusion schedule
for generating MNIST handwritten digits.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(tensor, t, shape):
    """Index into `tensor` using timestep indices `t`, reshape for broadcasting."""
    batch_size = t.shape[0]
    out = tensor.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(shape) - 1)))


class SinusoidalPositionEmbedding(nn.Module):
    """Maps integer timestep to dense vector via sinusoidal encoding."""

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
    """Two convolutions with GroupNorm + SiLU, time embedding injection, residual connection."""

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

        # Inject time embedding
        t = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """Spatial self-attention with multi-head QKV projections."""

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
    """Spatial downsampling via strided convolution."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling via nearest interpolation + convolution."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """
    Full noise prediction network.

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

        # Time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Class embedding (index num_classes = unconditional / null class)
        self.class_emb = nn.Embedding(num_classes + 1, time_emb_dim)

        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # Encoder
        self.enc_block1a = ResidualBlock(channels[0], channels[0], time_emb_dim)
        self.enc_block1b = ResidualBlock(channels[0], channels[0], time_emb_dim)
        self.down1 = Downsample(channels[0])

        self.enc_block2a = ResidualBlock(channels[0], channels[1], time_emb_dim)
        self.enc_block2b = ResidualBlock(channels[1], channels[1], time_emb_dim)
        self.enc_attn2 = AttentionBlock(channels[1])
        self.down2 = Downsample(channels[1])

        self.enc_block3a = ResidualBlock(channels[1], channels[2], time_emb_dim)
        self.enc_block3b = ResidualBlock(channels[2], channels[2], time_emb_dim)

        # Bottleneck
        self.mid_block1 = ResidualBlock(channels[2], channels[2], time_emb_dim)
        self.mid_attn = AttentionBlock(channels[2])
        self.mid_block2 = ResidualBlock(channels[2], channels[2], time_emb_dim)

        # Decoder
        self.dec_block3a = ResidualBlock(channels[2] + channels[2], channels[2], time_emb_dim)
        self.dec_block3b = ResidualBlock(channels[2] + channels[2], channels[2], time_emb_dim)
        self.up2 = Upsample(channels[2])

        self.dec_block2a = ResidualBlock(channels[2] + channels[1], channels[1], time_emb_dim)
        self.dec_block2b = ResidualBlock(channels[1] + channels[1], channels[1], time_emb_dim)
        self.dec_attn2 = AttentionBlock(channels[1])
        self.up1 = Upsample(channels[1])

        self.dec_block1a = ResidualBlock(channels[1] + channels[0], channels[0], time_emb_dim)
        self.dec_block1b = ResidualBlock(channels[0] + channels[0], channels[0], time_emb_dim)

        # Output
        self.out_norm = nn.GroupNorm(8, channels[0])
        self.out_conv = nn.Conv2d(channels[0], in_channels, 3, padding=1)

    def forward(self, x, t, class_labels=None):
        t_emb = self.time_emb(t)
        if class_labels is None:
            class_labels = torch.full((x.shape[0],), self.num_classes,
                                      device=x.device, dtype=torch.long)
        t_emb = t_emb + self.class_emb(class_labels)

        # Initial conv
        x0 = self.init_conv(x)  # (B, 64, 28, 28)

        # Encoder level 1 — 28x28
        h1a = self.enc_block1a(x0, t_emb)
        h1b = self.enc_block1b(h1a, t_emb)
        d1 = self.down1(h1b)  # (B, 64, 14, 14)

        # Encoder level 2 — 14x14 with attention
        h2a = self.enc_block2a(d1, t_emb)
        h2b = self.enc_block2b(h2a, t_emb)
        h2b = self.enc_attn2(h2b)
        d2 = self.down2(h2b)  # (B, 128, 7, 7)

        # Encoder level 3 — 7x7
        h3a = self.enc_block3a(d2, t_emb)
        h3b = self.enc_block3b(h3a, t_emb)

        # Bottleneck
        mid = self.mid_block1(h3b, t_emb)
        mid = self.mid_attn(mid)
        mid = self.mid_block2(mid, t_emb)

        # Decoder level 3 — 7x7
        x = self.dec_block3a(torch.cat([mid, h3b], dim=1), t_emb)
        x = self.dec_block3b(torch.cat([x, h3a], dim=1), t_emb)
        x = self.up2(x)  # (B, 256, 14, 14)

        # Decoder level 2 — 14x14 with attention
        x = self.dec_block2a(torch.cat([x, h2b], dim=1), t_emb)
        x = self.dec_block2b(torch.cat([x, h2a], dim=1), t_emb)
        x = self.dec_attn2(x)
        x = self.up1(x)  # (B, 128, 28, 28)

        # Decoder level 1 — 28x28
        x = self.dec_block1a(torch.cat([x, h1b], dim=1), t_emb)
        x = self.dec_block1b(torch.cat([x, h1a], dim=1), t_emb)

        # Output
        x = self.out_norm(x)
        x = F.silu(x)
        x = self.out_conv(x)
        return x


class DiffusionSchedule:
    """
    Precomputes noise schedule tensors and implements forward/reverse diffusion.

    Linear beta schedule from Ho et al. (2020).
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.sqrt_alphas = torch.sqrt(alphas)
        self.sqrt_alphas_bar = torch.sqrt(alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)
        # Coefficients for posterior mean
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_bar_prev) / (1.0 - alphas_bar)
        self.posterior_mean_coef2 = (1.0 - alphas_bar_prev) * torch.sqrt(alphas) / (1.0 - alphas_bar)
        self.posterior_variance = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)

    def to(self, device):
        self.device = device
        for attr in [
            "betas", "sqrt_alphas", "sqrt_alphas_bar",
            "sqrt_one_minus_alphas_bar", "posterior_mean_coef1",
            "posterior_mean_coef2", "posterior_variance",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def q_sample(self, x_0, t, noise=None):
        """Forward process: add noise to clean image x_0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)
        return (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0
            + extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )

    @torch.no_grad()
    def p_sample(self, model, x_t, t, class_labels=None, guidance_scale=1.0):
        """Single reverse step with optional classifier-free guidance."""
        if class_labels is not None and guidance_scale > 1.0:
            # Classifier-free guidance: run conditional + unconditional together
            null_labels = torch.full_like(class_labels, model.num_classes)
            x_double = torch.cat([x_t, x_t], dim=0)
            t_double = torch.cat([t, t], dim=0)
            labels_double = torch.cat([class_labels, null_labels], dim=0)
            pred_both = model(x_double, t_double, labels_double)
            pred_cond, pred_uncond = pred_both.chunk(2)
            pred_noise = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        else:
            pred_noise = model(x_t, t, class_labels)

        # Compute mean using the simpler formulation from Ho et al.
        alpha_t = extract(self.sqrt_alphas, t, x_t.shape)
        beta_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus_ab = extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape)

        mean = (x_t - beta_t / sqrt_one_minus_ab * pred_noise) / alpha_t

        # Add noise for t > 0
        if (t == 0).all():
            return mean
        variance = extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, class_labels=None, guidance_scale=1.0,
                      snapshot_steps=None):
        """Full reverse process: generate images from pure noise.

        Args:
            snapshot_steps: If set, capture intermediate states at this many
                evenly-spaced timesteps. Returns (final_images, snapshots) where
                snapshots is a list of tensors, one per captured step.
        """
        device = self.device
        x = torch.randn(shape, device=device)

        # Determine which timesteps to snapshot
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
            snapshots.append(x.clone())  # final result
            return x, snapshots

        return x
