"""VAR-mini：MNIST 上的 Visual AutoRegressive 建模。

兩階段 pipeline：
  1. 多尺度殘差 VQ-VAE (var.vqvae)
  2. 具 block-causal attention 的 scale-wise transformer (var.transformer)
"""

from var.vqvae import VARVQVAE, MultiScaleRVQ, VectorQuantizerEMA, Encoder, Decoder
from var.transformer import VARTransformer
from var.sample import generate, sample_top_k_top_p

__all__ = [
    "VARVQVAE", "MultiScaleRVQ", "VectorQuantizerEMA", "Encoder", "Decoder",
    "VARTransformer", "generate", "sample_top_k_top_p",
]
