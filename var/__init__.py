"""VAR-mini: Visual AutoRegressive modeling for MNIST.

Two-stage pipeline:
  1. Multi-scale residual VQ-VAE (var.vqvae)
  2. Scale-wise transformer with block-causal attention (var.transformer)
"""

from var.vqvae import VARVQVAE, MultiScaleRVQ, VectorQuantizerEMA, Encoder, Decoder
from var.transformer import VARTransformer
from var.sample import generate, sample_top_k_top_p

__all__ = [
    "VARVQVAE", "MultiScaleRVQ", "VectorQuantizerEMA", "Encoder", "Decoder",
    "VARTransformer", "generate", "sample_top_k_top_p",
]
