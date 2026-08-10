import torch
import torch.nn as nn
from typing import Optional


class RoPEPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 10000, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Inversen Frequenzen (Theta) berechnen
        inv_freq = 1.0 / \
            (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._precompute_cache(max_seq_len)

    def _precompute_cache(self, max_seq_len):
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, self.inv_freq)  # [MaxLen, Dim/2]
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch, Heads, SeqLen, Dim]
        position_ids: [Batch, SeqLen]
        """
        # Hole Cos/Sin aus dem Cache und füge Head-Dimension ein
        # [Batch, SeqLen, Dim/2] -> [Batch, 1, SeqLen, Dim/2]
        cos = self.cos_cached[position_ids].unsqueeze(1)
        sin = self.sin_cached[position_ids].unsqueeze(1)

        # RoPE Rotation: Teile Dim in gerade/ungerade Paare
        x_pairs = x.view(*x.shape[:-1], -1, 2)
        x_even, x_odd = x_pairs[..., 0], x_pairs[..., 1]

        # Standard RoPE Formel
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # Zusammenfügen: [Batch, Heads, SeqLen, Dim]
        return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)
