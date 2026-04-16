# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha
# Standard pre-norm causal transformer block. Used for the first
# (n_layers - n_fusion_layers) backbone layers. Patent EP25216372.0.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcortex.config import DCortexConfig


# ======================================================================
# MULTI-HEAD CAUSAL SELF-ATTENTION
# ======================================================================

class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention.

    Accepts an optional attention_mask [B, T] (1=valid, 0=pad).
    """

    def __init__(self, config: DCortexConfig) -> None:
        super().__init__()
        if config.hidden_dim % config.n_heads != 0:
            raise ValueError("hidden_dim must divide n_heads evenly")
        self.n_heads = config.n_heads
        self.head_dim = config.hidden_dim // config.n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim)
        self.out = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        h: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = h.shape
        qkv = self.qkv(h)                                         # [B, T, 3D]
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                          # [3, B, H, T, d]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale             # [B, H, T, T]

        # Causal
        causal = torch.triu(
            torch.ones(T, T, device=h.device, dtype=torch.bool),
            diagonal=1,
        )
        attn = attn.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Padding
        if attention_mask is not None:
            pad = (attention_mask == 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            attn = attn.masked_fill(pad, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


# ======================================================================
# GELU FFN
# ======================================================================

class FeedForward(nn.Module):
    """Two-layer GELU feedforward."""

    def __init__(self, config: DCortexConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_dim, config.ff_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.ff_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(h))))


# ======================================================================
# STANDARD BLOCK (pre-norm)
# ======================================================================

class StandardTransformerBlock(nn.Module):
    """Pre-norm transformer block without memory fusion.

    y = x + Attn(LN(x))
    y = y + FFN(LN(y))
    """

    def __init__(self, config: DCortexConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.attn = MultiHeadSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.ff = FeedForward(config)

    def forward(
        self,
        h: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = h + self.attn(self.norm1(h), attention_mask)
        h = h + self.ff(self.norm2(h))
        return h
