# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha
# SharedAddressEncoder: small shared module that produces address codes
# from raw embeddings. Used by BOTH writer (for keys) and reader (for queries).
# This guarantees address space compatibility STRUCTURALLY at initialization.
# Patent EP25216372.0.

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcortex.config import DCortexConfig


class SharedAddressEncoder(nn.Module):
    """C_sigma: shared address encoder over shared embeddings.

    Both writer and reader apply this SAME function to embeddings to extract
    an address code. The resulting codes for "the cat is red" (fact) and
    "what color is the cat" (question) share the entity token "cat" and
    therefore produce highly similar address codes BEFORE training.

    Architecture: 1 self-attention layer + learned-query attention pool.
    Small (~2M params for hidden_dim=768).
    """

    def __init__(self, config: DCortexConfig) -> None:
        super().__init__()
        D = config.hidden_dim
        H = max(4, config.n_enc_heads // 2)

        self.norm_in = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(
            D, num_heads=H, batch_first=True, dropout=config.dropout
        )
        self.norm_attn = nn.LayerNorm(D)

        # Learned query for attention pooling
        self.pool_q = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            D, num_heads=H, batch_first=True, dropout=0.0
        )
        self.norm_out = nn.LayerNorm(D)

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [B, T, D] raw token + pos embeddings (pre-normalized).
            attention_mask: [B, T] 1=valid, 0=pad. Optional.

        Returns:
            address: [B, D] pooled address code.
        """
        x = self.norm_in(embeddings)

        kpm = None
        if attention_mask is not None:
            kpm = (attention_mask == 0)

        # Self-attention to gather context (handle multi-token entities)
        h, _ = self.attn(x, x, x, key_padding_mask=kpm, need_weights=False)
        x = x + h
        x = self.norm_attn(x)

        # Attention pool with learned query
        B = x.shape[0]
        q = self.pool_q.expand(B, -1, -1)
        pooled, _ = self.pool_attn(q, x, x, key_padding_mask=kpm, need_weights=False)
        return self.norm_out(pooled.squeeze(1))
