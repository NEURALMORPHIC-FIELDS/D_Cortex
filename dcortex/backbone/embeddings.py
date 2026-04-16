# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha
# Token + learned absolute positional embeddings with pre-norm and dropout.
# Patent EP25216372.0.

import torch
import torch.nn as nn

from dcortex.config import DCortexConfig


class TokenEmbeddings(nn.Module):
    """Token embedding + learned absolute positional embedding.

    Pre-norm is applied after summation. RoPE is a candidate upgrade
    for a later iteration; Step 1 locks in absolute learned positions
    for simplicity.
    """

    def __init__(self, config: DCortexConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.max_seq_len = config.max_seq_len

        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.hidden_dim)
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

        nn.init.normal_(self.token_emb.weight, std=config.init_std)
        nn.init.normal_(self.pos_emb.weight, std=config.init_std)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Args: input_ids [B, T]. Returns: h [B, T, hidden_dim]."""
        if input_ids.dim() != 2:
            raise ValueError(
                f"TokenEmbeddings expects [B, T], got {tuple(input_ids.shape)}"
            )
        B, T = input_ids.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"
            )
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(input_ids) + self.pos_emb(positions)
        h = self.norm(h)
        h = self.dropout(h)
        return h
