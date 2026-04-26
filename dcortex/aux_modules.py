# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha
# Auxiliary heads: AuxAnswerHead + ValueToKeyProjector
# Bridge retrieval -> language and retrieval -> key cycle.
# Patent EP25216372.0.

import torch
import torch.nn as nn

from dcortex.config import DCortexConfig


class AuxAnswerHead(nn.Module):
    """Direct path from retrieved_value to answer token logits.

    Bypasses fusion blocks entirely. Forces retrieved_value to contain
    linguistically-decodable information about the answer.

    Tied to shared_token_emb to reduce params and align with LM head.
    """

    def __init__(self, config: DCortexConfig, shared_token_emb: nn.Embedding) -> None:
        super().__init__()
        D = config.hidden_dim
        self.norm = nn.LayerNorm(D)
        self.proj = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, D),
        )
        # Output head tied to shared token embeddings
        self.shared_token_emb = shared_token_emb

    def forward(self, retrieved_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            retrieved_value: [B, D] pooled retrieved value from memory.
        Returns:
            logits: [B, vocab_size] predicted distribution over answer token.
        """
        h = self.norm(retrieved_value)
        h = self.proj(h)
        # Tied projection
        return h @ self.shared_token_emb.weight.t()


class ValueToKeyProjector(nn.Module):
    """P: value space -> key space.

    Separate trainable projector used for L_cycle. Prevents forcing value
    to literally become the key (which would impoverish it semantically).

    Used as:
        k_tilde = P(retrieved_value)
        L_cycle = 1 - cos(k_tilde, k_target)
    """

    def __init__(self, config: DCortexConfig) -> None:
        super().__init__()
        D = config.hidden_dim
        d_ent = config.d_ent
        self.norm = nn.LayerNorm(D)
        self.proj = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, d_ent),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """value [B, D] -> [B, d_ent]"""
        return self.proj(self.norm(value))
