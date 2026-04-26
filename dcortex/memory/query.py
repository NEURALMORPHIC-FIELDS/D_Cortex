# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha
# QueryEngine: projects hidden state into three latent key spaces.
# Patent EP25216372.0.

from typing import Tuple

import torch
import torch.nn as nn

from dcortex.config import DCortexConfig


class QueryEngine(nn.Module):
    """Produce (q_ent, q_rel, q_typ) from a pooled hidden state.

    The three projections address three distinct semantic axes used by
    NN-semantic readers and the updater:

        q_ent : entity / subject / referent
        q_rel : relation / predicate
        q_typ : type / role / category

    Similarity at read and update time is a weighted combination of
    cosine similarities in each of these spaces.
    """

    def __init__(self, config: DCortexConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.d_ent = config.d_ent
        self.d_rel = config.d_rel
        self.d_typ = config.d_typ

        self.proj_ent = nn.Linear(config.hidden_dim, config.d_ent)
        self.proj_rel = nn.Linear(config.hidden_dim, config.d_rel)
        self.proj_typ = nn.Linear(config.hidden_dim, config.d_typ)

        # LayerNorm on input stabilizes similarity magnitudes
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        h_pool: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project pooled hidden state into key triplet.

        Args:
            h_pool: pooled hidden state, shape [B, hidden_dim].

        Returns:
            (q_ent [B, d_ent], q_rel [B, d_rel], q_typ [B, d_typ])
        """
        if h_pool.dim() != 2:
            raise ValueError(
                f"QueryEngine expects [B, hidden_dim], got shape {tuple(h_pool.shape)}"
            )
        h = self.norm(h_pool)
        return self.proj_ent(h), self.proj_rel(h), self.proj_typ(h)
