# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha
# MemoryWriter: gating over {state, episode_obj, conflict, archive, working, skip}.
# Produces per-candidate key triplet + value, routes through MemoryUpdater.
# Patent EP25216372.0.

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcortex.config import DCortexConfig
from dcortex.memory.banks import MemoryBank
from dcortex.memory.updater import MemoryUpdater


class MemoryWriter(nn.Module):
    """Gated writer.

    Given a pooled hidden state h_pool [B, D], produces:

        gate  : [B, 6]       softmax over {state, episode_obj, conflict,
                                           archive, working, skip}
        value : [B, D]       transformed value to store
        k_ent : [B, d_ent]   entity key
        k_rel : [B, d_rel]   relation key
        k_typ : [B, d_typ]   type key

    Routing policy (Step 1 MVP):
        For each batch element, choose argmax(gate). If the choice is
        "conflict", write the DIFFERENCE vs the matched State slot;
        if "skip", do nothing; else delegate to the updater on the chosen bank.

    Conflict auto-promotion:
        If the chosen bank is "state" but the updater detects a key-match
        with divergent value (via updater.detect_conflict), we additionally
        write the difference vector to ConflictMemory.

    All bank mutations are under torch.no_grad() inside the updater.
    Trainable parameters here are: the gate, value_head, and three key
    heads. Gradient into these heads comes from downstream aux losses
    (to be added in Step 2 training).
    """

    BANK_ORDER = ("state", "episode_obj", "conflict", "archive", "working", "skip")

    def __init__(self, config: DCortexConfig) -> None:
        super().__init__()
        self.config = config

        self.gate = nn.Linear(config.hidden_dim, 6)

        # Value projection (keeps dimensionality, shapes the stored rep)
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # Key heads (separate from QueryEngine's read heads so writes
        # can learn an independent address code).
        self.key_ent = nn.Linear(config.hidden_dim, config.d_ent)
        self.key_rel = nn.Linear(config.hidden_dim, config.d_rel)
        self.key_typ = nn.Linear(config.hidden_dim, config.d_typ)

        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        h_pool: torch.Tensor,
        updater: MemoryUpdater,
        banks: Dict[str, MemoryBank],
        step: int,
    ) -> torch.Tensor:
        """Route writes through the updater.

        Args:
            h_pool:   [B, hidden_dim]
            updater:  MemoryUpdater instance
            banks:    dict mapping {'state', 'episode_obj', 'conflict',
                                    'archive', 'working'} to MemoryBank
            step:     global step counter (for LRU tracking)

        Returns:
            gate_probs [B, 6] (for optional auxiliary losses)
        """
        if h_pool.dim() != 2:
            raise ValueError(
                f"MemoryWriter expects [B, hidden_dim], got {tuple(h_pool.shape)}"
            )

        h_norm = self.norm(h_pool)
        gate_logits = self.gate(h_norm)            # [B, 6]
        gate_probs = F.softmax(gate_logits, dim=-1)

        value = self.value_head(h_norm)            # [B, D]
        k_ent = self.key_ent(h_norm)
        k_rel = self.key_rel(h_norm)
        k_typ = self.key_typ(h_norm)

        # Hard routing per batch element (MVP)
        choices = gate_probs.argmax(dim=-1)        # [B]
        B = h_pool.shape[0]

        for b in range(B):
            choice_idx = int(choices[b].item())
            bank_name = self.BANK_ORDER[choice_idx]
            if bank_name == "skip":
                continue

            v  = value[b].detach()
            ke = k_ent[b].detach()
            kr = k_rel[b].detach()
            kt = k_typ[b].detach()

            if bank_name == "conflict":
                updater.update(banks["conflict"], v, ke, kr, kt, step, is_conflict=True)
                continue

            # Detect conflict on StateMemory even when gate chose "state"
            if bank_name == "state":
                is_conflict = updater.detect_conflict(
                    banks["state"], v, ke, kr, kt
                )
                if is_conflict:
                    # Dual write: state updated AND conflict recorded
                    updater.update(banks["state"], v, ke, kr, kt, step, is_conflict=False)
                    updater.update(banks["conflict"], v, ke, kr, kt, step, is_conflict=True)
                    continue

            updater.update(banks[bank_name], v, ke, kr, kt, step, is_conflict=False)

        return gate_probs
