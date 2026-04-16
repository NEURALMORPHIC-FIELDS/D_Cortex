# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha
# MemoryUpdater: nearest-neighbor semantic slot assignment.
# Patent EP25216372.0.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcortex.config import DCortexConfig
from dcortex.memory.banks import MemoryBank


class MemoryUpdater(nn.Module):
    """Nearest-neighbor semantic updater.

    Given a write candidate (value + key triplet), locate the most
    compatible existing slot by weighted cosine similarity on keys.

    Allocation policy:
        1. If best match s* >= theta_match AND bank has that slot:
               update the matched slot (EMA on value, fresh keys).
               If is_conflict=True, store (candidate - existing) as
               the new value rather than blending.
        2. Else if a free slot exists:
               allocate that free slot with the candidate.
        3. Else (bank full, no match):
               evict least-recently-used slot and write the candidate.

    All bank mutations are performed under torch.no_grad() on buffer
    tensors; the updater carries no learnable parameters itself.
    """

    def __init__(self, config: DCortexConfig) -> None:
        super().__init__()
        self.theta_match = config.theta_match
        self.theta_conflict = config.theta_conflict
        self.w_ent, self.w_rel, self.w_typ = config.query_weights
        self.ema_alpha = config.ema_alpha

    @torch.no_grad()
    def update(
        self,
        bank: MemoryBank,
        value: torch.Tensor,
        k_ent: torch.Tensor,
        k_rel: torch.Tensor,
        k_typ: torch.Tensor,
        step: int,
        is_conflict: bool = False,
    ) -> int:
        """Insert or update bank given a write candidate.

        All inputs are 1D (no batch dim). Caller is responsible for
        detaching gradients before calling.

        Returns:
            Index of the slot written to.
        """
        # Empty bank: direct write into slot 0
        if bank.n_occupied() == 0:
            self._write(bank, 0, value, k_ent, k_rel, k_typ, step)
            return 0

        # Compute per-slot weighted cosine similarity against candidate keys
        sim = self._compute_sim(bank, k_ent, k_rel, k_typ)
        sim = sim.masked_fill(~bank.occupied, float("-inf"))

        best_idx = int(sim.argmax().item())
        best_sim = float(sim[best_idx].item())

        # Rule 2: free slot exists AND no strong match -> allocate
        free = bank.free_slot()
        if free >= 0 and best_sim < self.theta_match:
            self._write(bank, free, value, k_ent, k_rel, k_typ, step)
            return free

        # Rule 1: strong match -> update in place (or write diff for conflict)
        if best_sim >= self.theta_match:
            if is_conflict:
                diff = value - bank.values[best_idx]
                self._write(bank, best_idx, diff, k_ent, k_rel, k_typ, step)
            else:
                blended = (1.0 - self.ema_alpha) * bank.values[best_idx] \
                    + self.ema_alpha * value
                self._write(bank, best_idx, blended, k_ent, k_rel, k_typ, step)
            return best_idx

        # Rule 3: bank full, no match -> evict LRU
        lru = bank.lru_slot()
        self._write(bank, lru, value, k_ent, k_rel, k_typ, step)
        return lru

    @torch.no_grad()
    def detect_conflict(
        self,
        bank: MemoryBank,
        value: torch.Tensor,
        k_ent: torch.Tensor,
        k_rel: torch.Tensor,
        k_typ: torch.Tensor,
    ) -> bool:
        """Return True if the candidate collides with an existing slot
        (high key similarity) but the value diverges significantly
        (low cosine on values). Used by the writer's gating logic to
        decide whether to route to ConflictMemory.
        """
        if bank.n_occupied() == 0:
            return False

        sim = self._compute_sim(bank, k_ent, k_rel, k_typ)
        sim = sim.masked_fill(~bank.occupied, float("-inf"))
        best_idx = int(sim.argmax().item())
        best_key_sim = float(sim[best_idx].item())

        if best_key_sim < self.theta_match:
            return False

        v_existing = bank.values[best_idx]
        value_sim = float(F.cosine_similarity(
            value.unsqueeze(0), v_existing.unsqueeze(0)
        ).item())
        # Conflict: same key signature, divergent value
        return value_sim < self.theta_conflict

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_sim(
        self,
        bank: MemoryBank,
        k_ent: torch.Tensor,
        k_rel: torch.Tensor,
        k_typ: torch.Tensor,
    ) -> torch.Tensor:
        """Return [capacity] similarity vector."""
        k_ent_n = F.normalize(bank.k_ent, dim=-1)
        k_rel_n = F.normalize(bank.k_rel, dim=-1)
        k_typ_n = F.normalize(bank.k_typ, dim=-1)

        q_ent_n = F.normalize(k_ent, dim=-1)
        q_rel_n = F.normalize(k_rel, dim=-1)
        q_typ_n = F.normalize(k_typ, dim=-1)

        sim_ent = k_ent_n @ q_ent_n                      # [C]
        sim_rel = k_rel_n @ q_rel_n
        sim_typ = k_typ_n @ q_typ_n

        return self.w_ent * sim_ent + self.w_rel * sim_rel + self.w_typ * sim_typ

    @torch.no_grad()
    def _write(
        self,
        bank: MemoryBank,
        idx: int,
        value: torch.Tensor,
        k_ent: torch.Tensor,
        k_rel: torch.Tensor,
        k_typ: torch.Tensor,
        step: int,
    ) -> None:
        bank.values[idx].copy_(value)
        bank.k_ent[idx].copy_(k_ent)
        bank.k_rel[idx].copy_(k_rel)
        bank.k_typ[idx].copy_(k_typ)
        bank.occupied[idx] = True
        bank.usage[idx] += 1.0
        bank.last_write_step[idx] = step
