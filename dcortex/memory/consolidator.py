# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha
# MemoryConsolidator: decay, prune (with optional migration), pairwise merge.
# Minimal operational policy for Step 1. Patent EP25216372.0.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcortex.config import DCortexConfig
from dcortex.memory.banks import MemoryBank


class MemoryConsolidator(nn.Module):
    """Minimal operational consolidator.

    Three operations per pass:
        1. Usage decay:     usage <- decay_rate * usage
        2. Prune low-usage: below prune_threshold -> free (optionally
                            migrated to a target bank first).
        3. Merge similar:   greedy pairwise merge on weighted cosine
                            similarity above merge_threshold.

    Intended schedule (Step 2 will wire this into training):
        - Step 1:  call consolidate() after every N writer steps.
        - Archive: consolidate(state_mem, target=archive_mem).

    No learnable parameters; all mutations under torch.no_grad().
    """

    def __init__(self, config: DCortexConfig) -> None:
        super().__init__()
        self.decay_rate = config.consolidate_decay_rate
        self.prune_threshold = config.consolidate_prune_threshold
        self.merge_threshold = config.consolidate_merge_threshold
        self.w_ent, self.w_rel, self.w_typ = config.query_weights

    @torch.no_grad()
    def consolidate(
        self,
        bank: MemoryBank,
        target: Optional[MemoryBank] = None,
        current_step: int = 0,
    ) -> dict:
        """Run one consolidation pass.

        Args:
            bank:         the bank to consolidate (source)
            target:       optional destination bank for pruned slots
            current_step: global step counter (used as write step in target)

        Returns:
            Diagnostic dict: {pruned, migrated, merged}.
        """
        self._decay_usage(bank)
        pruned, migrated = self._prune(bank, target, current_step)
        merged = self._merge_similar(bank)
        return {"pruned": pruned, "migrated": migrated, "merged": merged}

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _decay_usage(self, bank: MemoryBank) -> None:
        bank.usage.mul_(self.decay_rate)

    @torch.no_grad()
    def _prune(
        self,
        bank: MemoryBank,
        target: Optional[MemoryBank],
        current_step: int,
    ) -> (int, int):
        prune_mask = bank.occupied & (bank.usage < self.prune_threshold)
        n_prune = int(prune_mask.sum().item())
        if n_prune == 0:
            return 0, 0

        migrated = 0
        if target is not None:
            for idx in prune_mask.nonzero(as_tuple=False).flatten().tolist():
                idx = int(idx)
                if self._migrate_one(bank, idx, target, current_step):
                    migrated += 1

        # Clear pruned slots in source
        bank.occupied[prune_mask] = False
        bank.values[prune_mask] = 0
        bank.k_ent[prune_mask] = 0
        bank.k_rel[prune_mask] = 0
        bank.k_typ[prune_mask] = 0
        bank.usage[prune_mask] = 0
        bank.last_write_step[prune_mask] = -1

        return n_prune, migrated

    @torch.no_grad()
    def _migrate_one(
        self,
        src: MemoryBank,
        idx: int,
        dst: MemoryBank,
        current_step: int,
    ) -> bool:
        free = dst.free_slot()
        if free < 0:
            free = dst.lru_slot()
        dst.values[free].copy_(src.values[idx])
        dst.k_ent[free].copy_(src.k_ent[idx])
        dst.k_rel[free].copy_(src.k_rel[idx])
        dst.k_typ[free].copy_(src.k_typ[idx])
        dst.occupied[free] = True
        dst.usage[free] = src.usage[idx]
        dst.last_write_step[free] = current_step
        return True

    @torch.no_grad()
    def _merge_similar(self, bank: MemoryBank) -> int:
        occ = bank.occupied.nonzero(as_tuple=False).flatten()
        if occ.numel() < 2:
            return 0

        k_ent_n = F.normalize(bank.k_ent[occ], dim=-1)
        k_rel_n = F.normalize(bank.k_rel[occ], dim=-1)
        k_typ_n = F.normalize(bank.k_typ[occ], dim=-1)

        sim_ent = k_ent_n @ k_ent_n.t()
        sim_rel = k_rel_n @ k_rel_n.t()
        sim_typ = k_typ_n @ k_typ_n.t()

        sim = self.w_ent * sim_ent + self.w_rel * sim_rel + self.w_typ * sim_typ
        sim.fill_diagonal_(float("-inf"))

        merged_count = 0
        merged_local = set()

        # Greedy: in each iteration find the top pair above threshold
        max_iters = occ.numel()
        for _ in range(max_iters):
            max_val, flat_idx = sim.view(-1).max(dim=0)
            if float(max_val.item()) < self.merge_threshold:
                break
            i = int(flat_idx.item()) // sim.shape[1]
            j = int(flat_idx.item()) % sim.shape[1]

            if i in merged_local or j in merged_local:
                self._invalidate_row_col(sim, i)
                self._invalidate_row_col(sim, j)
                continue

            idx_i = int(occ[i].item())
            idx_j = int(occ[j].item())
            self._merge_into(bank, idx_i, idx_j)
            merged_count += 1
            merged_local.add(i)
            merged_local.add(j)
            self._invalidate_row_col(sim, i)
            self._invalidate_row_col(sim, j)

        return merged_count

    @staticmethod
    def _invalidate_row_col(sim: torch.Tensor, idx: int) -> None:
        sim[idx, :] = float("-inf")
        sim[:, idx] = float("-inf")

    @torch.no_grad()
    def _merge_into(self, bank: MemoryBank, idx_keep: int, idx_drop: int) -> None:
        bank.values[idx_keep] = 0.5 * (bank.values[idx_keep] + bank.values[idx_drop])
        bank.k_ent[idx_keep] = 0.5 * (bank.k_ent[idx_keep] + bank.k_ent[idx_drop])
        bank.k_rel[idx_keep] = 0.5 * (bank.k_rel[idx_keep] + bank.k_rel[idx_drop])
        bank.k_typ[idx_keep] = 0.5 * (bank.k_typ[idx_keep] + bank.k_typ[idx_drop])
        bank.usage[idx_keep] = bank.usage[idx_keep] + bank.usage[idx_drop]

        bank.occupied[idx_drop] = False
        bank.values[idx_drop] = 0
        bank.k_ent[idx_drop] = 0
        bank.k_rel[idx_drop] = 0
        bank.k_typ[idx_drop] = 0
        bank.usage[idx_drop] = 0
        bank.last_write_step[idx_drop] = -1
