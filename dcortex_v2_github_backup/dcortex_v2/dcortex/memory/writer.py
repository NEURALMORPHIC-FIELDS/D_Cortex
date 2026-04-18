# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha
# MemoryWriter: gating over {state, episode_obj, conflict, archive, working, skip}.
# Produces per-candidate key triplet + value, routes through MemoryUpdater.
# Patent EP25216372.0.

from typing import Dict, Tuple

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

    def __init__(self, config: DCortexConfig, shared_query_engine: nn.Module) -> None:
        super().__init__()
        self.config = config
        D = config.hidden_dim

        self.gate = nn.Linear(D, 6)

        # Contextual value path (kept for LM episodes - no answer_emb supplied)
        self.value_head = nn.Sequential(
            nn.Linear(D, D), nn.GELU(), nn.Linear(D, D),
        )

        # Lexical binding: linear projection of answer token embedding.
        # When answer_emb is supplied, value = alpha * W_v(answer_emb) + (1-alpha) * contextual.
        # Forces stored value to be lexically decodable.
        self.lexical_W_v = nn.Linear(D, D, bias=False)

        # SHARED query engine produces keys for write AND queries for read.
        self.query_engine = shared_query_engine

        self.norm = nn.LayerNorm(D)

    def forward(
        self,
        h_pool: torch.Tensor,
        addr_code: torch.Tensor,
        updater: MemoryUpdater,
        banks: Dict[str, MemoryBank],
        step: int,
        force_write: bool = False,
        answer_emb: torch.Tensor = None,
        lexical_alpha: float = 0.9,
        force_bank: str = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            ... (as before)
            force_bank: if provided (e.g. "working"), override gate and always
                use this bank. For validation/debugging and structural curriculum
                where we want all facts in one retrieval pool.
        """
        if h_pool.dim() != 2 or addr_code.dim() != 2:
            raise ValueError(
                f"Expected [B,D], got {tuple(h_pool.shape)} / {tuple(addr_code.shape)}"
            )

        h_norm = self.norm(h_pool)
        gate_logits = self.gate(h_norm)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Contextual value (always computed)
        value_ctx = self.value_head(h_norm)

        # Lexical binding when answer_emb supplied
        if answer_emb is not None:
            if answer_emb.dim() == 1:
                answer_emb = answer_emb.unsqueeze(0)
            value_lex = self.lexical_W_v(answer_emb)
            value = lexical_alpha * value_lex + (1.0 - lexical_alpha) * value_ctx
        else:
            value = value_ctx

        # Keys from SHARED query engine applied to ADDRESS code
        k_ent, k_rel, k_typ = self.query_engine(addr_code)

        # Hard routing per batch (force_write excludes skip)
        bank_probs = gate_probs[:, :5]
        choices = bank_probs.argmax(dim=-1)
        B = h_pool.shape[0]

        # Override with force_bank if specified
        if force_bank is not None:
            force_idx = self.BANK_ORDER.index(force_bank)
            choices = torch.full_like(choices, force_idx)

        slot_writes = []
        for b in range(B):
            choice_idx = int(choices[b].item())
            bank_name = self.BANK_ORDER[choice_idx]

            v  = value[b]
            ke = k_ent[b]
            kr = k_rel[b]
            kt = k_typ[b]

            v_d, ke_d, kr_d, kt_d = v.detach(), ke.detach(), kr.detach(), kt.detach()

            if bank_name == "conflict":
                slot = updater.update(banks["conflict"], v_d, ke_d, kr_d, kt_d, step, is_conflict=True)
                banks["conflict"].set_overlay(slot, v, ke, kr, kt)
                slot_writes.append(("conflict", slot))
                continue

            if bank_name == "state":
                is_conflict = updater.detect_conflict(
                    banks["state"], v_d, ke_d, kr_d, kt_d
                )
                if is_conflict:
                    slot1 = updater.update(banks["state"], v_d, ke_d, kr_d, kt_d, step, is_conflict=False)
                    banks["state"].set_overlay(slot1, v, ke, kr, kt)
                    slot2 = updater.update(banks["conflict"], v_d, ke_d, kr_d, kt_d, step, is_conflict=True)
                    banks["conflict"].set_overlay(slot2, v, ke, kr, kt)
                    slot_writes.append(("state", slot1))  # primary write recorded
                    continue

            slot = updater.update(banks[bank_name], v_d, ke_d, kr_d, kt_d, step, is_conflict=False)
            banks[bank_name].set_overlay(slot, v, ke, kr, kt)
            slot_writes.append((bank_name, slot))

        return {
            'gate_probs': gate_probs,
            'value': value,
            'k_ent': k_ent,
            'k_rel': k_rel,
            'k_typ': k_typ,
            'slot_writes': slot_writes,
        }
