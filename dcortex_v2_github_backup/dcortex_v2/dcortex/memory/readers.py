# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha
# Readers: SemanticReader (single-bank NN-attention), EpisodeReader
# (obj + SSM sub-fusion via W_theta), MemoryReadFusion (5-stream stack).
# Patent EP25216372.0.

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcortex.config import DCortexConfig
from dcortex.memory.banks import EpisodeObjectMemory, EpisodeSSM, MemoryBank


# ======================================================================
# SEMANTIC READER — single bank
# ======================================================================

class SemanticReader(nn.Module):
    """Read from a MemoryBank via NN-semantic attention.

    Similarity:
        s_i = w_ent cos(q_ent, k_i_ent)
            + w_rel cos(q_rel, k_i_rel)
            + w_typ cos(q_typ, k_i_typ)

    Unoccupied slots are masked out with -inf. Output is softmax(sim) @ values.
    Gradient enters through (q_ent, q_rel, q_typ); keys and values are
    buffers, so no grad flows into bank storage itself.
    """

    def __init__(self, config: DCortexConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.w_ent, self.w_rel, self.w_typ = config.query_weights

    def forward(
        self,
        q_ent: torch.Tensor,
        q_rel: torch.Tensor,
        q_typ: torch.Tensor,
        bank: MemoryBank,
    ) -> torch.Tensor:
        """Read from bank.

        Args:
            q_ent: [B, d_ent]
            q_rel: [B, d_rel]
            q_typ: [B, d_typ]
            bank:  a MemoryBank instance.

        Returns:
            r: [B, hidden_dim]
        """
        B = q_ent.shape[0]
        device = q_ent.device
        dtype = q_ent.dtype

        if bank.n_occupied() == 0:
            return torch.zeros(B, self.hidden_dim, device=device, dtype=dtype)

        q_ent_n = F.normalize(q_ent, dim=-1)                      # [B, d_ent]
        q_rel_n = F.normalize(q_rel, dim=-1)
        q_typ_n = F.normalize(q_typ, dim=-1)

        # Use get_diff_* to pick up overlay entries (with grad)
        k_ent_n = F.normalize(bank.get_diff_k_ent(), dim=-1)     # [C, d_ent]
        k_rel_n = F.normalize(bank.get_diff_k_rel(), dim=-1)
        k_typ_n = F.normalize(bank.get_diff_k_typ(), dim=-1)

        sim_ent = q_ent_n @ k_ent_n.t()                           # [B, C]
        sim_rel = q_rel_n @ k_rel_n.t()
        sim_typ = q_typ_n @ k_typ_n.t()

        sim = self.w_ent * sim_ent + self.w_rel * sim_rel + self.w_typ * sim_typ
        sim = sim.masked_fill(~bank.occupied.unsqueeze(0), float("-inf"))

        # Temperature: sharpens softmax on cosine sims in [-1, 1].
        # Without this, softmax([1, 0, 0, 0]) = [0.59, 0.14, 0.14, 0.14] - too diffuse.
        # With temp=20, softmax([20, 0, 0, 0]) = [~1, ~0, ~0, ~0] - concentrated.
        attn = F.softmax(sim * 20.0, dim=-1)                             # [B, C]
        r = attn @ bank.get_diff_values()                          # [B, hidden_dim]
        return r


# ======================================================================
# EPISODE READER — sub-fusion of obj read and SSM readout
# ======================================================================

class EpisodeReader(nn.Module):
    """Episode reader with dedicated W_theta sub-fusion.

    Flow:
        r_ep_obj = ReadEpisodeObjects(M_episode_obj, q)        via SemanticReader
        x_t      = EpisodeSSM.forward(pooled_h)                advances SSM
        r_ep_ssm = C_ssm(x_t)                                  (already inside SSM.forward)
        r_episode = W_theta( LayerNorm( [r_ep_obj ; r_ep_ssm] ) )

    This mirrors the spec: episodic fusion is a dedicated submodule,
    not a raw sum and not a direct concat into the backbone stream.
    """

    def __init__(self, config: DCortexConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.obj_reader = SemanticReader(config)
        self.norm = nn.LayerNorm(2 * config.hidden_dim)
        self.W_theta = nn.Linear(2 * config.hidden_dim, config.hidden_dim)

    def forward(
        self,
        q_ent: torch.Tensor,
        q_rel: torch.Tensor,
        q_typ: torch.Tensor,
        episode_obj_mem: EpisodeObjectMemory,
        episode_ssm: EpisodeSSM,
        ssm_input: torch.Tensor,
        ssm_readout: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Produce r_episode.

        Args:
            q_*: latent key queries [B, d_*]
            episode_obj_mem: EpisodeObjectMemory bank
            episode_ssm: EpisodeSSM recurrent module
            ssm_input: pooled hidden state used to advance SSM,
                       shape [B, hidden_dim].
            ssm_readout: if provided, use this pre-computed readout
                         instead of calling episode_ssm(ssm_input).
                         Used by the decoder to read without gradient
                         leak into encoder SSM parameters.

        Returns:
            r_episode: [B, hidden_dim]
        """
        B = q_ent.shape[0]

        # Obj read (B, D)
        r_obj = self.obj_reader(q_ent, q_rel, q_typ, episode_obj_mem)

        # SSM readout: either pre-computed (decoder) or live (encoder)
        if ssm_readout is not None:
            r_ssm_flat = ssm_readout                   # [hidden_dim], no grad
        else:
            r_ssm_flat = episode_ssm(ssm_input)        # [hidden_dim], with grad
        r_ssm = r_ssm_flat.unsqueeze(0).expand(B, -1)  # [B, hidden_dim]

        # Sub-fusion through W_theta
        fused = torch.cat([r_obj, r_ssm], dim=-1)      # [B, 2*D]
        fused = self.norm(fused)
        r_episode = self.W_theta(fused)                # [B, D]
        return r_episode


# ======================================================================
# GLOBAL READ FUSION — 5 streams
# ======================================================================

class MemoryReadFusion(nn.Module):
    """Aggregate the five read streams into a [B, 5, D] memory-token set.

    Streams are projected independently (per-stream identity cue via a
    Linear), then stacked as five "memory tokens" for downstream
    cross-attention inside each FusionBlock. A final LayerNorm stabilizes
    magnitudes across streams.

    This preserves each stream as an addressable entity (FusionBlock can
    attend selectively) rather than pre-averaging into a single vector.
    """

    STREAMS = ("state", "episode", "conflict", "archive", "working")

    def __init__(self, config: DCortexConfig) -> None:
        super().__init__()
        D = config.hidden_dim
        self.proj_state    = nn.Linear(D, D)
        self.proj_episode  = nn.Linear(D, D)
        self.proj_conflict = nn.Linear(D, D)
        self.proj_archive  = nn.Linear(D, D)
        self.proj_working  = nn.Linear(D, D)
        self.norm = nn.LayerNorm(D)

    def forward(
        self,
        r_state: torch.Tensor,
        r_episode: torch.Tensor,
        r_conflict: torch.Tensor,
        r_archive: torch.Tensor,
        r_working: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse into [B, 5, D]."""
        stacked = torch.stack(
            [
                self.proj_state(r_state),
                self.proj_episode(r_episode),
                self.proj_conflict(r_conflict),
                self.proj_archive(r_archive),
                self.proj_working(r_working),
            ],
            dim=1,
        )  # [B, 5, D]
        return self.norm(stacked)
