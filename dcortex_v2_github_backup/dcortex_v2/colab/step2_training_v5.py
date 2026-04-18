# -*- coding: utf-8 -*-
# ===========================================================================
# D_Cortex v2.0-alpha -- Step 2 (v5): Multi-Turn Memory Training
# Google Colab A100 GPU -- SDPA-optimized
# Single Monolithic Cell
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Patent EP25216372.0. Cluj-Napoca, Romania.
# ===========================================================================
#
# CHANGES vs v4:
#   [FIX-12 CRITICAL] MULTI-TURN CURRICULUM.
#     Root cause of memory irrelevance in v2/v3/v4: fact + question were
#     in the SAME 1024-token context window. Model solved everything via
#     in-context pattern matching without ever needing memory.
#
#     Now curriculum episodes are MULTI-TURN:
#       Turn 1: forward(facts, write_memory=True)  -> populate memory
#               Aux losses only (key align, val coherence, diversity).
#               Fact text is NOT in Turn 2 input.
#       Turn 2: forward(probe, write_memory=False) -> test recall
#               LM loss on question+answer. Answer is IMPOSSIBLE without
#               memory because the fact is not in the current context.
#
#     This makes memory STRUCTURALLY NECESSARY.
#     Fusion blocks MUST attend to memory tokens for the correct answer.
#     Gradient: LM loss -> fusion -> cross-attn -> readers -> query_engine.
#
# All previous fixes (FIX 1-11) remain in effect.
# ===========================================================================

import os, sys, time, math, json, gc, io, contextlib, subprocess, random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import numpy as np

# ======================== 1. ENVIRONMENT ====================================

from google.colab import drive
drive.mount('/content/drive')

PROJECT_ROOT = '/content/drive/MyDrive/dcortex_v2'
CHECKPOINT_DIR = f'{PROJECT_ROOT}/checkpoints'
RESULTS_DIR = f'{PROJECT_ROOT}/results'
BIN_DIR = f'{PROJECT_ROOT}/dataset_cache/bin'
LOCAL_DATA = '/content/tmp_data'
SEP = '=' * 70

for d in [PROJECT_ROOT, CHECKPOINT_DIR, RESULTS_DIR, BIN_DIR, LOCAL_DATA]:
    os.makedirs(d, exist_ok=True)

print(f"[INFO] Project root: {PROJECT_ROOT}", flush=True)

# ======================== 2. GPU DETECTION ==================================

import torch

assert torch.cuda.is_available(), "CUDA required. Connect to a GPU runtime."
GPU_NAME = torch.cuda.get_device_name(0)
# [FIX-1] total_memory (not total_mem). Critical bug in v1.
GPU_MEM_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3)
GPU_CAP = torch.cuda.get_device_capability(0)

print(SEP)
print(f"[INFO] GPU: {GPU_NAME} | VRAM: {GPU_MEM_GB:.1f} GB | SM {GPU_CAP[0]}.{GPU_CAP[1]}")

if 'A100' in GPU_NAME or GPU_CAP[0] >= 8:
    DTYPE = torch.bfloat16
    USE_SCALER = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("[INFO] A100 mode: bfloat16, TF32 enabled, NO GradScaler")
else:
    DTYPE = torch.float16
    USE_SCALER = True
    print(f"[WARN] {GPU_NAME}: fp16 + GradScaler (not optimal, but functional)")

torch.backends.cudnn.benchmark = True

# Check SDPA availability
_SDPA_AVAILABLE = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
print(f"[INFO] SDPA (F.scaled_dot_product_attention): "
      f"{'AVAILABLE' if _SDPA_AVAILABLE else 'NOT AVAILABLE (PyTorch < 2.0)'}")
print(SEP)

DEVICE = torch.device('cuda')
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ======================== 3. INLINE SOURCE + DEPENDENCIES ==================
#
# All dcortex source is written inline. No git clone, no external repo.
# ============================================================================

SRC_DIR = "/content/dcortex_src"
_SOURCE_FILES = {
    "dcortex/__init__.py": r'''"""D_Cortex v2.0-alpha — memory-native transformer."""

from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model

__all__ = ["DCortexConfig", "DCortexV2Model"]
__version__ = "2.0.0-alpha"
''',

    "dcortex/config.py": r'''# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha
# Configuration dataclass. Patent EP25216372.0.

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DCortexConfig:
    """Immutable configuration for D_Cortex v2.0-alpha.

    Demonstrator scale: 12 layers, 768 hidden, 12 heads, 3072 FFN,
    2048 context. Scope is architectural proof of principle.
    """

    # --- Backbone ---
    vocab_size: int = 50257
    hidden_dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    ff_dim: int = 3072
    max_seq_len: int = 2048
    dropout: float = 0.0

    # --- Fusion layers (last N backbone layers are native FusionBlocks) ---
    n_fusion_layers: int = 4

    # --- Memory bank capacities ---
    n_state_slots: int = 64
    n_episode_obj_slots: int = 128
    n_conflict_slots: int = 32
    n_archive_slots: int = 512
    n_work_slots: int = 16

    # --- Episode SSM ---
    ssm_hidden_dim: int = 256

    # --- Latent key dims (for NN-semantic reader/updater) ---
    d_ent: int = 128
    d_rel: int = 64
    d_typ: int = 64

    # --- Query similarity weights (w_ent, w_rel, w_typ) ---
    query_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)

    # --- Thresholds ---
    theta_match: float = 0.85
    theta_conflict: float = 0.3
    theta_write: float = 0.5

    # --- Consolidator ---
    consolidate_merge_threshold: float = 0.95
    consolidate_decay_rate: float = 0.99
    consolidate_prune_threshold: float = 0.05

    # --- Updater ---
    ema_alpha: float = 0.3

    # --- Initialization ---
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )
        if self.n_fusion_layers > self.n_layers:
            raise ValueError(
                f"n_fusion_layers ({self.n_fusion_layers}) must be "
                f"<= n_layers ({self.n_layers})"
            )
        if self.n_fusion_layers < 1:
            raise ValueError("n_fusion_layers must be >= 1 (memory unused otherwise)")
        if sum(self.query_weights) <= 0:
            raise ValueError("query_weights must sum > 0")
        if not (0.0 < self.ema_alpha < 1.0):
            raise ValueError("ema_alpha must be in (0, 1)")

    @property
    def n_standard_layers(self) -> int:
        """Number of plain StandardTransformerBlocks before the FusionBlocks."""
        return self.n_layers - self.n_fusion_layers

    def small_test(self) -> "DCortexConfig":
        """Return a tiny config for unit tests and CI."""
        return DCortexConfig(
            vocab_size=256,
            hidden_dim=64,
            n_layers=4,
            n_heads=4,
            ff_dim=128,
            max_seq_len=64,
            n_fusion_layers=2,
            n_state_slots=8,
            n_episode_obj_slots=16,
            n_conflict_slots=4,
            n_archive_slots=32,
            n_work_slots=4,
            ssm_hidden_dim=32,
            d_ent=16,
            d_rel=8,
            d_typ=8,
        )
''',

    "dcortex/memory/__init__.py": r'''"""D_Cortex v2.0-alpha memory subsystem."""

from dcortex.memory.banks import (
    ArchiveMemory,
    ConflictMemory,
    EpisodeObjectMemory,
    EpisodeSSM,
    MemoryBank,
    StateMemory,
    WorkingMemory,
)
from dcortex.memory.consolidator import MemoryConsolidator
from dcortex.memory.query import QueryEngine
from dcortex.memory.readers import EpisodeReader, MemoryReadFusion, SemanticReader
from dcortex.memory.updater import MemoryUpdater
from dcortex.memory.writer import MemoryWriter

__all__ = [
    "MemoryBank",
    "StateMemory",
    "EpisodeObjectMemory",
    "ConflictMemory",
    "ArchiveMemory",
    "WorkingMemory",
    "EpisodeSSM",
    "QueryEngine",
    "MemoryUpdater",
    "SemanticReader",
    "EpisodeReader",
    "MemoryReadFusion",
    "MemoryWriter",
    "MemoryConsolidator",
]
''',

    "dcortex/memory/banks.py": r'''# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha
# Memory Banks: State, EpisodeObject, Conflict, Archive, Working, EpisodeSSM.
# Patent EP25216372.0.

from typing import Optional

import torch
import torch.nn as nn


# ======================================================================
# BASE BANK
# ======================================================================

class MemoryBank(nn.Module):
    """Slot-based memory with three-field latent keys and value buffers.

    Keys (k_ent, k_rel, k_typ) are stored as buffers and updated in-place
    by the writer. Values are also buffers. Gradient flow enters through
    the attention weighting computed at read time
    (softmax(q @ k.T) @ v), where q comes from the trainable QueryEngine.
    """

    def __init__(
        self,
        capacity: int,
        hidden_dim: int,
        d_ent: int,
        d_rel: int,
        d_typ: int,
    ) -> None:
        super().__init__()
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.d_ent = d_ent
        self.d_rel = d_rel
        self.d_typ = d_typ

        # Latent keys (buffers)
        self.register_buffer("k_ent", torch.zeros(capacity, d_ent))
        self.register_buffer("k_rel", torch.zeros(capacity, d_rel))
        self.register_buffer("k_typ", torch.zeros(capacity, d_typ))

        # Values (buffers)
        self.register_buffer("values", torch.zeros(capacity, hidden_dim))

        # Metadata
        self.register_buffer("occupied", torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("usage", torch.zeros(capacity))
        self.register_buffer(
            "last_write_step",
            torch.full((capacity,), -1, dtype=torch.long),
        )

    def reset(self) -> None:
        """Clear all slots. Call between conversations."""
        self.k_ent.zero_()
        self.k_rel.zero_()
        self.k_typ.zero_()
        self.values.zero_()
        self.occupied.zero_()
        self.usage.zero_()
        self.last_write_step.fill_(-1)

    def n_occupied(self) -> int:
        return int(self.occupied.sum().item())

    def free_slot(self) -> int:
        """Return index of first free slot, or -1 if full."""
        free = (~self.occupied).nonzero(as_tuple=False)
        if free.numel() == 0:
            return -1
        return int(free[0].item())

    def lru_slot(self) -> int:
        """Return least-recently-used OCCUPIED slot.

        Falls back to slot 0 if no slots are occupied (defensive).
        """
        if self.n_occupied() == 0:
            return 0
        steps = self.last_write_step.float().clone()
        steps[~self.occupied] = float("inf")
        return int(steps.argmin().item())

    def snapshot(self) -> dict:
        """Diagnostic dictionary. Not used in forward."""
        return {
            "capacity": self.capacity,
            "occupied": self.n_occupied(),
            "usage_mean": float(self.usage[self.occupied].mean().item())
            if self.n_occupied() > 0 else 0.0,
            "usage_max": float(self.usage.max().item()),
        }


# ======================================================================
# CONCRETE BANKS
# ======================================================================

class StateMemory(MemoryBank):
    """Slot-based factual / stable memory.

    Holds the model's view of stable facts and ground-truth-like state.
    Populated by the writer through gating over the hidden stream.
    Consolidated (promoted) to ArchiveMemory on decay.
    """


class EpisodeObjectMemory(MemoryBank):
    """Discrete episodic objects.

    Holds events, scenes, or individuated context objects produced during
    a conversation. Read alongside EpisodeSSM by the EpisodeReader.
    """


class ConflictMemory(MemoryBank):
    """Difference-vector memory for contradictions.

    When a write candidate has high key-similarity but large value
    divergence with an existing state slot, the difference
    (candidate_value - existing_value) is written here rather than
    overwriting state. Preserves both facts for downstream resolution.
    """


class ArchiveMemory(MemoryBank):
    """Long-term consolidated storage.

    Target for slots migrated out of StateMemory by the consolidator.
    Larger capacity, lower write frequency.
    """


class WorkingMemory(MemoryBank):
    """Rolling short-term memory for the current turn or conversation.

    Small capacity, aggressively overwritten via LRU. Provides live
    recent-context recall inside the current session.
    """


# ======================================================================
# EPISODE SSM (trainable state-space recurrence)
# ======================================================================

class EpisodeSSM(nn.Module):
    """Continuous episodic state as a trainable state-space recurrence.

    Recurrence:
        x_t = sigmoid(a) * x_{t-1} + B * phi(u_t)

    Readout:
        r_ssm = C * x_t

    Parameters a, B, C are learned. phi is a GELU nonlinearity.
    The state `x` is a persistent buffer: within a forward pass gradients
    flow through a, B, C; across forward passes the state is detached
    so no graph persists across conversations or turns.
    """

    def __init__(self, input_dim: int, state_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim

        # Recurrent gate, per-dim scalar through sigmoid
        self.a_raw = nn.Parameter(torch.zeros(state_dim))

        # Input projection B with phi=GELU
        self.B = nn.Linear(input_dim, state_dim)
        self.phi = nn.GELU()

        # Output projection C
        self.C = nn.Linear(state_dim, input_dim)

        # Persistent state, session-scoped
        self.register_buffer("x", torch.zeros(state_dim))

    def reset(self) -> None:
        self.x.zero_()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Advance SSM one step and return the current readout.

        Args:
            u: representative input vector, shape [input_dim] or
               [B, input_dim]. If batched, inputs are averaged across
               the batch (single shared SSM state).

        Returns:
            Readout r_ssm, shape [input_dim].
        """
        if u.dim() == 2:
            u = u.mean(dim=0)
        elif u.dim() != 1:
            raise ValueError(f"EpisodeSSM input must be 1D or 2D, got {u.dim()}D")

        a = torch.sigmoid(self.a_raw)                # [state_dim]
        drive = self.B(self.phi(u))                  # [state_dim]
        # Current forward uses detached previous state as a constant seed,
        # then produces a fresh x with gradients; persistent state is
        # updated via .data to avoid cross-turn graph accumulation.
        x_new = a * self.x.detach() + drive          # [state_dim]
        self.x.data = x_new.detach()
        return self.C(x_new)                         # [input_dim]
''',

    "dcortex/memory/query.py": r'''# -*- coding: utf-8 -*-
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
''',

    "dcortex/memory/updater.py": r'''# -*- coding: utf-8 -*-
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
''',

    "dcortex/memory/readers.py": r'''# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha
# Readers: SemanticReader (single-bank NN-attention), EpisodeReader
# (obj + SSM sub-fusion via W_theta), MemoryReadFusion (5-stream stack).
# Patent EP25216372.0.

from typing import Tuple

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

        k_ent_n = F.normalize(bank.k_ent, dim=-1)                 # [C, d_ent]
        k_rel_n = F.normalize(bank.k_rel, dim=-1)
        k_typ_n = F.normalize(bank.k_typ, dim=-1)

        sim_ent = q_ent_n @ k_ent_n.t()                           # [B, C]
        sim_rel = q_rel_n @ k_rel_n.t()
        sim_typ = q_typ_n @ k_typ_n.t()

        sim = self.w_ent * sim_ent + self.w_rel * sim_rel + self.w_typ * sim_typ
        sim = sim.masked_fill(~bank.occupied.unsqueeze(0), float("-inf"))

        attn = F.softmax(sim, dim=-1)                             # [B, C]
        r = attn @ bank.values                                    # [B, hidden_dim]
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
    ) -> torch.Tensor:
        """Produce r_episode.

        Args:
            q_*: latent key queries [B, d_*]
            episode_obj_mem: EpisodeObjectMemory bank
            episode_ssm: EpisodeSSM recurrent module
            ssm_input: pooled hidden state used to advance SSM,
                       shape [B, hidden_dim].

        Returns:
            r_episode: [B, hidden_dim]
        """
        B = q_ent.shape[0]

        # Obj read (B, D)
        r_obj = self.obj_reader(q_ent, q_rel, q_typ, episode_obj_mem)

        # SSM advance + readout (shared state, single vector)
        r_ssm_flat = episode_ssm(ssm_input)           # [hidden_dim]
        r_ssm = r_ssm_flat.unsqueeze(0).expand(B, -1) # [B, hidden_dim]

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
''',

    "dcortex/memory/writer.py": r'''# -*- coding: utf-8 -*-
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
    ) -> Dict[str, torch.Tensor]:
        """Route writes through the updater.

        Args:
            h_pool:   [B, hidden_dim]
            updater:  MemoryUpdater instance
            banks:    dict mapping {'state', 'episode_obj', 'conflict',
                                    'archive', 'working'} to MemoryBank
            step:     global step counter (for LRU tracking)

        Returns:
            Dict with:
                gate_probs [B, 6]   softmax gate (with grad)
                value      [B, D]   writer value output (with grad)
                k_ent      [B, d_ent] writer entity key (with grad)
                k_rel      [B, d_rel] writer relation key (with grad)
                k_typ      [B, d_typ] writer type key (with grad)
            The non-detached tensors enable auxiliary losses that create
            gradient paths through the writer's heads.
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

            if bank_name == "state":
                is_conflict = updater.detect_conflict(
                    banks["state"], v, ke, kr, kt
                )
                if is_conflict:
                    updater.update(banks["state"], v, ke, kr, kt, step, is_conflict=False)
                    updater.update(banks["conflict"], v, ke, kr, kt, step, is_conflict=True)
                    continue

            updater.update(banks[bank_name], v, ke, kr, kt, step, is_conflict=False)

        return {
            'gate_probs': gate_probs,
            'value': value,
            'k_ent': k_ent,
            'k_rel': k_rel,
            'k_typ': k_typ,
        }
''',

    "dcortex/memory/consolidator.py": r'''# -*- coding: utf-8 -*-
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
''',

    "dcortex/backbone/__init__.py": r'''"""D_Cortex v2.0-alpha backbone layers."""

from dcortex.backbone.embeddings import TokenEmbeddings
from dcortex.backbone.fusion_block import CrossAttention, FusionBlock
from dcortex.backbone.transformer import (
    FeedForward,
    MultiHeadSelfAttention,
    StandardTransformerBlock,
)

__all__ = [
    "TokenEmbeddings",
    "MultiHeadSelfAttention",
    "FeedForward",
    "StandardTransformerBlock",
    "CrossAttention",
    "FusionBlock",
]
''',

    "dcortex/backbone/embeddings.py": r'''# -*- coding: utf-8 -*-
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
''',

    "dcortex/backbone/transformer.py": r'''# -*- coding: utf-8 -*-
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
''',

    "dcortex/backbone/fusion_block.py": r'''# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha
# FusionBlock: native backbone layer with self-attention, cross-attention
# to the 5-stream memory-token set, and FFN. Replaces hook-based injection.
# Patent EP25216372.0.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcortex.config import DCortexConfig
from dcortex.backbone.transformer import MultiHeadSelfAttention, FeedForward


# ======================================================================
# CROSS-ATTENTION (hidden stream -> memory tokens)
# ======================================================================

class CrossAttention(nn.Module):
    """Cross-attention from [B, T, D] hidden states to [B, K, D] memory tokens.

    K is typically 5 (one per fused read stream). No causal mask: memory
    tokens are not temporally ordered with respect to sequence positions.
    """

    def __init__(self, config: DCortexConfig) -> None:
        super().__init__()
        if config.hidden_dim % config.n_heads != 0:
            raise ValueError("hidden_dim must divide n_heads evenly")
        self.n_heads = config.n_heads
        self.head_dim = config.hidden_dim // config.n_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.kv = nn.Linear(config.hidden_dim, 2 * config.hidden_dim)
        self.out = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, h: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, T, D]
            memory: [B, K, D]
        Returns:
            [B, T, D]
        """
        B, T, D = h.shape
        _, K, _ = memory.shape

        q = self.q(h).reshape(B, T, self.n_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)                                   # [B, H, T, d]

        kv = self.kv(memory).reshape(B, K, 2, self.n_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)                              # [2, B, H, K, d]
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale               # [B, H, T, K]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)           # [B, T, D]
        return self.out(out)


# ======================================================================
# FUSION BLOCK (native layer; replaces StandardTransformerBlock in the
# last n_fusion_layers of the backbone)
# ======================================================================

class FusionBlock(nn.Module):
    """Native fusion layer.

    Flow (pre-norm):
        h  = h + SelfAttn(LN(h))
        m  = CrossAttn( LN_h(h), LN_m(memory) )
        h  = h + sigmoid(mem_gate) * m
        h  = h + FFN(LN(h))

    mem_gate is a learnable per-dim sigmoid that starts at 0.5 and can
    be driven toward 0 (ignore memory) or 1 (full contribution).
    """

    def __init__(self, config: DCortexConfig) -> None:
        super().__init__()
        D = config.hidden_dim

        self.norm_self = nn.LayerNorm(D)
        self.self_attn = MultiHeadSelfAttention(config)

        self.norm_h = nn.LayerNorm(D)
        self.norm_mem = nn.LayerNorm(D)
        self.cross_attn = CrossAttention(config)

        self.norm_ff = nn.LayerNorm(D)
        self.ff = FeedForward(config)

        # Per-dim sigmoid gate. Raw 0 -> sigmoid = 0.5.
        self.mem_gate = nn.Parameter(torch.zeros(D))

    def forward(
        self,
        h: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h: [B, T, D]
            memory: [B, K, D]
            attention_mask: [B, T] (1=valid) or None
        """
        # Self-attention
        h = h + self.self_attn(self.norm_self(h), attention_mask)

        # Cross-attention to memory tokens
        m = self.cross_attn(self.norm_h(h), self.norm_mem(memory))
        gate = torch.sigmoid(self.mem_gate)                         # [D]
        h = h + gate * m                                            # broadcast over B,T

        # FFN
        h = h + self.ff(self.norm_ff(h))
        return h
''',

    "dcortex/model.py": r'''# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha
# DCortexV2Model: end-to-end memory-native transformer.
# Patent EP25216372.0.

from typing import Dict, Optional

import torch
import torch.nn as nn

from dcortex.backbone.embeddings import TokenEmbeddings
from dcortex.backbone.fusion_block import FusionBlock
from dcortex.backbone.transformer import StandardTransformerBlock
from dcortex.config import DCortexConfig
from dcortex.memory.banks import (
    ArchiveMemory,
    ConflictMemory,
    EpisodeObjectMemory,
    EpisodeSSM,
    MemoryBank,
    StateMemory,
    WorkingMemory,
)
from dcortex.memory.consolidator import MemoryConsolidator
from dcortex.memory.query import QueryEngine
from dcortex.memory.readers import EpisodeReader, MemoryReadFusion, SemanticReader
from dcortex.memory.updater import MemoryUpdater
from dcortex.memory.writer import MemoryWriter


class DCortexV2Model(nn.Module):
    """End-to-end memory-native transformer.

    Forward flow:

        1. Embed tokens                       h <- Embed(input_ids)            [B, T, D]
        2. Standard transformer stack         h <- StandardBlocks(h)
        3. Query (pooled)                     q <- QueryEngine(pool(h))        [B, d_*]
        4. Five memory reads:
               r_state     <- SemanticReader(q, M_state)                       [B, D]
               r_episode   <- EpisodeReader(q, M_episode_obj, x_t_ep, pool(h)) [B, D]
               r_conflict  <- SemanticReader(q, M_conflict)                    [B, D]
               r_archive   <- SemanticReader(q, M_archive)                     [B, D]
               r_working   <- SemanticReader(q, M_working)                     [B, D]
        5. Fusion                             M <- MemoryReadFusion(...)       [B, 5, D]
        6. Fusion blocks                      h <- FusionBlocks(h, M)
        7. Write                              Writer(pool(h), Updater, banks)  (side-effect)
        8. Head                               logits <- LM_head(LN(h))         [B, T, V]

    `reset_memory()` clears all banks and SSM state (call between conversations).
    `consolidate()` runs a consolidator pass over state -> archive.
    """

    def __init__(self, config: DCortexConfig) -> None:
        super().__init__()
        self.config = config

        # ----------------- Embeddings -----------------
        self.embeddings = TokenEmbeddings(config)

        # ----------------- Memory banks ---------------
        self.state_mem = StateMemory(
            config.n_state_slots, config.hidden_dim,
            config.d_ent, config.d_rel, config.d_typ,
        )
        self.episode_obj_mem = EpisodeObjectMemory(
            config.n_episode_obj_slots, config.hidden_dim,
            config.d_ent, config.d_rel, config.d_typ,
        )
        self.conflict_mem = ConflictMemory(
            config.n_conflict_slots, config.hidden_dim,
            config.d_ent, config.d_rel, config.d_typ,
        )
        self.archive_mem = ArchiveMemory(
            config.n_archive_slots, config.hidden_dim,
            config.d_ent, config.d_rel, config.d_typ,
        )
        self.working_mem = WorkingMemory(
            config.n_work_slots, config.hidden_dim,
            config.d_ent, config.d_rel, config.d_typ,
        )
        self.episode_ssm = EpisodeSSM(config.hidden_dim, config.ssm_hidden_dim)

        # ----------------- Memory operators -----------
        self.query_engine = QueryEngine(config)
        self.state_reader = SemanticReader(config)
        self.episode_reader = EpisodeReader(config)
        self.conflict_reader = SemanticReader(config)
        self.archive_reader = SemanticReader(config)
        self.working_reader = SemanticReader(config)
        self.read_fusion = MemoryReadFusion(config)

        self.updater = MemoryUpdater(config)
        self.writer = MemoryWriter(config)
        self.consolidator = MemoryConsolidator(config)

        # ----------------- Backbone -------------------
        n_standard = config.n_standard_layers
        self.standard_blocks = nn.ModuleList(
            [StandardTransformerBlock(config) for _ in range(n_standard)]
        )
        self.fusion_blocks = nn.ModuleList(
            [FusionBlock(config) for _ in range(config.n_fusion_layers)]
        )

        # ----------------- Head -----------------------
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Tie embedding and LM head weights
        self.lm_head.weight = self.embeddings.token_emb.weight

        # Global step counter (for LRU + consolidation scheduling)
        self.register_buffer("step_counter", torch.zeros((), dtype=torch.long))

        # Auxiliary tensor store for training losses. Populated during
        # forward() when write_memory=True. Training script reads this
        # after forward to compute key-query alignment and value
        # reconstruction losses. Empty dict when write_memory=False.
        self._aux: Dict[str, torch.Tensor] = {}

        # ----------------- Init -----------------------
        self.apply(self._init_weights)

        self._print_summary()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.init_std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _print_summary(self) -> None:
        cfg = self.config
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        sep = "=" * 70
        print(sep)
        print("[INFO] D_Cortex v2.0-alpha instantiated")
        print(sep)
        print(f"  backbone       : hidden={cfg.hidden_dim}  layers={cfg.n_layers}  "
              f"heads={cfg.n_heads}  ff={cfg.ff_dim}")
        print(f"  fusion         : last {cfg.n_fusion_layers} of {cfg.n_layers} are FusionBlocks")
        print(f"  context        : max_seq_len={cfg.max_seq_len}")
        print(f"  vocab          : {cfg.vocab_size}")
        print(f"  memory banks   : state={cfg.n_state_slots}  "
              f"episode_obj={cfg.n_episode_obj_slots}  "
              f"conflict={cfg.n_conflict_slots}  "
              f"archive={cfg.n_archive_slots}  "
              f"working={cfg.n_work_slots}")
        print(f"  episode SSM    : state_dim={cfg.ssm_hidden_dim}")
        print(f"  latent keys    : ent={cfg.d_ent}  rel={cfg.d_rel}  typ={cfg.d_typ}")
        print(f"  thresholds     : match={cfg.theta_match}  "
              f"conflict={cfg.theta_conflict}  write={cfg.theta_write}")
        print(f"  parameters     : total={total/1e6:.2f}M  trainable={trainable/1e6:.2f}M")
        print(sep)

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------

    def reset_memory(self) -> None:
        """Clear all memory banks and SSM state. Call between conversations."""
        self.state_mem.reset()
        self.episode_obj_mem.reset()
        self.conflict_mem.reset()
        self.archive_mem.reset()
        self.working_mem.reset()
        self.episode_ssm.reset()
        self.step_counter.zero_()
        print("[INFO] Memory reset: all banks cleared, SSM state zeroed, step=0")

    def consolidate(self) -> Dict[str, Dict[str, int]]:
        """Run one consolidation pass.

        State -> Archive migration, plus in-place pairwise merging inside
        state and archive. Working memory is also decayed.

        Returns:
            Per-bank diagnostic: {bank_name: {pruned, migrated, merged}}.
        """
        step = int(self.step_counter.item())
        report = {
            "working": self.consolidator.consolidate(self.working_mem, None, step),
            "state":   self.consolidator.consolidate(self.state_mem, self.archive_mem, step),
            "archive": self.consolidator.consolidate(self.archive_mem, None, step),
            "episode_obj": self.consolidator.consolidate(self.episode_obj_mem, None, step),
            "conflict": self.consolidator.consolidate(self.conflict_mem, None, step),
        }
        for bank, r in report.items():
            print(f"[INFO] consolidate[{bank}]: "
                  f"pruned={r['pruned']} migrated={r['migrated']} merged={r['merged']}")
        return report

    # ------------------------------------------------------------------
    # Bank registry
    # ------------------------------------------------------------------

    def _bank_dict(self) -> Dict[str, MemoryBank]:
        return {
            "state": self.state_mem,
            "episode_obj": self.episode_obj_mem,
            "conflict": self.conflict_mem,
            "archive": self.archive_mem,
            "working": self.working_mem,
        }

    def memory_snapshot(self) -> Dict[str, dict]:
        """Diagnostic: occupancy summary for each bank."""
        return {name: bank.snapshot() for name, bank in self._bank_dict().items()}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        write_memory: bool = True,
    ) -> torch.Tensor:
        """Run one forward pass and optionally write the result to memory.

        Args:
            input_ids:      [B, T] token ids.
            attention_mask: [B, T] (1=valid, 0=pad) or None.
            write_memory:   if True, route pooled final hidden state
                            through the writer into memory.

        Returns:
            logits: [B, T, vocab_size].
        """
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be [B, T], got {tuple(input_ids.shape)}"
            )

        # 1. Embed
        h = self.embeddings(input_ids)                               # [B, T, D]

        # 2. Standard transformer stack
        for block in self.standard_blocks:
            h = block(h, attention_mask)

        # 3. Query (pooled across sequence)
        h_pool_early = self._pool(h, attention_mask)                 # [B, D]
        q_ent, q_rel, q_typ = self.query_engine(h_pool_early)

        # 4. Five memory reads
        r_state = self.state_reader(q_ent, q_rel, q_typ, self.state_mem)
        r_episode = self.episode_reader(
            q_ent, q_rel, q_typ,
            self.episode_obj_mem, self.episode_ssm,
            ssm_input=h_pool_early,
        )
        r_conflict = self.conflict_reader(q_ent, q_rel, q_typ, self.conflict_mem)
        r_archive = self.archive_reader(q_ent, q_rel, q_typ, self.archive_mem)
        r_working = self.working_reader(q_ent, q_rel, q_typ, self.working_mem)

        # 5. Fuse reads
        memory_tokens = self.read_fusion(
            r_state, r_episode, r_conflict, r_archive, r_working,
        )                                                             # [B, 5, D]

        # 6. Fusion blocks
        for block in self.fusion_blocks:
            h = block(h, memory_tokens, attention_mask)

        # 7. Write candidate into memory
        if write_memory:
            h_pool_final = self._pool(h, attention_mask)             # [B, D]
            self.step_counter += 1
            step = int(self.step_counter.item())
            write_out = self.writer(
                h_pool_final, self.updater, self._bank_dict(), step
            )
            # Store aux tensors for training losses. All carry gradients.
            self._aux = {
                'gate_probs': write_out['gate_probs'],   # [B, 6]
                'w_value': write_out['value'],            # [B, D]
                'w_k_ent': write_out['k_ent'],            # [B, d_ent]
                'w_k_rel': write_out['k_rel'],            # [B, d_rel]
                'w_k_typ': write_out['k_typ'],            # [B, d_typ]
                'q_ent': q_ent,                           # [B, d_ent]
                'q_rel': q_rel,                           # [B, d_rel]
                'q_typ': q_typ,                           # [B, d_typ]
                'h_pool': h_pool_final,                   # [B, D]
            }
        else:
            self._aux = {}

        # 8. Final norm + LM head
        h = self.final_norm(h)
        logits = self.lm_head(h)                                     # [B, T, V]
        return logits

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pool(
        h: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Mean-pool across sequence length with optional mask.

        Args:
            h: [B, T, D]
            attention_mask: [B, T] or None.

        Returns:
            [B, D]
        """
        if attention_mask is None:
            return h.mean(dim=1)
        mask = attention_mask.float().unsqueeze(-1)                   # [B, T, 1]
        denom = mask.sum(dim=1).clamp_min(1.0)                        # [B, 1]
        return (h * mask).sum(dim=1) / denom
''',

}


def write_source() -> None:
    """Write all dcortex source files to /content/dcortex_src."""
    for fpath, content in _SOURCE_FILES.items():
        full = os.path.join(SRC_DIR, fpath)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
    print(f"[INFO] {len(_SOURCE_FILES)} source files written to {SRC_DIR}")


write_source()

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "tiktoken", "datasets", "matplotlib"],
    check=True,
)
print("[INFO] Dependencies installed")

# ======================== 4. IMPORTS ========================================

import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import matplotlib.pyplot as plt
from datasets import load_dataset

from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model
from dcortex.backbone.transformer import MultiHeadSelfAttention
from dcortex.backbone.fusion_block import CrossAttention

print("[INFO] All imports OK")

# ======================== 4B. SDPA MONKEY-PATCH =============================
#
# The repo's attention modules use manual q @ k.T -> softmax -> attn @ v.
# On A100 with PyTorch >= 2.0, F.scaled_dot_product_attention dispatches
# to flash or memory-efficient kernels automatically.
#
# We monkey-patch MultiHeadSelfAttention.forward and CrossAttention.forward
# AFTER import so the source files stay untouched.
# ============================================================================

if _SDPA_AVAILABLE:
    def _sdpa_self_attn_forward(
        self,
        h: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = h.shape
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, d]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Build attention mask for SDPA: combine causal + padding
        attn_mask = None
        if attention_mask is not None:
            pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            causal = torch.triu(
                torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1
            )
            combined = causal.unsqueeze(0).unsqueeze(0) | pad_mask
            attn_mask = torch.zeros(B, 1, T, T, device=h.device, dtype=q.dtype)
            attn_mask.masked_fill_(combined, float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=(attention_mask is None),
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out(out)

    def _sdpa_cross_attn_forward(
        self, h: torch.Tensor, memory: torch.Tensor,
    ) -> torch.Tensor:
        B, T, D = h.shape
        _, K, _ = memory.shape

        q = self.q(h).reshape(B, T, self.n_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        kv = self.kv(memory).reshape(B, K, 2, self.n_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out(out)

    MultiHeadSelfAttention.forward = _sdpa_self_attn_forward
    CrossAttention.forward = _sdpa_cross_attn_forward
    print("[INFO] SDPA monkey-patch applied to MultiHeadSelfAttention + CrossAttention")
    print("       Flash/memory-efficient kernels will be used when eligible")
else:
    print("[WARN] SDPA not available. Using manual attention (slower, more VRAM)")

# ======================== 5. TOKENIZER + DATA PIPELINE ======================

ENC = tiktoken.get_encoding("gpt2")
EOT = ENC.eot_token


def tokenize_to_bin(split: str, max_tokens: int) -> str:
    """Stream TinyStories, tokenize with GPT-2 BPE, save as uint16 .bin."""
    path = os.path.join(BIN_DIR, f'tinystories_{split}.bin')
    if os.path.exists(path):
        n = os.path.getsize(path) // 2
        print(f"[INFO] {split} cached: {path} ({n:,} tokens)")
        return path

    print(f"[INFO] Tokenizing {split} from TinyStories...", flush=True)
    try:
        ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load TinyStories ({e}). "
            "Check internet or run: pip install datasets"
        ) from e

    tokens: List[int] = []
    t0 = time.time()
    for i, ex in enumerate(ds):
        text = ex.get('text', '') or ex.get('story', '')
        if not text:
            continue
        enc_ids = ENC.encode_ordinary(text)
        enc_ids.append(EOT)
        tokens.extend(enc_ids)
        if i > 0 and i % 50000 == 0:
            elapsed = time.time() - t0
            print(f"  {len(tokens):,} tokens ({len(tokens)/elapsed:.0f} tok/s)",
                  flush=True)
        if len(tokens) >= max_tokens:
            break

    arr = np.array(tokens[:max_tokens], dtype=np.uint16)
    tmp = path + '.tmp'
    arr.tofile(tmp)
    os.rename(tmp, path)
    elapsed = time.time() - t0
    print(f"[INFO] {split}: {len(arr):,} tokens -> {path} ({elapsed:.1f}s)")
    return path


train_bin = tokenize_to_bin('train', max_tokens=80_000_000)
val_bin = tokenize_to_bin('validation', max_tokens=5_000_000)


def copy_to_local_ssd(src: str) -> str:
    """Copy .bin to Colab local SSD for fast memmap."""
    dst = os.path.join(LOCAL_DATA, os.path.basename(src))
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        return dst
    stat = os.statvfs('/content')
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    need_gb = os.path.getsize(src) / (1024**3)
    if free_gb < need_gb + 1.0:
        print(f"[WARN] Low disk ({free_gb:.1f} GB). Using Drive path directly.")
        return src
    subprocess.run(["cp", src, dst], check=True)
    print(f"[INFO] {os.path.basename(src)} -> local SSD ({need_gb:.2f} GB)")
    return dst


train_data = np.memmap(copy_to_local_ssd(train_bin), dtype=np.uint16, mode='r')
val_data = np.memmap(copy_to_local_ssd(val_bin), dtype=np.uint16, mode='r')
print(f"[INFO] Data ready: {len(train_data):,} train / {len(val_data):,} val tokens")

# ======================== 6. TRAINING CONFIG ================================


@dataclass
class TrainConfig:
    """Hyperparameters for A100 validation run."""
    seq_len: int = 1024
    batch_size: int = 16
    grad_accum: int = 4
    lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    total_steps: int = 5000
    warmup_steps: int = 200
    session_len: int = 8

    # [FIX-3] Margin entropy config
    gate_entropy_w: float = 0.02
    gate_entropy_min_H: float = 1.0

    # [FIX-8] Writer gradient losses
    w_key_align: float = 0.1       # key-query alignment cosine loss
    w_val_coherence: float = 0.05  # value-h_pool coherence cosine loss

    # [FIX-10] Key diversity loss on state/episode_obj/working
    w_diversity: float = 0.05

    log_every: int = 50
    eval_every: int = 250
    eval_batches: int = 20
    ckpt_every: int = 1000

    # [FIX-9] Memory-dominant curriculum: constant 85% from step 0.
    # No ramp. Memory tasks are the primary training signal.
    curriculum_ratio: float = 0.85


TC = TrainConfig()
TOK_PER_STEP = TC.batch_size * TC.seq_len * TC.grad_accum


print(SEP)
print(f"[INFO] Config: {TC.total_steps} steps | "
      f"batch {TC.batch_size}x{TC.grad_accum}={TC.batch_size*TC.grad_accum} | "
      f"seq {TC.seq_len} | {TOK_PER_STEP:,} tok/step")
print(f"[INFO] LR {TC.lr}->{TC.min_lr} cosine | warmup {TC.warmup_steps} | "
      f"clip {TC.grad_clip} | wd {TC.weight_decay}")
print(f"[INFO] Memory session {TC.session_len} micro-batches")
print(f"[INFO] Gate entropy: margin penalty, w={TC.gate_entropy_w}, "
      f"H_min={TC.gate_entropy_min_H:.2f} (max ln(6)={math.log(6):.2f})")
print(f"[INFO] Curriculum: DOMINANT {TC.curriculum_ratio:.0%} memory-essential from step 0")
print(f"[INFO] Key diversity: w={TC.w_diversity} on state/episode_obj/working")
print(f"[INFO] Total tokens: {TC.total_steps * TOK_PER_STEP:,}")
print(SEP)

# ======================== 7. MODEL + OPTIMIZER ==============================

cfg = DCortexConfig()
model = DCortexV2Model(cfg).to(DEVICE)

N_PARAMS = sum(p.numel() for p in model.parameters())
N_TRAINABLE = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[INFO] {N_PARAMS/1e6:.2f}M params ({N_TRAINABLE/1e6:.2f}M trainable) on {DEVICE}")

decay_p, nodecay_p = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.dim() < 2 or 'norm' in name or 'bias' in name:
        nodecay_p.append(p)
    else:
        decay_p.append(p)

optimizer = torch.optim.AdamW([
    {'params': decay_p, 'weight_decay': TC.weight_decay},
    {'params': nodecay_p, 'weight_decay': 0.0},
], lr=TC.lr, betas=(TC.beta1, TC.beta2))

print(f"[INFO] AdamW: {len(decay_p)} decay + {len(nodecay_p)} no-decay groups")

scaler = torch.amp.GradScaler('cuda') if USE_SCALER else None


def get_lr(step: int) -> float:
    """Cosine schedule with linear warmup."""
    if step < TC.warmup_steps:
        return TC.lr * (step + 1) / TC.warmup_steps
    if step >= TC.total_steps:
        return TC.min_lr
    t = (step - TC.warmup_steps) / (TC.total_steps - TC.warmup_steps)
    return TC.min_lr + 0.5 * (TC.lr - TC.min_lr) * (1.0 + math.cos(math.pi * t))


# ======================== 7B. PER-SUBMODULE GRADIENT TRACKING ===============
# [FIX-6] episode_ssm is now tracked as its own group.

SUBMODULE_GROUPS = {
    'embeddings': [],
    'standard_blocks': [],
    'fusion_blocks': [],
    'query_engine': [],
    'readers': [],
    'read_fusion': [],
    'writer': [],
    'episode_ssm': [],
    'lm_head': [],
}

for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if name.startswith('embeddings'):
        SUBMODULE_GROUPS['embeddings'].append(p)
    elif name.startswith('standard_blocks'):
        SUBMODULE_GROUPS['standard_blocks'].append(p)
    elif name.startswith('fusion_blocks'):
        SUBMODULE_GROUPS['fusion_blocks'].append(p)
    elif name.startswith('query_engine'):
        SUBMODULE_GROUPS['query_engine'].append(p)
    elif any(name.startswith(r) for r in
             ('state_reader', 'episode_reader', 'conflict_reader',
              'archive_reader', 'working_reader')):
        SUBMODULE_GROUPS['readers'].append(p)
    elif name.startswith('read_fusion'):
        SUBMODULE_GROUPS['read_fusion'].append(p)
    elif name.startswith('writer'):
        SUBMODULE_GROUPS['writer'].append(p)
    elif name.startswith('episode_ssm'):
        SUBMODULE_GROUPS['episode_ssm'].append(p)
    elif name.startswith('lm_head') or name.startswith('final_norm'):
        SUBMODULE_GROUPS['lm_head'].append(p)

_assigned = sum(len(v) for v in SUBMODULE_GROUPS.values())
_total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
print(f"[INFO] Submodule groups: {_assigned}/{_total_trainable} params assigned")
if _assigned != _total_trainable:
    # Identify unassigned params for debugging
    all_assigned = set()
    for gp in SUBMODULE_GROUPS.values():
        for p in gp:
            all_assigned.add(id(p))
    unassigned = []
    for name, p in model.named_parameters():
        if p.requires_grad and id(p) not in all_assigned:
            unassigned.append(name)
    print(f"[WARN] Unassigned trainable params: {unassigned[:10]}"
          f"{' ... (truncated)' if len(unassigned) > 10 else ''}")

for gname, gparams in SUBMODULE_GROUPS.items():
    n = sum(p.numel() for p in gparams)
    print(f"  {gname:20s}: {len(gparams):3d} tensors, {n/1e6:.2f}M params")


def compute_submodule_grad_norms() -> Dict[str, float]:
    """Compute L2 grad norm per submodule group."""
    norms = {}
    for gname, gparams in SUBMODULE_GROUPS.items():
        total_sq = 0.0
        for p in gparams:
            if p.grad is not None:
                total_sq += p.grad.data.norm(2).item() ** 2
        norms[gname] = math.sqrt(total_sq)
    return norms


# ======================== 7C. AUX TENSOR ACCESS ============================
# [FIX-8] No forward hook needed. After model(x, write_memory=True),
# model._aux contains: gate_probs, w_value, w_k_ent/rel/typ,
# q_ent/rel/typ, h_pool. All carry gradients for aux losses.
# ============================================================================

_gate_accum: List[torch.Tensor] = []  # still track gate stats for monitoring

# ======================== 8. BATCH LOADER ===================================


def get_batch(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random batch from memmap binary."""
    data = train_data if split == 'train' else val_data
    ix = np.random.randint(0, len(data) - TC.seq_len - 1, size=(TC.batch_size,))
    x = np.stack([data[i : i + TC.seq_len].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1 : i + 1 + TC.seq_len].astype(np.int64) for i in ix])
    x = torch.from_numpy(x).pin_memory().to(DEVICE, non_blocking=True)
    y = torch.from_numpy(y).pin_memory().to(DEVICE, non_blocking=True)
    return x, y


def get_seeded_batches(
    split: str, n_batches: int, seed: int = 9999,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Pre-generate n_batches with a fixed seed for reproducible ablation."""
    rng_state = np.random.get_state()
    np.random.seed(seed)
    batches = [get_batch(split) for _ in range(n_batches)]
    np.random.set_state(rng_state)
    return batches


# ======================== 8B. MEMORY CURRICULUM =============================

_ENTITIES = [
    "cat", "dog", "bird", "fish", "rabbit", "horse", "bear", "fox",
    "lion", "tiger", "elephant", "monkey", "penguin", "dolphin", "owl",
]
_COLORS = ["red", "blue", "green", "yellow", "black", "white", "brown", "pink"]
_PLACES = ["Paris", "London", "Tokyo", "Rome", "Berlin", "Madrid", "Cairo"]
_NAMES = ["Tom", "Lily", "Max", "Sara", "Ben", "Emma", "Jack", "Anna"]
_FILLER = (
    "Once upon a time, there was a little town by the river. "
    "The sun was shining and the birds were singing in the trees. "
    "Everyone was happy and the day was beautiful. "
    "The children played in the garden while the adults talked. "
)


def _make_fact_recall() -> str:
    ent = random.choice(_ENTITIES)
    col = random.choice(_COLORS)
    return (
        f"The {ent} has color {col}. {_FILLER}"
        f"What color is the {ent}? The {ent} has color {col}."
    )


def _make_contradiction() -> str:
    ent = random.choice(_ENTITIES)
    c1, c2 = random.sample(_COLORS, 2)
    return (
        f"The {ent} is {c1}. {_FILLER}"
        f"Actually, the {ent} is {c2}. The {ent} is {c2}."
    )


def _make_update() -> str:
    name = random.choice(_NAMES)
    p1, p2 = random.sample(_PLACES, 2)
    return (
        f"{name} lives in {p1}. {_FILLER}"
        f"{name} moved to {p2}. Where does {name} live? "
        f"{name} lives in {p2}."
    )


def _make_entity_confusion() -> str:
    """Two entities with different attributes. Probe each separately."""
    e1, e2 = random.sample(_ENTITIES, 2)
    c1, c2 = random.sample(_COLORS, 2)
    return (
        f"The {e1} is {c1}. The {e2} is {c2}. {_FILLER}"
        f"What color is the {e1}? The {e1} is {c1}. "
        f"What color is the {e2}? The {e2} is {c2}."
    )


def _make_delayed_multi_fact() -> str:
    """Three facts, long filler, then recall all three."""
    entities = random.sample(_ENTITIES, 3)
    colors = random.sample(_COLORS, 3)
    facts = " ".join(f"The {e} is {c}." for e, c in zip(entities, colors))
    recalls = " ".join(
        f"What color is the {e}? The {e} is {c}."
        for e, c in zip(entities, colors)
    )
    return f"{facts} {_FILLER} {_FILLER} {recalls}"


_CURRICULUM_GENERATORS = [
    _make_fact_recall, _make_fact_recall,     # 2x weight (core task)
    _make_contradiction, _make_contradiction, # 2x weight
    _make_update,
    _make_entity_confusion,
    _make_delayed_multi_fact,
]


def get_curriculum_batch() -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of curriculum sequences, tokenize, pack with EOT.

    To avoid diluting the signal with huge EOT-pad tails, we concatenate
    multiple curriculum strings per sample until seq_len + 1 is reached.
    """
    xs, ys = [], []
    for _ in range(TC.batch_size):
        ids: List[int] = []
        while len(ids) < TC.seq_len + 1:
            gen = random.choice(_CURRICULUM_GENERATORS)
            text = gen()
            ids.extend(ENC.encode_ordinary(text))
            ids.append(EOT)
        ids = ids[:TC.seq_len + 1]
        xs.append(ids[:TC.seq_len])
        ys.append(ids[1:TC.seq_len + 1])
    x = torch.tensor(xs, dtype=torch.long).pin_memory().to(DEVICE, non_blocking=True)
    y = torch.tensor(ys, dtype=torch.long).pin_memory().to(DEVICE, non_blocking=True)
    return x, y


# ======================== 8C. MULTI-TURN CURRICULUM ========================
#
# [FIX-12] Each episode produces (fact_text, probe_text) as SEPARATE strings.
# Turn 1: fact_text is forwarded with write_memory=True (populates banks).
# Turn 2: probe_text is forwarded with write_memory=False (tests recall).
# At Turn 2, the fact text is NOT in the context window.
# Memory is the ONLY path to the correct answer.
# ============================================================================


def _mt_fact_recall() -> Tuple[str, str]:
    ent = random.choice(_ENTITIES)
    col = random.choice(_COLORS)
    fact = f"The {ent} has color {col}. The {ent} is {col}. Remember: the {ent} is {col}."
    probe = f"What color is the {ent}? The {ent} has color {col}."
    return fact, probe


def _mt_contradiction() -> Tuple[str, str]:
    ent = random.choice(_ENTITIES)
    c1, c2 = random.sample(_COLORS, 2)
    fact = (f"The {ent} is {c1}. The {ent} is definitely {c1}. "
            f"Wait, actually the {ent} is {c2}. The {ent} is now {c2}.")
    probe = f"What color is the {ent} now? The {ent} is {c2}."
    return fact, probe


def _mt_update() -> Tuple[str, str]:
    name = random.choice(_NAMES)
    p1, p2 = random.sample(_PLACES, 2)
    fact = (f"{name} lives in {p1}. {name} is from {p1}. "
            f"Then {name} moved to {p2}. {name} now lives in {p2}.")
    probe = f"Where does {name} live? {name} lives in {p2}."
    return fact, probe


def _mt_entity_confusion() -> Tuple[str, str]:
    e1, e2 = random.sample(_ENTITIES, 2)
    c1, c2 = random.sample(_COLORS, 2)
    fact = (f"The {e1} is {c1}. The {e2} is {c2}. "
            f"Remember: {e1} is {c1} and {e2} is {c2}.")
    probe = (f"What color is the {e1}? The {e1} is {c1}. "
             f"What color is the {e2}? The {e2} is {c2}.")
    return fact, probe


def _mt_multi_fact() -> Tuple[str, str]:
    entities = random.sample(_ENTITIES, 3)
    colors = random.sample(_COLORS, 3)
    fact = " ".join(
        f"The {e} is {c}. Remember: {e} is {c}."
        for e, c in zip(entities, colors)
    )
    probe = " ".join(
        f"What color is the {e}? The {e} is {c}."
        for e, c in zip(entities, colors)
    )
    return fact, probe


_MT_GENERATORS = [
    _mt_fact_recall, _mt_fact_recall,         # 2x weight
    _mt_contradiction, _mt_contradiction,     # 2x weight
    _mt_update,
    _mt_entity_confusion,
    _mt_multi_fact,
]


def get_multi_turn_batch() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate batch of multi-turn episodes.

    Returns:
        x_fact  [B, seq_len] : fact injection sequence (Turn 1 input)
        x_probe [B, seq_len] : probe sequence (Turn 2 input)
        y_probe [B, seq_len] : probe targets (shifted)

    At Turn 2, the fact text is NOT in x_probe. Memory is the only path.
    """
    facts, probes_x, probes_y = [], [], []
    for _ in range(TC.batch_size):
        gen = random.choice(_MT_GENERATORS)
        fact, probe = gen()

        # Fact: tokenize and pad
        f_ids = ENC.encode_ordinary(fact)
        f_ids.append(EOT)
        f_ids = (f_ids + [EOT] * TC.seq_len)[:TC.seq_len]
        facts.append(f_ids)

        # Probe: tokenize, create x/y shift
        p_ids = ENC.encode_ordinary(probe)
        p_ids.append(EOT)
        if len(p_ids) > TC.seq_len + 1:
            p_ids = p_ids[:TC.seq_len + 1]
        else:
            p_ids = p_ids + [EOT] * (TC.seq_len + 1 - len(p_ids))
        probes_x.append(p_ids[:TC.seq_len])
        probes_y.append(p_ids[1:TC.seq_len + 1])

    x_f = torch.tensor(facts, dtype=torch.long).pin_memory().to(DEVICE, non_blocking=True)
    x_p = torch.tensor(probes_x, dtype=torch.long).pin_memory().to(DEVICE, non_blocking=True)
    y_p = torch.tensor(probes_y, dtype=torch.long).pin_memory().to(DEVICE, non_blocking=True)
    return x_f, x_p, y_p


# ======================== 9. LOSSES =========================================


def compute_key_diversity(
    w_k_ent: torch.Tensor,
    bank_keys: torch.Tensor,
    n_occupied: int,
) -> torch.Tensor:
    """Penalize similarity of current write key to existing bank keys.

    [FIX-10] Current write key has grad (from writer heads).
    Bank stored keys are buffers (no grad). Gradient flows only into
    writer, pushing it to produce keys dissimilar to what's stored.

    Args:
        w_k_ent: [B, d_ent] current write candidate entity key (with grad)
        bank_keys: [capacity, d_ent] stored entity keys (buffer, no grad)
        n_occupied: number of occupied slots

    Returns:
        scalar loss: mean cosine of write key vs occupied bank keys.
    """
    if n_occupied == 0:
        return torch.tensor(0.0, device=w_k_ent.device)
    # Use first batch element as representative (all elements similar due to pooling)
    wk = F.normalize(w_k_ent[0:1], dim=-1)                    # [1, d]
    bk = F.normalize(bank_keys[:n_occupied], dim=-1)           # [n, d]
    sims = (wk @ bk.t()).squeeze(0)                            # [n]
    return sims.mean()


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    model: nn.Module,
) -> Tuple[torch.Tensor, float, float, float, float, float, float]:
    """LM cross-entropy + memory auxiliary losses.

    Components:
      1. Margin gate entropy: max(0, H_min - H)
      2. Key-query alignment: 1 - cos(w_keys, q_keys)
      3. Value coherence: 1 - cos(value, h_pool)
      4. [FIX-10] Key diversity: cos(w_key, stored_keys) on state/episode_obj/working

    Returns: (total_loss, lm_val, ge_val, margin_val, key_align_val,
              val_coh_val, div_val)
    """
    B, T, V = logits.shape
    lm_loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

    ge_val = 0.0
    margin_val = 0.0
    key_align_val = 0.0
    val_coh_val = 0.0
    div_val = 0.0

    margin_pen = torch.tensor(0.0, device=logits.device)
    key_align_loss = torch.tensor(0.0, device=logits.device)
    val_coh_loss = torch.tensor(0.0, device=logits.device)
    div_loss = torch.tensor(0.0, device=logits.device)

    gp = aux.get('gate_probs')
    if gp is not None:
        H = -(gp * (gp + 1e-8).log()).sum(dim=-1).mean()
        ge_val = H.item()
        margin_pen = F.relu(TC.gate_entropy_min_H - H)
        margin_val = margin_pen.item()

    w_ent = aux.get('w_k_ent')
    q_ent = aux.get('q_ent')
    if w_ent is not None and q_ent is not None:
        cos_ent = F.cosine_similarity(w_ent, q_ent, dim=-1)
        cos_rel = F.cosine_similarity(aux['w_k_rel'], aux['q_rel'], dim=-1)
        cos_typ = F.cosine_similarity(aux['w_k_typ'], aux['q_typ'], dim=-1)
        alignment = 0.5 * cos_ent + 0.3 * cos_rel + 0.2 * cos_typ
        key_align_loss = (1.0 - alignment).mean()
        key_align_val = key_align_loss.item()

    w_val = aux.get('w_value')
    h_pool = aux.get('h_pool')
    if w_val is not None and h_pool is not None:
        val_coh = F.cosine_similarity(w_val, h_pool.detach(), dim=-1)
        val_coh_loss = (1.0 - val_coh).mean()
        val_coh_val = val_coh_loss.item()

    # [FIX-10] Key diversity: penalize similarity of write key to stored keys
    if w_ent is not None:
        div_banks = [
            ('state', model.state_mem),
            ('episode_obj', model.episode_obj_mem),
            ('working', model.working_mem),
        ]
        n_div = 0
        for bname, bank in div_banks:
            nocc = bank.n_occupied()
            if nocc > 0:
                div_loss = div_loss + compute_key_diversity(
                    w_ent, bank.k_ent, nocc
                )
                n_div += 1
        if n_div > 0:
            div_loss = div_loss / n_div
        div_val = div_loss.item()

    total = (lm_loss
             + TC.gate_entropy_w * margin_pen
             + TC.w_key_align * key_align_loss
             + TC.w_val_coherence * val_coh_loss
             + TC.w_diversity * div_loss)

    return total, lm_loss.item(), ge_val, margin_val, key_align_val, val_coh_val, div_val


def compute_aux_losses(
    aux: Dict[str, torch.Tensor],
    model: nn.Module,
) -> Tuple[torch.Tensor, float, float, float, float, float]:
    """Compute ONLY auxiliary losses (no LM cross-entropy).

    [FIX-12] Used during Turn 1 (fact injection) of multi-turn episodes.
    The fact turn has no targets for LM loss, but we still want gradients
    through the writer (via key align, value coherence, diversity) and
    through the gate (via margin entropy).

    Returns: (total_aux_loss, ge_val, margin_val, ka_val, vc_val, dv_val)
    """
    device = next(iter(aux.values())).device

    ge_val = 0.0
    margin_val = 0.0
    ka_val = 0.0
    vc_val = 0.0
    dv_val = 0.0

    margin_pen = torch.tensor(0.0, device=device)
    key_align_loss = torch.tensor(0.0, device=device)
    val_coh_loss = torch.tensor(0.0, device=device)
    div_loss = torch.tensor(0.0, device=device)

    gp = aux.get('gate_probs')
    if gp is not None:
        H = -(gp * (gp + 1e-8).log()).sum(dim=-1).mean()
        ge_val = H.item()
        margin_pen = F.relu(TC.gate_entropy_min_H - H)
        margin_val = margin_pen.item()

    w_ent = aux.get('w_k_ent')
    q_ent = aux.get('q_ent')
    if w_ent is not None and q_ent is not None:
        cos_ent = F.cosine_similarity(w_ent, q_ent, dim=-1)
        cos_rel = F.cosine_similarity(aux['w_k_rel'], aux['q_rel'], dim=-1)
        cos_typ = F.cosine_similarity(aux['w_k_typ'], aux['q_typ'], dim=-1)
        alignment = 0.5 * cos_ent + 0.3 * cos_rel + 0.2 * cos_typ
        key_align_loss = (1.0 - alignment).mean()
        ka_val = key_align_loss.item()

    w_val = aux.get('w_value')
    h_pool = aux.get('h_pool')
    if w_val is not None and h_pool is not None:
        val_coh = F.cosine_similarity(w_val, h_pool.detach(), dim=-1)
        val_coh_loss = (1.0 - val_coh).mean()
        vc_val = val_coh_loss.item()

    if w_ent is not None:
        div_banks = [
            ('state', model.state_mem),
            ('episode_obj', model.episode_obj_mem),
            ('working', model.working_mem),
        ]
        n_div = 0
        for bname, bank in div_banks:
            nocc = bank.n_occupied()
            if nocc > 0:
                div_loss = div_loss + compute_key_diversity(w_ent, bank.k_ent, nocc)
                n_div += 1
        if n_div > 0:
            div_loss = div_loss / n_div
        dv_val = div_loss.item()

    total_aux = (TC.gate_entropy_w * margin_pen
                 + TC.w_key_align * key_align_loss
                 + TC.w_val_coherence * val_coh_loss
                 + TC.w_diversity * div_loss)

    return total_aux, ge_val, margin_val, ka_val, vc_val, dv_val


# ======================== 10. EVAL + GENERATE ===============================


@torch.no_grad()
def evaluate(model: nn.Module, step: int) -> float:
    """Eval on val set."""
    model.eval()
    total_loss = 0.0
    for _ in range(TC.eval_batches):
        x, y = get_batch('val')
        with torch.amp.autocast('cuda', dtype=DTYPE):
            logits = model(x, write_memory=False)
            total_loss += F.cross_entropy(
                logits.view(-1, cfg.vocab_size), y.view(-1)
            ).item()
    avg = total_loss / TC.eval_batches
    ppl = math.exp(min(avg, 20.0))
    print(f"  [EVAL] step={step} | loss={avg:.4f} | ppl={ppl:.2f}", flush=True)
    model.train()
    return avg


@torch.no_grad()
def generate_sample(
    model: nn.Module,
    prompt: str = "Once upon a time",
    max_new: int = 80,
    temp: float = 0.8,
    top_k: int = 40,
) -> str:
    """Top-k sampling."""
    model.eval()
    ids = ENC.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    for _ in range(max_new):
        if x.shape[1] > cfg.max_seq_len:
            x = x[:, -cfg.max_seq_len:]
        with torch.amp.autocast('cuda', dtype=DTYPE):
            logits = model(x, write_memory=False)
        logits = logits[:, -1, :] / temp
        if top_k > 0:
            vals, idx = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(1, idx, vals)
        probs = F.softmax(logits, dim=-1)
        tok = torch.multinomial(probs, 1)
        x = torch.cat([x, tok], dim=1)
        if tok.item() == EOT:
            break
    model.train()
    return ENC.decode(x[0].tolist())


# ======================== 11. CHECKPOINTING =================================

_last_saved_step = -1  # [FIX-2] guard against double-save at same step


def save_ckpt(
    model: nn.Module, optimizer, scaler, step: int, losses: list,
) -> None:
    """Atomic checkpoint with duplicate guard."""
    global _last_saved_step
    if step == _last_saved_step:
        return  # [FIX-2] already saved at this step, skip
    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'losses': losses[-200:],
        'config_model': asdict(cfg),
        'config_train': asdict(TC),
    }
    if scaler is not None:
        ckpt['scaler'] = scaler.state_dict()
    fname = f'ckpt_step{step:06d}.pt'
    tmp = os.path.join(CHECKPOINT_DIR, fname + '.tmp')
    final = os.path.join(CHECKPOINT_DIR, fname)
    torch.save(ckpt, tmp)
    os.rename(tmp, final)
    _last_saved_step = step
    print(f"[INFO] Checkpoint: {fname}", flush=True)


def load_latest_ckpt(model: nn.Module, optimizer, scaler) -> int:
    """Resume from latest checkpoint."""
    ckpts = list(Path(CHECKPOINT_DIR).glob('ckpt_step*.pt'))
    if not ckpts:
        print("[INFO] No checkpoint found. Starting from step 0.")
        return 0
    ckpts.sort(key=lambda p: int(p.stem.split('step')[1]))
    latest = ckpts[-1]
    print(f"[INFO] Loading {latest.name}...")
    c = torch.load(latest, map_location=DEVICE, weights_only=False)
    model.load_state_dict(c['model'])
    optimizer.load_state_dict(c['optimizer'])
    if scaler is not None and 'scaler' in c:
        scaler.load_state_dict(c['scaler'])
    s = c['step']
    print(f"[INFO] Resumed from step {s}")
    return s


# ======================== 12. TRAINING LOOP =================================

print(SEP)
print("[INFO] TRAINING START")
print(SEP)

start_step = load_latest_ckpt(model, optimizer, scaler)
model.train()
torch.cuda.reset_peak_memory_stats()

# Metrics storage
M_steps: List[int] = []
M_lm: List[float] = []
M_ge: List[float] = []       # raw gate entropy (monitoring)
M_margin: List[float] = []   # margin-penalty value
M_keyalign: List[float] = [] # key-query alignment loss
M_valcoh: List[float] = []   # value coherence loss
M_gn: List[float] = []
M_lr: List[float] = []
M_tps: List[float] = []
M_gate: List[List[float]] = []
M_occ: List[Dict[str, float]] = []
M_fgate: List[List[float]] = []
M_subgrad: List[Dict[str, float]] = []
M_curric: List[float] = []   # curriculum ratio per step (constant in v4)
M_div: List[float] = []     # key diversity loss
E_steps: List[int] = []
E_loss: List[float] = []
E_ppl: List[float] = []
loss_log: List[float] = []

micro_ctr = 0
best_val = float('inf')
t0_train = time.time()
tok_done = 0

with contextlib.redirect_stdout(io.StringIO()):
    model.reset_memory()

init_val = evaluate(model, start_step)
E_steps.append(start_step)
E_loss.append(init_val)
E_ppl.append(math.exp(min(init_val, 20.0)))
expected_init = math.log(cfg.vocab_size)
print(f"[INFO] Init val loss: {init_val:.4f} (expected ~{expected_init:.2f} for random)")
print(f"  [SAMPLE] {generate_sample(model)[:200]}")

for step in range(start_step, TC.total_steps):
    t_step = time.time()

    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    # [FIX-9] Constant 85% curriculum from step 0
    curric_ratio = TC.curriculum_ratio

    optimizer.zero_grad(set_to_none=True)

    acc_lm = 0.0
    acc_ge = 0.0
    acc_margin = 0.0
    acc_keyalign = 0.0
    acc_valcoh = 0.0
    acc_div = 0.0
    _gate_accum.clear()

    for _mi in range(TC.grad_accum):
        if random.random() < TC.curriculum_ratio:
            # ---- MULTI-TURN CURRICULUM EPISODE [FIX-12] ----
            # Turn 1: inject facts into memory (aux losses, no LM loss)
            x_fact, x_probe, y_probe = get_multi_turn_batch()

            with torch.amp.autocast('cuda', dtype=DTYPE):
                model(x_fact, write_memory=True)
                aux_fact = model._aux
                aux_loss, ge_v, mg_v, ka_v, vc_v, dv_v = compute_aux_losses(
                    aux_fact, model,
                )
                scaled_aux = aux_loss / TC.grad_accum

            if USE_SCALER:
                scaler.scale(scaled_aux).backward()
            else:
                scaled_aux.backward()

            # Track gate from fact injection turn
            gp = aux_fact.get('gate_probs')
            if gp is not None:
                _gate_accum.append(gp.detach().cpu())

            # Turn 2: probe recall (LM loss, memory is the ONLY path)
            with torch.amp.autocast('cuda', dtype=DTYPE):
                logits_probe = model(x_probe, write_memory=False)
                lm_loss = F.cross_entropy(
                    logits_probe.view(-1, cfg.vocab_size),
                    y_probe.view(-1),
                )
                scaled_lm = lm_loss / TC.grad_accum

            if USE_SCALER:
                scaler.scale(scaled_lm).backward()
            else:
                scaled_lm.backward()

            acc_lm += lm_loss.item() / TC.grad_accum
            acc_ge += ge_v / TC.grad_accum
            acc_margin += mg_v / TC.grad_accum
            acc_keyalign += ka_v / TC.grad_accum
            acc_valcoh += vc_v / TC.grad_accum
            acc_div += dv_v / TC.grad_accum
            tok_done += TC.batch_size * TC.seq_len * 2  # 2 turns

        else:
            # ---- STANDARD LM BATCH ----
            x, y = get_batch('train')

            with torch.amp.autocast('cuda', dtype=DTYPE):
                logits = model(x, write_memory=True)
                aux = model._aux
                loss, lm_val, ge_val, margin_val, ka_val, vc_val, dv_val = compute_loss(
                    logits, y, aux, model,
                )
                loss = loss / TC.grad_accum

            if USE_SCALER:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            acc_lm += lm_val / TC.grad_accum
            acc_ge += ge_val / TC.grad_accum
            acc_margin += margin_val / TC.grad_accum
            acc_keyalign += ka_val / TC.grad_accum
            acc_valcoh += vc_val / TC.grad_accum
            acc_div += dv_val / TC.grad_accum
            tok_done += TC.batch_size * TC.seq_len

            gp = aux.get('gate_probs')
            if gp is not None:
                _gate_accum.append(gp.detach().cpu())

        micro_ctr += 1

        if micro_ctr % TC.session_len == 0:
            with contextlib.redirect_stdout(io.StringIO()):
                model.reset_memory()

    sub_grads = compute_submodule_grad_norms()

    if USE_SCALER:
        scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), TC.grad_clip
    ).item()

    if USE_SCALER:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    step_time = time.time() - t_step
    tps = TOK_PER_STEP / max(step_time, 1e-6)
    loss_log.append(acc_lm)

    if _gate_accum:
        gate_all = torch.cat(_gate_accum, dim=0)
        gate_avg = gate_all.mean(0).tolist()
    else:
        gate_avg = [0.0] * 6

    snap = model.memory_snapshot()
    occ = {k: v['occupied'] / v['capacity'] for k, v in snap.items()}
    fgates = [torch.sigmoid(b.mem_gate).mean().item() for b in model.fusion_blocks]

    M_steps.append(step)
    M_lm.append(acc_lm)
    M_ge.append(acc_ge)
    M_margin.append(acc_margin)
    M_keyalign.append(acc_keyalign)
    M_valcoh.append(acc_valcoh)
    M_div.append(acc_div)
    M_gn.append(grad_norm)
    M_lr.append(lr)
    M_tps.append(tps)
    M_gate.append(gate_avg)
    M_occ.append(occ)
    M_fgate.append(fgates)
    M_subgrad.append(sub_grads)
    M_curric.append(curric_ratio)

    if step % TC.log_every == 0 or step == start_step:
        elapsed = time.time() - t0_train
        done = step - start_step + 1
        eta_s = elapsed / done * (TC.total_steps - step - 1)
        vram = torch.cuda.max_memory_allocated() / (1024**3)
        print(
            f"Step {step:5d}/{TC.total_steps} | loss={acc_lm:.4f} | "
            f"ka={acc_keyalign:.3f} vc={acc_valcoh:.3f} dv={acc_div:.3f} | "
            f"ge={acc_ge:.3f} | gn={grad_norm:.2f} | "
            f"lr={lr:.2e} | {tps:.0f} tok/s | VRAM={vram:.1f}GB | "
            f"ETA {int(eta_s//60)}m{int(eta_s%60):02d}s",
            flush=True,
        )
        sg_str = " | ".join(f"{k}={v:.3f}" for k, v in sub_grads.items() if v > 0)
        print(f"  grad/sub: {sg_str}", flush=True)

    if (step + 1) % TC.eval_every == 0:
        vl = evaluate(model, step + 1)
        E_steps.append(step + 1)
        E_loss.append(vl)
        E_ppl.append(math.exp(min(vl, 20.0)))
        print(f"  [SAMPLE] {generate_sample(model)[:200]}", flush=True)
        if vl < best_val:
            best_val = vl
            save_ckpt(model, optimizer, scaler, step + 1, loss_log)

    if (step + 1) % TC.ckpt_every == 0:
        save_ckpt(model, optimizer, scaler, step + 1, loss_log)  # guarded by FIX-2

    if math.isnan(acc_lm) or math.isinf(acc_lm):
        print(f"[ERROR] Loss={acc_lm} at step {step}. Aborting.", flush=True)
        break

total_time = time.time() - t0_train
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

print(SEP)
print(f"[INFO] Training complete: {total_time:.0f}s ({total_time/60:.1f}min)")
print(f"[INFO] {tok_done:,} tokens | {tok_done/total_time:,.0f} tok/s | "
      f"peak VRAM {peak_vram:.1f} GB")
if M_lm:
    print(f"[INFO] Final train loss: {M_lm[-1]:.4f}")
print(f"[INFO] Best val loss: {best_val:.4f} "
      f"(ppl={math.exp(min(best_val, 20.0)):.2f})")
print(SEP)

save_ckpt(model, optimizer, scaler, TC.total_steps, loss_log)

# ======================== 13. SAME-BATCH MEMORY ABLATION ====================

print(SEP)
print("[INFO] SAME-BATCH MEMORY ABLATION STUDY")
print(SEP)

N_POPULATE = 10
N_EVAL_ABL = 50


@torch.no_grad()
def same_batch_ablation(
    model: nn.Module,
    n_populate: int = N_POPULATE,
    n_eval: int = N_EVAL_ABL,
    seed: int = 12345,
) -> Tuple[float, float]:
    """Compare eval loss with populated vs empty memory on SAME batches."""
    model.eval()
    _gate_accum.clear()  # [FIX-7]

    eval_batches = get_seeded_batches('val', n_eval, seed=seed)

    with contextlib.redirect_stdout(io.StringIO()):
        model.reset_memory()
    populate_batches = get_seeded_batches('val', n_populate, seed=seed + 1)
    for x, _ in populate_batches:
        with torch.amp.autocast('cuda', dtype=DTYPE):
            model(x, write_memory=True)

    loss_with = 0.0
    for x, y in eval_batches:
        with torch.amp.autocast('cuda', dtype=DTYPE):
            logits = model(x, write_memory=False)
            loss_with += F.cross_entropy(
                logits.view(-1, cfg.vocab_size), y.view(-1)
            ).item()
    loss_with /= n_eval

    with contextlib.redirect_stdout(io.StringIO()):
        model.reset_memory()
    loss_without = 0.0
    for x, y in eval_batches:
        with torch.amp.autocast('cuda', dtype=DTYPE):
            logits = model(x, write_memory=False)
            loss_without += F.cross_entropy(
                logits.view(-1, cfg.vocab_size), y.view(-1)
            ).item()
    loss_without /= n_eval

    model.train()
    return loss_with, loss_without


loss_w_mem, loss_wo_mem = same_batch_ablation(model)
abl_delta = loss_wo_mem - loss_w_mem
ppl_w = math.exp(min(loss_w_mem, 20.0))
ppl_wo = math.exp(min(loss_wo_mem, 20.0))

print(f"  WITH memory   : loss={loss_w_mem:.4f}  ppl={ppl_w:.2f}")
print(f"  WITHOUT memory: loss={loss_wo_mem:.4f}  ppl={ppl_wo:.2f}")
print(f"  Delta (without - with): {abl_delta:+.4f}")
print(f"  Method: same {N_EVAL_ABL} batches (seed=12345), "
      f"populated with {N_POPULATE} passes")
print("  Note: this measures SHORT-TERM PRIMING on adjacent val samples,")
print("        not durable long-horizon memory. Full validation is Step 3.")
if abl_delta > 0.005:
    print(f"  RESULT: Priming benefit (+{abl_delta:.4f} loss reduction)")
elif abl_delta > 0:
    print(f"  RESULT: Slight priming benefit ({abl_delta:+.4f})")
else:
    print(f"  RESULT: No priming benefit ({abl_delta:+.4f}). "
          "Normal at this training budget.")

# ======================== 14. CONFLICTMEMORY SEMANTIC TEST ==================
#
# [FIX-5] Verifies the diff-vector mechanism properly. The updater stores
# (v_new - v_existing) when called with is_conflict=True on a bank that
# already holds a key-similar slot. We check the cosine of the stored value
# against the expected difference.
# ============================================================================

print(SEP)
print("[INFO] CONFLICTMEMORY SEMANTIC VALIDATION")
print(SEP)


@torch.no_grad()
def test_conflict_memory_semantics(model: nn.Module) -> Dict[str, object]:
    """Validate the diff-vector and EMA mechanisms in MemoryUpdater."""
    model.eval()
    results: Dict[str, object] = {}

    with contextlib.redirect_stdout(io.StringIO()):
        model.reset_memory()

    # --- End-to-end probe via forward pass ---
    fact_text = "The big red cat sat on the old wooden chair in the kitchen."
    fact_ids = ENC.encode_ordinary(fact_text)
    fact_ids = (fact_ids + [EOT] * TC.seq_len)[:TC.seq_len]
    x_fact = torch.tensor([fact_ids], dtype=torch.long, device=DEVICE)

    with torch.amp.autocast('cuda', dtype=DTYPE):
        model(x_fact, write_memory=True)

    snap1 = model.memory_snapshot()
    results['state_after_fact'] = snap1['state']['occupied']
    results['conflict_after_fact'] = snap1['conflict']['occupied']

    for _ in range(5):
        with torch.amp.autocast('cuda', dtype=DTYPE):
            model(x_fact, write_memory=True)
    results['state_after_repeat'] = model.memory_snapshot()['state']['occupied']

    contra_text = "The big blue dog sat on the new metal table in the bedroom."
    contra_ids = ENC.encode_ordinary(contra_text)
    contra_ids = (contra_ids + [EOT] * TC.seq_len)[:TC.seq_len]
    x_contra = torch.tensor([contra_ids], dtype=torch.long, device=DEVICE)

    conflict_before = model.memory_snapshot()['conflict']['occupied']
    with torch.amp.autocast('cuda', dtype=DTYPE):
        model(x_contra, write_memory=True)
    conflict_after = model.memory_snapshot()['conflict']['occupied']
    results['conflict_slots_gained_end_to_end'] = conflict_after - conflict_before

    # --- Direct mechanism test: diff-vector correctness ---
    D = cfg.hidden_dim
    d_ent, d_rel, d_typ = cfg.d_ent, cfg.d_rel, cfg.d_typ

    key_base = torch.randn(d_ent, device=DEVICE)
    k_ent_1 = F.normalize(key_base, dim=0)
    k_ent_2 = F.normalize(key_base + 0.05 * torch.randn(d_ent, device=DEVICE), dim=0)
    key_sim = F.cosine_similarity(
        k_ent_1.unsqueeze(0), k_ent_2.unsqueeze(0)
    ).item()
    results['direct_test_key_sim'] = round(key_sim, 4)

    k_rel = torch.randn(d_rel, device=DEVICE)
    k_typ = torch.randn(d_typ, device=DEVICE)

    v1 = torch.randn(D, device=DEVICE)
    v2 = -v1 + 0.1 * torch.randn(D, device=DEVICE)
    val_sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
    results['direct_test_value_sim'] = round(val_sim, 4)

    # Fresh test bank holding v1
    test_bank = model.state_mem.__class__(8, D, d_ent, d_rel, d_typ).to(DEVICE)
    model.updater.update(test_bank, v1, k_ent_1, k_rel, k_typ, step=1)
    results['test_bank_after_v1'] = test_bank.n_occupied()

    is_conflict = model.updater.detect_conflict(
        test_bank, v2, k_ent_2, k_rel, k_typ,
    )
    results['conflict_detected'] = is_conflict

    diff_correct = False
    if is_conflict:
        v1_before = test_bank.values[0].clone()
        model.updater.update(
            test_bank, v2, k_ent_2, k_rel, k_typ,
            step=2, is_conflict=True,
        )
        stored_diff = test_bank.values[0]
        expected_diff = v2 - v1_before
        diff_cosine = F.cosine_similarity(
            stored_diff.unsqueeze(0), expected_diff.unsqueeze(0),
        ).item()
        results['diff_vector_cosine_vs_expected'] = round(diff_cosine, 4)
        diff_correct = diff_cosine > 0.99
        results['diff_vector_correct'] = diff_correct

        # EMA path
        test_bank.reset()
        model.updater.update(test_bank, v1, k_ent_1, k_rel, k_typ, step=3)
        v1_stored = test_bank.values[0].clone()
        model.updater.update(
            test_bank, v2, k_ent_2, k_rel, k_typ,
            step=4, is_conflict=False,
        )
        ema_val = test_bank.values[0]
        alpha = model.updater.ema_alpha
        expected_ema = (1.0 - alpha) * v1_stored + alpha * v2
        ema_cosine = F.cosine_similarity(
            ema_val.unsqueeze(0), expected_ema.unsqueeze(0),
        ).item()
        results['ema_cosine_vs_expected'] = round(ema_cosine, 4)
        results['ema_update_correct'] = ema_cosine > 0.99

    results['mechanism_functional'] = (
        bool(results.get('conflict_detected', False)) and diff_correct
    )

    model.train()
    return results


conflict_results = test_conflict_memory_semantics(model)
print("ConflictMemory semantic test results:")
for k, v in conflict_results.items():
    print(f"  {k:38s}: {v}")

if conflict_results.get('mechanism_functional'):
    print("  VERDICT: plumbing functional (detection + diff storage verified)")
    print("           NOTE: does not prove the MODEL generates conflict-")
    print("                 inducing keys on its own. That is Step 3.")
else:
    print("  VERDICT: mechanism did not trigger. Check theta_match / "
          "theta_conflict or seed.")

# ======================== 15. FINAL DIAGNOSTICS =============================

print(SEP)
print("[INFO] FINAL DIAGNOSTICS")
print(SEP)

with contextlib.redirect_stdout(io.StringIO()):
    model.reset_memory()
model.eval()
for _ in range(20):
    x, _ = get_batch('val')
    with torch.amp.autocast('cuda', dtype=DTYPE):
        model(x, write_memory=True)

snap_final = model.memory_snapshot()
print("Memory bank occupancy (after 20 forward passes):")
for name, info in snap_final.items():
    pct = info['occupied'] / info['capacity'] * 100
    print(f"  {name:12s}: {info['occupied']:3d}/{info['capacity']:3d} "
          f"({pct:5.1f}%)  avg_usage={info['usage_mean']:.1f}  "
          f"max_usage={info['usage_max']:.0f}")

print("\nFusion block mem_gate (sigmoid) after training:")
for i, block in enumerate(model.fusion_blocks):
    g = torch.sigmoid(block.mem_gate)
    print(f"  Block {i}: mean={g.mean():.4f}  std={g.std():.4f}  "
          f"min={g.min():.4f}  max={g.max():.4f}")

if M_gate:
    final_gate = M_gate[-1]
    gate_names = ['state', 'episode_obj', 'conflict', 'archive', 'working', 'skip']
    print("\nWriter gate distribution (final step, averaged over micro-batches):")
    for i, gn in enumerate(gate_names):
        bar = '#' * int(final_gate[i] * 60)
        print(f"  {gn:12s}: {final_gate[i]:.4f}  {bar}")

if M_subgrad:
    print("\nPer-submodule gradient norms (final step):")
    final_sg = M_subgrad[-1]
    for gname, gnorm in final_sg.items():
        status = "OK" if gnorm > 0 else "ZERO"
        print(f"  {gname:20s}: {gnorm:.4f}  [{status}]")

print("\nConsolidation pass on populated memory:")
with contextlib.redirect_stdout(io.StringIO()):
    cons_report = model.consolidate()
for bank, r in cons_report.items():
    print(f"  {bank:12s}: pruned={r['pruned']}  migrated={r['migrated']}  "
          f"merged={r['merged']}")

print("\nGeneration samples (post-training):")
test_prompts = [
    "Once upon a time",
    "The little cat went to",
    "In a big forest there was",
]
for prompt in test_prompts:
    sample = generate_sample(model, prompt, max_new=100, temp=0.8)
    print(f"  Prompt: \"{prompt}\"")
    print(f"  -> {sample[:300]}")
    print()

# ======================== 16. PLOTS =========================================

print(SEP)
print("[INFO] Generating plots")
print(SEP)

fig, axes = plt.subplots(4, 3, figsize=(18, 20))
fig.suptitle(
    'D_Cortex v2.0-alpha -- Step 2 (v5)): Training + Memory Aux Losses',
    fontsize=14, y=0.98,
)

# 1. Training loss
ax = axes[0, 0]
ax.plot(M_steps, M_lm, alpha=0.25, color='blue', linewidth=0.5)
if len(M_lm) > 50:
    w = 50
    smoothed = [np.mean(M_lm[max(0, i - w) : i + 1]) for i in range(len(M_lm))]
    ax.plot(M_steps, smoothed, color='blue', linewidth=1.5, label='smoothed (w=50)')
ax.set_xlabel('Step')
ax.set_ylabel('LM Loss')
ax.set_title(f'Training Loss (curriculum {TC.curriculum_ratio:.0%} from step 0)')
ax.grid(True, alpha=0.3)
ax.legend()

# 2. Eval loss + perplexity
ax = axes[0, 1]
if E_steps:
    ln1 = ax.plot(E_steps, E_loss, 'o-', color='red', markersize=4, label='loss')
    ax2 = ax.twinx()
    ln2 = ax2.plot(E_steps, E_ppl, 's--', color='orange', markersize=4,
                   alpha=0.7, label='ppl')
    ax2.set_ylabel('Perplexity', color='orange')
    lns = ln1 + ln2
    ax.legend(lns, [l.get_label() for l in lns], loc='upper right')
ax.set_xlabel('Step')
ax.set_ylabel('Loss', color='red')
ax.set_title('Eval Loss / Perplexity')
ax.grid(True, alpha=0.3)

# 3. Aux losses: entropy + key_align + val_coherence
ax = axes[0, 2]
ax.plot(M_steps, M_ge, alpha=0.6, color='green', label='H (entropy)')
ax.plot(M_steps, M_keyalign, alpha=0.7, color='blue', label='key_align')
ax.plot(M_steps, M_valcoh, alpha=0.7, color='orange', label='val_coherence')
ax.plot(M_steps, M_div, alpha=0.7, color='magenta', label='key_diversity')
ax.plot(M_steps, M_margin, alpha=0.4, color='red', ls=':', label='margin pen')
max_ent = math.log(6)
ax.axhline(y=TC.gate_entropy_min_H, color='gray', ls='--', alpha=0.5,
           label=f'H_min={TC.gate_entropy_min_H:.2f}')
ax.set_xlabel('Step')
ax.set_ylabel('Loss value')
ax.set_title('Aux Losses (key-query alignment, value coherence, entropy)')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7)

# 4. Gate distribution
ax = axes[1, 0]
gate_arr = np.array(M_gate)
g_names = ['state', 'ep_obj', 'conflict', 'archive', 'working', 'skip']
g_colors = ['#2196F3', '#4CAF50', '#F44336', '#FF9800', '#9C27B0', '#607D8B']
bottom = np.zeros(len(M_steps))
for i, (gn, gc) in enumerate(zip(g_names, g_colors)):
    ax.fill_between(M_steps, bottom, bottom + gate_arr[:, i],
                    alpha=0.7, color=gc, label=gn)
    bottom += gate_arr[:, i]
ax.set_ylim(0, 1.05)
ax.set_xlabel('Step')
ax.set_ylabel('Probability')
ax.set_title('Writer Gate Distribution')
ax.legend(fontsize=7, loc='upper right')
ax.grid(True, alpha=0.3)

# 5. Memory bank occupancy
ax = axes[1, 1]
bank_keys = list(M_occ[0].keys()) if M_occ else []
for bk, bc in zip(bank_keys, g_colors[:len(bank_keys)]):
    vals = [d.get(bk, 0) for d in M_occ]
    ax.plot(M_steps, vals, label=bk, color=bc, alpha=0.7)
ax.set_xlabel('Step')
ax.set_ylabel('Fraction occupied')
ax.set_title('Memory Bank Occupancy')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 6. Gradient norm
ax = axes[1, 2]
ax.plot(M_steps, M_gn, alpha=0.4, color='purple', linewidth=0.5)
if len(M_gn) > 50:
    sm_gn = [np.mean(M_gn[max(0, i - 50) : i + 1]) for i in range(len(M_gn))]
    ax.plot(M_steps, sm_gn, color='purple', linewidth=1.5)
ax.axhline(y=TC.grad_clip, color='red', ls='--', label=f'clip={TC.grad_clip}')
ax.set_xlabel('Step')
ax.set_ylabel('Norm')
ax.set_title('Gradient Norm (global)')
ax.legend()
ax.grid(True, alpha=0.3)

# 7. Learning rate
ax = axes[2, 0]
ax.plot(M_steps, M_lr, color='teal', label='LR')
ax.set_xlabel('Step')
ax.set_ylabel('LR')
ax.set_title(f'LR Schedule (curriculum={TC.curriculum_ratio:.0%} constant)')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7)

# 8. Throughput
ax = axes[2, 1]
ax.plot(M_steps, M_tps, alpha=0.4, color='brown', linewidth=0.5)
if M_tps:
    avg_tps = np.mean(M_tps)
    ax.axhline(y=avg_tps, color='red', ls='--', label=f'avg={avg_tps:.0f}')
ax.set_xlabel('Step')
ax.set_ylabel('tok/s')
ax.set_title('Training Throughput')
ax.legend()
ax.grid(True, alpha=0.3)

# 9. Same-batch ablation
ax = axes[2, 2]
bars = ax.bar(
    ['With Memory', 'Without Memory'],
    [loss_w_mem, loss_wo_mem],
    color=['#2196F3', '#F44336'],
)
ax.set_ylabel('Eval Loss')
ax.set_title(f'Same-Batch Ablation (delta={abl_delta:+.4f})')
for b, v in zip(bars, [loss_w_mem, loss_wo_mem]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
            f'{v:.4f}', ha='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# 10. Per-submodule grad norms
ax = axes[3, 0]
if M_subgrad:
    for gname in SUBMODULE_GROUPS:
        vals = [d.get(gname, 0.0) for d in M_subgrad]
        ax.plot(M_steps, vals, alpha=0.7, label=gname)
    ax.set_xlabel('Step')
    ax.set_ylabel('Grad L2 norm')
    ax.set_title('Per-Submodule Gradient Norms')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

# 11. Fusion block mem_gate
ax = axes[3, 1]
if M_fgate:
    fgate_arr = np.array(M_fgate)
    for i in range(fgate_arr.shape[1]):
        ax.plot(M_steps, fgate_arr[:, i], alpha=0.7, label=f'FusionBlock {i}')
    ax.set_xlabel('Step')
    ax.set_ylabel('sigmoid(mem_gate) mean')
    ax.set_title('Fusion Block Memory Gate')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

# 12. Conflict test summary
ax = axes[3, 2]
ax.axis('off')
conflict_text = "ConflictMemory Test Results\n" + "-" * 30 + "\n"
for k, v in conflict_results.items():
    conflict_text += f"{k}: {v}\n"
ax.text(0.05, 0.95, conflict_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'step2v5_training_report.png')
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"[INFO] Plots saved: {plot_path}")

# ======================== 17. SUMMARY REPORT ================================

report = {
    'project': 'D_Cortex v2.0-alpha',
    'step': 'Step 2 (v5): Training with writer gradient fix (key-query alignment + value coherence)',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'fixes_applied': [
        'FIX-1: total_memory (was total_mem)',
        'FIX-2: double-checkpoint guard',
        'FIX-3: margin entropy penalty (H_min=1.0)',
        'FIX-4: curriculum ramp (superseded by FIX-9)',
        'FIX-5: conflict test verifies diff-vector cosine',
        'FIX-6: episode_ssm in submodule gradient tracking',
        'FIX-7: _gate_accum cleared in ablation',
        'FIX-8: writer aux losses (key-query alignment + value coherence)',
        'FIX-9: memory-dominant curriculum 85% from step 0',
        'FIX-10: key diversity loss on state/episode_obj/working',
        'FIX-11: theta_match raised to 0.85',
        'FIX-12: MULTI-TURN curriculum (fact in Turn 1, probe in Turn 2)',
    ],
    'hardware': {
        'gpu': GPU_NAME,
        'vram_gb': round(GPU_MEM_GB, 1),
        'peak_vram_gb': round(peak_vram, 1),
        'dtype': str(DTYPE),
        'grad_scaler': USE_SCALER,
        'sdpa_patched': _SDPA_AVAILABLE,
    },
    'model': {
        'params_total': N_PARAMS,
        'params_trainable': N_TRAINABLE,
        'n_layers': cfg.n_layers,
        'hidden_dim': cfg.hidden_dim,
        'n_heads': cfg.n_heads,
        'n_fusion_layers': cfg.n_fusion_layers,
        'vocab_size': cfg.vocab_size,
    },
    'data': {
        'dataset': 'roneneldan/TinyStories + synthetic curriculum',
        'train_tokens': len(train_data),
        'val_tokens': len(val_data),
        'tokenizer': 'tiktoken gpt2',
        'curriculum_ratio': TC.curriculum_ratio,
    },
    'training': {
        'total_steps': TC.total_steps,
        'tokens_processed': tok_done,
        'time_seconds': round(total_time, 1),
        'avg_tok_per_sec': round(tok_done / total_time),
        'peak_vram_gb': round(peak_vram, 1),
        'init_val_loss': round(init_val, 4),
        'final_train_loss': round(M_lm[-1], 4) if M_lm else None,
        'best_val_loss': round(best_val, 4),
        'best_val_ppl': round(math.exp(min(best_val, 20.0)), 2),
    },
    'aux_losses': {
        'gate_entropy': {
            'H_min': TC.gate_entropy_min_H,
            'weight': TC.gate_entropy_w,
            'H_initial': round(M_ge[0], 4) if M_ge else None,
            'H_final': round(M_ge[-1], 4) if M_ge else None,
        },
        'key_query_alignment': {
            'weight': TC.w_key_align,
            'initial': round(M_keyalign[0], 4) if M_keyalign else None,
            'final': round(M_keyalign[-1], 4) if M_keyalign else None,
        },
        'value_coherence': {
            'weight': TC.w_val_coherence,
            'initial': round(M_valcoh[0], 4) if M_valcoh else None,
            'final': round(M_valcoh[-1], 4) if M_valcoh else None,
        },
        'key_diversity': {
            'weight': TC.w_diversity,
            'initial': round(M_div[0], 4) if M_div else None,
            'final': round(M_div[-1], 4) if M_div else None,
        },
    },
    'memory_ablation': {
        'method': 'same-batch short-term priming (seed=12345)',
        'n_eval_batches': N_EVAL_ABL,
        'n_populate_passes': N_POPULATE,
        'loss_with_memory': round(loss_w_mem, 4),
        'loss_without_memory': round(loss_wo_mem, 4),
        'delta': round(abl_delta, 4),
        'priming_helps': abl_delta > 0,
    },
    'conflict_memory_test': conflict_results,
    'submodule_grad_norms_final': M_subgrad[-1] if M_subgrad else {},
    'memory_snapshot_final': {
        k: {'occupied': v['occupied'], 'capacity': v['capacity']}
        for k, v in snap_final.items()
    },
}

report_path = os.path.join(RESULTS_DIR, 'step2v5_training_report.json')
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, default=str)

print(SEP)
print("D_CORTEX v2.0-alpha -- STEP 2 v5 REPORT")
print(SEP)
print(f"  Fixes applied   : 12 (FIX-1..11 from v2/v3/v4 + FIX-12 multi-turn)")
print(f"  Model           : {N_PARAMS/1e6:.2f}M params | "
      f"{cfg.n_layers}L/{cfg.hidden_dim}d/{cfg.n_heads}h")
print(f"  GPU             : {GPU_NAME} ({DTYPE}) | peak VRAM {peak_vram:.1f} GB")
print(f"  Training        : {TC.total_steps} steps | {tok_done:,} tokens | "
      f"{total_time:.0f}s ({total_time/60:.1f}min)")
print(f"  Throughput      : {tok_done/total_time:,.0f} tok/s")
print(f"  Curriculum      : {TC.curriculum_ratio:.0%} MULTI-TURN from step 0")
print(f"  theta_match     : 0.85")
print(f"  Init val loss   : {init_val:.4f}")
final_lm = M_lm[-1] if M_lm else float('nan')
print(f"  Final train loss: {final_lm:.4f}")
print(f"  Best val loss   : {best_val:.4f} (ppl={math.exp(min(best_val,20)):.2f})")
if M_ge:
    print(f"  Gate entropy    : {M_ge[0]:.4f} -> {M_ge[-1]:.4f}")
if M_keyalign:
    print(f"  Key-query align : {M_keyalign[0]:.4f} -> {M_keyalign[-1]:.4f}")
if M_valcoh:
    print(f"  Value coherence : {M_valcoh[0]:.4f} -> {M_valcoh[-1]:.4f}")
if M_div:
    print(f"  Key diversity   : {M_div[0]:.4f} -> {M_div[-1]:.4f}")
print(f"  Ablation delta  : {abl_delta:+.4f}")
print(f"  Conflict test   : mechanism_functional="
      f"{conflict_results.get('mechanism_functional', 'N/A')}")
print(f"  Bank occupancy  :")
for name, info in snap_final.items():
    print(f"    {name:12s}: {info['occupied']:3d}/{info['capacity']:3d}")
if M_subgrad:
    dead = [k for k, v in M_subgrad[-1].items() if v == 0.0]
    if dead:
        print(f"  [WARN] Dead submodules (zero grad): {dead}")
    else:
        print(f"  Grad flow       : all {len(SUBMODULE_GROUPS)} submodules nonzero")
print(f"  Report JSON     : {report_path}")
print(f"  Plots PNG       : {plot_path}")
print(SEP)
print("v5 KEY CHANGE: MULTI-TURN CURRICULUM [FIX-12]")
print("  v2/v3/v4: fact + question in SAME 1024-token context window.")
print("            Model solved via in-context pattern matching.")
print("            Memory was structurally unnecessary.")
print()
print("  v5: fact injected in Turn 1, question asked in Turn 2.")
print("      At Turn 2, fact text is NOT in the input.")
print("      Memory is the ONLY path to the correct answer.")
print("      Gradient: LM loss -> fusion -> cross-attn -> readers -> query_engine")
print()
print("WHAT TO LOOK FOR:")
print("  1. query_engine and readers grad norms >> 0 (were ~0 in v2/v3/v4)")
print("  2. Fusion mem_gate moves AWAY from 0.5 (memory becomes useful)")
print("  3. Bank occupancy > 2 per bank")
print("  4. Same-batch ablation delta > 0")
print("  5. Step 3 Suite A delta > 0 (memory helps fact recall)")
print("  6. Gate distribution becomes non-uniform (specialization)")
print(SEP)
print("STATUS: READY TO RUN (fresh training from step 0)")
print(SEP)
