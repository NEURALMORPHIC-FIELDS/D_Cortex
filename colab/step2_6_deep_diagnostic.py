# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha -- Step 2.5: Ablation on trained checkpoint
# Tests: normal memory vs zero memory vs permuted memory
# Tells us whether decoder uses retrieved_value at all
# Patent EP25216372.0.

import os, sys, time, json, math, random, subprocess, io, contextlib
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

PROJECT_ROOT = '/content/drive/MyDrive/dcortex_v2'
for d in ['checkpoints', 'results']:
    os.makedirs(os.path.join(PROJECT_ROOT, d), exist_ok=True)

import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[INFO] GPU: {gpu_name} | {vram_gb:.1f} GB")
# ======================== 3. INLINE SOURCE ================================
SRC_DIR = "/content/dcortex_src"
_SOURCE_FILES = {
    "dcortex/__init__.py": r'''"""D_Cortex v2.0-alpha -- dual-agent memory-native transformer."""

from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model
from dcortex.encoder import MemoryEncoder

__all__ = ["DCortexConfig", "DCortexV2Model", "MemoryEncoder"]
__version__ = "2.0.0-alpha"
''',
    "dcortex/config.py": r'''# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha (dual-agent architecture)
# Configuration dataclass. Patent EP25216372.0.

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DCortexConfig:
    """Configuration for D_Cortex v2.0-alpha dual-agent architecture.

    Two separate agents meet ONLY through memory banks:
        Encoder: sees facts, writes to memory. Own embeddings + blocks.
        Decoder: sees questions, reads from memory. Own embeddings + blocks.
    No weight sharing. Memory is the only bridge.
    """

    # --- Shared dims (must match for memory bank compatibility) ---
    vocab_size: int = 50257
    hidden_dim: int = 768
    max_seq_len: int = 2048
    dropout: float = 0.0

    # --- Encoder (fact processor, memory writer) ---
    n_enc_layers: int = 4
    n_enc_heads: int = 12
    enc_ff_dim: int = 3072

    # --- Decoder (question processor, memory reader, language producer) ---
    n_dec_layers: int = 12
    n_dec_heads: int = 12
    dec_ff_dim: int = 3072
    n_fusion_layers: int = 4

    # --- Memory bank capacities ---
    n_state_slots: int = 64
    n_episode_obj_slots: int = 128
    n_conflict_slots: int = 32
    n_archive_slots: int = 512
    n_work_slots: int = 16

    # --- Episode SSM ---
    ssm_hidden_dim: int = 256

    # --- Latent key dims ---
    d_ent: int = 128
    d_rel: int = 64
    d_typ: int = 64

    # --- Query similarity weights ---
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
        if self.hidden_dim % self.n_enc_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"n_enc_heads ({self.n_enc_heads})"
            )
        if self.hidden_dim % self.n_dec_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"n_dec_heads ({self.n_dec_heads})"
            )
        if self.n_fusion_layers > self.n_dec_layers:
            raise ValueError(
                f"n_fusion_layers ({self.n_fusion_layers}) must be "
                f"<= n_dec_layers ({self.n_dec_layers})"
            )
        if self.n_fusion_layers < 1:
            raise ValueError("n_fusion_layers must be >= 1")

    @property
    def n_dec_standard_layers(self) -> int:
        """Decoder standard blocks before fusion blocks."""
        return self.n_dec_layers - self.n_fusion_layers

    @property
    def n_heads(self) -> int:
        """Backward compat for modules that read config.n_heads."""
        return self.n_dec_heads

    @property
    def n_layers(self) -> int:
        """Backward compat for modules that read config.n_layers."""
        return self.n_dec_layers

    @property
    def ff_dim(self) -> int:
        """Backward compat for modules that read config.ff_dim."""
        return self.dec_ff_dim

    @property
    def n_standard_layers(self) -> int:
        """Backward compat."""
        return self.n_dec_standard_layers

    def small_test(self) -> "DCortexConfig":
        """Tiny config for unit tests."""
        return DCortexConfig(
            vocab_size=256,
            hidden_dim=64,
            max_seq_len=64,
            n_enc_layers=2,
            n_enc_heads=4,
            enc_ff_dim=128,
            n_dec_layers=4,
            n_dec_heads=4,
            dec_ff_dim=128,
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
    "dcortex/encoder.py": r'''# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha (dual-agent architecture)
# MemoryEncoder: Agent A. Sees facts, writes to memory banks.
# Has own embeddings, own transformer blocks, own writer.
# Does NOT read memory. Does NOT produce language.
# Meets the Decoder ONLY through the memory banks.
# Patent EP25216372.0.

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
from dcortex.memory.query import QueryEngine
from dcortex.memory.updater import MemoryUpdater
from dcortex.memory.writer import MemoryWriter


class EncoderBlock(nn.Module):
    """Pre-norm transformer block for the encoder.

    Identical structure to StandardTransformerBlock but parameterized
    independently (own n_heads, ff_dim) so encoder and decoder have
    separate capacity.
    """

    def __init__(self, hidden_dim: int, n_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = _EncoderMHSA(hidden_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h + self.attn(self.norm1(h))
        h = h + self.ff(self.norm2(h))
        return h


class _EncoderMHSA(nn.Module):
    """Non-causal multi-head self-attention for the encoder.

    The encoder processes facts as a whole, not autoregressively.
    No causal mask needed: the encoder sees the entire fact at once.
    """

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, T, D = h.shape
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = attn @ v

        return self.out(out.transpose(1, 2).reshape(B, T, D))


class MemoryEncoder(nn.Module):
    """Agent A: fact processor and memory writer.

    Forward flow:
        1. Embed fact tokens          h <- Embed(input_ids)     [B, T, D]
        2. Encoder blocks             h <- EncoderBlocks(h)
        3. Pool                       h_pool <- mean(h)         [B, D]
        4. Write to memory            Writer(h_pool, updater, banks)
        5. Advance EpisodeSSM         ssm(h_pool)

    The encoder has:
        - Own token embeddings (NOT shared with decoder)
        - Own transformer blocks (NOT shared with decoder)
        - Own query engine for write keys
        - Writer + Updater

    The encoder does NOT have:
        - LM head (does not produce language)
        - FusionBlocks (does not read memory)
        - Readers (does not query memory)

    It meets the Decoder ONLY through the memory bank buffers.
    """

    def __init__(
        self,
        config: DCortexConfig,
        shared_token_emb: nn.Embedding,
        shared_pos_emb: nn.Embedding,
        shared_query_engine: 'QueryEngine',
        shared_address_encoder: 'nn.Module',
    ) -> None:
        super().__init__()
        self.config = config
        D = config.hidden_dim

        # SHARED embeddings (same as decoder, prevents semantic drift)
        self.token_emb = shared_token_emb
        self.pos_emb = shared_pos_emb
        self.emb_norm = nn.LayerNorm(D)
        self.emb_drop = nn.Dropout(config.dropout)

        # Own transformer blocks (separate processing - for VALUE extraction)
        self.blocks = nn.ModuleList([
            EncoderBlock(D, config.n_enc_heads, config.enc_ff_dim, config.dropout)
            for _ in range(config.n_enc_layers)
        ])

        # SHARED query engine (same key space as decoder readers)
        self.query_engine = shared_query_engine

        # SHARED address encoder (same address space for keys and queries)
        self.address_encoder = shared_address_encoder

        # Own write infrastructure (writer uses shared query engine + shared address)
        self.writer = MemoryWriter(config, shared_query_engine=shared_query_engine)
        self.updater = MemoryUpdater(config)
        self.episode_ssm = EpisodeSSM(D, config.ssm_hidden_dim)

        self.final_norm = nn.LayerNorm(D)

    def forward(
        self,
        input_ids: torch.Tensor,
        banks: Dict[str, MemoryBank],
        step: int,
    ) -> Dict[str, torch.Tensor]:
        """Process facts and write to memory.

        Args:
            input_ids: [B, T] fact tokens.
            banks: dict of memory banks (shared with decoder).
            step: global step counter.

        Returns:
            Dict with grad-carrying aux tensors for training losses.
        """
        B, T = input_ids.shape

        # 1. Embed (raw, before encoder blocks)
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        emb_raw = self.token_emb(input_ids) + self.pos_emb(positions)

        # 2. ADDRESS CODE from shared address encoder (operates on raw embeddings)
        # SAME function applied to SAME embeddings as decoder reader.
        # Key generated from this is GUARANTEED structurally compatible.
        addr_code = self.address_encoder(emb_raw)                    # [B, D]

        # 3. Encoder blocks for VALUE extraction (contextual)
        h = self.emb_norm(emb_raw)
        h = self.emb_drop(h)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        h_pool = h.mean(dim=1)                                       # [B, D]

        # 4. Query engine outputs (for diagnostics; same projection as keys)
        q_ent, q_rel, q_typ = self.query_engine(addr_code)

        # 5. Write: keys from addr_code, value from h_pool
        write_out = self.writer(h_pool, addr_code, self.updater, banks, step)

        # 6. Advance EpisodeSSM
        self.episode_ssm(h_pool)

        return {
            'gate_probs': write_out['gate_probs'],
            'w_value': write_out['value'],
            'w_k_ent': write_out['k_ent'],
            'w_k_rel': write_out['k_rel'],
            'w_k_typ': write_out['k_typ'],
            'q_ent': q_ent,
            'q_rel': q_rel,
            'q_typ': q_typ,
            'h_pool': h_pool,
            'addr_code': addr_code,
            'slot_writes': write_out['slot_writes'],
        }

    def reset(self) -> None:
        """Reset encoder-owned state (EpisodeSSM)."""
        self.episode_ssm.reset()
''',
    "dcortex/shared_address.py": r'''# -*- coding: utf-8 -*-
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
''',
    "dcortex/aux_modules.py": r'''# -*- coding: utf-8 -*-
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
    """Slot-based memory with differentiable overlay for gradient flow.

    Keys and values are stored as buffers (persistent, no grad).
    During a training episode, the writer additionally stores grad-carrying
    tensors in an overlay dict. Readers use get_diff_*() methods to get
    tensors that combine buffer (no grad) and overlay (with grad).
    After backward(), clear_overlay() detaches everything.

    This enables gradient from decoder loss to flow through memory values
    back to the encoder's writer heads, without requiring persistent
    computation graphs across episodes.
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

        self.register_buffer("k_ent", torch.zeros(capacity, d_ent))
        self.register_buffer("k_rel", torch.zeros(capacity, d_rel))
        self.register_buffer("k_typ", torch.zeros(capacity, d_typ))
        self.register_buffer("values", torch.zeros(capacity, hidden_dim))
        self.register_buffer("occupied", torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("usage", torch.zeros(capacity))
        self.register_buffer(
            "last_write_step",
            torch.full((capacity,), -1, dtype=torch.long),
        )

        # Differentiable overlay: {slot_idx: {value, k_ent, k_rel, k_typ}}
        # Populated by writer WITH grad, used by reader for gradient flow.
        self._overlay: dict = {}

    def set_overlay(
        self,
        idx: int,
        value: torch.Tensor,
        k_ent: torch.Tensor,
        k_rel: torch.Tensor,
        k_typ: torch.Tensor,
    ) -> None:
        """Store grad-carrying tensors for current episode."""
        self._overlay[idx] = {
            'value': value, 'k_ent': k_ent, 'k_rel': k_rel, 'k_typ': k_typ,
        }

    def clear_overlay(self) -> None:
        """Remove all overlay entries. Call after backward()."""
        self._overlay.clear()

    def get_diff_values(self) -> torch.Tensor:
        """Return [C, D] values with overlay rows carrying grad."""
        if not self._overlay:
            return self.values
        rows = []
        for i in range(self.capacity):
            if i in self._overlay:
                rows.append(self._overlay[i]['value'])
            else:
                rows.append(self.values[i])
        return torch.stack(rows)

    def get_diff_k_ent(self) -> torch.Tensor:
        if not self._overlay:
            return self.k_ent
        rows = []
        for i in range(self.capacity):
            if i in self._overlay:
                rows.append(self._overlay[i]['k_ent'])
            else:
                rows.append(self.k_ent[i])
        return torch.stack(rows)

    def get_diff_k_rel(self) -> torch.Tensor:
        if not self._overlay:
            return self.k_rel
        rows = []
        for i in range(self.capacity):
            if i in self._overlay:
                rows.append(self._overlay[i]['k_rel'])
            else:
                rows.append(self.k_rel[i])
        return torch.stack(rows)

    def get_diff_k_typ(self) -> torch.Tensor:
        if not self._overlay:
            return self.k_typ
        rows = []
        for i in range(self.capacity):
            if i in self._overlay:
                rows.append(self._overlay[i]['k_typ'])
            else:
                rows.append(self.k_typ[i])
        return torch.stack(rows)

    def reset(self) -> None:
        """Clear all slots and overlay."""
        self.k_ent.zero_()
        self.k_rel.zero_()
        self.k_typ.zero_()
        self.values.zero_()
        self.occupied.zero_()
        self.usage.zero_()
        self.last_write_step.fill_(-1)
        self._overlay.clear()

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

        # Readout buffer: updated after each forward(), readable by decoder
        # without gradient flow through encoder parameters.
        self.register_buffer("readout", torch.zeros(input_dim))

    def reset(self) -> None:
        self.x.zero_()
        self.readout.zero_()

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
        x_new = a * self.x.detach() + drive          # [state_dim]
        self.x.data = x_new.detach()
        r = self.C(x_new)                            # [input_dim]
        # Store readout as buffer for decoder (no grad)
        self.readout.data = r.detach()
        return r                                     # [input_dim]

    def get_readout(self) -> torch.Tensor:
        """Return the last readout as a detached buffer.

        Used by the decoder to read SSM state without gradient flow
        through encoder parameters. Returns [input_dim].
        """
        return self.readout.detach()
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

        attn = F.softmax(sim, dim=-1)                             # [B, C]
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

    def __init__(self, config: DCortexConfig, shared_query_engine: nn.Module) -> None:
        super().__init__()
        self.config = config

        self.gate = nn.Linear(config.hidden_dim, 6)

        # Value projection
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # SHARED query engine produces keys for write AND queries for read.
        # Writer no longer has own key heads. Same projections as reader.
        # k_ent = shared_query_engine.proj_ent(h)
        # q_ent = shared_query_engine.proj_ent(h)  <-- same projection
        self.query_engine = shared_query_engine

        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        h_pool: torch.Tensor,
        addr_code: torch.Tensor,
        updater: MemoryUpdater,
        banks: Dict[str, MemoryBank],
        step: int,
        force_write: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Route writes through the updater.

        Args:
            h_pool:    [B, hidden_dim] contextual encoder output (for value).
            addr_code: [B, hidden_dim] address code from SharedAddressEncoder
                       (for key generation - SAME function as reader queries).
            updater, banks, step, force_write: as before.

        Returns aux dict including 'slot_writes': list of (bank_name, slot_idx)
        per batch element, in batch order.
        """
        if h_pool.dim() != 2 or addr_code.dim() != 2:
            raise ValueError(
                f"Expected [B,D] for h_pool and addr_code, got {tuple(h_pool.shape)} and {tuple(addr_code.shape)}"
            )

        h_norm = self.norm(h_pool)
        gate_logits = self.gate(h_norm)
        gate_probs = F.softmax(gate_logits, dim=-1)

        value = self.value_head(h_norm)

        # Keys from SHARED query engine applied to ADDRESS code
        # (SAME projection AND SAME input function as reader queries)
        k_ent, k_rel, k_typ = self.query_engine(addr_code)

        # Hard routing per batch (force_write excludes skip)
        bank_probs = gate_probs[:, :5]
        choices = bank_probs.argmax(dim=-1)
        B = h_pool.shape[0]

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
        force_attend: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            h: [B, T, D]
            memory: [B, K, D]
            attention_mask: [B, T] (1=valid) or None
            force_attend: if True, bypass mem_gate (full memory contribution).
                          Prevents decoder from learning to ignore memory
                          during curriculum episodes.
        """
        # Self-attention
        h = h + self.self_attn(self.norm_self(h), attention_mask)

        # Cross-attention to memory tokens
        m = self.cross_attn(self.norm_h(h), self.norm_mem(memory))
        if force_attend:
            h = h + m                                                # no gate
        else:
            gate = torch.sigmoid(self.mem_gate)                      # [D]
            h = h + gate * m                                         # gated

        # FFN
        h = h + self.ff(self.norm_ff(h))
        return h
''',
    "dcortex/model.py": r'''# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v2.0-alpha (dual-agent architecture)
# DCortexV2Model: two agents that meet ONLY through memory banks.
#
# Agent A (MemoryEncoder): sees facts, writes to memory.
#   Own embeddings, own transformer blocks, own writer, own query engine.
#   Does NOT read memory. Does NOT produce language.
#
# Agent B (Decoder): sees questions, reads from memory, produces language.
#   Own embeddings, own transformer blocks (standard + fusion), own readers,
#   own query engine, own LM head.
#   Does NOT write to memory. Does NOT see fact text.
#
# The ONLY connection is the memory bank buffer tensors.
# No weight sharing. No hidden state sharing. No gradient shortcut.
#
# Patent EP25216372.0.

from typing import Dict, Optional

import torch
import torch.nn as nn

from dcortex.backbone.embeddings import TokenEmbeddings
from dcortex.backbone.fusion_block import FusionBlock
from dcortex.backbone.transformer import StandardTransformerBlock
from dcortex.config import DCortexConfig
from dcortex.encoder import MemoryEncoder
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
    """Dual-agent memory-native transformer.

    Usage:
        # Agent A writes facts to memory:
        enc_aux = model.encode(fact_ids)

        # Agent B reads memory and answers:
        logits = model.decode(question_ids)

    The encode() and decode() methods use SEPARATE neural networks.
    They share NOTHING except the memory bank buffers.
    """

    def __init__(self, config: DCortexConfig) -> None:
        super().__init__()
        self.config = config

        # ================= SHARED MEMORY BANKS (buffers only) =================
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

        # ================= SHARED SEMANTIC INFRASTRUCTURE =====================
        # Shared token + position embeddings: encoder and decoder see the same
        # latent alphabet. "cat" means the same vector for both agents.
        self.shared_token_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.shared_pos_emb = nn.Embedding(config.max_seq_len, config.hidden_dim)
        nn.init.normal_(self.shared_token_emb.weight, std=config.init_std)
        nn.init.normal_(self.shared_pos_emb.weight, std=config.init_std)

        # Shared query engine: writer keys and reader queries live in the
        # same geometric space.
        self.shared_query_engine = QueryEngine(config)

        # Shared address encoder: produces address codes from raw embeddings.
        # Same function applied by writer (for keys) and reader (for queries).
        # GUARANTEES address compatibility structurally at initialization.
        from dcortex.shared_address import SharedAddressEncoder
        self.shared_address_encoder = SharedAddressEncoder(config)

        # Auxiliary heads: direct retrieval -> answer and retrieval -> key cycle
        from dcortex.aux_modules import AuxAnswerHead, ValueToKeyProjector
        self.aux_answer_head = AuxAnswerHead(config, self.shared_token_emb)
        self.value_to_key_proj = ValueToKeyProjector(config)

        # ================= AGENT A: ENCODER (writes memory) ===================
        self.encoder = MemoryEncoder(
            config,
            shared_token_emb=self.shared_token_emb,
            shared_pos_emb=self.shared_pos_emb,
            shared_query_engine=self.shared_query_engine,
            shared_address_encoder=self.shared_address_encoder,
        )

        # ================= AGENT B: DECODER (reads memory) ====================
        # Decoder embeddings use same token_emb + pos_emb
        self.dec_emb_norm = nn.LayerNorm(config.hidden_dim)
        self.dec_emb_drop = nn.Dropout(config.dropout)

        # Own standard blocks (separate processing from encoder)
        self.dec_standard_blocks = nn.ModuleList([
            StandardTransformerBlock(config)
            for _ in range(config.n_dec_standard_layers)
        ])

        # Own readers (but use shared_query_engine for queries)
        self.dec_state_reader = SemanticReader(config)
        self.dec_episode_reader = EpisodeReader(config)
        self.dec_conflict_reader = SemanticReader(config)
        self.dec_archive_reader = SemanticReader(config)
        self.dec_working_reader = SemanticReader(config)
        self.dec_read_fusion = MemoryReadFusion(config)

        # Own fusion blocks
        self.dec_fusion_blocks = nn.ModuleList([
            FusionBlock(config)
            for _ in range(config.n_fusion_layers)
        ])

        # Own LM head (tied to shared embeddings)
        self.dec_final_norm = nn.LayerNorm(config.hidden_dim)
        self.dec_lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.dec_lm_head.weight = self.shared_token_emb.weight

        # ================= CONSOLIDATOR =======================================
        self.consolidator = MemoryConsolidator(config)

        # ================= GLOBAL STATE =======================================
        self.register_buffer("step_counter", torch.zeros((), dtype=torch.long))
        self._enc_aux: Dict[str, torch.Tensor] = {}

        # Init
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
        enc_p = sum(p.numel() for p in self.encoder.parameters())
        dec_p = sum(
            p.numel() for n, p in self.named_parameters()
            if n.startswith('dec_')
        )
        total = sum(p.numel() for p in self.parameters())
        sep = "=" * 70
        print(sep)
        print("[INFO] D_Cortex v2.0-alpha (DUAL-AGENT) instantiated")
        print(sep)
        print(f"  ENCODER (Agent A, writes memory):")
        print(f"    layers={cfg.n_enc_layers}  heads={cfg.n_enc_heads}  "
              f"ff={cfg.enc_ff_dim}  params={enc_p/1e6:.2f}M")
        print(f"  DECODER (Agent B, reads memory, produces language):")
        print(f"    layers={cfg.n_dec_layers} ({cfg.n_dec_standard_layers} std + "
              f"{cfg.n_fusion_layers} fusion)  heads={cfg.n_dec_heads}  "
              f"ff={cfg.dec_ff_dim}  params={dec_p/1e6:.2f}M")
        print(f"  SHARED semantic infrastructure:")
        print(f"    token_emb + pos_emb: {sum(p.numel() for p in [self.shared_token_emb.weight, self.shared_pos_emb.weight])/1e6:.2f}M")
        print(f"    query_engine: {sum(p.numel() for p in self.shared_query_engine.parameters())/1e6:.2f}M")
        print(f"  SHARED: memory banks (buffers)")
        print(f"  memory banks : state={cfg.n_state_slots}  "
              f"episode_obj={cfg.n_episode_obj_slots}  "
              f"conflict={cfg.n_conflict_slots}  "
              f"archive={cfg.n_archive_slots}  "
              f"working={cfg.n_work_slots}")
        print(f"  episode SSM  : state_dim={cfg.ssm_hidden_dim} (owned by encoder)")
        print(f"  latent keys  : ent={cfg.d_ent}  rel={cfg.d_rel}  typ={cfg.d_typ}")
        print(f"  thresholds   : match={cfg.theta_match}  conflict={cfg.theta_conflict}")
        print(f"  total params : {total/1e6:.2f}M")
        print(sep)

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
        return {name: bank.snapshot() for name, bank in self._bank_dict().items()}

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------

    def reset_memory(self) -> None:
        """Clear all memory banks, overlays, and encoder SSM state."""
        for bank in self._bank_dict().values():
            bank.reset()
        self.encoder.reset()
        self.step_counter.zero_()
        print("[INFO] Memory reset: all banks cleared, SSM zeroed, step=0")

    def begin_episode(self) -> None:
        """Clear overlays. Call before each multi-turn training episode."""
        for bank in self._bank_dict().values():
            bank.clear_overlay()

    def clear_overlays(self) -> None:
        """Clear all overlays. Call after backward() to detach the graph."""
        for bank in self._bank_dict().values():
            bank.clear_overlay()

    def consolidate(self) -> Dict[str, Dict[str, int]]:
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
    # ENCODE (Agent A): see facts, write to memory
    # ------------------------------------------------------------------

    def encode(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Agent A: process fact tokens and write to memory banks.

        Args:
            input_ids: [B, T] fact token ids.

        Returns:
            Dict of aux tensors with gradients (for encoder training losses).
        """
        if input_ids.dim() != 2:
            raise ValueError(f"encode expects [B, T], got {tuple(input_ids.shape)}")

        self.step_counter += 1
        step = int(self.step_counter.item())

        self._enc_aux = self.encoder(input_ids, self._bank_dict(), step)
        return self._enc_aux

    # ------------------------------------------------------------------
    # DECODE (Agent B): see question, read memory, produce language
    # ------------------------------------------------------------------

    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        force_attend: bool = False,
        return_retrieved: bool = False,
    ) -> torch.Tensor:
        """Agent B: process question tokens, read memory, produce logits.

        Args:
            input_ids:      [B, T] question token ids.
            attention_mask: [B, T] (1=valid, 0=pad) or None.
            force_attend:   if True, fusion blocks bypass mem_gate.
            return_retrieved: if True, returns (logits, retrieved_value).
        """
        if input_ids.dim() != 2:
            raise ValueError(f"decode expects [B, T], got {tuple(input_ids.shape)}")

        # 1. Shared embeddings (raw, before decoder blocks)
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        emb_raw = self.shared_token_emb(input_ids) + self.shared_pos_emb(positions)

        # 2. ADDRESS CODE from shared address encoder (SAME function as writer)
        # This is the structural guarantee of address compatibility.
        addr_code = self.shared_address_encoder(emb_raw)             # [B, D]

        # 3. Decoder embeddings continue through normal path
        h = self.dec_emb_norm(emb_raw)
        h = self.dec_emb_drop(h)

        for block in self.dec_standard_blocks:
            h = block(h, attention_mask)

        h_pool = self._pool(h, attention_mask)
        # Query from ADDRESS CODE (same function as writer's keys, structural guarantee)
        q_ent, q_rel, q_typ = self.shared_query_engine(addr_code)

        r_state = self.dec_state_reader(q_ent, q_rel, q_typ, self.state_mem)
        r_episode = self.dec_episode_reader(
            q_ent, q_rel, q_typ,
            self.episode_obj_mem, self.encoder.episode_ssm,
            ssm_input=h_pool,
            ssm_readout=self.encoder.episode_ssm.get_readout(),
        )
        r_conflict = self.dec_conflict_reader(q_ent, q_rel, q_typ, self.conflict_mem)
        r_archive = self.dec_archive_reader(q_ent, q_rel, q_typ, self.archive_mem)
        r_working = self.dec_working_reader(q_ent, q_rel, q_typ, self.working_mem)

        memory_tokens = self.dec_read_fusion(
            r_state, r_episode, r_conflict, r_archive, r_working,
        )

        # retrieved_value: pooled memory for aux head + cycle loss
        retrieved_value = memory_tokens.mean(dim=1)                   # [B, D]

        for block in self.dec_fusion_blocks:
            h = block(h, memory_tokens, attention_mask, force_attend=force_attend)

        h = self.dec_final_norm(h)
        logits = self.dec_lm_head(h)

        if return_retrieved:
            return logits, retrieved_value
        return logits

    # ------------------------------------------------------------------
    # BACKWARD COMPAT: forward() for single-turn usage
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        write_memory: bool = True,
    ) -> torch.Tensor:
        """Backward-compatible single-turn forward.

        When write_memory=True: acts as encoder+decoder on same input.
        When write_memory=False: acts as decoder-only (reads existing memory).

        For proper dual-agent usage, call encode() and decode() separately.
        """
        if write_memory:
            self.encode(input_ids)
        return self.decode(input_ids, attention_mask)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pool(
        h: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is None:
            return h.mean(dim=1)
        mask = attention_mask.float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (h * mask).sum(dim=1) / denom
''',
}
def write_source():
    for fpath, content in _SOURCE_FILES.items():
        full = os.path.join(SRC_DIR, fpath)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f: f.write(content)
    print(f"[INFO] {len(_SOURCE_FILES)} source files written to {SRC_DIR}")
write_source()
if SRC_DIR not in sys.path: sys.path.insert(0, SRC_DIR)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tiktoken", "datasets", "matplotlib"], check=True)
print("[INFO] Dependencies installed")
# ======================== 4. IMPORTS ========================================

import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model
from dcortex.backbone.transformer import MultiHeadSelfAttention
from dcortex.backbone.fusion_block import CrossAttention

print("[INFO] Imports OK")

if hasattr(F, 'scaled_dot_product_attention'):
    def _sdpa_self(self, h, attention_mask=None):
        B, T, D = h.shape
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_mask = None
        if attention_mask is not None:
            pad = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            causal = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), 1)
            combined = causal.unsqueeze(0).unsqueeze(0) | pad
            attn_mask = torch.zeros(B, 1, T, T, device=h.device, dtype=q.dtype)
            attn_mask.masked_fill_(combined, float("-inf"))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
            dropout_p=0.0, is_causal=(attention_mask is None))
        return self.out(out.transpose(1, 2).reshape(B, T, D))
    def _sdpa_cross(self, h, memory):
        B, T, D = h.shape; _, K, _ = memory.shape
        q = self.q(h).reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(memory).reshape(B, K, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        return self.out(out.transpose(1, 2).reshape(B, T, D))
    MultiHeadSelfAttention.forward = _sdpa_self
    CrossAttention.forward = _sdpa_cross

ENC = tiktoken.get_encoding("gpt2")
EOT = ENC.eot_token
SEQ_LEN = 64

def _pad(ids, length=SEQ_LEN):
    if len(ids) > length: return ids[:length]
    return ids + [EOT] * (length - len(ids))

cfg = DCortexConfig()
model = DCortexV2Model(cfg).to(DEVICE)
model.eval()

ckpts = sorted(Path(os.path.join(PROJECT_ROOT, 'checkpoints')).glob('ckpt_step*.pt'),
               key=lambda p: int(p.stem.split('step')[1]))
if not ckpts: raise RuntimeError("No checkpoint.")
ckpt = torch.load(ckpts[-1], map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model'])
ckpt_step = ckpt['step']
print(f"[INFO] Loaded: {ckpts[-1].name} (step {ckpt_step})")

_ENTITIES = ["cat","dog","bird","fish","rabbit","horse","bear","fox",
             "lion","tiger","monkey","penguin","owl","wolf","deer",
             "dragon","knight","wizard","princess","fairy","goblin","witch",
             "pirate","giant","ghost","robot","queen","king","dwarf","elf"]
_COLORS = ["red","blue","green","yellow","black","white","brown","pink",
           "orange","purple","golden","silver","crimson","gray","violet"]

# First token ids for each color (with leading space as expected in answer)
COLOR_TOKENS = {c: ENC.encode_ordinary(f" {c}")[0] for c in _COLORS}
COLOR_TOKEN_IDS = list(COLOR_TOKENS.values())
print(f"[INFO] {len(COLOR_TOKEN_IDS)} color tokens tracked")


def gen_ep(n_facts=4):
    ents = random.sample(_ENTITIES, n_facts)
    cols = random.sample(_COLORS, n_facts)
    target = random.randint(0, n_facts - 1)
    fact_texts = [f"The {e} is {c}." for e, c in zip(ents, cols)]
    prompt = f"What color is the {ents[target]}? The {ents[target]} is"
    answer = f" {cols[target]}"
    return fact_texts, prompt, answer, target, ents, cols


@torch.no_grad()
def encode_facts(fact_texts):
    slots = []
    for f in fact_texts:
        ids = _pad(ENC.encode_ordinary(f) + [EOT])
        xf = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            aux = model.encode(xf)
        slots.append(aux['slot_writes'][0])
    return slots


# ======================== TEST 1: Aux head direct accuracy ==================

print(f"\n{'='*70}")
print("TEST 1: Aux head direct top1 on answer token")
print(f"{'='*70}")
print("  Question: does retrieved_value contain the answer?")

N = 500
random.seed(42)

aux_correct = 0
decode_correct = 0
both_correct = 0
aux_in_colors = 0
decode_in_colors = 0

aux_rank_in_colors = []
decode_rank_in_colors = []

for i in range(N):
    facts, prompt, answer, target, _, cols = gen_ep(4)
    with contextlib.redirect_stdout(io.StringIO()): model.reset_memory()
    encode_facts(facts)
    target_tok = ENC.encode_ordinary(answer)[0]

    p_ids = ENC.encode_ordinary(prompt)
    xp = torch.tensor([_pad(p_ids)], dtype=torch.long, device=DEVICE)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, retrieved_value = model.decode(xp, return_retrieved=True)
        aux_logits = model.aux_answer_head(retrieved_value).float()
    
    # Aux head prediction
    aux_pred = aux_logits.argmax(dim=-1).item()
    if aux_pred == target_tok: aux_correct += 1
    if aux_pred in COLOR_TOKEN_IDS: aux_in_colors += 1
    
    # Decode prediction
    pred_pos = min(len(p_ids) - 1, SEQ_LEN - 1)
    dec_pred = logits[0, pred_pos, :].float().argmax().item()
    if dec_pred == target_tok: decode_correct += 1
    if dec_pred in COLOR_TOKEN_IDS: decode_in_colors += 1
    
    if aux_pred == target_tok and dec_pred == target_tok: both_correct += 1

    # Rank within colors only
    color_logits_aux = aux_logits[0, COLOR_TOKEN_IDS]
    target_in_colors_idx = COLOR_TOKEN_IDS.index(target_tok)
    sorted_color = color_logits_aux.argsort(descending=True).tolist()
    aux_rank_in_colors.append(sorted_color.index(target_in_colors_idx) + 1)

    color_logits_dec = logits[0, pred_pos, COLOR_TOKEN_IDS].float()
    sorted_color_d = color_logits_dec.argsort(descending=True).tolist()
    decode_rank_in_colors.append(sorted_color_d.index(target_in_colors_idx) + 1)

print(f"  N = {N} episodes, 4 facts each")
print(f"  {'Aux head top1':<30s} {aux_correct/N:>6.1%}")
print(f"  {'Decode top1':<30s} {decode_correct/N:>6.1%}")
print(f"  {'Both correct':<30s} {both_correct/N:>6.1%}")
print(f"  {'Aux predicts a color':<30s} {aux_in_colors/N:>6.1%}")
print(f"  {'Decode predicts a color':<30s} {decode_in_colors/N:>6.1%}")
print(f"  {'Aux rank (within 15 colors)':<30s} mean={sum(aux_rank_in_colors)/N:.2f}")
print(f"  {'Decode rank (within 15 colors)':<30s} mean={sum(decode_rank_in_colors)/N:.2f}")

# ======================== TEST 2: mem_gate inspection ========================

print(f"\n{'='*70}")
print("TEST 2: mem_gate values across fusion blocks")
print(f"{'='*70}")
print("  Question: how much does fusion actually use memory?")

for i, block in enumerate(model.dec_fusion_blocks):
    gate_raw = block.mem_gate.data                  # [D]
    gate_sig = torch.sigmoid(gate_raw)
    print(f"  Fusion block {i}: mean={gate_sig.mean().item():.4f}  "
          f"std={gate_sig.std().item():.4f}  "
          f"min={gate_sig.min().item():.4f}  max={gate_sig.max().item():.4f}")

# ======================== TEST 3: Per-stream contribution ====================

print(f"\n{'='*70}")
print("TEST 3: Per-stream contribution (zero out one stream at a time)")
print(f"{'='*70}")
print("  Question: which memory streams carry signal?")

def run_with_masked_stream(facts, prompt, answer, mask_stream=None):
    """Run with one stream masked to zero in memory_tokens."""
    with contextlib.redirect_stdout(io.StringIO()): model.reset_memory()
    encode_facts(facts)
    p_ids = ENC.encode_ordinary(prompt)
    xp = torch.tensor([_pad(p_ids)], dtype=torch.long, device=DEVICE)
    target_tok = ENC.encode_ordinary(answer)[0]

    # We need to hook into decode to mask a specific stream
    # Stream order in dec_read_fusion: state=0, episode=1, conflict=2, archive=3, working=4
    stream_names = ['state', 'episode', 'conflict', 'archive', 'working']

    original_fusion_forward = model.dec_read_fusion.forward
    def patched_fusion_forward(r_state, r_episode, r_conflict, r_archive, r_working):
        streams = [r_state, r_episode, r_conflict, r_archive, r_working]
        if mask_stream is not None:
            idx = stream_names.index(mask_stream)
            streams[idx] = torch.zeros_like(streams[idx])
        return original_fusion_forward(*streams)

    model.dec_read_fusion.forward = patched_fusion_forward
    try:
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model.decode(xp)
        pred = logits[0, min(len(p_ids)-1, SEQ_LEN-1), :].float().argmax().item()
    finally:
        model.dec_read_fusion.forward = original_fusion_forward

    return pred == target_tok

stream_acc = {'normal': 0, 'mask_state': 0, 'mask_episode': 0,
              'mask_conflict': 0, 'mask_archive': 0, 'mask_working': 0}

N2 = 200
for _ in range(N2):
    facts, prompt, answer, target, _, _ = gen_ep(4)
    stream_acc['normal'] += run_with_masked_stream(facts, prompt, answer, None)
    stream_acc['mask_state'] += run_with_masked_stream(facts, prompt, answer, 'state')
    stream_acc['mask_episode'] += run_with_masked_stream(facts, prompt, answer, 'episode')
    stream_acc['mask_conflict'] += run_with_masked_stream(facts, prompt, answer, 'conflict')
    stream_acc['mask_archive'] += run_with_masked_stream(facts, prompt, answer, 'archive')
    stream_acc['mask_working'] += run_with_masked_stream(facts, prompt, answer, 'working')

print(f"  N = {N2} episodes")
print(f"  {'Normal (all streams)':<30s} {stream_acc['normal']/N2:>6.1%}")
for s in ['state', 'episode', 'conflict', 'archive', 'working']:
    key = f'mask_{s}'
    drop = (stream_acc['normal'] - stream_acc[key]) / N2
    print(f"  {'Mask ' + s:<30s} {stream_acc[key]/N2:>6.1%}  "
          f"(drop: {drop:+.1%})")

# ======================== TEST 4: Color-space accuracy =====================

print(f"\n{'='*70}")
print("TEST 4: Accuracy restricted to color vocabulary")
print(f"{'='*70}")
print("  Question: given the model chose a color, how often is it RIGHT color?")

N3 = 500
color_space_correct = 0
color_space_total = 0
# Among episodes where decode predicted a color:
for _ in range(N3):
    facts, prompt, answer, target, _, _ = gen_ep(4)
    with contextlib.redirect_stdout(io.StringIO()): model.reset_memory()
    encode_facts(facts)
    target_tok = ENC.encode_ordinary(answer)[0]
    p_ids = ENC.encode_ordinary(prompt)
    xp = torch.tensor([_pad(p_ids)], dtype=torch.long, device=DEVICE)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model.decode(xp)
    # Restrict to colors
    pred_pos = min(len(p_ids) - 1, SEQ_LEN - 1)
    color_logits = logits[0, pred_pos, COLOR_TOKEN_IDS].float()
    color_pred_idx = color_logits.argmax().item()
    color_pred_tok = COLOR_TOKEN_IDS[color_pred_idx]
    if color_pred_tok == target_tok: color_space_correct += 1
    color_space_total += 1

print(f"  N = {N3}, restrict to 15 colors")
print(f"  Accuracy when forced to pick a color: {color_space_correct/color_space_total:>6.1%}")
print(f"  (Random baseline: {1/15:.1%})")

# ======================== TEST 5: Direct value decode =======================

print(f"\n{'='*70}")
print("TEST 5: Raw value -> LM head (bypass fusion)")
print(f"{'='*70}")
print("  Question: does the stored value naturally decode to the answer?")

N4 = 100
value_decode_correct = 0
for _ in range(N4):
    facts, prompt, answer, target, _, _ = gen_ep(4)
    with contextlib.redirect_stdout(io.StringIO()): model.reset_memory()
    slots = encode_facts(facts)

    target_tok = ENC.encode_ordinary(answer)[0]
    target_bank_name, target_slot_idx = slots[target]

    # Read the target slot's value directly
    bank = getattr(model, f"{target_bank_name}_mem")
    value = bank.values[target_slot_idx]              # [D]
    
    # Decode through LM head (value -> vocab)
    with torch.no_grad():
        # final_norm + lm_head
        h = model.dec_final_norm(value.unsqueeze(0))
        direct_logits = model.dec_lm_head(h).float()
    pred = direct_logits.argmax(dim=-1).item()
    if pred == target_tok: value_decode_correct += 1

print(f"  N = {N4}")
print(f"  Raw value -> LM head top1 = {value_decode_correct/N4:>6.1%}")

# ======================== TEST 6: Token-color specific ======================

print(f"\n{'='*70}")
print("TEST 6: Aux head top-5 for 5 test cases (qualitative)")
print(f"{'='*70}")

test_cases = [
    (["The cat is red.", "The dog is blue.", "The bird is green.", "The fish is yellow."], "cat", "red"),
    (["The knight is silver.", "The wizard is purple.", "The princess is pink.", "The dragon is crimson."], "dragon", "crimson"),
    (["The lion is golden.", "The tiger is orange.", "The bear is brown.", "The fox is red."], "tiger", "orange"),
]

for facts, ent, col in test_cases:
    with contextlib.redirect_stdout(io.StringIO()): model.reset_memory()
    encode_facts(facts)
    prompt = f"What color is the {ent}? The {ent} is"
    p_ids = ENC.encode_ordinary(prompt)
    xp = torch.tensor([_pad(p_ids)], dtype=torch.long, device=DEVICE)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, retrieved = model.decode(xp, return_retrieved=True)
        aux_logits = model.aux_answer_head(retrieved).float()
    
    aux_top5 = aux_logits[0].argsort(descending=True)[:5]
    dec_top5 = logits[0, min(len(p_ids)-1, SEQ_LEN-1), :].float().argsort(descending=True)[:5]
    
    print(f"  Target: {ent} -> {col}")
    print(f"    Aux head top5:   {[ENC.decode([t.item()]) for t in aux_top5]}")
    print(f"    Decode top5:     {[ENC.decode([t.item()]) for t in dec_top5]}")

# ======================== VERDICT ===========================================

print(f"\n{'='*70}")
print("VERDICT - WHERE EMISSION BREAKS")
print(f"{'='*70}")

aux_rate = aux_correct / N
decode_rate = decode_correct / N
aux_color_match = aux_in_colors / N
decode_color_match = decode_in_colors / N
avg_gate = sum(torch.sigmoid(b.mem_gate.data).mean().item() for b in model.dec_fusion_blocks) / 4
working_drop = (stream_acc['normal'] - stream_acc['mask_working']) / N2
value_direct = value_decode_correct / N4
color_space_acc = color_space_correct / color_space_total

print(f"  Aux head top1        : {aux_rate:.1%}")
print(f"  Decode top1          : {decode_rate:.1%}")
print(f"  Color-restricted     : {color_space_acc:.1%}")
print(f"  Mean mem_gate (sig)  : {avg_gate:.3f}")
print(f"  Working stream drop  : {working_drop:+.1%}")
print(f"  Raw value -> LM      : {value_direct:.1%}")

verdict = []
if aux_rate > 0.25 and decode_rate < 0.15:
    verdict.append("AUX HEAD WORKS but DECODE DOES NOT - emission path rupted")
if avg_gate < 0.3:
    verdict.append(f"FUSION GATE SUPPRESSED ({avg_gate:.2f}) - memory barely reaches output")
if working_drop < 0.05:
    verdict.append("MASKING WORKING STREAM doesn't hurt - memory contribution is marginal")
if color_space_acc > 0.3 and decode_rate < 0.15:
    verdict.append("COLORS ARE DISCRIMINATED but model picks non-color tokens at full vocab")
if value_direct < 0.05:
    verdict.append("RAW VALUE doesn't decode to answer - value is structurally non-lexical")

if not verdict:
    verdict.append("Mixed signals - review numbers manually")

for v in verdict:
    print(f"  - {v}")

report = {
    'checkpoint_step': ckpt_step,
    'test1_aux_direct': {'n': N, 'aux_top1': aux_rate, 'decode_top1': decode_rate,
                          'both': both_correct/N, 'aux_color': aux_color_match,
                          'decode_color': decode_color_match,
                          'aux_rank_in_colors': sum(aux_rank_in_colors)/N,
                          'decode_rank_in_colors': sum(decode_rank_in_colors)/N},
    'test2_mem_gate': [torch.sigmoid(b.mem_gate.data).mean().item() for b in model.dec_fusion_blocks],
    'test3_streams': {k: v/N2 for k, v in stream_acc.items()},
    'test4_color_space': color_space_acc,
    'test5_value_direct': value_direct,
    'verdict': verdict,
}

with open(os.path.join(PROJECT_ROOT, 'results', 'deep_diagnostic.json'), 'w') as f:
    json.dump(report, f, indent=2, default=str)
print(f"\n[INFO] Report saved: {os.path.join(PROJECT_ROOT, 'results', 'deep_diagnostic.json')}")
print(f"{'='*70}\n")
