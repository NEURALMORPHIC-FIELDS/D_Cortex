# -*- coding: utf-8 -*-
# ===========================================================================
# D_Cortex v2.0-alpha :: step12_training_v15_6_pas6.py
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# v15.6 PAS 6 — Role-of-Modifier Resolver (RoMR)
# ===========================================================================
# ======================== 1. ENVIRONMENT ====================================

import os
import sys

from google.colab import drive
drive.mount('/content/drive')

PROJECT_ROOT = '/content/drive/MyDrive/dcortex_v2'

# v15 uses ISOLATED subfolder (separate from v11-v14 artifacts)
V15_ROOT = f'{PROJECT_ROOT}/v15'
CHECKPOINT_DIR = f'{V15_ROOT}/checkpoints'
RESULTS_DIR = f'{V15_ROOT}/results'
LOGS_DIR = f'{V15_ROOT}/logs'

# Dataset cache SHARED with v11/v12 (same tokenized bin files)
BIN_DIR = f'{PROJECT_ROOT}/dataset_cache/bin'
LOCAL_DATA = '/content/tmp_data'
SEP = '=' * 70

for d in [PROJECT_ROOT, V15_ROOT, CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR, BIN_DIR, LOCAL_DATA]:
    os.makedirs(d, exist_ok=True)

print(f"[INFO] Project root: {PROJECT_ROOT}", flush=True)
print(f"[INFO] v15 workspace: {V15_ROOT}", flush=True)
print(f"[INFO] v15 checkpoints: {CHECKPOINT_DIR}", flush=True)
print(f"[INFO] v15 results: {RESULTS_DIR}", flush=True)
print(f"[INFO] Shared bin cache: {BIN_DIR}", flush=True)

# ======================== 2. GPU DETECTION ==================================

import torch
import numpy as np
import random
import math
import json
import time
import subprocess

assert torch.cuda.is_available(), "CUDA required."
GPU_NAME = torch.cuda.get_device_name(0)
GPU_MEM_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3)
GPU_CAP = torch.cuda.get_device_capability(0)

print(SEP)
print(f"[INFO] GPU: {GPU_NAME} | VRAM: {GPU_MEM_GB:.1f} GB | SM {GPU_CAP[0]}.{GPU_CAP[1]}")

if 'A100' in GPU_NAME or GPU_CAP[0] >= 8:
    DTYPE = torch.bfloat16
    USE_SCALER = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("[INFO] A100 mode: bfloat16, TF32, NO GradScaler")
else:
    DTYPE = torch.float16
    USE_SCALER = True
    print(f"[WARN] {GPU_NAME}: fp16 + GradScaler")

torch.backends.cudnn.benchmark = True
_SDPA_AVAILABLE = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
print(f"[INFO] SDPA: {'AVAILABLE' if _SDPA_AVAILABLE else 'NOT AVAILABLE'}")
print(SEP)

DEVICE = torch.device('cuda')
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
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
    query_weights: Tuple[float, float, float] = (1.0, 0.0, 0.0)

    # --- Thresholds ---
    theta_match: float = 0.85
    theta_conflict: float = 0.3
    theta_write: float = 0.5

    # --- Consolidator ---
    consolidate_merge_threshold: float = 0.95
    consolidate_decay_rate: float = 0.99
    consolidate_prune_threshold: float = 0.05

    # --- Updater ---
    ema_alpha: float = 0.9

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
        answer_token_id: torch.Tensor = None,
        lexical_alpha: float = 0.9,
        force_bank: str = None,
    ) -> Dict[str, torch.Tensor]:
        """Process facts and write to memory.

        Args:
            input_ids: [B, T] fact tokens.
            banks, step: as before.
            answer_token_id: [B] answer token ids. If provided, writer binds
                value lexically to the answer embedding. Required for
                structural episodes; None for LM/free-form encoding.
            lexical_alpha: weight on lexical component (default 0.9).
        """
        B, T = input_ids.shape

        # 1. Embed (raw, before encoder blocks)
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        emb_raw = self.token_emb(input_ids) + self.pos_emb(positions)

        # 2. ADDRESS CODE from shared address encoder (operates on raw embeddings)
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

        # 5. Compute answer embedding if provided (lexical binding)
        answer_emb = None
        if answer_token_id is not None:
            answer_emb = self.token_emb(answer_token_id)             # [B, D]

        # 6. Write: keys from addr_code, value from h_pool + optional lexical
        write_out = self.writer(
            h_pool, addr_code, self.updater, banks, step,
            answer_emb=answer_emb, lexical_alpha=lexical_alpha,
            force_bank=force_bank,
        )

        # 7. Advance EpisodeSSM
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

    def encode(
        self,
        input_ids: torch.Tensor,
        answer_token_id: torch.Tensor = None,
        lexical_alpha: float = 0.9,
        force_bank: str = None,
    ) -> Dict[str, torch.Tensor]:
        """Agent A: process fact tokens and write to memory banks.

        Args:
            input_ids: [B, T] fact token ids.
            answer_token_id: [B] answer token ids for lexical value binding.
                If provided, stored value is biased toward the answer embedding.
                Required for structural episodes.
            lexical_alpha: weight on lexical component of value (0..1).

        Returns:
            Dict of aux tensors with gradients.
        """
        if input_ids.dim() != 2:
            raise ValueError(f"encode expects [B, T], got {tuple(input_ids.shape)}")

        self.step_counter += 1
        step = int(self.step_counter.item())

        self._enc_aux = self.encoder(
            input_ids, self._bank_dict(), step,
            answer_token_id=answer_token_id,
            lexical_alpha=lexical_alpha,
            force_bank=force_bank,
        )
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

        # retrieved_value: SUM of raw reader outputs (BEFORE fusion projections).
        # Why not use memory_tokens.sum: each fusion proj has a bias, so
        # proj_state(zeros) = bias != 0, polluting signal from unpopulated streams.
        # Summing raw reader outputs: zero streams contribute exact zero,
        # populated stream contributes actual value.
        retrieved_value = r_state + r_episode + r_conflict + r_archive + r_working  # [B, D]

        memory_tokens = self.dec_read_fusion(
            r_state, r_episode, r_conflict, r_archive, r_working,
        )

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
import subprocess  # cell-local import guard (Colab cell may not inherit from top)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tiktoken", "datasets", "matplotlib"], check=True)
print("[INFO] Dependencies installed")
# ======================== 4. IMPORTS ========================================

import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from datasets import load_dataset

from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model
from dcortex.backbone.transformer import MultiHeadSelfAttention
from dcortex.backbone.fusion_block import CrossAttention

print("[INFO] All imports OK")

if _SDPA_AVAILABLE:
    def _sdpa_self_attn_forward(self, h, attention_mask=None):
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
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=(attention_mask is None))
        return self.out(out.transpose(1, 2).reshape(B, T, D))
    def _sdpa_cross_attn_forward(self, h, memory):
        B, T, D = h.shape; _, K, _ = memory.shape
        q = self.q(h).reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(memory).reshape(B, K, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0, is_causal=False)
        return self.out(out.transpose(1, 2).reshape(B, T, D))
    MultiHeadSelfAttention.forward = _sdpa_self_attn_forward
    CrossAttention.forward = _sdpa_cross_attn_forward
    print("[INFO] SDPA patched")

# ======================== 5. TOKENIZER + DATA ===============================

ENC = tiktoken.get_encoding("gpt2")
EOT = ENC.eot_token

def tokenize_to_bin(split, max_tokens):
    path = os.path.join(BIN_DIR, f'tinystories_{split}.bin')
    if os.path.exists(path):
        n = os.path.getsize(path) // 2
        print(f"[INFO] {split} cached: {n:,} tokens"); return path
    print(f"[INFO] Tokenizing {split}...", flush=True)
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    tokens = []
    for i, ex in enumerate(ds):
        text = ex.get('text', '') or ex.get('story', '')
        if not text: continue
        tokens.extend(ENC.encode_ordinary(text)); tokens.append(EOT)
        if i > 0 and i % 50000 == 0: print(f"  {len(tokens):,} tok", flush=True)
        if len(tokens) >= max_tokens: break
    arr = np.array(tokens[:max_tokens], dtype=np.uint16)
    tmp = path + '.tmp'; arr.tofile(tmp); os.rename(tmp, path)
    print(f"[INFO] {split}: {len(arr):,} tokens"); return path

train_bin = tokenize_to_bin('train', 80_000_000)
val_bin = tokenize_to_bin('validation', 5_000_000)

def copy_to_local_ssd(src):
    dst = os.path.join(LOCAL_DATA, os.path.basename(src))
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src): return dst
    stat = os.statvfs('/content')
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    if free_gb < os.path.getsize(src) / (1024**3) + 1.0: return src
    subprocess.run(["cp", src, dst], check=True); return dst

train_data = np.memmap(copy_to_local_ssd(train_bin), dtype=np.uint16, mode='r')
val_data = np.memmap(copy_to_local_ssd(val_bin), dtype=np.uint16, mode='r')
print(f"[INFO] Data: {len(train_data):,} train / {len(val_data):,} val tokens")

# ======================== 5b. V15 STAGE 1 COMPONENTS (INLINE) =============
#
# All v15 components inlined here after v14 infrastructure imports succeed.
# These extend (not replace) the v14 dcortex source tree. The v14 source
# tree at /content/dcortex_src remains functional for backward inspection.

# --- v15 attribute vocabulary --------------------------------------------

V15_ATTR_TYPES = ["color", "size", "location", "state", "unknown"]
V15_ATTR_TO_IDX = {a: i for i, a in enumerate(V15_ATTR_TYPES)}
V15_N_ATTR_TYPES = len(V15_ATTR_TYPES)  # 5
V15_UNKNOWN_ATTR_IDX = V15_ATTR_TO_IDX["unknown"]

# --- v15 class vocabulary ------------------------------------------------

V15_CLASSES = ["creature", "person", "object", "unknown"]
V15_CLASS_TO_IDX = {c: i for i, c in enumerate(V15_CLASSES)}
V15_N_CLASSES = len(V15_CLASSES)  # 4
V15_UNKNOWN_CLASS_IDX = V15_CLASS_TO_IDX["unknown"]

# Special token for no-match output (added to GPT-2 vocab as sentinel)
# GPT-2 vocab size is 50257, EOT is 50256. We use a phrase token instead
# of modifying the tokenizer: the literal answer text "unknown" already
# tokenizes to a single BPE token, which we use as UNKNOWN_TOKEN.
_UNKNOWN_WORD_TOKENS = ENC.encode(" unknown")
assert len(_UNKNOWN_WORD_TOKENS) == 1, f"' unknown' should tokenize to 1 token, got {_UNKNOWN_WORD_TOKENS}"
V15_UNKNOWN_ANSWER_TOKEN = _UNKNOWN_WORD_TOKENS[0]
print(f"[v15] UNKNOWN_TOKEN = {V15_UNKNOWN_ANSWER_TOKEN} (from ' unknown')")

# ==========================================================================
# V15 COMPONENT 1: AttributeSlot + ObjectRecord + ObjectBank
# ==========================================================================
#
# Verbatim port from /dcortex_v2/dcortex/memory/object_bank.py, tested by
# tests/test_object_bank.py (94/94 assertions passed).

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class AttributeSlot:
    """One attribute slot within an ObjectRecord."""
    value: torch.Tensor                   # [d_val] attribute value embedding
    answer_token: int                     # answer token id
    presence: bool = True
    confidence: float = 1.0
    last_write_step: int = -1
    write_count: int = 1


@dataclass
class ObjectRecord:
    """One object in cognitive memory. Follows m_i spec from Part IV."""
    # Identity
    id_anchor: torch.Tensor               # [d_id]
    lexical_token: int
    # Semantic
    sem_anchor: torch.Tensor              # [d_sem]
    class_id: int = -1
    class_confidence: float = 0.0
    # Type
    type: str = "object"
    # Temporal
    write_step: int = -1
    last_access_step: int = -1
    # Context signature
    context_signature: Optional[torch.Tensor] = None   # [d_ctx]
    # Epistemic
    confidence: float = 1.0
    salience: float = 1.0
    novelty: float = 1.0
    uncertainty: float = 0.0
    # Attribute slots
    attributes: Dict[str, AttributeSlot] = field(default_factory=dict)
    # Links (stub for Stage 2)
    links: Dict[str, List[int]] = field(default_factory=dict)
    # Version history (stub for Stage 2)
    version_history: List[dict] = field(default_factory=list)
    # Provisional flag
    is_provisional: bool = False

    def has_attribute(self, attr_name: str) -> bool:
        slot = self.attributes.get(attr_name)
        return slot is not None and slot.presence

    def set_attribute(self, attr_name, value, answer_token, step, confidence=1.0):
        if attr_name in self.attributes and self.attributes[attr_name].presence:
            existing = self.attributes[attr_name]
            self.attributes[attr_name] = AttributeSlot(
                value=value, answer_token=answer_token, presence=True,
                confidence=confidence, last_write_step=step,
                write_count=existing.write_count + 1,
            )
        else:
            self.attributes[attr_name] = AttributeSlot(
                value=value, answer_token=answer_token, presence=True,
                confidence=confidence, last_write_step=step, write_count=1,
            )

    def get_attribute(self, attr_name):
        slot = self.attributes.get(attr_name)
        if slot is None or not slot.presence:
            return None
        return slot

    def snapshot(self):
        return {
            "lexical_token": self.lexical_token, "class_id": self.class_id,
            "class_confidence": self.class_confidence, "type": self.type,
            "write_step": self.write_step, "last_access_step": self.last_access_step,
            "confidence": self.confidence, "salience": self.salience,
            "novelty": self.novelty, "uncertainty": self.uncertainty,
            "is_provisional": self.is_provisional,
            "attributes": {
                name: {
                    "answer_token": s.answer_token, "presence": s.presence,
                    "confidence": s.confidence, "last_write_step": s.last_write_step,
                    "write_count": s.write_count,
                } for name, s in self.attributes.items()
            },
            "n_links": sum(len(v) for v in self.links.values()),
            "n_versions": len(self.version_history),
        }


class ObjectBank(nn.Module):
    """Memory bank storing ObjectRecord instances keyed by slot index."""

    def __init__(self, capacity, d_id, d_sem, d_val, d_ctx=None, theta_match=0.85):
        super().__init__()
        self.capacity = capacity
        self.d_id = d_id
        self.d_sem = d_sem
        self.d_val = d_val
        self.d_ctx = d_ctx if d_ctx is not None else d_id
        self.theta_match = theta_match

        self.register_buffer("k_ent", torch.zeros(capacity, d_id))
        self.register_buffer("occupied", torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("last_access", torch.full((capacity,), -1, dtype=torch.long))

        self._records: Dict[int, ObjectRecord] = {}
        self._overlay: Dict[int, Dict[str, torch.Tensor]] = {}

    def n_occupied(self):
        return int(self.occupied.sum().item())

    def free_slot(self):
        free = (~self.occupied).nonzero(as_tuple=False)
        if free.numel() == 0:
            return -1
        return int(free[0].item())

    def lru_slot(self):
        if self.n_occupied() == 0:
            return 0
        steps = self.last_access.float().clone()
        steps[~self.occupied] = float("inf")
        return int(steps.argmin().item())

    def reset(self):
        self.k_ent.zero_()
        self.occupied.zero_()
        self.last_access.fill_(-1)
        self._records.clear()
        self._overlay.clear()

    def clear_grads(self):
        self._overlay.clear()

    def find_by_identity(self, query_id, min_similarity=None):
        if self.n_occupied() == 0:
            return -1, -1.0
        q_n = F.normalize(query_id.unsqueeze(0), dim=-1)
        k_n = F.normalize(self.k_ent, dim=-1)
        sims = (q_n @ k_n.T).squeeze(0)
        sims = sims.masked_fill(~self.occupied, float("-inf"))
        best_sim, best_idx = sims.max(dim=-1)
        best_sim_val = float(best_sim.item())
        best_idx_val = int(best_idx.item())
        if min_similarity is not None and best_sim_val < min_similarity:
            return -1, best_sim_val
        return best_idx_val, best_sim_val

    @torch.no_grad()
    def write_object(
        self, id_anchor, lexical_token, step,
        sem_anchor=None, class_id=-1, class_confidence=0.0,
        context_signature=None, confidence=1.0, salience=1.0,
        novelty=1.0, uncertainty=0.0, is_provisional=False,
    ):
        match_idx, match_sim = self.find_by_identity(
            id_anchor, min_similarity=self.theta_match
        )
        if match_idx >= 0:
            self.last_access[match_idx] = step
            return match_idx

        slot = self.free_slot()
        if slot < 0:
            slot = self.lru_slot()
            if slot in self._records:
                del self._records[slot]

        if sem_anchor is None:
            sem_anchor = torch.zeros(self.d_sem, device=id_anchor.device)

        record = ObjectRecord(
            id_anchor=id_anchor.detach().clone(),
            lexical_token=lexical_token,
            sem_anchor=sem_anchor.detach().clone(),
            class_id=class_id, class_confidence=class_confidence,
            type="object",
            write_step=step, last_access_step=step,
            context_signature=context_signature.detach().clone()
                if context_signature is not None else None,
            confidence=confidence, salience=salience,
            novelty=novelty, uncertainty=uncertainty,
            attributes={}, links={}, version_history=[],
            is_provisional=is_provisional,
        )
        self._records[slot] = record
        self.k_ent[slot] = id_anchor.detach().clone()
        self.occupied[slot] = True
        self.last_access[slot] = step
        return slot

    def read_object(self, query_id, min_similarity=None):
        idx, sim = self.find_by_identity(query_id, min_similarity=min_similarity)
        if idx < 0:
            return None, -1, sim
        record = self._records.get(idx)
        if record is None:
            return None, -1, sim
        return record, idx, sim

    @torch.no_grad()
    def update_attribute(self, slot_idx, attr_name, value, answer_token, step, confidence=1.0):
        record = self._records.get(slot_idx)
        if record is None:
            return False
        record.set_attribute(attr_name, value.detach().clone(), answer_token, step, confidence)
        record.last_access_step = step
        self.last_access[slot_idx] = step
        return True

    @torch.no_grad()
    def merge_object(self, slot_idx, id_anchor=None, sem_anchor=None,
                     class_id=None, class_confidence=None, step=None):
        record = self._records.get(slot_idx)
        if record is None:
            return False
        if id_anchor is not None:
            record.id_anchor = 0.8 * record.id_anchor + 0.2 * id_anchor.detach().clone()
            self.k_ent[slot_idx] = record.id_anchor
        if sem_anchor is not None:
            record.sem_anchor = 0.5 * record.sem_anchor + 0.5 * sem_anchor.detach().clone()
        if class_id is not None and class_id >= 0:
            if record.class_confidence < (class_confidence or 0.0):
                record.class_id = class_id
                record.class_confidence = class_confidence or 0.5
                record.uncertainty = max(0.0, record.uncertainty - 0.3)
                record.is_provisional = False
        if step is not None:
            record.last_access_step = step
            self.last_access[slot_idx] = step
        return True

    @torch.no_grad()
    def no_match_create_provisional(self, id_anchor, lexical_token, step, context_signature=None):
        return self.write_object(
            id_anchor=id_anchor, lexical_token=lexical_token, step=step,
            sem_anchor=None, class_id=-1, class_confidence=0.0,
            context_signature=context_signature,
            confidence=0.5, salience=1.0, novelty=1.0, uncertainty=1.0,
            is_provisional=True,
        )

    def snapshot(self):
        return {
            "capacity": self.capacity,
            "occupied": self.n_occupied(),
            "records": {idx: rec.snapshot() for idx, rec in self._records.items()},
        }

    def all_records(self):
        return [(idx, rec) for idx, rec in self._records.items()]


# ==========================================================================
# V15 COMPONENT 2: ClassEncoder
# ==========================================================================

class ClassEncoder(nn.Module):
    """Parses anchor sentences, emits class_id + class_emb + confidence.

    Input: pooled anchor sentence embedding [B, d_model]
    Output:
        z_class      - [B, d_sem]        semantic class vector (for sem_anchor)
        class_id     - [B]               argmax class index
        class_conf   - [B]               softmax max probability
        class_logits - [B, n_classes]   raw logits (for supervised CE loss)
    """

    def __init__(self, d_model: int, d_sem: int, n_classes: int):
        super().__init__()
        self.d_model = d_model
        self.d_sem = d_sem
        self.n_classes = n_classes
        self.class_head = nn.Linear(d_model, n_classes)
        self.class_emb = nn.Embedding(n_classes, d_sem)

    def forward(self, pooled_anchor: torch.Tensor):
        # pooled_anchor: [B, d_model] or [d_model]
        if pooled_anchor.dim() == 1:
            pooled_anchor = pooled_anchor.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        logits = self.class_head(pooled_anchor)              # [B, n_classes]
        probs = torch.softmax(logits, dim=-1)                # [B, n_classes]
        class_conf, class_id = probs.max(dim=-1)             # [B], [B]
        z_class = self.class_emb(class_id)                   # [B, d_sem]

        if squeeze:
            return z_class.squeeze(0), int(class_id.item()), float(class_conf.item()), logits.squeeze(0)
        return z_class, class_id, class_conf, logits


# ==========================================================================
# V15 COMPONENT 3 + 4: QueryClassifier, FactClassifier
# ==========================================================================

class AttributeTypeClassifier(nn.Module):
    """5-class MLP head: color / size / location / state / unknown.

    Shared structure for QueryClassifier (on pooled question) and
    FactClassifier (on pooled fact). Different training signals.
    """

    def __init__(self, d_model: int, n_attr_types: int = V15_N_ATTR_TYPES):
        super().__init__()
        self.d_model = d_model
        self.n_attr_types = n_attr_types
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_attr_types),
        )

    def forward(self, pooled: torch.Tensor):
        # pooled: [B, d_model]
        return self.head(pooled)

    def predict(self, pooled: torch.Tensor):
        """Return (attr_id, confidence, logits)."""
        logits = self.head(pooled)
        probs = torch.softmax(logits, dim=-1)
        conf, attr_id = probs.max(dim=-1)
        return attr_id, conf, logits


class QueryClassifier(AttributeTypeClassifier):
    """Alias for clarity. Same structure, trained on question embeddings."""
    pass


class FactClassifier(AttributeTypeClassifier):
    """Alias for clarity. Same structure, trained on fact embeddings."""
    pass


# ==========================================================================
# V15 helper: rule-based class anchor detection (Stage 1)
# ==========================================================================

# Map known class nouns in anchor sentences to class ids.
V15_CLASS_KEYWORDS = {
    "creature": V15_CLASS_TO_IDX["creature"],
    "animal":   V15_CLASS_TO_IDX["creature"],
    "being":    V15_CLASS_TO_IDX["creature"],
    "person":   V15_CLASS_TO_IDX["person"],
    "human":    V15_CLASS_TO_IDX["person"],
    "man":      V15_CLASS_TO_IDX["person"],
    "woman":    V15_CLASS_TO_IDX["person"],
    "object":   V15_CLASS_TO_IDX["object"],
    "thing":    V15_CLASS_TO_IDX["object"],
    "item":     V15_CLASS_TO_IDX["object"],
    "tool":     V15_CLASS_TO_IDX["object"],
}

def detect_class_anchor(text: str):
    """Rule-based parser for 'A/The X is a/an Y' patterns.

    Returns:
        (entity, class_id) if matched, else (None, None).
    """
    import re
    # Pattern: "A/The {entity} is a/an {class}"
    # Case-insensitive, simple word boundaries.
    pattern = r"(?:^|\s)(?:A|The)\s+(\w+)\s+is\s+(?:a|an)\s+(\w+)"
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if m is None:
        return None, None
    entity = m.group(1).lower()
    class_word = m.group(2).lower()
    class_id = V15_CLASS_KEYWORDS.get(class_word)
    if class_id is None:
        return entity, V15_UNKNOWN_CLASS_IDX
    return entity, class_id


# ==========================================================================
# V15 rule-based fact type detection (for supervising FactClassifier)
# ==========================================================================

V15_COLOR_VOCAB = {
    "red", "blue", "green", "yellow", "black", "white", "brown", "pink",
    "orange", "purple", "golden", "silver", "crimson", "gray", "violet",
}
V15_SIZE_VOCAB = {"tiny", "small", "big", "huge", "large", "little"}
V15_LOCATION_VOCAB = {
    "forest", "cave", "castle", "river", "mountain", "garden",
    "cellar", "tower", "ocean", "desert", "room", "field",
}
V15_STATE_VOCAB = {
    "asleep", "awake", "angry", "calm", "hungry", "tired",
    "happy", "afraid", "sad", "scared",
}


def detect_attribute_type(text: str) -> int:
    """Infer attribute type from fact text by keyword presence.

    Used to generate labels for FactClassifier supervised training.
    Returns V15_ATTR_TO_IDX value (color=0, size=1, location=2, state=3, unknown=4).
    """
    t = text.lower()
    tokens = set(t.replace(".", "").replace(",", "").replace("!", "").split())
    if tokens & V15_COLOR_VOCAB:
        return V15_ATTR_TO_IDX["color"]
    if tokens & V15_SIZE_VOCAB:
        return V15_ATTR_TO_IDX["size"]
    if tokens & V15_LOCATION_VOCAB:
        return V15_ATTR_TO_IDX["location"]
    if tokens & V15_STATE_VOCAB:
        return V15_ATTR_TO_IDX["state"]
    return V15_ATTR_TO_IDX["unknown"]


def detect_query_type(text: str) -> int:
    """Infer query type from question text.

    Used for QueryClassifier supervised training.
    """
    t = text.lower()
    if "color" in t or "colour" in t or "colored" in t:
        return V15_ATTR_TO_IDX["color"]
    if any(kw in t for kw in ["size", "large", "big", "small", "tiny", "huge", "how big"]):
        return V15_ATTR_TO_IDX["size"]
    if any(kw in t for kw in ["where", "location", "place", "in what"]):
        return V15_ATTR_TO_IDX["location"]
    if any(kw in t for kw in ["state", "how does", "how is", "feel", "condition"]):
        return V15_ATTR_TO_IDX["state"]
    return V15_ATTR_TO_IDX["unknown"]


print("[v15] Core components defined: ObjectBank, ObjectRecord, AttributeSlot, "
      "ClassEncoder, AttributeTypeClassifier (Query/Fact), rule helpers")

# ======================== 5c. V15 COGNITIVE MEMORY WRAPPER =================
#
# Minimal wrapper that holds ObjectBank + classifiers + class encoder and
# exposes a compact API to be hooked into the DCortexV2Model's encode/decode
# paths. Does NOT replace v14 banks yet (parallel operation in Stage 1).


class V15CognitiveMemory(nn.Module):
    """Holds all v15 Stage 1 memory components in one nn.Module.

    Components:
        object_bank       - structured memory (working scope, 16 slots)
        class_encoder     - semantic class inference from anchor sentences
        query_classifier  - query type routing (which attribute slot)
        fact_classifier   - fact type routing (which attribute slot)
        identity_projection - reused from v14 (frozen at init, low-LR after)
        theta_nomatch     - learnable no-match threshold

    Initial version: stateless between episodes (object_bank is reset
    at episode boundaries, same as v14 working memory).
    """

    def __init__(self, d_model: int, d_id: int, d_sem: int, d_val: int,
                 capacity: int = 16, theta_nomatch_init: float = 0.75):
        super().__init__()
        self.d_model = d_model
        self.d_id = d_id
        self.d_sem = d_sem
        self.d_val = d_val
        self.capacity = capacity

        # Memory bank
        self.object_bank = ObjectBank(
            capacity=capacity, d_id=d_id, d_sem=d_sem, d_val=d_val,
            theta_match=0.85,
        )

        # Semantic class encoder
        self.class_encoder = ClassEncoder(
            d_model=d_model, d_sem=d_sem, n_classes=V15_N_CLASSES,
        )

        # Attribute routing
        self.query_classifier = QueryClassifier(d_model=d_model)
        self.fact_classifier = FactClassifier(d_model=d_model)

        # Identity projection (retained from v14 dual-channel)
        self.identity_projection = nn.Linear(d_model, d_id)
        nn.init.orthogonal_(self.identity_projection.weight)
        nn.init.zeros_(self.identity_projection.bias)
        # Freeze initially (same strategy as v14)
        for p in self.identity_projection.parameters():
            p.requires_grad = False

        # Learnable no-match threshold (logit, sigmoid applied when used)
        self.theta_nomatch_logit = nn.Parameter(
            torch.tensor(float(math.log(theta_nomatch_init / (1 - theta_nomatch_init))))
        )

    def theta_nomatch(self):
        return torch.sigmoid(self.theta_nomatch_logit)

    def unfreeze_identity(self):
        """Call after warmup step count (e.g. 2000) to enable low-LR update."""
        for p in self.identity_projection.parameters():
            p.requires_grad = True

    def reset_episode(self):
        """Clear the object bank and overlay at episode boundary."""
        self.object_bank.reset()

    def compute_identity_key(self, raw_pooled: torch.Tensor, addr_code: torch.Tensor,
                              lambda_id: float = 0.75) -> torch.Tensor:
        """v14 dual-channel hybrid key, retained.

        Args:
            raw_pooled: [d_model] pooled GPT-2 embedding
            addr_code: [d_id]   semantic address code from existing K_phi
            lambda_id: scalar mixing weight (0.75 default, identity-dominant)
        """
        k_id = F.normalize(self.identity_projection(raw_pooled), dim=-1)
        k_sem = F.normalize(addr_code, dim=-1)
        mixed = math.sqrt(lambda_id) * k_id + math.sqrt(1.0 - lambda_id) * k_sem
        return F.normalize(mixed, dim=-1)


print("[v15] V15CognitiveMemory wrapper defined")

# ======================== 5d. V15 ENCODE/DECODE HOOKS (MINIMAL) ==========
#
# Stage 1: we do NOT replace DCortexV2Model.encode / .decode. Instead we
# add a thin set of helper functions that operate on a V15CognitiveMemory
# instance and produce the same answer logits as v14 would, but through
# the object-centric path. This lets training still run via existing v14
# pipeline, with v15 components learning in parallel. Full integration
# happens after Stage 1 smoke tests pass.

@torch.no_grad()
def v15_write_fact(
    memory: V15CognitiveMemory,
    raw_pooled_fact: torch.Tensor,     # [d_model] from encoder over fact tokens
    fact_text: str,                     # original text for rule-based detection
    entity_lexical_token: int,
    answer_token: int,
    step: int,
    addr_code: torch.Tensor,            # [d_id] from query_engine K_phi path
    anchor_context: Optional[str] = None,  # e.g. "A phoenix is a creature"
):
    """Write one fact as an object + attribute.

    Flow:
      1. Compute dual-channel identity key.
      2. Run FactClassifier to determine attribute type.
      3. If entity not in bank: write_object (with class inferred from anchor).
      4. Set attribute slot (selective, per attribute type).
    """
    id_anchor = memory.compute_identity_key(raw_pooled_fact, addr_code)

    # Attribute type from rule (for training label) + from classifier (for pred)
    attr_idx_rule = detect_attribute_type(fact_text)

    # Class prior from anchor context if provided
    class_id = -1
    class_conf = 0.0
    z_sem = None
    if anchor_context is not None:
        entity_str, detected_cls = detect_class_anchor(anchor_context)
        if detected_cls is not None:
            class_id = detected_cls
            class_conf = 0.9
            z_sem = memory.class_encoder.class_emb(
                torch.tensor(class_id, device=raw_pooled_fact.device)
            )

    # Find or create object
    existing_idx, existing_sim = memory.object_bank.find_by_identity(
        id_anchor, min_similarity=memory.object_bank.theta_match
    )
    if existing_idx < 0:
        # New entity - write object
        slot = memory.object_bank.write_object(
            id_anchor=id_anchor, lexical_token=entity_lexical_token,
            step=step, sem_anchor=z_sem,
            class_id=class_id, class_confidence=class_conf,
        )
    else:
        slot = existing_idx
        # If new class info arrived, merge
        if class_id >= 0:
            memory.object_bank.merge_object(
                slot_idx=slot, class_id=class_id,
                class_confidence=class_conf, step=step,
            )

    # Write attribute to correct slot
    if attr_idx_rule != V15_UNKNOWN_ATTR_IDX:
        attr_name = V15_ATTR_TYPES[attr_idx_rule]
        memory.object_bank.update_attribute(
            slot_idx=slot, attr_name=attr_name,
            value=raw_pooled_fact.detach().clone(),
            answer_token=answer_token, step=step,
        )
    return slot


@torch.no_grad()
def v15_read_for_query(
    memory: V15CognitiveMemory,
    raw_pooled_query: torch.Tensor,
    query_text: str,
    addr_code: torch.Tensor,
) -> Tuple[int, str, Optional[AttributeSlot], float]:
    """Route query to the correct object + attribute slot.

    Returns:
        (slot_idx, attr_name, attr_slot_or_None, similarity)
        If no-match: (-1, 'unknown', None, similarity)
    """
    q_id = memory.compute_identity_key(raw_pooled_query, addr_code)

    # Query type from rules (supervised label for training) + classifier at runtime
    attr_idx = detect_query_type(query_text)
    attr_name = V15_ATTR_TYPES[attr_idx]

    # No-match gate
    theta = float(memory.theta_nomatch().item())
    best_idx, best_sim = memory.object_bank.find_by_identity(q_id, min_similarity=theta)
    if best_idx < 0:
        return -1, "unknown", None, best_sim

    record, _, _ = memory.object_bank.read_object(q_id, min_similarity=theta)
    if record is None:
        return -1, "unknown", None, best_sim

    if attr_name == "unknown":
        return best_idx, "unknown", None, best_sim

    attr_slot = record.get_attribute(attr_name)
    return best_idx, attr_name, attr_slot, best_sim


print("[v15] Encode/decode hooks defined: v15_write_fact, v15_read_for_query")


# ======================== 6. V15 CURRICULUM + LOSSES ======================
#
# Full Stage 1 implementation: entity pool, attribute vocabularies,
# fact/query templates, 8 episode generators (with supervised labels for
# all v15 classifier heads), 9 loss functions. No benchmark yet (Section 7
# below is smoke test; full benchmark v1 implementation in Section 9).

# --- 6.1 Entity pool (40 entities, 10 per class, stratified) ------------
#
# Matches Benchmark v1 spec exactly. Classes align with V15_CLASSES indices:
#   0 = creature, 1 = person, 2 = object, 3 = unknown (unused in pool)

V15_POOL_CREATURES = [
    "bear", "dog", "tiger", "fox", "rabbit",
    "wolf", "bird", "cat", "horse", "deer",
]
V15_POOL_FANTASY = [
    "dragon", "phoenix", "unicorn", "mermaid", "minotaur",
    "griffin", "chimera", "hydra", "pegasus", "basilisk",
]
V15_POOL_PERSONS = [
    "teacher", "doctor", "farmer", "pilot", "chef",
    "judge", "dancer", "sailor", "priest", "warrior",
]
V15_POOL_OBJECTS = [
    "lantern", "compass", "sword", "mirror", "chest",
    "crown", "telescope", "key", "shield", "scroll",
]

# All fantasy creatures are class=creature (combined with real animals)
V15_POOL_ALL = (
    [(e, V15_CLASS_TO_IDX["creature"]) for e in V15_POOL_CREATURES] +
    [(e, V15_CLASS_TO_IDX["creature"]) for e in V15_POOL_FANTASY] +
    [(e, V15_CLASS_TO_IDX["person"])   for e in V15_POOL_PERSONS] +
    [(e, V15_CLASS_TO_IDX["object"])   for e in V15_POOL_OBJECTS]
)
assert len(V15_POOL_ALL) == 40, f"pool size wrong: {len(V15_POOL_ALL)}"

# Stratified 80/20 held-out split per Benchmark v1 spec (seed=20260418).
# Training uses 8 entities (2 per class of creature/fantasy/person/object);
# benchmark uses remaining 32.
_split_rng = random.Random(20260418)
_creatures = list(V15_POOL_CREATURES)
_fantasy = list(V15_POOL_FANTASY)
_persons = list(V15_POOL_PERSONS)
_objects = list(V15_POOL_OBJECTS)
_split_rng.shuffle(_creatures)
_split_rng.shuffle(_fantasy)
_split_rng.shuffle(_persons)
_split_rng.shuffle(_objects)

V15_TRAIN_ENTITIES = (
    [(e, V15_CLASS_TO_IDX["creature"]) for e in _creatures[:2]] +
    [(e, V15_CLASS_TO_IDX["creature"]) for e in _fantasy[:2]] +
    [(e, V15_CLASS_TO_IDX["person"])   for e in _persons[:2]] +
    [(e, V15_CLASS_TO_IDX["object"])   for e in _objects[:2]]
)
V15_HELDOUT_ENTITIES = (
    [(e, V15_CLASS_TO_IDX["creature"]) for e in _creatures[2:]] +
    [(e, V15_CLASS_TO_IDX["creature"]) for e in _fantasy[2:]] +
    [(e, V15_CLASS_TO_IDX["person"])   for e in _persons[2:]] +
    [(e, V15_CLASS_TO_IDX["object"])   for e in _objects[2:]]
)
assert len(V15_TRAIN_ENTITIES) == 8
assert len(V15_HELDOUT_ENTITIES) == 32

# --- 6.2 Attribute vocabularies (per Benchmark v1 spec) -----------------

V15_COLORS = [
    "red", "blue", "green", "yellow", "black", "white", "brown", "pink",
    "orange", "purple", "golden", "silver", "crimson", "gray", "violet",
]
V15_SIZES = ["tiny", "small", "big", "huge"]
V15_LOCATIONS = [
    "forest", "cave", "castle", "river", "mountain",
    "garden", "cellar", "tower", "ocean", "desert",
]
V15_STATES = [
    "asleep", "awake", "angry", "calm",
    "hungry", "tired", "happy", "afraid",
]
V15_ATTR_VALUES = {
    "color":    V15_COLORS,
    "size":     V15_SIZES,
    "location": V15_LOCATIONS,
    "state":    V15_STATES,
}

# --- 6.3 Class anchor templates -------------------------------------------

V15_CLASS_ANCHORS = {
    V15_CLASS_TO_IDX["creature"]: "A {e} is a creature.",
    V15_CLASS_TO_IDX["person"]:   "The {e} is a person.",
    V15_CLASS_TO_IDX["object"]:   "The {e} is an object.",
}

def v15_render_class_anchor(entity: str, class_id: int) -> str:
    tpl = V15_CLASS_ANCHORS[class_id]
    return tpl.format(e=entity)

# --- 6.4 Fact templates (5 per attribute) ----------------------------------

V15_FACT_TEMPLATES = {
    "color": [
        "The {e} is {v}.",
        "The {e} was painted {v}.",
        "A {v} {e} stood nearby.",
        "The {e} appeared {v}.",
        "The {e}, quite clearly, was {v}.",
    ],
    "size": [
        "The {e} is {v}.",
        "A {v} {e} was there.",
        "Everyone noticed how {v} the {e} was.",
        "The {e} looked {v}.",
        "The {e} seemed {v}.",
    ],
    "location": [
        "The {e} is in the {v}.",
        "The {e} was found in the {v}.",
        "A {e} lived in the {v}.",
        "In the {v}, a {e} appeared.",
        "The {e} remained in the {v}.",
    ],
    "state": [
        "The {e} is {v}.",
        "The {e} looked {v}.",
        "The {e} seemed {v}.",
        "A {v} {e} rested there.",
        "The {e} was clearly {v}.",
    ],
}

# --- 6.5 Query templates (3 per attribute) --------------------------------
# Each query produces (prompt_ending_in_space, attr_type, target_value_fn).
# Target is always the attribute's value as a single-token continuation,
# preceded by a space for GPT-2 BPE.

V15_QUERY_TEMPLATES = {
    "color": [
        "What color is the {e}? The {e} is",
        "Tell me the color of the {e}. It is",
        "The {e} has what color? The {e} is",
    ],
    "size": [
        "How large is the {e}? The {e} is",
        "What size is the {e}? The {e} is",
        "Describe the size of the {e}. It is",
    ],
    "location": [
        "Where is the {e}? The {e} is in the",
        "In what place is the {e}? The {e} is in the",
        "Where can the {e} be found? It is in the",
    ],
    "state": [
        "What state is the {e} in? The {e} is",
        "How does the {e} feel? The {e} is",
        "The {e} is in what condition? The {e} is",
    ],
}

def v15_render_fact(entity: str, attr: str, value: str, rng: random.Random) -> str:
    tpl = rng.choice(V15_FACT_TEMPLATES[attr])
    return tpl.format(e=entity, v=value)

def v15_render_query(entity: str, attr: str, rng: random.Random) -> str:
    tpl = rng.choice(V15_QUERY_TEMPLATES[attr])
    return tpl.format(e=entity)

# --- 6.6 Tokenization helpers for target answers --------------------------

def v15_first_token(text: str) -> int:
    """GPT-2 BPE first token of a ' word' (leading space). Used for answers."""
    ids = ENC.encode(" " + text.strip())
    return ids[0]

# Pre-compute answer token tables
V15_ANSWER_TOKENS = {
    attr: {v: v15_first_token(v) for v in values}
    for attr, values in V15_ATTR_VALUES.items()
}
V15_ANSWER_TOKENS["unknown"] = {"unknown": V15_UNKNOWN_ANSWER_TOKEN}

# --- 6.7 Episode dataclass ------------------------------------------------

@dataclass
class V15Episode:
    """One generated training episode with ALL labels for v15 losses."""
    episode_type: str
    # Facts to write into memory
    facts: List[str]                         # raw text per fact
    fact_entity_tokens: List[int]             # lexical token id of entity in each fact
    fact_attr_labels: List[int]               # V15_ATTR_TO_IDX per fact (for L_ftype)
    fact_answer_tokens: List[int]             # answer token per fact
    fact_class_labels: List[int]              # V15_CLASS_TO_IDX per fact entity (for L_class)
    fact_is_anchor: List[bool]                # True if fact is a class anchor ("A X is a Y")
    # Query + target
    query: str                                # raw text
    query_attr_label: int                     # V15_ATTR_TO_IDX (for L_qtype)
    query_entity_token: int                   # lexical token id of queried entity
    target_answer_token: int                  # target for emit
    target_is_unknown: bool                   # True iff entity absent from facts (for L_nomatch)
    # Which fact slot holds the right answer (for L_select, L_slot)
    target_fact_idx: int                      # -1 if no-match
    target_slot_name: str                     # attribute slot name or 'unknown'

# --- 6.8 Base episode builders (reusable primitives) --------------------

def _v15_pick_entities(n: int, pool, rng: random.Random,
                      allow_repeat: bool = False):
    """Select n (entity, class_id) pairs from pool."""
    if allow_repeat or n > len(pool):
        return [rng.choice(pool) for _ in range(n)]
    return rng.sample(pool, n)

def _v15_pick_attr_value(attr: str, rng: random.Random) -> str:
    return rng.choice(V15_ATTR_VALUES[attr])

# --- 6.9 Eight episode generators ----------------------------------------

def _gen_single_attr_simple(rng: random.Random, pool) -> V15Episode:
    """One attribute per entity, 3-5 entities, query one."""
    n = rng.randint(3, 5)
    ents = _v15_pick_entities(n, pool, rng)
    attr = rng.choice(["color", "size", "location", "state"])
    facts, fact_entity_toks, fact_attr_lbls, fact_ans_toks = [], [], [], []
    fact_cls_lbls, fact_is_anchor = [], []
    values = []
    for (e, cls) in ents:
        v = _v15_pick_attr_value(attr, rng)
        fact = v15_render_fact(e, attr, v, rng)
        facts.append(fact)
        fact_entity_toks.append(v15_first_token(e))
        fact_attr_lbls.append(V15_ATTR_TO_IDX[attr])
        fact_ans_toks.append(V15_ANSWER_TOKENS[attr][v])
        fact_cls_lbls.append(cls)
        fact_is_anchor.append(False)
        values.append(v)
    t_idx = rng.randint(0, n - 1)
    (t_ent, _) = ents[t_idx]
    t_val = values[t_idx]
    query = v15_render_query(t_ent, attr, rng)
    return V15Episode(
        episode_type="single_attr_simple",
        facts=facts, fact_entity_tokens=fact_entity_toks,
        fact_attr_labels=fact_attr_lbls, fact_answer_tokens=fact_ans_toks,
        fact_class_labels=fact_cls_lbls, fact_is_anchor=fact_is_anchor,
        query=query, query_attr_label=V15_ATTR_TO_IDX[attr],
        query_entity_token=v15_first_token(t_ent),
        target_answer_token=V15_ANSWER_TOKENS[attr][t_val],
        target_is_unknown=False, target_fact_idx=t_idx,
        target_slot_name=attr,
    )

def _gen_multi_attr_object(rng: random.Random, pool) -> V15Episode:
    """2-3 entities, each with ALL 4 attributes, query one random attr."""
    n = rng.randint(2, 3)
    ents = _v15_pick_entities(n, pool, rng)
    attrs = ["color", "size", "location", "state"]
    facts, fact_entity_toks, fact_attr_lbls, fact_ans_toks = [], [], [], []
    fact_cls_lbls, fact_is_anchor = [], []
    # Store per-(entity,attr) value map for query
    value_map = {}     # (ent_idx, attr) -> value
    for ent_idx, (e, cls) in enumerate(ents):
        order = list(attrs); rng.shuffle(order)
        for a in order:
            v = _v15_pick_attr_value(a, rng)
            fact = v15_render_fact(e, a, v, rng)
            facts.append(fact)
            fact_entity_toks.append(v15_first_token(e))
            fact_attr_lbls.append(V15_ATTR_TO_IDX[a])
            fact_ans_toks.append(V15_ANSWER_TOKENS[a][v])
            fact_cls_lbls.append(cls)
            fact_is_anchor.append(False)
            value_map[(ent_idx, a)] = v
    t_ent_idx = rng.randint(0, n - 1)
    t_attr = rng.choice(attrs)
    t_val = value_map[(t_ent_idx, t_attr)]
    (t_ent, _) = ents[t_ent_idx]
    # Find the fact index for this (entity, attr) pair
    t_fact_idx = -1
    for i, (e_tok, a_lbl) in enumerate(zip(fact_entity_toks, fact_attr_lbls)):
        if e_tok == v15_first_token(t_ent) and a_lbl == V15_ATTR_TO_IDX[t_attr]:
            t_fact_idx = i
            break
    query = v15_render_query(t_ent, t_attr, rng)
    return V15Episode(
        episode_type="multi_attr_object",
        facts=facts, fact_entity_tokens=fact_entity_toks,
        fact_attr_labels=fact_attr_lbls, fact_answer_tokens=fact_ans_toks,
        fact_class_labels=fact_cls_lbls, fact_is_anchor=fact_is_anchor,
        query=query, query_attr_label=V15_ATTR_TO_IDX[t_attr],
        query_entity_token=v15_first_token(t_ent),
        target_answer_token=V15_ANSWER_TOKENS[t_attr][t_val],
        target_is_unknown=False, target_fact_idx=t_fact_idx,
        target_slot_name=t_attr,
    )

def _gen_selective_update(rng: random.Random, pool) -> V15Episode:
    """Write all 4 attrs for 2-3 entities, then UPDATE one attr for one entity.
    Query can be either the updated attr (new value) or an unchanged attr (old value).
    This tests M4 selective update.
    """
    n = rng.randint(2, 3)
    ents = _v15_pick_entities(n, pool, rng)
    attrs = ["color", "size", "location", "state"]
    facts, fact_entity_toks, fact_attr_lbls, fact_ans_toks = [], [], [], []
    fact_cls_lbls, fact_is_anchor = [], []
    value_map = {}
    # Phase 1: write all 4 attrs per entity
    for ent_idx, (e, cls) in enumerate(ents):
        order = list(attrs); rng.shuffle(order)
        for a in order:
            v = _v15_pick_attr_value(a, rng)
            fact = v15_render_fact(e, a, v, rng)
            facts.append(fact)
            fact_entity_toks.append(v15_first_token(e))
            fact_attr_lbls.append(V15_ATTR_TO_IDX[a])
            fact_ans_toks.append(V15_ANSWER_TOKENS[a][v])
            fact_cls_lbls.append(cls)
            fact_is_anchor.append(False)
            value_map[(ent_idx, a)] = v
    # Phase 2: update ONE attr on ONE entity
    u_ent_idx = rng.randint(0, n - 1)
    u_attr = rng.choice(attrs)
    (u_ent, u_cls) = ents[u_ent_idx]
    old_val = value_map[(u_ent_idx, u_attr)]
    new_val = old_val
    while new_val == old_val:
        new_val = _v15_pick_attr_value(u_attr, rng)
    update_fact = v15_render_fact(u_ent, u_attr, new_val, rng)
    facts.append(update_fact)
    fact_entity_toks.append(v15_first_token(u_ent))
    fact_attr_lbls.append(V15_ATTR_TO_IDX[u_attr])
    fact_ans_toks.append(V15_ANSWER_TOKENS[u_attr][new_val])
    fact_cls_lbls.append(u_cls)
    fact_is_anchor.append(False)
    value_map[(u_ent_idx, u_attr)] = new_val  # overwrite
    # Phase 3: query - 50/50 the updated attr or an unchanged attr
    if rng.random() < 0.5:
        # Query updated attr -> expect new value
        t_ent_idx, t_attr = u_ent_idx, u_attr
    else:
        # Query a DIFFERENT attribute of the same entity -> expect unchanged
        other_attrs = [a for a in attrs if a != u_attr]
        t_attr = rng.choice(other_attrs)
        t_ent_idx = u_ent_idx
    (t_ent, _) = ents[t_ent_idx]
    t_val = value_map[(t_ent_idx, t_attr)]
    # Find LAST fact matching (entity, attr) - that's the authoritative value
    t_fact_idx = -1
    for i in range(len(facts) - 1, -1, -1):
        if (fact_entity_toks[i] == v15_first_token(t_ent) and
                fact_attr_lbls[i] == V15_ATTR_TO_IDX[t_attr]):
            t_fact_idx = i; break
    query = v15_render_query(t_ent, t_attr, rng)
    return V15Episode(
        episode_type="selective_update",
        facts=facts, fact_entity_tokens=fact_entity_toks,
        fact_attr_labels=fact_attr_lbls, fact_answer_tokens=fact_ans_toks,
        fact_class_labels=fact_cls_lbls, fact_is_anchor=fact_is_anchor,
        query=query, query_attr_label=V15_ATTR_TO_IDX[t_attr],
        query_entity_token=v15_first_token(t_ent),
        target_answer_token=V15_ANSWER_TOKENS[t_attr][t_val],
        target_is_unknown=False, target_fact_idx=t_fact_idx,
        target_slot_name=t_attr,
    )

def _gen_no_match(rng: random.Random, pool) -> V15Episode:
    """Write 3-5 entities. Query an ABSENT entity from same pool.
    Target: UNKNOWN_TOKEN. Tests M5 unknown handling.
    """
    n = rng.randint(3, 5)
    ents = _v15_pick_entities(n, pool, rng)
    attr = rng.choice(["color", "size", "location", "state"])
    facts, fact_entity_toks, fact_attr_lbls, fact_ans_toks = [], [], [], []
    fact_cls_lbls, fact_is_anchor = [], []
    present_entities = set()
    for (e, cls) in ents:
        v = _v15_pick_attr_value(attr, rng)
        fact = v15_render_fact(e, attr, v, rng)
        facts.append(fact)
        fact_entity_toks.append(v15_first_token(e))
        fact_attr_lbls.append(V15_ATTR_TO_IDX[attr])
        fact_ans_toks.append(V15_ANSWER_TOKENS[attr][v])
        fact_cls_lbls.append(cls)
        fact_is_anchor.append(False)
        present_entities.add(e)
    # Pick absent entity
    candidates = [p for p in pool if p[0] not in present_entities]
    absent_ent, _ = rng.choice(candidates)
    query = v15_render_query(absent_ent, attr, rng)
    return V15Episode(
        episode_type="no_match",
        facts=facts, fact_entity_tokens=fact_entity_toks,
        fact_attr_labels=fact_attr_lbls, fact_answer_tokens=fact_ans_toks,
        fact_class_labels=fact_cls_lbls, fact_is_anchor=fact_is_anchor,
        query=query, query_attr_label=V15_ATTR_TO_IDX[attr],
        query_entity_token=v15_first_token(absent_ent),
        target_answer_token=V15_UNKNOWN_ANSWER_TOKEN,
        target_is_unknown=True, target_fact_idx=-1,
        target_slot_name="unknown",
    )

def _gen_paraphrase(rng: random.Random, pool) -> V15Episode:
    """Same structure as single_attr_simple, but each fact uses a DIFFERENT
    template (forced mix), and query template chosen from 3 variants.
    """
    ep = _gen_single_attr_simple(rng, pool)
    ep.episode_type = "paraphrase"
    # Regenerate facts with forced template diversity
    attr = V15_ATTR_TYPES[ep.fact_attr_labels[0]]
    tpls = V15_FACT_TEMPLATES[attr]
    if len(ep.facts) > len(tpls):
        # pool of templates reused
        order = [rng.choice(tpls) for _ in range(len(ep.facts))]
    else:
        order = rng.sample(tpls, len(ep.facts))
    new_facts = []
    for i, fact in enumerate(ep.facts):
        # Extract entity and value from original fact via attr_label + ans_token
        ent_tok = ep.fact_entity_tokens[i]
        # Decode entity token back to word (expensive but simple)
        ent_word = ENC.decode([ent_tok]).strip()
        ans_tok = ep.fact_answer_tokens[i]
        val_word = ENC.decode([ans_tok]).strip()
        new_facts.append(order[i].format(e=ent_word, v=val_word))
    ep.facts = new_facts
    return ep

def _gen_coreference_distant(rng: random.Random, pool) -> V15Episode:
    """Templates with coreference or distant mention between entity and value.
    Tests P7 from benchmark.
    """
    n = rng.randint(2, 4)
    ents = _v15_pick_entities(n, pool, rng)
    attr = "color"   # coreference templates tuned for color primarily
    coref_templates = [
        "A {e} appeared. It was {v}.",
        "The {e} walked in. Everyone looked. Eventually, it was revealed to be {v}.",
        "There was a {e}. It turned out to be {v}.",
        "Consider the {e}. This one, as we learned, was {v}.",
        "A {e} was present. Looking carefully: {v}.",
    ]
    facts, fact_entity_toks, fact_attr_lbls, fact_ans_toks = [], [], [], []
    fact_cls_lbls, fact_is_anchor = [], []
    values = []
    for (e, cls) in ents:
        v = _v15_pick_attr_value(attr, rng)
        tpl = rng.choice(coref_templates)
        fact = tpl.format(e=e, v=v)
        facts.append(fact)
        fact_entity_toks.append(v15_first_token(e))
        fact_attr_lbls.append(V15_ATTR_TO_IDX[attr])
        fact_ans_toks.append(V15_ANSWER_TOKENS[attr][v])
        fact_cls_lbls.append(cls)
        fact_is_anchor.append(False)
        values.append(v)
    t_idx = rng.randint(0, n - 1)
    (t_ent, _) = ents[t_idx]
    t_val = values[t_idx]
    query = v15_render_query(t_ent, attr, rng)
    return V15Episode(
        episode_type="coreference_distant",
        facts=facts, fact_entity_tokens=fact_entity_toks,
        fact_attr_labels=fact_attr_lbls, fact_answer_tokens=fact_ans_toks,
        fact_class_labels=fact_cls_lbls, fact_is_anchor=fact_is_anchor,
        query=query, query_attr_label=V15_ATTR_TO_IDX[attr],
        query_entity_token=v15_first_token(t_ent),
        target_answer_token=V15_ANSWER_TOKENS[attr][t_val],
        target_is_unknown=False, target_fact_idx=t_idx,
        target_slot_name=attr,
    )

def _gen_lm_pretraining(rng: random.Random, pool) -> V15Episode:
    """Pure language modeling placeholder: no facts/queries, just a text span.
    Implemented as a degenerate episode - the training loop handles LM separately
    via a dedicated path (standard next-token prediction). Here we return
    an empty structural episode; loss will skip memory losses for this type.
    """
    return V15Episode(
        episode_type="lm_pretraining",
        facts=[], fact_entity_tokens=[], fact_attr_labels=[],
        fact_answer_tokens=[], fact_class_labels=[], fact_is_anchor=[],
        query="", query_attr_label=V15_UNKNOWN_ATTR_IDX,
        query_entity_token=0, target_answer_token=0,
        target_is_unknown=False, target_fact_idx=-1,
        target_slot_name="unknown",
    )

def _gen_provisional_entity(rng: random.Random, pool) -> V15Episode:
    """Introduce entity WITH class anchor, then a fact, then query.
    Tests that class anchor reduces uncertainty for provisional node.
    """
    n = rng.randint(2, 3)
    ents = _v15_pick_entities(n, pool, rng)
    attr = rng.choice(["color", "size", "location", "state"])
    facts, fact_entity_toks, fact_attr_lbls, fact_ans_toks = [], [], [], []
    fact_cls_lbls, fact_is_anchor = [], []
    values = []
    for (e, cls) in ents:
        # First a class anchor sentence
        anchor = v15_render_class_anchor(e, cls)
        facts.append(anchor)
        fact_entity_toks.append(v15_first_token(e))
        fact_attr_lbls.append(V15_UNKNOWN_ATTR_IDX)    # anchors are not attributes
        fact_ans_toks.append(0)                          # dummy - no answer for anchor
        fact_cls_lbls.append(cls)
        fact_is_anchor.append(True)
        # Then the attribute fact
        v = _v15_pick_attr_value(attr, rng)
        fact = v15_render_fact(e, attr, v, rng)
        facts.append(fact)
        fact_entity_toks.append(v15_first_token(e))
        fact_attr_lbls.append(V15_ATTR_TO_IDX[attr])
        fact_ans_toks.append(V15_ANSWER_TOKENS[attr][v])
        fact_cls_lbls.append(cls)
        fact_is_anchor.append(False)
        values.append(v)
    t_idx = rng.randint(0, n - 1)
    (t_ent, _) = ents[t_idx]
    t_val = values[t_idx]
    # target_fact_idx: position of the (non-anchor) attribute fact for t_ent
    # Each entity contributes 2 facts, so attribute fact idx = 2*t_idx + 1
    t_fact_idx = 2 * t_idx + 1
    query = v15_render_query(t_ent, attr, rng)
    return V15Episode(
        episode_type="provisional_entity",
        facts=facts, fact_entity_tokens=fact_entity_toks,
        fact_attr_labels=fact_attr_lbls, fact_answer_tokens=fact_ans_toks,
        fact_class_labels=fact_cls_lbls, fact_is_anchor=fact_is_anchor,
        query=query, query_attr_label=V15_ATTR_TO_IDX[attr],
        query_entity_token=v15_first_token(t_ent),
        target_answer_token=V15_ANSWER_TOKENS[attr][t_val],
        target_is_unknown=False, target_fact_idx=t_fact_idx,
        target_slot_name=attr,
    )

V15_EPISODE_GENERATORS = {
    "single_attr_simple":  _gen_single_attr_simple,
    "multi_attr_object":   _gen_multi_attr_object,
    "selective_update":    _gen_selective_update,
    "no_match":            _gen_no_match,
    "paraphrase":          _gen_paraphrase,
    "coreference_distant": _gen_coreference_distant,
    "lm_pretraining":      _gen_lm_pretraining,
    "provisional_entity":  _gen_provisional_entity,
}

# Episode type distribution (matches v15 architecture Part XVIII)
V15_EPISODE_TYPES = [
    ("single_attr_simple",      0.25),
    ("multi_attr_object",       0.20),
    ("selective_update",        0.15),
    ("no_match",                0.10),
    ("paraphrase",              0.10),
    ("coreference_distant",     0.10),
    ("lm_pretraining",          0.05),
    ("provisional_entity",      0.05),
]

def v15_sample_episode_type(rng: random.Random) -> str:
    """Weighted sampler over V15_EPISODE_TYPES."""
    r = rng.random()
    cum = 0.0
    for name, p in V15_EPISODE_TYPES:
        cum += p
        if r < cum:
            return name
    return V15_EPISODE_TYPES[-1][0]

def v15_generate_episode(episode_type: str, rng: random.Random,
                          use_heldout: bool = False) -> V15Episode:
    """Top-level episode dispatch.
    
    use_heldout=True samples from V15_HELDOUT_ENTITIES (for eval);
    use_heldout=False samples from V15_TRAIN_ENTITIES (for training).
    """
    gen = V15_EPISODE_GENERATORS.get(episode_type)
    if gen is None:
        raise ValueError(f"Unknown episode type: {episode_type}")
    pool = V15_HELDOUT_ENTITIES if use_heldout else V15_TRAIN_ENTITIES
    return gen(rng, pool)

# --- 6.10 Batch tensorization ---------------------------------------------

def v15_tokenize_episode(ep: V15Episode, device) -> Dict:
    """Convert episode text to tensors for model input.
    
    Returns dict with:
        fact_ids:      List[Tensor [T_i]]    - token ids per fact
        fact_attn:     List[Tensor [T_i]]    - attention masks (all ones)
        query_ids:     Tensor [T_q]
        query_attn:    Tensor [T_q]
        target:        Tensor []              - scalar answer token
        + all scalar labels as tensors
    """
    fact_ids = []
    fact_attn = []
    for fact in ep.facts:
        ids = ENC.encode(fact)
        t = torch.tensor(ids, dtype=torch.long, device=device)
        fact_ids.append(t)
        fact_attn.append(torch.ones_like(t))
    
    if ep.query:
        q_ids = ENC.encode(ep.query)
        q_t = torch.tensor(q_ids, dtype=torch.long, device=device)
    else:
        q_t = torch.zeros(1, dtype=torch.long, device=device)
    
    return {
        "fact_ids":          fact_ids,
        "fact_attn":         fact_attn,
        "fact_entity_toks":  torch.tensor(ep.fact_entity_tokens, dtype=torch.long, device=device)
                                if ep.fact_entity_tokens else torch.zeros(0, dtype=torch.long, device=device),
        "fact_attr_labels":  torch.tensor(ep.fact_attr_labels, dtype=torch.long, device=device)
                                if ep.fact_attr_labels else torch.zeros(0, dtype=torch.long, device=device),
        "fact_class_labels": torch.tensor(ep.fact_class_labels, dtype=torch.long, device=device)
                                if ep.fact_class_labels else torch.zeros(0, dtype=torch.long, device=device),
        "fact_answer_toks":  torch.tensor(ep.fact_answer_tokens, dtype=torch.long, device=device)
                                if ep.fact_answer_tokens else torch.zeros(0, dtype=torch.long, device=device),
        "fact_is_anchor":    torch.tensor(ep.fact_is_anchor, dtype=torch.bool, device=device)
                                if ep.fact_is_anchor else torch.zeros(0, dtype=torch.bool, device=device),
        "query_ids":         q_t,
        "query_attn":        torch.ones_like(q_t),
        "query_attr_label":  torch.tensor(ep.query_attr_label, dtype=torch.long, device=device),
        "target":            torch.tensor(ep.target_answer_token, dtype=torch.long, device=device),
        "target_is_unknown": torch.tensor(int(ep.target_is_unknown), dtype=torch.long, device=device),
        "target_fact_idx":   torch.tensor(ep.target_fact_idx, dtype=torch.long, device=device),
    }

# --- 6.11 Forward pass: embed facts + query through shared token_emb ----

def v15_embed_pooled(base_model, token_ids: torch.Tensor) -> torch.Tensor:
    """Mean-pool raw GPT-2 embeddings from shared_token_emb.
    
    Args:
        base_model: DCortexV2Model (has .shared_token_emb)
        token_ids: [T] token ids
    Returns:
        pooled: [d_model] mean-pooled raw embedding
    """
    emb = base_model.shared_token_emb(token_ids)        # [T, d_model]
    return emb.mean(dim=0)                                # [d_model]

def v15_addr_code_from_pooled(base_model, pooled: torch.Tensor) -> torch.Tensor:
    """Compute addr_code via existing query_engine path (K_phi).
    
    shared_query_engine.forward returns (k_ent, k_rel, k_typ); we use k_ent.
    Input shape expected: [B, D]. We unsqueeze then squeeze.
    """
    h = pooled.unsqueeze(0)                               # [1, D]
    k_ent, _, _ = base_model.shared_query_engine(h)       # [1, d_ent]
    return k_ent.squeeze(0)                                # [d_ent]

# --- 6.12 Loss computation ------------------------------------------------
#
# 9 losses per spec. Not all fire for every episode type:
#   - L_lm only fires for lm_pretraining episodes
#   - L_class only fires when class anchor is present
#   - L_nomatch fires for no_match episodes (and symmetric signal on others)
#   - L_emit, L_qtype, L_ftype fire for all non-LM episodes
#   - L_select, L_slot fire for non-LM, non-no-match episodes

V15_LOSS_WEIGHTS = {
    "emit":       1.0,
    "select":     1.0,
    "slot":       1.0,
    "preserve":   0.2,
    "qtype":      0.5,
    "ftype":      0.5,
    "class":      0.3,
    "nomatch":    0.5,
    "lm":         0.5,
}

def v15_compute_losses(
    base_model,
    v15_memory: V15CognitiveMemory,
    batch_episodes: List[V15Episode],
    device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute total loss + per-component breakdown for one batch of episodes.
    
    Each episode is processed individually (not stacked - variable-length facts).
    Returns (total_loss, component_dict).
    """
    # Accumulators
    losses = {k: torch.zeros(1, device=device) for k in V15_LOSS_WEIGHTS}
    counts = {k: 0 for k in V15_LOSS_WEIGHTS}
    
    for ep in batch_episodes:
        tok = v15_tokenize_episode(ep, device)
        
        # --- LM pretraining branch: just next-token on facts (skip here,
        #     handled in outer loop via base_model standard LM head)
        if ep.episode_type == "lm_pretraining":
            continue
        
        # Reset v15 memory for this episode
        v15_memory.reset_episode()
        
        # --- Process each fact ---
        fact_pooled_list = []         # for FactClassifier input
        fact_addr_list = []           # for identity key
        fact_anchor_pooled = []       # for ClassEncoder (only anchors)
        fact_anchor_class_labels = []
        
        for i, fact_ids in enumerate(tok["fact_ids"]):
            pooled = v15_embed_pooled(base_model, fact_ids)      # [d_model] with grad
            addr = v15_addr_code_from_pooled(base_model, pooled)  # [d_id] with grad
            fact_pooled_list.append(pooled)
            fact_addr_list.append(addr)
            
            is_anchor = bool(tok["fact_is_anchor"][i].item()) if i < len(tok["fact_is_anchor"]) else False
            if is_anchor:
                fact_anchor_pooled.append(pooled)
                fact_anchor_class_labels.append(int(tok["fact_class_labels"][i].item()))
            else:
                # Write non-anchor facts to object bank (detached side)
                ent_tok = int(tok["fact_entity_toks"][i].item())
                ans_tok = int(tok["fact_answer_toks"][i].item())
                # Anchor context is the PREVIOUS fact if it was an anchor for same entity
                anchor_ctx = None
                if i > 0 and bool(tok["fact_is_anchor"][i-1].item()):
                    anchor_ctx = ep.facts[i-1]
                v15_write_fact(
                    memory=v15_memory,
                    raw_pooled_fact=pooled.detach(),
                    fact_text=ep.facts[i],
                    entity_lexical_token=ent_tok,
                    answer_token=ans_tok,
                    step=i, addr_code=addr.detach(),
                    anchor_context=anchor_ctx,
                )
        
        # --- L_ftype: FactClassifier supervised on non-anchor facts ---
        non_anchor_idxs = [i for i in range(len(ep.facts))
                           if i < len(tok["fact_is_anchor"])
                           and not bool(tok["fact_is_anchor"][i].item())]
        if len(non_anchor_idxs) > 0:
            pooled_batch = torch.stack([fact_pooled_list[i] for i in non_anchor_idxs])
            ftype_labels = torch.stack([tok["fact_attr_labels"][i] for i in non_anchor_idxs])
            ftype_logits = v15_memory.fact_classifier(pooled_batch)
            l_ftype = F.cross_entropy(ftype_logits, ftype_labels)
            losses["ftype"] = losses["ftype"] + l_ftype
            counts["ftype"] += 1
        
        # --- L_class: ClassEncoder supervised on anchor facts ---
        #
        # Two sub-losses:
        #   (a) CE on class_head logits (anchor pooled -> class id)
        #   (b) Contrastive pull on class_emb: InfoNCE over class_emb rows
        #       so class_emb.weight receives a direct gradient signal
        #       (not just via class_head in the encoder forward).
        if len(fact_anchor_pooled) > 0:
            pooled_batch = torch.stack(fact_anchor_pooled)         # [K, d_model]
            class_labels = torch.tensor(fact_anchor_class_labels, dtype=torch.long, device=device)
            _, _, _, class_logits = v15_memory.class_encoder(pooled_batch)
            l_class_ce = F.cross_entropy(class_logits, class_labels)
            
            # Contrastive over class_emb: cosine(target_class_emb, all_class_emb)
            # must concentrate on target class. This is the ONLY loss that
            # directly gradients class_emb.weight.
            target_class_emb = v15_memory.class_encoder.class_emb(class_labels)  # [K, d_sem]
            all_class_emb = v15_memory.class_encoder.class_emb.weight            # [n_classes, d_sem]
            emb_sims = F.cosine_similarity(
                target_class_emb.unsqueeze(1),                        # [K, 1, d_sem]
                all_class_emb.unsqueeze(0),                            # [1, n_classes, d_sem]
                dim=-1,
            ) / 0.1                                                   # temperature
            l_class_contrast = F.cross_entropy(emb_sims, class_labels)
            
            losses["class"] = losses["class"] + l_class_ce + 0.5 * l_class_contrast
            counts["class"] += 1
        
        # --- Process query ---
        q_pooled = v15_embed_pooled(base_model, tok["query_ids"])
        q_addr = v15_addr_code_from_pooled(base_model, q_pooled)
        
        # --- L_qtype: QueryClassifier supervised ---
        qtype_logits = v15_memory.query_classifier(q_pooled.unsqueeze(0))
        l_qtype = F.cross_entropy(qtype_logits, tok["query_attr_label"].unsqueeze(0))
        losses["qtype"] = losses["qtype"] + l_qtype
        counts["qtype"] += 1
        
        # --- L_preserve: cosine alignment between addr_code and raw_pooled ---
        # Project raw_pooled through identity_projection (same space as addr)
        # identity_projection is frozen initially; L_preserve encourages addr
        # to stay close to the projected raw signal.
        q_id_projected = v15_memory.identity_projection(q_pooled)       # [d_id]
        l_preserve = 1.0 - F.cosine_similarity(
            q_addr.unsqueeze(0), q_id_projected.unsqueeze(0), dim=-1
        ).mean()
        losses["preserve"] = losses["preserve"] + l_preserve
        counts["preserve"] += 1
        
        # --- L_select: hybrid query key should be closer to target fact key
        #     than to distractor fact keys ---
        #     Uses dual-channel identity key on queries and facts, trains
        #     identity_projection (post-unfreeze) + query_engine (K_phi) to
        #     prefer the target fact.
        if (ep.target_fact_idx >= 0 and len(fact_pooled_list) >= 2
                and ep.target_fact_idx < len(fact_pooled_list)):
            q_k = v15_memory.compute_identity_key(q_pooled, q_addr)       # [d_id]
            fact_keys = torch.stack([
                v15_memory.compute_identity_key(fact_pooled_list[i], fact_addr_list[i])
                for i in range(len(fact_pooled_list))
            ])                                                            # [N, d_id]
            # Cosine similarities (both are already normalized)
            logits = (q_k.unsqueeze(0) @ fact_keys.T).squeeze(0)           # [N]
            # Temperature
            logits = logits / 0.1
            target = torch.tensor(ep.target_fact_idx, dtype=torch.long, device=device)
            l_select = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
            losses["select"] = losses["select"] + l_select
            counts["select"] += 1
        
        # --- L_slot: query_classifier prediction must match which attribute
        #     slot contains the answer. Already covered by L_qtype in Stage 1
        #     (single head predicts slot from query text). Retained as alias
        #     for when Stage 2 adds a separate slot-routing head.
        losses["slot"] = losses["slot"] + l_qtype.detach() * 0.0  # placeholder zero
        
        # --- L_nomatch: gate supervision ---
        #     For no_match episodes, best_sim should be below theta.
        #     For non-no_match, best_sim should be above theta.
        if len(fact_pooled_list) > 0:
            q_k_now = v15_memory.compute_identity_key(q_pooled, q_addr)
            fact_keys_now = torch.stack([
                v15_memory.compute_identity_key(fact_pooled_list[i], fact_addr_list[i])
                for i in range(len(fact_pooled_list))
            ])
            sims_now = (q_k_now.unsqueeze(0) @ fact_keys_now.T).squeeze(0)  # [N]
            best_sim, _ = sims_now.max(dim=-1)
            theta = v15_memory.theta_nomatch()                               # scalar, sigmoid
            # Binary cross-entropy: predict P(match) = sigmoid((sim - theta) * scale)
            scale = 10.0
            p_match = torch.sigmoid((best_sim - theta) * scale)
            y_match = 1.0 - tok["target_is_unknown"].float()                 # 1 if not unknown
            eps = 1e-7
            l_nomatch = -(y_match * torch.log(p_match + eps) +
                          (1 - y_match) * torch.log(1 - p_match + eps))
            losses["nomatch"] = losses["nomatch"] + l_nomatch
            counts["nomatch"] += 1
        
        # --- L_emit: primary answer cross-entropy ---
        # Stage 1: a simple path that lets v15 signal propagate end-to-end.
        # We project q_pooled to vocabulary via a temporary linear layer
        # built on the fly from shared token embeddings - this is a STAGE 1
        # training signal, not the final emit path. Stage 2 will replace
        # this with proper decoder-fused memory attention.
        vocab_embed = base_model.shared_token_emb.weight                 # [V, d_model]
        logits_emit = q_pooled @ vocab_embed.T                            # [V]
        l_emit = F.cross_entropy(logits_emit.unsqueeze(0), tok["target"].unsqueeze(0))
        losses["emit"] = losses["emit"] + l_emit
        counts["emit"] += 1
        
        # L_lm handled separately (skipped for structural episodes)
    
    # Normalize by counts
    for k in losses:
        if counts[k] > 0:
            losses[k] = losses[k] / counts[k]
    
    # Total
    total = torch.zeros(1, device=device)
    for k, w in V15_LOSS_WEIGHTS.items():
        total = total + w * losses[k]
    
    return total.squeeze(), losses

print("[v15] Section 6 loaded: entity pool (40), vocab (15+4+10+8), "
      "templates (5*4 + 3*4), 8 generators, 9 losses")



# ======================== A. V15.1 SUBSTRATE ================================
#
# Deterministic object memory substrate.
# - Canonicalizer: string -> canonical entity_id
# - Parser: rule-based, scope-limited, reports coverage per dimension
# - ObjectBank: exact string match via find_by_entity_id, NO cosine in
#   critical path
# - Write/Read: fully deterministic, returns value_idx as integer or
#   symbolic NONE_OBJECT / NONE_ATTRIBUTE / PARSER_FAILURE
# ===========================================================================

# --- A.1 Canonicalizer ----------------------------------------------------

V15_1_ALIAS_MAP: Dict[str, str] = {
    # Stage 1: start with identity map.
    # Extended explicitly when curriculum introduces aliases.
    # Example: "warrior": "warrior", "fighter": "warrior"
}

V15_1_DETERMINERS = ("the ", "a ", "an ")
V15_1_TRAILING_PUNCT = ".,;:!?"

def canonicalize_entity(text: str) -> str:
    """Canonicalize entity text to entity_id.
    
    Steps:
      1. lowercase + strip all whitespace
      2. collapse internal whitespace
      3. strip leading determiners (the/a/an)
      4. strip trailing punctuation
      5. map known aliases
    """
    s = text.lower().strip()
    # Collapse internal whitespace
    s = " ".join(s.split())
    for det in V15_1_DETERMINERS:
        if s.startswith(det):
            s = s[len(det):]
            break
    s = s.rstrip(V15_1_TRAILING_PUNCT).strip()
    return V15_1_ALIAS_MAP.get(s, s)


# --- A.2 Rule-based parser ------------------------------------------------

# Attribute keywords (must match template set hash exactly)
V15_1_ATTR_KEYWORDS = {
    "color":    {"color", "colour", "shade", "pigment"},
    "size":     {"size", "dimension"},
    "location": {"location", "place", "where", "site"},
    "state":    {"state", "condition", "mood"},
}

# Implicit attribute triggers in query ("how large" -> size, "how does X feel" -> state)
V15_1_IMPLICIT_QUERY_ATTR = {
    # size
    "large":     "size",
    "big":       "size",
    "small":     "size",
    "measures":  "size",
    # state
    "feel":      "state",
    "feels":     "state",
    "feeling":   "state",
    "presently": "state",
    # location
    "situated":  "location",
    "found":     "location",
}

# All attribute keyword strings flattened (used for entity exclusion)
V15_1_ALL_ATTR_KEYWORDS = set()
for _kws in V15_1_ATTR_KEYWORDS.values():
    V15_1_ALL_ATTR_KEYWORDS |= _kws
V15_1_ALL_ATTR_KEYWORDS |= set(V15_1_IMPLICIT_QUERY_ATTR.keys())

# All attribute values flattened (used for entity exclusion)
V15_1_ALL_ATTR_VALUES: set = set()
for _attr in ("color", "size", "location", "state"):
    V15_1_ALL_ATTR_VALUES |= set(V15_ATTR_VALUES.get(_attr, []))

# Stopwords that cannot be entity
V15_1_STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "are", "it", "that",
    "this", "and", "or", "but", "of", "in", "on", "at", "to",
    "what", "which", "who", "where", "when", "how", "why",
    "describe", "tell", "me", "be", "been", "being",
    "has", "have", "had", "does", "do", "did",
    "i", "he", "she", "they", "we", "you",
    "my", "your", "his", "her", "its", "their", "our",
    "there", "here", "now", "then",
    "very", "quite", "so", "too", "also",
    "appeared", "appears", "seemed", "seems", "looked", "looks",
    "noticed", "everyone", "stood", "nearby", "walked", "walks",
    "painted", "found", "lived", "remained", "rested", "clearly",
    "situated", "holds", "site", "dimension", "measures",
    "presently", "pigment", "shade", "mood", "condition",
    "say", "can", "can't", "cannot",
}


def _keywords_to_attr_type(tokens_lower: List[str]) -> Optional[str]:
    """Scan tokens for attribute keyword; return attr_type or None."""
    # First check explicit keywords
    for t in tokens_lower:
        for attr, kws in V15_1_ATTR_KEYWORDS.items():
            if t in kws:
                return attr
    # Then check implicit triggers
    for t in tokens_lower:
        if t in V15_1_IMPLICIT_QUERY_ATTR:
            return V15_1_IMPLICIT_QUERY_ATTR[t]
    return None


def _find_value_in_text(text_lower: str, attr_type: str) -> Optional[str]:
    """For a fact, scan for a known value of the declared attribute type.
    Returns value string or None. Uses word-boundary-aware check.
    """
    vocab = V15_ATTR_VALUES.get(attr_type, [])
    # Tokenize text for clean word match
    tokens = text_lower.split()
    tokens_clean = [t.rstrip(V15_1_TRAILING_PUNCT) for t in tokens]
    for v in vocab:
        if v in tokens_clean:
            return v
    return None


def _find_entity_span(text: str) -> Optional[str]:
    """Find entity span in fact or query text.
    
    Strategy: find first noun that is NOT a stopword, NOT an attribute
    keyword, and NOT an attribute value.
    """
    words = text.split()
    lower = [w.lower().rstrip(V15_1_TRAILING_PUNCT) for w in words]
    
    # Prefer: first noun after "the/a/an" that is not stopword/keyword/value
    for i, w in enumerate(lower):
        if w in ("the", "a", "an") and i + 1 < len(lower):
            candidate = lower[i + 1]
            if (candidate and candidate.isalpha() and
                candidate not in V15_1_STOPWORDS and
                candidate not in V15_1_ALL_ATTR_KEYWORDS and
                candidate not in V15_1_ALL_ATTR_VALUES):
                return candidate
    # Fallback: first alpha word that is not stopword/keyword/value
    for w in lower:
        if (w.isalpha() and
            w not in V15_1_STOPWORDS and
            w not in V15_1_ALL_ATTR_KEYWORDS and
            w not in V15_1_ALL_ATTR_VALUES):
            return w
    return None


@dataclass
class ParseResult:
    """Result of parsing one fact or query."""
    kind:           str   # "fact" | "query"
    entity_id:      Optional[str]   # None if parse failed
    attr_type:      Optional[str]   # None if parse failed
    value_idx:      Optional[int]   # for facts only; None otherwise
    class_hint:     Optional[int]   # from anchor; None otherwise
    is_anchor:      bool            # True if "X is a <class>"
    entity_ok:      bool
    attr_ok:        bool
    value_ok:       bool
    anchor_ok:      bool            # True if no anchor OR anchor parsed
    
    @property
    def parse_failed(self) -> bool:
        if self.kind == "fact":
            return not (self.entity_ok and self.attr_ok and self.value_ok)
        return not (self.entity_ok and self.attr_ok)


def parse_fact(text: str) -> ParseResult:
    """Parse a fact into (entity_id, attr_type, value_idx, class_hint?).
    
    Handles:
      - class anchor:   "A phoenix is a creature."
      - attribute fact: "The cat has red color."
      - paraphrase:     "The cat is red."  (implicit color)
      - coref:          "It was gold."     (no entity - rely on prior anchor)
    """
    # Check class anchor first
    anchor_ent, anchor_cls = detect_class_anchor(text)
    if anchor_ent is not None and anchor_cls is not None:
        return ParseResult(
            kind="fact",
            entity_id=canonicalize_entity(anchor_ent),
            attr_type=None,
            value_idx=None,
            class_hint=anchor_cls,
            is_anchor=True,
            entity_ok=True, attr_ok=True, value_ok=True, anchor_ok=True,
        )
    
    tokens = text.split()
    lower = [t.lower().rstrip(V15_1_TRAILING_PUNCT) for t in tokens]
    
    # Entity
    ent_raw = _find_entity_span(text)
    entity_ok = ent_raw is not None
    entity_id = canonicalize_entity(ent_raw) if ent_raw else None
    
    # Attribute type (explicit keyword or implicit via value lookup)
    attr_type = _keywords_to_attr_type(lower)
    # Implicit: if no keyword, infer from value span
    if attr_type is None:
        for candidate_attr in ("color", "size", "location", "state"):
            v = _find_value_in_text(text.lower(), candidate_attr)
            if v is not None:
                attr_type = candidate_attr
                break
    attr_ok = attr_type is not None
    
    # Value
    value_idx = None
    value_ok = False
    if attr_type is not None:
        v = _find_value_in_text(text.lower(), attr_type)
        if v is not None:
            vocab = V15_ATTR_VALUES[attr_type]
            value_idx = vocab.index(v)
            value_ok = True
    
    return ParseResult(
        kind="fact",
        entity_id=entity_id,
        attr_type=attr_type,
        value_idx=value_idx,
        class_hint=None,
        is_anchor=False,
        entity_ok=entity_ok, attr_ok=attr_ok, value_ok=value_ok, anchor_ok=True,
    )


def parse_query(text: str) -> ParseResult:
    """Parse a query into (entity_id, attr_type). No value_idx."""
    tokens = text.split()
    lower = [t.lower().rstrip(V15_1_TRAILING_PUNCT) for t in tokens]
    
    ent_raw = _find_entity_span(text)
    entity_ok = ent_raw is not None
    entity_id = canonicalize_entity(ent_raw) if ent_raw else None
    
    attr_type = _keywords_to_attr_type(lower)
    attr_ok = attr_type is not None
    
    return ParseResult(
        kind="query",
        entity_id=entity_id,
        attr_type=attr_type,
        value_idx=None,
        class_hint=None,
        is_anchor=False,
        entity_ok=entity_ok, attr_ok=attr_ok, value_ok=True, anchor_ok=True,
    )


# --- A.3 AttributeSlot + ObjectRecord (deterministic) ---------------------

@dataclass
class AttributeSlot:
    """Attribute slot within an object record.
    value_idx is the PRIMARY TRUTH. value_emb is auxiliary for shadow only.
    """
    present:    bool = False
    value_idx:  int = -1
    version:    int = 0
    write_step: int = -1
    value_emb:  Optional[torch.Tensor] = None   # auxiliary for shadow training


@dataclass
class ObjectRecord:
    """Single object record. entity_id is PRIMARY KEY."""
    entity_id:       str
    entity_emb:      torch.Tensor       # auxiliary for shadow
    class_id:        int = -1
    class_emb:       Optional[torch.Tensor] = None
    uncertainty:     float = 0.0
    last_write_step: int = -1
    attr_slots:      Dict[str, AttributeSlot] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.attr_slots:
            for a in ("color", "size", "location", "state"):
                self.attr_slots[a] = AttributeSlot()


# --- A.4 Deterministic ObjectBank -----------------------------------------

class MemoryFullError(Exception):
    """Raised when bank is full and cannot allocate a new object."""
    pass


class DeterministicObjectBank:
    """Object bank with exact string-match addressing via entity_id.
    
    No cosine. No threshold. No learned routing in critical path.
    """
    
    def __init__(self, capacity: int = 32, d_model: int = 768):
        self.capacity = capacity
        self.d_model  = d_model
        self._records: Dict[int, ObjectRecord] = {}
        self._entity_id_to_slot: Dict[str, int] = {}
        self._next_slot = 0
    
    def find_by_entity_id(self, entity_id: str) -> Optional[int]:
        """Exact canonical match. Returns slot index or None."""
        canonical = canonicalize_entity(entity_id)
        return self._entity_id_to_slot.get(canonical)
    
    def is_full(self) -> bool:
        return len(self._records) >= self.capacity
    
    def allocate_new(self, entity_id: str, entity_emb: torch.Tensor,
                      class_hint: Optional[int] = None,
                      class_emb:  Optional[torch.Tensor] = None,
                      step: int = 0) -> int:
        """Allocate a new slot for entity_id. Raises MemoryFullError if full."""
        if self.is_full():
            raise MemoryFullError(
                f"Bank capacity ({self.capacity}) exceeded. "
                f"Not NONE_OBJECT - this is physical memory exhaustion."
            )
        canonical = canonicalize_entity(entity_id)
        if canonical in self._entity_id_to_slot:
            return self._entity_id_to_slot[canonical]
        slot = self._next_slot
        self._next_slot += 1
        rec = ObjectRecord(
            entity_id=canonical,
            entity_emb=entity_emb.detach().clone(),
            class_id=(class_hint if class_hint is not None else -1),
            class_emb=(class_emb.detach().clone() if class_emb is not None else None),
            uncertainty=(0.0 if class_hint is not None else 1.0),
            last_write_step=step,
        )
        self._records[slot] = rec
        self._entity_id_to_slot[canonical] = slot
        return slot
    
    def write_attribute(self, entity_id: str, attr_type: str, value_idx: int,
                         step: int, value_emb: Optional[torch.Tensor] = None):
        """Write a single attribute slot. OTHER SLOTS REMAIN BYTE-IDENTICAL."""
        slot = self.find_by_entity_id(entity_id)
        if slot is None:
            raise KeyError(f"entity_id '{entity_id}' not in bank - call allocate_new first")
        rec = self._records[slot]
        old = rec.attr_slots[attr_type]
        new = AttributeSlot(
            present=True,
            value_idx=value_idx,
            version=old.version + 1,
            write_step=step,
            value_emb=(value_emb.detach().clone() if value_emb is not None else old.value_emb),
        )
        rec.attr_slots[attr_type] = new
        rec.last_write_step = step
    
    def read_attribute(self, entity_id: str, attr_type: str) -> Tuple[str, Optional[int]]:
        """Deterministic read.
        
        Returns (status, value_idx):
          - ("FOUND", value_idx)       if object exists and attribute present
          - ("NONE_OBJECT", None)      if entity_id not in bank
          - ("NONE_ATTRIBUTE", None)   if object exists but attribute not written
        """
        slot = self.find_by_entity_id(entity_id)
        if slot is None:
            return ("NONE_OBJECT", None)
        rec = self._records[slot]
        if attr_type not in rec.attr_slots:
            return ("NONE_ATTRIBUTE", None)
        slot_data = rec.attr_slots[attr_type]
        if not slot_data.present:
            return ("NONE_ATTRIBUTE", None)
        return ("FOUND", slot_data.value_idx)
    
    def snapshot_slots(self) -> Dict[int, Dict[str, Tuple[bool, int, int]]]:
        """Take a snapshot of all (present, value_idx, version) for preserve check."""
        snap = {}
        for slot_idx, rec in self._records.items():
            snap[slot_idx] = {}
            for attr, s in rec.attr_slots.items():
                snap[slot_idx][attr] = (s.present, s.value_idx, s.version)
        return snap
    
    def reset(self):
        """Clear all records. Used at episode boundaries."""
        self._records.clear()
        self._entity_id_to_slot.clear()
        self._next_slot = 0
    
    def occupied_slots(self) -> List[int]:
        return sorted(self._records.keys())
    
    def get_record(self, slot: int) -> Optional[ObjectRecord]:
        return self._records.get(slot)


# --- A.5 Deterministic Write/Read Pipeline --------------------------------

READ_STATUS_FOUND          = "FOUND"
READ_STATUS_NONE_OBJECT    = "NONE_OBJECT"
READ_STATUS_NONE_ATTRIBUTE = "NONE_ATTRIBUTE"
READ_STATUS_PARSER_FAIL    = "PARSER_FAILURE"


def v15_1_write_fact(bank: DeterministicObjectBank, parse: ParseResult,
                      entity_emb_fn, class_emb_fn, value_emb_fn,
                      step: int) -> str:
    """Critical-path write. Returns status string.
    
    entity_emb_fn(entity_id) -> Tensor
    class_emb_fn(class_id, entity_emb) -> Tensor
    value_emb_fn(attr_type, value_idx) -> Tensor
    """
    if parse.parse_failed and not parse.is_anchor:
        return "PARSER_FAILURE_WRITE"
    
    # Anchor: allocate object with class hint, no attribute write
    if parse.is_anchor:
        ent_emb = entity_emb_fn(parse.entity_id)
        cls_emb = class_emb_fn(parse.class_hint, ent_emb) if parse.class_hint is not None else None
        slot = bank.find_by_entity_id(parse.entity_id)
        if slot is None:
            bank.allocate_new(parse.entity_id, ent_emb,
                              class_hint=parse.class_hint, class_emb=cls_emb, step=step)
        else:
            rec = bank.get_record(slot)
            if rec.class_id == -1 and parse.class_hint is not None:
                rec.class_id = parse.class_hint
                rec.class_emb = cls_emb
                rec.uncertainty = 0.0
        return "ANCHOR_WRITTEN"
    
    # Regular attribute fact
    ent_emb = entity_emb_fn(parse.entity_id)
    slot = bank.find_by_entity_id(parse.entity_id)
    if slot is None:
        try:
            bank.allocate_new(parse.entity_id, ent_emb, step=step)
        except MemoryFullError:
            return "MEMORY_FULL"
    val_emb = value_emb_fn(parse.attr_type, parse.value_idx)
    bank.write_attribute(parse.entity_id, parse.attr_type, parse.value_idx,
                          step=step, value_emb=val_emb)
    return "WRITTEN"


def v15_1_read_query(bank: DeterministicObjectBank,
                      parse: ParseResult) -> Tuple[str, Optional[int]]:
    """Critical-path read. Returns (status, value_idx_or_None).
    
    Status is one of: FOUND, NONE_OBJECT, NONE_ATTRIBUTE, PARSER_FAILURE.
    """
    if not parse.entity_ok:
        return (READ_STATUS_PARSER_FAIL, None)
    if not parse.attr_ok:
        return (READ_STATUS_PARSER_FAIL, None)
    return bank.read_attribute(parse.entity_id, parse.attr_type)


print("[v15.1] Section A: substrate components defined")
print("         - canonicalize_entity")
print("         - parse_fact / parse_query with ParseResult")
print("         - DeterministicObjectBank (exact entity_id match, NO cosine)")
print("         - v15_1_write_fact / v15_1_read_query (critical path)")
print(f"         - alias_map size: {len(V15_1_ALIAS_MAP)}")

# ======================== B. V15.1 SUBSTRATE VALIDATION =====================
#
# Validates the deterministic substrate WITHOUT any training.
# If substrate doesn't score ~100% here, we have an implementation bug,
# not a learning problem.
#
# Outputs three separate verdicts:
#   1. Memory Substrate  (P1-P5, A2, A3, A5)
#   2. Parser Robustness (parser coverage, P6, P7, A6)
#   3. Shadow Readiness  (A1 critical_vs_shadow - only after shadow training)
# ===========================================================================

import hashlib

# --- B.1 Frozen seeds and hashes -----------------------------------------

V15_1_BENCH_SEED       = 20260601
V15_1_AUDIT_SEED       = 20260614
V15_1_BENCH_SPLIT_SEED = 20260418   # same as v15 (shared held-out split)

V15_1_BENCHMARK_CONFIG = {
    "n_per_cell":  500,
    "bench_seed":  V15_1_BENCH_SEED,
    "audit_seed":  V15_1_AUDIT_SEED,
    "split_seed":  V15_1_BENCH_SPLIT_SEED,
}

print(f"[v15.1 BENCHMARK] seeds: bench={V15_1_BENCH_SEED} audit={V15_1_AUDIT_SEED} "
      f"split={V15_1_BENCH_SPLIT_SEED}")


# --- B.2 Auxiliary embedding functions (used by write) -------------------

def _make_entity_emb_fn(base_model):
    def fn(entity_id: str) -> torch.Tensor:
        toks = ENC.encode(" " + entity_id)
        if not toks:
            toks = ENC.encode(entity_id)
        tok_ids = torch.tensor(toks, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            emb = base_model.shared_token_emb(tok_ids).mean(dim=0)
        return emb
    return fn


def _make_class_emb_fn(v15_1_memory):
    def fn(class_id: int, entity_emb: torch.Tensor) -> torch.Tensor:
        if class_id is None or class_id < 0:
            return torch.zeros(v15_1_memory.class_encoder.d_sem, device=DEVICE)
        with torch.no_grad():
            z, _, _, _ = v15_1_memory.class_encoder(entity_emb.unsqueeze(0))
        return z.squeeze(0)
    return fn


def _make_value_emb_fn(base_model):
    def fn(attr_type: str, value_idx: int) -> torch.Tensor:
        v_string = V15_ATTR_VALUES[attr_type][value_idx]
        toks = ENC.encode(" " + v_string)
        tok_ids = torch.tensor(toks, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            emb = base_model.shared_token_emb(tok_ids).mean(dim=0)
        return emb
    return fn


# --- B.3 Trial record (for memory substrate probes) ----------------------

@dataclass
class V15_1_TrialRecord:
    probe:                 str
    episode_type:          str
    episode_seed:          int
    # Parser status (PER TRIAL)
    query_parser_ok:       bool
    facts_all_parsed:      bool
    # Critical path outcome
    read_status:           str       # FOUND / NONE_OBJECT / NONE_ATTRIBUTE / PARSER_FAILURE
    predicted_value_idx:   Optional[int]
    target_value_idx:      Optional[int]
    target_is_unknown_obj: bool
    target_is_unknown_attr: bool
    # Correctness
    correct:               bool


# --- B.4 Run one episode on the deterministic substrate ------------------

def _v15_1_run_trial(bank: DeterministicObjectBank, base_model, v15_1_memory,
                     ep: V15Episode, probe_name: str,
                     episode_seed: int) -> V15_1_TrialRecord:
    """Run one episode: write all facts then read query, all deterministic."""
    bank.reset()
    
    entity_emb_fn = _make_entity_emb_fn(base_model)
    class_emb_fn  = _make_class_emb_fn(v15_1_memory)
    value_emb_fn  = _make_value_emb_fn(base_model)
    
    facts_all_parsed = True
    # Write all facts
    for step_idx, fact_text in enumerate(ep.facts):
        parse = parse_fact(fact_text)
        if parse.parse_failed and not parse.is_anchor:
            facts_all_parsed = False
            continue
        status = v15_1_write_fact(bank, parse,
                                    entity_emb_fn, class_emb_fn, value_emb_fn,
                                    step=step_idx)
    
    # Read query
    q_parse = parse_query(ep.query)
    read_status, pred_idx = v15_1_read_query(bank, q_parse)
    
    # Determine target
    target_idx = None
    if not ep.target_is_unknown:
        attr_type = V15_ATTR_TYPES[ep.query_attr_label]
        if attr_type in V15_ATTR_VALUES:
            vocab = V15_ATTR_VALUES[attr_type]
            t_tok = int(ep.target_answer_token)
            # Reconstruct target value string from answer token via vocab
            for i, v_str in enumerate(vocab):
                if V15_ANSWER_TOKENS[attr_type].get(v_str) == t_tok:
                    target_idx = i
                    break
    
    # Determine correctness
    target_is_none_obj  = False
    target_is_none_attr = False
    if ep.target_is_unknown:
        # Check if it's AE (absent entity) or AA (absent attribute) by
        # inspecting whether the query entity appears among facts
        q_ent_canonical = canonicalize_entity(q_parse.entity_id) if q_parse.entity_id else ""
        fact_entity_ids = set()
        for f in ep.facts:
            p = parse_fact(f)
            if p.entity_id is not None:
                fact_entity_ids.add(p.entity_id)
        if q_ent_canonical not in fact_entity_ids:
            target_is_none_obj = True
        else:
            target_is_none_attr = True
    
    # Correct if status and value_idx match target
    if target_is_none_obj:
        correct = (read_status == READ_STATUS_NONE_OBJECT)
    elif target_is_none_attr:
        correct = (read_status == READ_STATUS_NONE_ATTRIBUTE)
    else:
        correct = (read_status == READ_STATUS_FOUND and pred_idx == target_idx)
    
    return V15_1_TrialRecord(
        probe=probe_name, episode_type=ep.episode_type, episode_seed=episode_seed,
        query_parser_ok=(q_parse.entity_ok and q_parse.attr_ok),
        facts_all_parsed=facts_all_parsed,
        read_status=read_status,
        predicted_value_idx=pred_idx,
        target_value_idx=target_idx,
        target_is_unknown_obj=target_is_none_obj,
        target_is_unknown_attr=target_is_none_attr,
        correct=correct,
    )


def _v15_1_agg(trials: List[V15_1_TrialRecord]) -> Dict:
    n = len(trials)
    if n == 0:
        return {"n": 0}
    # Parser stats
    parsed = [t for t in trials if t.query_parser_ok and t.facts_all_parsed]
    n_parsed = len(parsed)
    parser_coverage = n_parsed / n
    # Accuracy on parsed trials only
    acc_parsed = sum(1 for t in parsed if t.correct) / max(1, n_parsed)
    # Status distribution
    status_dist = {}
    for t in trials:
        status_dist[t.read_status] = status_dist.get(t.read_status, 0) + 1
    return {
        "n":               n,
        "n_parsed":        n_parsed,
        "parser_coverage": parser_coverage,
        "accuracy":        acc_parsed,
        "status_dist":     status_dist,
    }


# --- B.5 Memory Substrate Probes (P1-P5, A2, A3, A5) ---------------------

def _bench_P1(bank, base_model, v15_1_mem, cfg):
    """Single attribute retrieval (4 attribute types)."""
    rng = random.Random(cfg["bench_seed"] + 1)
    results = {}
    for attr_type in ("color", "size", "location", "state"):
        trials = []
        for i in range(cfg["n_per_cell"]):
            ep = v15_generate_episode("single_attr_simple", rng, use_heldout=True)
            tries = 0
            while (V15_ATTR_TYPES[ep.fact_attr_labels[0]] != attr_type and tries < 10):
                ep = v15_generate_episode("single_attr_simple", rng, use_heldout=True)
                tries += 1
            trials.append(_v15_1_run_trial(bank, base_model, v15_1_mem, ep,
                                             f"P1_{attr_type}", cfg["bench_seed"] * 100 + i))
        results[attr_type] = _v15_1_agg(trials)
    return results


def _bench_P2(bank, base_model, v15_1_mem, cfg):
    rng = random.Random(cfg["bench_seed"] + 2)
    results = {}
    for attr_type in ("color", "size", "location", "state"):
        trials = []
        for i in range(cfg["n_per_cell"]):
            ep = v15_generate_episode("multi_attr_object", rng, use_heldout=True)
            tries = 0
            while (V15_ATTR_TYPES[ep.query_attr_label] != attr_type and tries < 20):
                ep = v15_generate_episode("multi_attr_object", rng, use_heldout=True)
                tries += 1
            trials.append(_v15_1_run_trial(bank, base_model, v15_1_mem, ep,
                                             f"P2_{attr_type}", cfg["bench_seed"] * 200 + i))
        results[attr_type] = _v15_1_agg(trials)
    return results


def _bench_P3(bank, base_model, v15_1_mem, cfg):
    rng = random.Random(cfg["bench_seed"] + 3)
    trials = []
    for i in range(cfg["n_per_cell"]):
        ep = v15_generate_episode("selective_update", rng, use_heldout=True)
        trials.append(_v15_1_run_trial(bank, base_model, v15_1_mem, ep,
                                         "P3", cfg["bench_seed"] * 300 + i))
    return _v15_1_agg(trials)


def _bench_P4_AE(bank, base_model, v15_1_mem, cfg):
    rng = random.Random(cfg["bench_seed"] + 4)
    trials = []
    for i in range(cfg["n_per_cell"]):
        ep = v15_generate_episode("no_match", rng, use_heldout=True)
        trials.append(_v15_1_run_trial(bank, base_model, v15_1_mem, ep,
                                         "P4_AE", cfg["bench_seed"] * 410 + i))
    return _v15_1_agg(trials)


def _bench_P4_AA(bank, base_model, v15_1_mem, cfg):
    """Build AA episodes explicitly: multi_attr + query attribute not written."""
    rng = random.Random(cfg["bench_seed"] + 45)
    trials = []
    for i in range(cfg["n_per_cell"]):
        ep_full = v15_generate_episode("multi_attr_object", rng, use_heldout=True)
        # Modify to query attribute NOT in facts for query entity
        q_ent_tok = ep_full.query_entity_token
        attrs_written = set()
        for j, etok in enumerate(ep_full.fact_entity_tokens):
            if etok == q_ent_tok:
                attrs_written.add(V15_ATTR_TYPES[ep_full.fact_attr_labels[j]])
        all_attrs = {"color", "size", "location", "state"}
        absent_attrs = list(all_attrs - attrs_written)
        if not absent_attrs:
            removed = rng.choice(list(all_attrs))
            keep = []
            for j in range(len(ep_full.facts)):
                if (ep_full.fact_entity_tokens[j] == q_ent_tok and
                    V15_ATTR_TYPES[ep_full.fact_attr_labels[j]] == removed):
                    continue
                keep.append(j)
            ep_full.facts              = [ep_full.facts[j] for j in keep]
            ep_full.fact_entity_tokens = [ep_full.fact_entity_tokens[j] for j in keep]
            ep_full.fact_attr_labels   = [ep_full.fact_attr_labels[j] for j in keep]
            ep_full.fact_answer_tokens = [ep_full.fact_answer_tokens[j] for j in keep]
            ep_full.fact_class_labels  = [ep_full.fact_class_labels[j] for j in keep]
            ep_full.fact_is_anchor     = [ep_full.fact_is_anchor[j] for j in keep]
            absent_attrs = [removed]
        ent_word = ENC.decode([q_ent_tok]).strip()
        chosen = rng.choice(absent_attrs)
        ep_full.query = v15_render_query(ent_word, chosen, rng)
        ep_full.query_attr_label    = V15_ATTR_TO_IDX[chosen]
        ep_full.target_answer_token = V15_UNKNOWN_ANSWER_TOKEN
        ep_full.target_is_unknown   = True
        ep_full.target_fact_idx     = -1
        ep_full.target_slot_name    = "unknown"
        ep_full.episode_type        = "absent_attribute"
        trials.append(_v15_1_run_trial(bank, base_model, v15_1_mem, ep_full,
                                         "P4_AA", cfg["bench_seed"] * 450 + i))
    return _v15_1_agg(trials)


def _generate_scaling_episode(rng: random.Random, n_facts: int,
                                use_heldout: bool = True) -> V15Episode:
    """Generate single_attr_simple episode with EXACT n_facts entities.
    
    Used by P5 to test scaling at fixed n=3,5,8,12.
    """
    pool = V15_HELDOUT_ENTITIES if use_heldout else V15_TRAIN_ENTITIES
    if n_facts > len(pool):
        raise ValueError(f"n_facts={n_facts} > pool size {len(pool)}")
    ents = rng.sample(pool, n_facts)
    attr = rng.choice(["color", "size", "location", "state"])
    facts, fact_entity_toks, fact_attr_lbls, fact_ans_toks = [], [], [], []
    fact_cls_lbls, fact_is_anchor = [], []
    values = []
    for (e, cls) in ents:
        v = rng.choice(V15_ATTR_VALUES[attr])
        fact = v15_render_fact(e, attr, v, rng)
        facts.append(fact)
        fact_entity_toks.append(v15_first_token(e))
        fact_attr_lbls.append(V15_ATTR_TO_IDX[attr])
        fact_ans_toks.append(V15_ANSWER_TOKENS[attr][v])
        fact_cls_lbls.append(cls)
        fact_is_anchor.append(False)
        values.append(v)
    t_idx = rng.randint(0, n_facts - 1)
    (t_ent, _) = ents[t_idx]
    t_val = values[t_idx]
    query = v15_render_query(t_ent, attr, rng)
    return V15Episode(
        episode_type=f"scaling_n{n_facts}",
        facts=facts, fact_entity_tokens=fact_entity_toks,
        fact_attr_labels=fact_attr_lbls, fact_answer_tokens=fact_ans_toks,
        fact_class_labels=fact_cls_lbls, fact_is_anchor=fact_is_anchor,
        query=query, query_attr_label=V15_ATTR_TO_IDX[attr],
        query_entity_token=v15_first_token(t_ent),
        target_answer_token=V15_ANSWER_TOKENS[attr][t_val],
        target_is_unknown=False, target_fact_idx=t_idx,
        target_slot_name=attr,
    )


def _bench_P5(bank, base_model, v15_1_mem, cfg):
    """P5: scaling 3/5/8/12 facts. Substrate must be invariant to n_facts."""
    rng = random.Random(cfg["bench_seed"] + 5)
    by_n_facts = {}
    n_per_bucket = cfg["n_per_cell"] // 4
    for n_facts_target in (3, 5, 8, 12):
        if n_facts_target > len(V15_HELDOUT_ENTITIES):
            by_n_facts[f"n{n_facts_target}"] = {
                "n": 0, "n_parsed": 0, "parser_coverage": 0.0, "accuracy": 0.0,
                "status_dist": {}, "skipped": "n_facts > heldout pool size",
            }
            continue
        trials = []
        for i in range(n_per_bucket):
            ep = _generate_scaling_episode(rng, n_facts_target, use_heldout=True)
            trials.append(_v15_1_run_trial(bank, base_model, v15_1_mem, ep,
                                             f"P5_n{n_facts_target}",
                                             cfg["bench_seed"] * 500 + i))
        by_n_facts[f"n{n_facts_target}"] = _v15_1_agg(trials)
    return by_n_facts


def _audit_A2(bank, base_model, v15_1_mem, cfg):
    """A2: Query about absent entity. Must return NONE_OBJECT."""
    rng = random.Random(cfg["audit_seed"] + 200)
    trials = []
    for i in range(cfg["n_per_cell"]):
        ep = v15_generate_episode("no_match", rng, use_heldout=True)
        trials.append(_v15_1_run_trial(bank, base_model, v15_1_mem, ep,
                                         "A2", cfg["audit_seed"] * 200 + i))
    agg = _v15_1_agg(trials)
    # For A2, correct means NONE_OBJECT was returned
    none_obj_count = agg["status_dist"].get(READ_STATUS_NONE_OBJECT, 0)
    agg["nomatch_rate"] = none_obj_count / max(1, agg["n_parsed"])
    agg["_fail"] = agg["nomatch_rate"] < 0.99
    return agg


def _audit_A3(bank, base_model, v15_1_mem, cfg):
    """A3: Permute value_idx between slots after write. Read should change."""
    rng = random.Random(cfg["audit_seed"] + 300)
    correct_before = 0
    correct_after = 0
    n = min(cfg["n_per_cell"], 100)
    entity_emb_fn = _make_entity_emb_fn(base_model)
    class_emb_fn  = _make_class_emb_fn(v15_1_mem)
    value_emb_fn  = _make_value_emb_fn(base_model)
    for i in range(n):
        ep = v15_generate_episode("single_attr_simple", rng, use_heldout=True)
        # Before: normal deterministic read
        tr_before = _v15_1_run_trial(bank, base_model, v15_1_mem, ep,
                                       "A3_before", cfg["audit_seed"] * 300 + i)
        if tr_before.correct:
            correct_before += 1
        # After: write again, permute value_idx, re-read (bank is already populated from tr_before)
        q_parse = parse_query(ep.query)
        if q_parse.attr_ok:
            slots = bank.occupied_slots()
            if len(slots) >= 2:
                vals = []
                for s in slots:
                    rec = bank.get_record(s)
                    a = rec.attr_slots.get(q_parse.attr_type)
                    if a is not None and a.present:
                        vals.append((s, a.value_idx))
                if len(vals) >= 2:
                    rng_perm = random.Random(cfg["audit_seed"] + i * 17)
                    perm = [v for (_, v) in vals]
                    rng_perm.shuffle(perm)
                    tries = 0
                    while any(p == orig[1] for p, orig in zip(perm, vals)) and tries < 10:
                        rng_perm.shuffle(perm)
                        tries += 1
                    for (s, _), new_v in zip(vals, perm):
                        bank.get_record(s).attr_slots[q_parse.attr_type].value_idx = new_v
        status, pred = v15_1_read_query(bank, q_parse)
        target_idx = None
        if not ep.target_is_unknown:
            attr_type = V15_ATTR_TYPES[ep.query_attr_label]
            vocab = V15_ATTR_VALUES[attr_type]
            t_tok = int(ep.target_answer_token)
            for k, v_str in enumerate(vocab):
                if V15_ANSWER_TOKENS[attr_type].get(v_str) == t_tok:
                    target_idx = k
                    break
        if status == READ_STATUS_FOUND and pred == target_idx:
            correct_after += 1
    acc_before = correct_before / n
    acc_after  = correct_after  / n
    drop       = acc_before - acc_after
    return {
        "n":                n,
        "accuracy_before":  acc_before,
        "accuracy_after":   acc_after,
        "drop":             drop,
        "_fail":            (drop < 0.50),
    }


def _audit_A4_canonicalization(bank, base_model, v15_1_mem, cfg):
    """A4 (replaced): Canonicalization Stress Test.
    Tests whether entity_id pipeline handles case, determiners, punctuation.
    
    Uses the parser's own entity_id extraction as ground truth (not GPT-2
    tokenization, which can split words like 'mermaid' into 'mer'+'maid').
    """
    rng = random.Random(cfg["audit_seed"] + 400)
    variants_tested = 0
    variants_ok = 0
    for i in range(min(cfg["n_per_cell"], 100)):
        ep = v15_generate_episode("single_attr_simple", rng, use_heldout=True)
        # Use parser's entity_id extraction as ground truth
        q_parse = parse_query(ep.query)
        if not q_parse.entity_ok:
            continue
        ent_word = q_parse.entity_id   # already canonicalized via parser
        # Generate orthographic/format variants of the SAME entity word
        variants = [
            f"The {ent_word.upper()}",
            f"the {ent_word}",
            f"A {ent_word}",
            f"{ent_word.capitalize()}",
            f"{ent_word},",
            f" {ent_word}  ",
            f"AN {ent_word}",
            f"{ent_word}.",
        ]
        for v in variants:
            variants_tested += 1
            cid = canonicalize_entity(v)
            if cid == ent_word:
                variants_ok += 1
    rate = variants_ok / max(1, variants_tested)
    return {
        "variants_tested": variants_tested,
        "variants_ok":     variants_ok,
        "pass_rate":       rate,
        "_fail":           rate < 0.95,
    }


def _audit_A5(bank, base_model, v15_1_mem, cfg):
    """A5: Cross-entity transfer. A has color, B has size. Query color of B.
    Must return NONE_ATTRIBUTE."""
    rng = random.Random(cfg["audit_seed"] + 500)
    trials = []
    leakage_count = 0
    for i in range(cfg["n_per_cell"]):
        ents = rng.sample(V15_HELDOUT_ENTITIES, 2)
        (ent_A, cls_A), (ent_B, cls_B) = ents
        col = rng.choice(V15_COLORS)
        siz = rng.choice(V15_SIZES)
        fact_a = v15_render_fact(ent_A, "color", col, rng)
        fact_b = v15_render_fact(ent_B, "size", siz, rng)
        query  = v15_render_query(ent_B, "color", rng)
        ep = V15Episode(
            episode_type="A5_transfer",
            facts=[fact_a, fact_b],
            fact_entity_tokens=[v15_first_token(ent_A), v15_first_token(ent_B)],
            fact_attr_labels=[V15_ATTR_TO_IDX["color"], V15_ATTR_TO_IDX["size"]],
            fact_answer_tokens=[V15_ANSWER_TOKENS["color"][col],
                                V15_ANSWER_TOKENS["size"][siz]],
            fact_class_labels=[cls_A, cls_B], fact_is_anchor=[False, False],
            query=query, query_attr_label=V15_ATTR_TO_IDX["color"],
            query_entity_token=v15_first_token(ent_B),
            target_answer_token=V15_UNKNOWN_ANSWER_TOKEN,
            target_is_unknown=True, target_fact_idx=-1, target_slot_name="unknown",
        )
        tr = _v15_1_run_trial(bank, base_model, v15_1_mem, ep,
                                "A5", cfg["audit_seed"] * 500 + i)
        trials.append(tr)
        # Leakage: returned a value instead of NONE
        if tr.read_status == READ_STATUS_FOUND:
            leakage_count += 1
    agg = _v15_1_agg(trials)
    agg["cross_entity_leakage_rate"] = leakage_count / max(1, agg["n_parsed"])
    none_attr_count = agg["status_dist"].get(READ_STATUS_NONE_ATTRIBUTE, 0)
    agg["nomatch_attr_rate"] = none_attr_count / max(1, agg["n_parsed"])
    agg["_fail"] = (agg["cross_entity_leakage_rate"] > 0.01 or
                    agg["nomatch_attr_rate"] < 0.99)
    return agg


# --- B.6 Parser Robustness Probes (P6, P7, A6) ---------------------------

def _bench_P6(bank, base_model, v15_1_mem, cfg):
    rng = random.Random(cfg["bench_seed"] + 6)
    trials = []
    for i in range(cfg["n_per_cell"]):
        ep = v15_generate_episode("paraphrase", rng, use_heldout=True)
        trials.append(_v15_1_run_trial(bank, base_model, v15_1_mem, ep,
                                         "P6", cfg["bench_seed"] * 600 + i))
    return _v15_1_agg(trials)


def _bench_P7(bank, base_model, v15_1_mem, cfg):
    rng = random.Random(cfg["bench_seed"] + 7)
    trials = []
    for i in range(cfg["n_per_cell"]):
        ep = v15_generate_episode("coreference_distant", rng, use_heldout=True)
        trials.append(_v15_1_run_trial(bank, base_model, v15_1_mem, ep,
                                         "P7", cfg["bench_seed"] * 700 + i))
    return _v15_1_agg(trials)


def _audit_A6(bank, base_model, v15_1_mem, cfg):
    """A6: template ablation - novel templates."""
    rng = random.Random(cfg["audit_seed"] + 600)
    novel_queries = {
        "color":    ["The {e} appears in what shade? The {e} appears",
                     "Say the pigment of the {e}. It is"],
        "size":     ["The {e} measures how? The {e} is",
                     "What is the dimension of the {e}? It is"],
        "location": ["The {e} is situated where? The {e} is in the",
                     "Which site holds the {e}? The {e} is in the"],
        "state":    ["The {e} is presently how? The {e} is",
                     "Describe the mood of the {e}. It is"],
    }
    trials = []
    for i in range(min(cfg["n_per_cell"], 200)):
        ep = v15_generate_episode("single_attr_simple", rng, use_heldout=True)
        attr_type = V15_ATTR_TYPES[ep.fact_attr_labels[0]]
        ent_word = ENC.decode([ep.query_entity_token]).strip()
        novel_tpl = rng.choice(novel_queries[attr_type])
        ep.query = novel_tpl.format(e=ent_word)
        trials.append(_v15_1_run_trial(bank, base_model, v15_1_mem, ep,
                                         "A6", cfg["audit_seed"] * 600 + i))
    return _v15_1_agg(trials)


# --- B.7 Validation runner (substrate-only, no training) -----------------

def v15_1_validate_substrate(bank, base_model, v15_1_mem, cfg=None) -> Dict:
    """Run full substrate validation: no training required."""
    if cfg is None:
        cfg = V15_1_BENCHMARK_CONFIG
    
    print()
    print(SEP)
    print("[v15.1 SUBSTRATE VALIDATION] (deterministic, no training)")
    print(f"  bench_seed = {cfg['bench_seed']}")
    print(f"  audit_seed = {cfg['audit_seed']}")
    print(f"  n_per_cell = {cfg['n_per_cell']}")
    print(SEP)
    
    results = {"config": dict(cfg)}
    
    # Memory substrate probes
    print()
    print("--- Memory Substrate Probes ---")
    results["P1"]    = _bench_P1(bank, base_model, v15_1_mem, cfg)
    for k, v in results["P1"].items():
        print(f"P1_{k}: parser_coverage={v['parser_coverage']:.1%}  "
              f"accuracy_on_parsed={v['accuracy']:.1%}  n={v['n']}")
    results["P2"]    = _bench_P2(bank, base_model, v15_1_mem, cfg)
    for k, v in results["P2"].items():
        print(f"P2_{k}: parser_coverage={v['parser_coverage']:.1%}  "
              f"accuracy_on_parsed={v['accuracy']:.1%}  n={v['n']}")
    results["P3"]    = _bench_P3(bank, base_model, v15_1_mem, cfg)
    print(f"P3:    parser_coverage={results['P3']['parser_coverage']:.1%}  "
          f"accuracy_on_parsed={results['P3']['accuracy']:.1%}")
    results["P4_AE"] = _bench_P4_AE(bank, base_model, v15_1_mem, cfg)
    print(f"P4_AE: parser_coverage={results['P4_AE']['parser_coverage']:.1%}  "
          f"accuracy_on_parsed={results['P4_AE']['accuracy']:.1%}")
    results["P4_AA"] = _bench_P4_AA(bank, base_model, v15_1_mem, cfg)
    print(f"P4_AA: parser_coverage={results['P4_AA']['parser_coverage']:.1%}  "
          f"accuracy_on_parsed={results['P4_AA']['accuracy']:.1%}")
    results["P5"]    = _bench_P5(bank, base_model, v15_1_mem, cfg)
    for k, v in results["P5"].items():
        cov = v.get("parser_coverage", 0.0)
        acc = v.get("accuracy", 0.0)
        print(f"P5_{k}: parser_coverage={cov:.1%}  "
              f"accuracy_on_parsed={acc:.1%}  n={v.get('n', 0)}")
    
    results["A2"] = _audit_A2(bank, base_model, v15_1_mem, cfg)
    print(f"A2:    nomatch_rate={results['A2']['nomatch_rate']:.1%}  "
          f"FAIL={results['A2']['_fail']}")
    results["A3"] = _audit_A3(bank, base_model, v15_1_mem, cfg)
    print(f"A3:    before={results['A3']['accuracy_before']:.1%}  "
          f"after={results['A3']['accuracy_after']:.1%}  "
          f"drop={results['A3']['drop']:.1%}  FAIL={results['A3']['_fail']}")
    results["A4"] = _audit_A4_canonicalization(bank, base_model, v15_1_mem, cfg)
    print(f"A4 (canonicalization): pass_rate={results['A4']['pass_rate']:.1%}  "
          f"FAIL={results['A4']['_fail']}")
    results["A5"] = _audit_A5(bank, base_model, v15_1_mem, cfg)
    print(f"A5:    leakage={results['A5']['cross_entity_leakage_rate']:.1%}  "
          f"nomatch_attr={results['A5']['nomatch_attr_rate']:.1%}  "
          f"FAIL={results['A5']['_fail']}")
    
    # Parser robustness probes
    print()
    print("--- Parser Robustness Probes ---")
    results["P6"] = _bench_P6(bank, base_model, v15_1_mem, cfg)
    print(f"P6:    parser_coverage={results['P6']['parser_coverage']:.1%}  "
          f"accuracy_on_parsed={results['P6']['accuracy']:.1%}")
    results["P7"] = _bench_P7(bank, base_model, v15_1_mem, cfg)
    print(f"P7:    parser_coverage={results['P7']['parser_coverage']:.1%}  "
          f"accuracy_on_parsed={results['P7']['accuracy']:.1%}")
    results["A6"] = _audit_A6(bank, base_model, v15_1_mem, cfg)
    print(f"A6:    parser_coverage={results['A6']['parser_coverage']:.1%}  "
          f"accuracy_on_parsed={results['A6']['accuracy']:.1%}")
    
    # Compute verdicts
    print()
    print(SEP)
    print("--- VERDICTS ---")
    print(SEP)
    
    # P5 scaling status: COMPLETE iff all 4 buckets have n>=10 trials AND all pass
    p5_buckets = results["P5"]
    p5_status = "COMPLETE"
    p5_min_acc = 1.0
    p5_min_n = float('inf')
    p5_skipped = []
    for k, v in p5_buckets.items():
        n = v.get("n", 0)
        acc = v.get("accuracy", 0.0)
        if n < 10:
            p5_status = "INCOMPLETE"
            p5_skipped.append(f"{k} (n={n})")
        if n > 0:
            p5_min_acc = min(p5_min_acc, acc)
            p5_min_n = min(p5_min_n, n)
    if p5_min_n == float('inf'):
        p5_min_n = 0
    
    # Verdict 1: Memory Substrate
    mem_probes = {
        "P1_min":  min(v["accuracy"] for v in results["P1"].values()),
        "P2_min":  min(v["accuracy"] for v in results["P2"].values()),
        "P3":      results["P3"]["accuracy"],
        "P4_AE":   results["P4_AE"]["accuracy"],
        "P4_AA":   results["P4_AA"]["accuracy"],
        "P5_min":  p5_min_acc if p5_status == "COMPLETE" else 0.0,
        "A2_nomatch": results["A2"]["nomatch_rate"],
        "A3_drop":    results["A3"]["drop"],
        "A5_unknown": results["A5"]["nomatch_attr_rate"],
        "A5_leakage": results["A5"]["cross_entity_leakage_rate"],
    }
    mem_pass = (
        mem_probes["P1_min"] >= 0.99 and
        mem_probes["P2_min"] >= 0.99 and
        mem_probes["P3"]     >= 0.99 and
        mem_probes["P4_AE"]  >= 0.99 and
        mem_probes["P4_AA"]  >= 0.99 and
        (p5_status == "COMPLETE" and mem_probes["P5_min"] >= 0.99) and
        mem_probes["A2_nomatch"] >= 0.99 and
        mem_probes["A3_drop"]    >= 0.50 and
        mem_probes["A5_unknown"] >= 0.99 and
        mem_probes["A5_leakage"] <= 0.01
    )
    
    # Verdict 2: Parser Coverage (parser found SOMETHING)
    coverage_metrics = {
        "P6_coverage":  results["P6"]["parser_coverage"],
        "P7_coverage":  results["P7"]["parser_coverage"],
        "A4_canonical": results["A4"]["pass_rate"],
        "A6_coverage":  results["A6"]["parser_coverage"],
    }
    coverage_pass = (
        coverage_metrics["P6_coverage"]  >= 0.90 and
        coverage_metrics["P7_coverage"]  >= 0.90 and
        coverage_metrics["A4_canonical"] >= 0.95 and
        coverage_metrics["A6_coverage"]  >= 0.50
    )
    
    # Verdict 2b: Parser Fidelity (parser found the RIGHT thing)
    # accuracy_on_parsed: of the cases the parser handled, how many ended correct
    fidelity_metrics = {
        "P6_fidelity": results["P6"].get("accuracy", 0.0),
        "P7_fidelity": results["P7"].get("accuracy", 0.0),
        "A6_fidelity": results["A6"].get("accuracy", 0.0),
    }
    fid_min = min(fidelity_metrics.values())
    if fid_min >= 0.95:
        fidelity_status = "PASS"
    elif fid_min >= 0.80:
        fidelity_status = "PARTIAL"
    else:
        fidelity_status = "FAIL"
    
    # Print verdicts
    print()
    print("Verdict 1: MEMORY SUBSTRATE")
    for k, v in mem_probes.items():
        threshold = ""
        if "A3_drop" in k:
            threshold = ">= 0.50"
        elif "leakage" in k:
            threshold = "<= 0.01"
        else:
            threshold = ">= 0.99"
        print(f"  {k}: {v:.3f}  ({threshold})")
    print(f"  => {'PASS' if mem_pass else 'FAIL'}")
    
    print()
    print("Verdict 2a: PARSER COVERAGE (parser found something)")
    for k, v in coverage_metrics.items():
        print(f"  {k}: {v:.3f}")
    print(f"  => {'PASS' if coverage_pass else 'FAIL'}")
    
    print()
    print("Verdict 2b: PARSER FIDELITY (parser found the RIGHT thing)")
    for k, v in fidelity_metrics.items():
        print(f"  {k}: {v:.3f}")
    print(f"  => {fidelity_status}  (PASS>=95%, PARTIAL>=80%, else FAIL)")
    
    print()
    print(f"P5 SCALING: {p5_status}")
    for k in ("n3", "n5", "n8", "n12"):
        bucket = p5_buckets.get(k, {})
        n = bucket.get("n", 0)
        acc = bucket.get("accuracy", 0.0)
        if n == 0:
            print(f"  {k}: SKIPPED (n=0)")
        else:
            print(f"  {k}: n={n}  acc={acc:.3f}")
    
    print()
    print("Verdict 3: SHADOW READINESS")
    print("  Not applicable (shadow training not performed in this mode)")
    print("  Run with MODE='train_shadow' to produce shadow readiness verdict.")
    
    # Decision:
    #   READY_FOR_SHADOW_TRUSTED_ONLY iff substrate PASS, coverage PASS, P5 COMPLETE,
    #                                   fidelity at least PARTIAL
    #   READY_FOR_SHADOW               iff all above AND fidelity PASS
    #   NOT_READY_FOR_SHADOW           otherwise
    decision_blockers = []
    if not mem_pass:
        decision_blockers.append("Memory Substrate FAIL")
    if not coverage_pass:
        decision_blockers.append("Parser Coverage FAIL")
    if p5_status != "COMPLETE":
        decision_blockers.append(f"P5 INCOMPLETE ({', '.join(p5_skipped)})")
    if fidelity_status == "FAIL":
        decision_blockers.append(f"Parser Fidelity FAIL ({fid_min:.1%})")
    
    if decision_blockers:
        decision = "NOT_READY_FOR_SHADOW"
    elif fidelity_status == "PASS":
        decision = "READY_FOR_SHADOW"
    else:  # PARTIAL
        decision = "READY_FOR_SHADOW_TRUSTED_ONLY"
    
    print()
    print(f"DECISION: {decision}")
    if decision_blockers:
        print(f"  Blockers: {'; '.join(decision_blockers)}")
    if decision == "READY_FOR_SHADOW_TRUSTED_ONLY":
        print(f"  Caveat: fidelity is PARTIAL ({fid_min:.1%}). Shadow training MUST be")
        print(f"          restricted to parser-trusted episode types (trust mask).")
    
    # Branch determination
    if mem_pass:
        branch = "MEMORY_SUBSTRATE_VALIDATED"
        text = ("Stage 1 deterministic memory substrate VALIDATED. "
                "Parser coverage validated. Parser fidelity is " + fidelity_status + " (P6/A6 accuracy on "
                "paraphrase and template-ablation conditions). If fidelity < PASS, "
                "shadow training must restrict supervision to parser-trusted "
                "episode types only (trust mask), otherwise shadow heads will "
                "learn parser mistakes as ground truth.")
    else:
        branch = "SUBSTRATE_BROKEN"
        failed = []
        for k, v in mem_probes.items():
            if "A3_drop" in k and v < 0.50:
                failed.append(f"{k}={v:.3f}")
            elif "leakage" in k and v > 0.01:
                failed.append(f"{k}={v:.3f}")
            elif v < 0.99 and "leakage" not in k and "A3_drop" not in k:
                failed.append(f"{k}={v:.3f}")
        text = (f"Memory substrate FAILED on: {', '.join(failed)}. "
                f"This is an IMPLEMENTATION BUG, not a learning problem. "
                f"Fix bank/parser before any training.")
    
    print()
    print(SEP)
    print(f"BRANCH: {branch}")
    print(text)
    print(SEP)
    
    results["_verdicts"] = {
        "memory_substrate":   {"pass": mem_pass,    "metrics": mem_probes},
        "parser_coverage":    {"pass": coverage_pass, "metrics": coverage_metrics},
        "parser_fidelity":    {"status": fidelity_status, "metrics": fidelity_metrics,
                                "min": fid_min},
        "p5_scaling":         {"status": p5_status, "min_acc": p5_min_acc,
                                "min_n": p5_min_n, "skipped": p5_skipped,
                                "buckets": {k: {"n": v.get("n", 0),
                                                  "accuracy": v.get("accuracy", 0.0)}
                                              for k, v in p5_buckets.items()}},
        "shadow_readiness":   {"pass": None, "metrics": None, "note": "not run"},
        "decision":           decision,
        "decision_blockers":  decision_blockers,
        "branch":             branch,
        "text":               text,
    }
    return results


def v15_1_write_memo(results: Dict, path: str, shadow_results: Optional[Dict] = None):
    """Write internal memo. Only legitimate output format.
    
    Schema (required at top):
      Memory Substrate: PASS/FAIL
      Parser Robustness: PASS/FAIL
      P5 scaling: COMPLETE/INCOMPLETE
      Decision: READY_FOR_SHADOW / NOT_READY_FOR_SHADOW
    """
    v = results["_verdicts"]
    cfg = results["config"]
    lines = []
    lines.append("# v15.1 Stage 1 Internal Memo")
    lines.append("")
    
    # ===== REQUIRED SCHEMA (top) =====
    mem_str      = "PASS" if v["memory_substrate"]["pass"]    else "FAIL"
    coverage_str = "PASS" if v["parser_coverage"]["pass"]     else "FAIL"
    fidelity_str = v["parser_fidelity"]["status"]
    p5_str       = v["p5_scaling"]["status"]
    decision     = v["decision"]
    lines.append("## Status")
    lines.append("")
    lines.append(f"- **Memory Substrate**: {mem_str}")
    lines.append(f"- **Parser Coverage**: {coverage_str}")
    lines.append(f"- **Parser Fidelity**: {fidelity_str}")
    lines.append(f"- **P5 scaling**: {p5_str}")
    lines.append(f"- **Decision**: {decision}")
    if v["decision_blockers"]:
        lines.append(f"- **Blockers**: {'; '.join(v['decision_blockers'])}")
    lines.append("")
    
    lines.append("## Immutables")
    lines.append(f"- bench_seed: {cfg['bench_seed']}")
    lines.append(f"- audit_seed: {cfg['audit_seed']}")
    lines.append(f"- split_seed: {cfg['split_seed']}")
    lines.append(f"- n_per_cell: {cfg['n_per_cell']}")
    lines.append("")
    lines.append("## Verdict 1: Memory Substrate")
    lines.append(f"**{mem_str}**")
    lines.append("```")
    lines.append(json.dumps(v["memory_substrate"]["metrics"], indent=2, default=str))
    lines.append("```")
    lines.append("")
    lines.append("## Verdict 2a: Parser Coverage")
    lines.append(f"**{coverage_str}**")
    lines.append("```")
    lines.append(json.dumps(v["parser_coverage"]["metrics"], indent=2, default=str))
    lines.append("```")
    lines.append("")
    lines.append("## Verdict 2b: Parser Fidelity")
    lines.append(f"**{fidelity_str}**  (PASS >= 95%, PARTIAL >= 80%, else FAIL)")
    lines.append("```")
    lines.append(json.dumps(v["parser_fidelity"]["metrics"], indent=2, default=str))
    lines.append("```")
    lines.append("")
    lines.append("## P5 Scaling Detail")
    lines.append(f"**{p5_str}**")
    lines.append("```")
    lines.append(json.dumps(v["p5_scaling"], indent=2, default=str))
    lines.append("```")
    lines.append("")
    lines.append("## Verdict 3: Shadow Readiness")
    if shadow_results is not None:
        lines.append(f"**{'PASS' if v['shadow_readiness']['pass'] else 'FAIL'}**")
        lines.append("```")
        lines.append(json.dumps(shadow_results, indent=2, default=str))
        lines.append("```")
    else:
        lines.append("*Not evaluated (shadow training not run).*")
    lines.append("")
    lines.append("## Full Benchmark Results")
    for probe in ("P1", "P2", "P3", "P4_AE", "P4_AA", "P5", "P6", "P7"):
        lines.append(f"### {probe}")
        lines.append("```")
        lines.append(json.dumps(results.get(probe, {}), indent=2, default=str))
        lines.append("```")
    lines.append("")
    lines.append("## Audit Results")
    for probe in ("A2", "A3", "A4", "A5", "A6"):
        lines.append(f"### {probe}")
        lines.append("```")
        lines.append(json.dumps(results.get(probe, {}), indent=2, default=str))
        lines.append("```")
    lines.append("")
    lines.append(f"## Branch: {v['branch']}")
    lines.append("")
    lines.append(v["text"])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


print("[v15.1] Section B: substrate validation framework defined")
print("         - 5 memory probes (P1, P2, P3, P4_AE, P4_AA, P5)")
print("         - 4 memory audit probes (A2, A3, A4_canonicalization, A5)")
print("         - 3 parser-robustness probes (P6, P7, A6)")
print("         - 3 separate verdicts produced")

# ======================== C. V15.1 SHADOW MODULES ==========================
#
# Shadow heads: learned modules trained in parallel with critical path but
# NOT in the critical path. Used for Stage 2 when parser alone no longer
# suffices (paraphrase variation, coreference, OOV).
#
# Three shadow heads:
#   - ShadowAttributeRouter: predict attr_type from query
#   - ShadowTypedValueHeads: predict value_idx from value_emb
#   - ShadowObjectResolver:  predict slot index from query features
# ===========================================================================

class ShadowAttributeRouter(nn.Module):
    """Predicts attr_type {color, size, location, state, NONE_ATTR} from query."""
    
    def __init__(self, d_model: int, n_classes: int = 5):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes  # color, size, location, state, NONE_ATTR
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_classes),
        )
    
    def forward(self, q_pooled: torch.Tensor) -> torch.Tensor:
        """q_pooled: [B, d_model] -> logits [B, n_classes]."""
        return self.head(q_pooled)


class ShadowTypedValueHeads(nn.Module):
    """Four separate value heads, one per attribute type.
    Input: value_emb stored in AttributeSlot.
    Output per head: logits over typed vocabulary (NO UNKNOWN class).
    """
    
    def __init__(self, d_model: int, attr_vocab_sizes: Dict[str, int]):
        super().__init__()
        self.d_model = d_model
        self.attr_vocab_sizes = attr_vocab_sizes
        self.heads = nn.ModuleDict({
            attr: nn.Linear(d_model, size)
            for attr, size in attr_vocab_sizes.items()
        })
    
    def forward(self, attr: str, value_emb: torch.Tensor) -> torch.Tensor:
        """value_emb: [B, d_model] -> logits [B, vocab_size[attr]]."""
        if attr not in self.heads:
            raise ValueError(f"Unknown attribute type: {attr}")
        return self.heads[attr](value_emb)


class ShadowObjectResolver(nn.Module):
    """Predicts object slot index (or NONE_OBJECT) from query features.
    
    Input features per candidate slot: [cos_sim_entity, class_match,
    uncertainty, recency, salience]. Concatenated with a global query
    feature for NONE_OBJECT scoring.
    """
    
    def __init__(self, d_model: int, d_feat: int = 5):
        super().__init__()
        self.d_model = d_model
        self.d_feat = d_feat
        # Per-slot score head (shared)
        self.per_slot_head = nn.Sequential(
            nn.Linear(d_feat, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        # NONE_OBJECT score head (from query alone)
        self.none_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
    
    def forward(self, q_entity_emb: torch.Tensor,
                 slot_features: torch.Tensor) -> torch.Tensor:
        """q_entity_emb: [d_model]  (single query)
        slot_features:  [K, d_feat]  (K occupied slots)
        Returns: logits [K+1]  (K slot scores + NONE_OBJECT).
        """
        K = slot_features.shape[0]
        if K > 0:
            slot_scores = self.per_slot_head(slot_features).squeeze(-1)  # [K]
        else:
            slot_scores = torch.zeros(0, device=q_entity_emb.device)
        none_score = self.none_head(q_entity_emb).squeeze(-1).unsqueeze(0)  # [1]
        return torch.cat([slot_scores, none_score], dim=0)   # [K+1]


class V15_1_ShadowHeads(nn.Module):
    """Container wrapping all three shadow heads."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.attr_router = ShadowAttributeRouter(d_model, n_classes=5)
        vocab_sizes = {
            "color":    len(V15_COLORS),
            "size":     len(V15_SIZES),
            "location": len(V15_LOCATIONS),
            "state":    len(V15_STATES),
        }
        self.value_heads = ShadowTypedValueHeads(d_model, vocab_sizes)
        self.object_resolver = ShadowObjectResolver(d_model)
    
    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


print("[v15.1] Section C: shadow modules defined")
print("         - ShadowAttributeRouter (5 classes: 4 attrs + NONE_ATTR)")
print("         - ShadowTypedValueHeads (4 heads, typed vocab, no UNKNOWN)")
print("         - ShadowObjectResolver (per-slot + NONE_OBJECT)")

# ======================== D. V15.1 SHADOW TRAINING ========================
#
# Trains shadow heads on episodes from the curriculum. Critical path is
# used to compute ground truth, shadow heads try to match it.
# ===========================================================================

V15_1_SHADOW_CONFIG = {
    "n_steps":         2000,
    "batch_episodes":  4,
    "lr":              3e-4,
    "weight_decay":    0.01,
    "betas":           (0.9, 0.95),
    "warmup_steps":    200,
    "grad_clip":       1.0,
    "log_every":       50,
    "ckpt_every":      500,
    "eval_every":      500,
    "seed":            7777,
    # Loss weights
    "w_attr":    1.0,
    "w_value":   1.0,
    "w_object":  1.0,
    "w_preserve": 0.5,   # critical constraint on slot integrity
    "w_class":    0.1,   # auxiliary
}


def _shadow_lr_at(step: int, cfg: Dict) -> float:
    """Warmup + cosine decay."""
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    t = (step - cfg["warmup_steps"]) / max(1, cfg["n_steps"] - cfg["warmup_steps"])
    t = min(1.0, max(0.0, t))
    return cfg["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def _build_slot_features(bank: DeterministicObjectBank,
                          q_entity_emb: torch.Tensor,
                          q_class_hint: Optional[torch.Tensor],
                          current_step: int) -> torch.Tensor:
    """Build per-slot features [K, d_feat] for ShadowObjectResolver."""
    slots = bank.occupied_slots()
    if not slots:
        return torch.zeros(0, 5, device=q_entity_emb.device)
    feats = []
    for s in slots:
        rec = bank.get_record(s)
        # Cosine with entity_emb
        e_sim = F.cosine_similarity(q_entity_emb.unsqueeze(0),
                                      rec.entity_emb.unsqueeze(0), dim=-1).item()
        # Class compatibility
        if q_class_hint is not None and rec.class_emb is not None:
            c_sim = F.cosine_similarity(q_class_hint.unsqueeze(0),
                                          rec.class_emb.unsqueeze(0), dim=-1).item()
        else:
            c_sim = 0.0
        unc = rec.uncertainty
        rec_age = max(0, current_step - rec.last_write_step) / 10.0
        salience = sum(1 for s_ in rec.attr_slots.values() if s_.present) / 4.0
        feats.append([e_sim, c_sim, unc, rec_age, salience])
    return torch.tensor(feats, dtype=torch.float32, device=q_entity_emb.device)


def _compute_shadow_losses(base_model, v15_1_memory: V15_1_ShadowHeads,
                            bank: DeterministicObjectBank,
                            batch_episodes: List[V15Episode], cfg: Dict,
                            current_step: int = 0) -> Tuple[torch.Tensor, Dict]:
    """Compute shadow losses for one batch of episodes.
    
    For each episode:
      - Write all facts to bank via critical path.
      - For query: compute shadow predictions and compare to ground truth
        from critical path.
    """
    device = DEVICE
    shadow = v15_1_memory.shadow
    losses = {
        "attr":     torch.zeros((), device=device),
        "value":    torch.zeros((), device=device),
        "object":   torch.zeros((), device=device),
        "preserve": torch.zeros((), device=device),
        "class":    torch.zeros((), device=device),
    }
    counts = {k: 0 for k in losses}
    
    entity_emb_fn = _make_entity_emb_fn(base_model)
    class_emb_fn  = _make_class_emb_fn(v15_1_memory)
    value_emb_fn  = _make_value_emb_fn(base_model)
    
    for ep in batch_episodes:
        bank.reset()
        # Snapshot before selective_update to check preserve
        all_parsed_facts = []
        for idx, fact_text in enumerate(ep.facts):
            p = parse_fact(fact_text)
            all_parsed_facts.append(p)
            if p.parse_failed and not p.is_anchor:
                continue
            v15_1_write_fact(bank, p, entity_emb_fn, class_emb_fn, value_emb_fn,
                              step=idx)
        
        # Query
        q_parse = parse_query(ep.query)
        if not q_parse.entity_ok:
            continue
        q_entity_emb = entity_emb_fn(q_parse.entity_id)
        
        # ---- Shadow AttributeRouter ----
        # Target: q_parse.attr_type (index in 5-class space)
        if q_parse.attr_ok:
            attr_tgt = V15_ATTR_TO_IDX[q_parse.attr_type]
        else:
            attr_tgt = 4  # NONE_ATTR
        # If target is genuinely NONE_ATTR (absent attribute), set target accordingly
        status, _ = bank.read_attribute(q_parse.entity_id, q_parse.attr_type) if q_parse.attr_ok else (None, None)
        if q_parse.attr_ok and status == READ_STATUS_NONE_ATTRIBUTE:
            attr_tgt = 4  # NONE_ATTR when slot not written
        # Get pooled query emb via shared_token_emb mean (for shadow input)
        with torch.no_grad():
            q_ids = torch.tensor(ENC.encode(ep.query), dtype=torch.long, device=device)
            q_pooled = base_model.shared_token_emb(q_ids).mean(dim=0)
        attr_logits = shadow.attr_router(q_pooled.unsqueeze(0))
        tgt = torch.tensor([attr_tgt], dtype=torch.long, device=device)
        losses["attr"] = losses["attr"] + F.cross_entropy(attr_logits, tgt)
        counts["attr"] += 1
        
        # ---- Shadow ObjectResolver ----
        slot_feats = _build_slot_features(bank, q_entity_emb, None, current_step)
        resolver_logits = shadow.object_resolver(q_entity_emb, slot_feats)
        # Target: index of slot with matching entity_id, else K (NONE_OBJECT)
        target_slot = bank.find_by_entity_id(q_parse.entity_id)
        K = slot_feats.shape[0]
        if target_slot is None:
            obj_tgt = K  # NONE_OBJECT
        else:
            slot_list = bank.occupied_slots()
            obj_tgt = slot_list.index(target_slot)
        tgt_obj = torch.tensor([obj_tgt], dtype=torch.long, device=device)
        losses["object"] = losses["object"] + F.cross_entropy(
            resolver_logits.unsqueeze(0), tgt_obj)
        counts["object"] += 1
        
        # ---- Shadow TypedValueHeads ----
        # Only when target_slot exists and attr present
        if target_slot is not None and q_parse.attr_ok:
            rec = bank.get_record(target_slot)
            a = rec.attr_slots.get(q_parse.attr_type)
            if a is not None and a.present and a.value_emb is not None:
                value_logits = shadow.value_heads(q_parse.attr_type,
                                                   a.value_emb.unsqueeze(0))
                tgt_v = torch.tensor([a.value_idx], dtype=torch.long, device=device)
                losses["value"] = losses["value"] + F.cross_entropy(value_logits, tgt_v)
                counts["value"] += 1
    
    # Normalize
    total = torch.zeros((), device=device)
    total = total + cfg["w_attr"]   * (losses["attr"]   / max(1, counts["attr"]))
    total = total + cfg["w_value"]  * (losses["value"]  / max(1, counts["value"]))
    total = total + cfg["w_object"] * (losses["object"] / max(1, counts["object"]))
    
    out = {}
    for k in losses:
        out[k] = losses[k] / max(1, counts[k])
    return total, out


def v15_1_train_shadow_main(bank, base_model, v15_1_memory) -> Dict:
    """Main shadow training loop."""
    cfg = V15_1_SHADOW_CONFIG
    print()
    print(SEP)
    print("[v15.1 SHADOW TRAINING]")
    print(f"  steps: {cfg['n_steps']}  batch: {cfg['batch_episodes']}")
    print(f"  warmup: {cfg['warmup_steps']}  LR: {cfg['lr']:.0e}")
    print(SEP)
    
    shadow = v15_1_memory.shadow
    shadow.train()
    params = [p for p in shadow.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=cfg["betas"],
                              weight_decay=cfg["weight_decay"])
    
    rng = random.Random(cfg["seed"])
    loss_hist = []
    t0 = time.time()
    
    # Shadow training: parser-TRUSTED episode types only.
    # EXCLUDED:
    #   - lm_pretraining (no query)
    #   - paraphrase (parser fidelity drops: ~82%)
    #   - coreference_distant (parser fidelity drops)
    # INCLUDED (parser-trusted, accuracy_on_parsed near 100%):
    #   - single_attr_simple
    #   - multi_attr_object
    #   - selective_update
    #   - no_match
    #   - provisional_entity
    SHADOW_EPISODE_TYPES = [
        ("single_attr_simple",      0.35),
        ("multi_attr_object",       0.25),
        ("selective_update",        0.20),
        ("no_match",                0.15),
        ("provisional_entity",      0.05),
    ]
    
    def _sample_shadow_episode_type(rng_: random.Random) -> str:
        r = rng_.random()
        cum = 0.0
        for name, p in SHADOW_EPISODE_TYPES:
            cum += p
            if r < cum:
                return name
        return SHADOW_EPISODE_TYPES[-1][0]
    
    skipped_steps = 0
    for step in range(cfg["n_steps"]):
        # Sample batch (exclude lm_pretraining - no query)
        batch = []
        for _ in range(cfg["batch_episodes"]):
            ep_type = _sample_shadow_episode_type(rng)
            batch.append(v15_generate_episode(ep_type, rng, use_heldout=False))
        
        # LR
        for g in opt.param_groups:
            g["lr"] = _shadow_lr_at(step, cfg)
        
        opt.zero_grad(set_to_none=True)
        total, parts = _compute_shadow_losses(base_model, v15_1_memory, bank,
                                                batch, cfg, current_step=step)
        # Safety: skip step if total has no grad path (edge case)
        if not total.requires_grad:
            skipped_steps += 1
            continue
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        opt.step()
        
        loss_hist.append({
            "step":   step + 1,
            "total":  float(total.item()),
            **{k: float(v.detach().item() if v.numel()==1 else v.detach().mean().item())
               for k, v in parts.items()},
        })
        
        if (step + 1) % cfg["log_every"] == 0:
            elapsed = time.time() - t0
            eta_s = (cfg["n_steps"] - step - 1) * (elapsed / max(1, step + 1))
            parts_str = " ".join(f"{k}={float(v.item() if v.numel()==1 else v.mean().item()):.3f}"
                                   for k, v in parts.items())
            print(f"[v15.1 SHADOW] step {step+1}/{cfg['n_steps']} "
                  f"total={float(total.item()):.3f} lr={opt.param_groups[0]['lr']:.2e} "
                  f"ETA={int(eta_s//60)}m{int(eta_s%60)}s", flush=True)
            print(f"             {parts_str}", flush=True)
    
    if skipped_steps > 0:
        print(f"[v15.1 SHADOW] skipped {skipped_steps} steps (no grad path - edge cases)")
    shadow.eval()
    return {
        "loss_history":       loss_hist,
        "final_total":        float(loss_hist[-1]["total"]) if loss_hist else None,
        "skipped_steps":      skipped_steps,
    }


print("[v15.1] Section D: shadow training defined")
print("         - shadow losses: attr + value + object")
print("         - AdamW + warmup + cosine decay")
print("         - critical-path bank used to produce ground truth")

# ======================== E. V15.1 SHADOW AUDIT ===========================
#
# A1: Critical vs Shadow comparison.
# Runs the benchmark in three modes:
#   1. critical_only: deterministic read (substrate)
#   2. shadow_only:   use shadow heads for attr + value (no critical path)
#   3. mixed:         critical for object, shadow for value
#
# Interpretation:
#   - critical >= 99% AND shadow < critical: Stage 1 valid, shadow progressing
#   - critical < 99%: substrate broken
#   - mixed > critical and critical < 99%: hidden shortcut
# ===========================================================================

@torch.no_grad()
def _run_single_mode_trial(bank, base_model, v15_1_mem, ep: V15Episode,
                            mode: str, probe_name: str, episode_seed: int
                            ) -> V15_1_TrialRecord:
    """Run one episode in specified mode.
    
    mode: "critical_only" | "shadow_only" | "mixed"
    """
    shadow = v15_1_mem.shadow
    bank.reset()
    
    entity_emb_fn = _make_entity_emb_fn(base_model)
    class_emb_fn  = _make_class_emb_fn(v15_1_mem)
    value_emb_fn  = _make_value_emb_fn(base_model)
    
    facts_all_parsed = True
    for idx, fact_text in enumerate(ep.facts):
        p = parse_fact(fact_text)
        if p.parse_failed and not p.is_anchor:
            facts_all_parsed = False
            continue
        v15_1_write_fact(bank, p, entity_emb_fn, class_emb_fn, value_emb_fn, step=idx)
    
    q_parse = parse_query(ep.query)
    
    # Determine target (same logic as _v15_1_run_trial)
    target_idx = None
    if not ep.target_is_unknown:
        attr_type = V15_ATTR_TYPES[ep.query_attr_label]
        vocab = V15_ATTR_VALUES[attr_type]
        t_tok = int(ep.target_answer_token)
        for i, v_str in enumerate(vocab):
            if V15_ANSWER_TOKENS[attr_type].get(v_str) == t_tok:
                target_idx = i
                break
    
    target_is_none_obj  = False
    target_is_none_attr = False
    if ep.target_is_unknown:
        q_ent = canonicalize_entity(q_parse.entity_id) if q_parse.entity_id else ""
        fact_eids = set()
        for f in ep.facts:
            pf = parse_fact(f)
            if pf.entity_id is not None:
                fact_eids.add(pf.entity_id)
        if q_ent not in fact_eids:
            target_is_none_obj = True
        else:
            target_is_none_attr = True
    
    # Now produce prediction based on mode
    read_status = READ_STATUS_PARSER_FAIL
    pred_idx = None
    
    if not q_parse.entity_ok:
        read_status = READ_STATUS_PARSER_FAIL
    elif mode == "critical_only":
        if not q_parse.attr_ok:
            read_status = READ_STATUS_PARSER_FAIL
        else:
            read_status, pred_idx = bank.read_attribute(q_parse.entity_id, q_parse.attr_type)
    elif mode == "shadow_only":
        # Use shadow attr_router + shadow value heads + shadow obj_resolver
        q_ids = torch.tensor(ENC.encode(ep.query), dtype=torch.long, device=DEVICE)
        q_pooled = base_model.shared_token_emb(q_ids).mean(dim=0)
        attr_logits = shadow.attr_router(q_pooled.unsqueeze(0))
        attr_pred = int(attr_logits.argmax(dim=-1).item())
        # Resolve object
        q_ent_emb = entity_emb_fn(q_parse.entity_id)
        slot_feats = _build_slot_features(bank, q_ent_emb, None, current_step=1000)
        resolver_logits = shadow.object_resolver(q_ent_emb, slot_feats)
        obj_pred = int(resolver_logits.argmax(dim=-1).item())
        K = slot_feats.shape[0]
        if obj_pred == K:
            read_status = READ_STATUS_NONE_OBJECT
        elif attr_pred == 4:
            read_status = READ_STATUS_NONE_ATTRIBUTE
        else:
            attr_type = V15_ATTR_TYPES[attr_pred]
            slot_list = bank.occupied_slots()
            rec = bank.get_record(slot_list[obj_pred])
            a = rec.attr_slots.get(attr_type)
            if a is None or not a.present or a.value_emb is None:
                read_status = READ_STATUS_NONE_ATTRIBUTE
            else:
                value_logits = shadow.value_heads(attr_type, a.value_emb.unsqueeze(0))
                pred_idx = int(value_logits.argmax(dim=-1).item())
                read_status = READ_STATUS_FOUND
    elif mode == "mixed":
        # Critical path for object resolution; shadow for value prediction
        if not q_parse.attr_ok:
            read_status = READ_STATUS_PARSER_FAIL
        else:
            slot = bank.find_by_entity_id(q_parse.entity_id)
            if slot is None:
                read_status = READ_STATUS_NONE_OBJECT
            else:
                rec = bank.get_record(slot)
                a = rec.attr_slots.get(q_parse.attr_type)
                if a is None or not a.present or a.value_emb is None:
                    read_status = READ_STATUS_NONE_ATTRIBUTE
                else:
                    value_logits = shadow.value_heads(q_parse.attr_type,
                                                       a.value_emb.unsqueeze(0))
                    pred_idx = int(value_logits.argmax(dim=-1).item())
                    read_status = READ_STATUS_FOUND
    
    if target_is_none_obj:
        correct = (read_status == READ_STATUS_NONE_OBJECT)
    elif target_is_none_attr:
        correct = (read_status == READ_STATUS_NONE_ATTRIBUTE)
    else:
        correct = (read_status == READ_STATUS_FOUND and pred_idx == target_idx)
    
    return V15_1_TrialRecord(
        probe=probe_name, episode_type=ep.episode_type, episode_seed=episode_seed,
        query_parser_ok=(q_parse.entity_ok and q_parse.attr_ok),
        facts_all_parsed=facts_all_parsed,
        read_status=read_status,
        predicted_value_idx=pred_idx,
        target_value_idx=target_idx,
        target_is_unknown_obj=target_is_none_obj,
        target_is_unknown_attr=target_is_none_attr,
        correct=correct,
    )


def _audit_A1_critical_vs_shadow(bank, base_model, v15_1_mem, cfg) -> Dict:
    """A1: Run the substrate in three modes (critical_only, shadow_only, mixed)
    on TWO disjoint query sets:
      - TRUSTED: single_attr_simple + multi_attr_object (parser fidelity near 100%)
      - HARD:    paraphrase + coreference_distant (parser fidelity ~82%)
    
    Brutal interpretation per GPT:
      - critical_only must stay >= 99.5% on both sets
      - shadow_only P1/P2 >= 85% expected on TRUSTED; HARD separate
      - shadow_only AE/AA >= 95% expected on TRUSTED
      - mixed on TRUSTED must be >= critical_only - 0.5pp
      - mixed on HARD can degrade but must NOT drag TRUSTED down
    """
    n = min(cfg["n_per_cell"], 200)
    trusted_types = [("single_attr_simple", 0.7), ("multi_attr_object", 0.3)]
    hard_types    = [("paraphrase", 0.5),         ("coreference_distant", 0.5)]
    
    def _run_on(ep_types: List[Tuple[str, float]], tag: str) -> Dict:
        result = {}
        for mode in ("critical_only", "shadow_only", "mixed"):
            rng = random.Random(cfg["audit_seed"] + 100 + hash(tag) % 10000)
            trials = []
            for i in range(n):
                # Sample episode type from the set
                r = rng.random()
                cum = 0.0
                chosen = ep_types[-1][0]
                for name, p in ep_types:
                    cum += p
                    if r < cum:
                        chosen = name
                        break
                ep = v15_generate_episode(chosen, rng, use_heldout=True)
                tr = _run_single_mode_trial(bank, base_model, v15_1_mem, ep,
                                              mode=mode, probe_name=f"A1_{tag}_{mode}",
                                              episode_seed=cfg["audit_seed"] * 1000 + i)
                trials.append(tr)
            result[mode] = _v15_1_agg(trials)
        return result
    
    trusted = _run_on(trusted_types, "trusted")
    hard    = _run_on(hard_types,    "hard")
    
    # Interpretation per GPT's thresholds
    crit_trusted   = trusted["critical_only"]["accuracy"]
    shadow_trusted = trusted["shadow_only"]["accuracy"]
    mixed_trusted  = trusted["mixed"]["accuracy"]
    crit_hard      = hard["critical_only"]["accuracy"]
    shadow_hard    = hard["shadow_only"]["accuracy"]
    mixed_hard     = hard["mixed"]["accuracy"]
    
    interp = {
        # Critical must stay clean on trusted (brutal threshold)
        "critical_trusted_clean":   crit_trusted >= 0.995,
        "critical_hard_clean":      crit_hard    >= 0.995,
        # Shadow progress (on trusted set only, per GPT)
        "shadow_trusted_p_target":  shadow_trusted >= 0.85,
        # Mixed must not drag trusted down (within 0.5pp)
        "mixed_trusted_not_worse":  mixed_trusted >= (crit_trusted - 0.005),
        # Hidden shortcut detection
        "no_hidden_shortcut":       not (mixed_trusted > crit_trusted - 0.005
                                          and crit_trusted < 0.99),
    }
    
    return {
        "trusted": trusted,
        "hard":    hard,
        "_interpretation": interp,
        "_thresholds": {
            "critical_floor":        0.995,
            "shadow_trusted_min":    0.85,
            "mixed_trusted_slack":   0.005,
        },
    }


def v15_1_run_shadow_audit(bank, base_model, v15_1_mem, cfg=None) -> Dict:
    """Run shadow audit (A1) and assemble Shadow Readiness verdict.
    Splits TRUSTED vs HARD sets; applies GPT's brutal thresholds.
    """
    if cfg is None:
        cfg = V15_1_BENCHMARK_CONFIG
    
    print()
    print(SEP)
    print("[v15.1 SHADOW AUDIT] A1 critical_vs_shadow (TRUSTED | HARD)")
    print(SEP)
    
    results = _audit_A1_critical_vs_shadow(bank, base_model, v15_1_mem, cfg)
    
    print("\n--- TRUSTED set (single_attr_simple, multi_attr_object) ---")
    for mode in ("critical_only", "shadow_only", "mixed"):
        v = results["trusted"][mode]
        print(f"A1 trusted {mode}: acc={v['accuracy']:.1%} parser_cov={v['parser_coverage']:.1%} n={v['n']}")
    
    print("\n--- HARD set (paraphrase, coreference_distant) ---")
    for mode in ("critical_only", "shadow_only", "mixed"):
        v = results["hard"][mode]
        print(f"A1 hard    {mode}: acc={v['accuracy']:.1%} parser_cov={v['parser_coverage']:.1%} n={v['n']}")
    
    interp = results["_interpretation"]
    print("\n--- Interpretation (GPT thresholds) ---")
    for k, val in interp.items():
        print(f"  {k}: {val}")
    
    # Shadow readiness verdict per GPT's brutal thresholds
    crit_trusted_ok    = interp["critical_trusted_clean"]
    shadow_target_met  = interp["shadow_trusted_p_target"]
    mixed_safe         = interp["mixed_trusted_not_worse"]
    no_shortcut        = interp["no_hidden_shortcut"]
    
    if not crit_trusted_ok:
        shadow_pass = False
        shadow_text = (f"Critical path degraded on TRUSTED set "
                       f"({results['trusted']['critical_only']['accuracy']:.1%} < 99.5%). "
                       f"Substrate not preserved under shadow. INVESTIGATE BEFORE PROCEEDING.")
    elif not mixed_safe:
        shadow_pass = False
        shadow_text = (f"Mixed mode drags TRUSTED set below critical - 0.5pp: "
                       f"critical={results['trusted']['critical_only']['accuracy']:.1%}, "
                       f"mixed={results['trusted']['mixed']['accuracy']:.1%}. "
                       f"Shadow value heads contaminate the critical path.")
    elif not no_shortcut:
        shadow_pass = False
        shadow_text = "Hidden shortcut detected. Retracting verdict."
    elif shadow_target_met:
        shadow_pass = True
        shadow_text = (f"Stage 1 preserved; shadow reached target on TRUSTED "
                       f"(shadow_only={results['trusted']['shadow_only']['accuracy']:.1%} >= 85%). "
                       f"HARD set reported separately: "
                       f"shadow_only_hard={results['hard']['shadow_only']['accuracy']:.1%}.")
    else:
        shadow_pass = True  # Stage 1 preserved even if shadow not yet at target
        shadow_text = (f"Stage 1 preserved (critical clean, mixed safe, no shortcut), "
                       f"but shadow not yet at target on TRUSTED "
                       f"(shadow_only={results['trusted']['shadow_only']['accuracy']:.1%} < 85%). "
                       f"More training needed.")
    
    return {
        "A1_critical_vs_shadow": results,
        "_shadow_verdict": {
            "pass": shadow_pass,
            "text": shadow_text,
            "interpretation": interp,
        },
    }


print("[v15.1] Section E: shadow audit defined")
print("         - A1 critical_vs_shadow (3 modes: critical, shadow, mixed)")
print("         - Shadow Readiness verdict logic")


# ======================== F. V15.1 MEMORY WRAPPER ============================

class V15_1_Memory(nn.Module):
    """Thin container: a class_encoder (auxiliary) + shadow heads.
    The deterministic bank is NOT an nn.Module; it's external state.
    """
    
    def __init__(self, d_model: int, d_sem: int = 64, n_classes: int = 4):
        super().__init__()
        self.d_model = d_model
        self.class_encoder = ClassEncoder(d_model=d_model, d_sem=d_sem, n_classes=n_classes)
        self.shadow = V15_1_ShadowHeads(d_model=d_model)
    
    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


print("[v15.1] Section F: V15_1_Memory wrapper defined")
print("         - class_encoder (auxiliary)")
print("         - shadow heads (attr_router, value_heads, object_resolver)")


# ======================== A2. V15.2 PROTOCOL — STEP 1 ======================
#
# Step 1 of Stage 1.2: protocol data structures.
#
# NOT in critical path:
#   - the old v15.1 parse_fact/parse_query remain available for reference
#   - v15.1 DeterministicObjectBank remains the memory substrate (frozen)
#
# What's new:
#   - OpType enum: WRITE, READ, UPDATE, ANCHOR_DEFINE
#   - AmbiguityFlag enum: 7 closed values (not free strings)
#   - ParsePacket: rich intermediate structure with candidate lists +
#     confidence + evidence_span + ambiguity_flags
#   - READ_STATUS_PARSE_UNCERTAIN: 5th symbolic output, distinct from
#     PARSER_FAILURE, NONE_OBJECT, NONE_ATTRIBUTE, FOUND
# ===========================================================================

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set


class OpType(Enum):
    """Operation type declared by parser, verified against bank state."""
    WRITE          = "WRITE"           # new fact, entity does NOT exist yet in bank
    READ           = "READ"            # query
    UPDATE         = "UPDATE"          # overwrite existing slot (requires bank state check)
    ANCHOR_DEFINE  = "ANCHOR_DEFINE"   # define entity class (A dragon is a creature)


class AmbiguityFlag(Enum):
    """Closed set of 7 ambiguity flags. Verifier emits ONLY these."""
    MULTIPLE_ATTR_TRIGGERS     = "MULTIPLE_ATTR_TRIGGERS"
    REFERENT_AMBIGUOUS         = "REFERENT_AMBIGUOUS"
    ATTR_VALUE_MISMATCH        = "ATTR_VALUE_MISMATCH"
    TEMPLATE_UNKNOWN           = "TEMPLATE_UNKNOWN"
    MULTI_ENTITY_SAME_TYPE     = "MULTI_ENTITY_SAME_TYPE"
    OP_TYPE_AMBIGUOUS          = "OP_TYPE_AMBIGUOUS"
    VALUE_MISSING_OR_UNCLEAR   = "VALUE_MISSING_OR_UNCLEAR"


# ============ Candidate tuples ============
# Typed tuples stored inside ParsePacket. Kept as plain tuples for
# JSON serialization and cheap hashing.
#
# entity_candidate:    (entity_id: str, confidence: float, span: Tuple[int, int])
# attribute_candidate: (attr_type: str, confidence: float, evidence: str)
# value_candidate:     (attr_type: str, value_idx: int, confidence: float, evidence_span: Tuple[int, int])
# reference_candidate: (entity_id: str, antecedent_idx: int)


@dataclass
class ParsePacket:
    """Rich intermediate structure between lexical extractor and memory.
    
    Produced by v15_2_parse_fact or v15_2_parse_query.
    Consumed by v15_2_verify.
    
    The point: no premature commitment. If the extractor sees 2 plausible
    entities, both live in `entity_candidates` with their confidence.
    The verifier decides. The memory never sees ambiguity.
    """
    # Provenance
    source_text:           str
    source_kind:           str              # "fact" | "query"
    
    # Declared operation (parser's best guess; verifier may reject)
    op_type:               OpType
    op_type_confidence:    float            # how certain parser is about op_type
    
    # Candidate lists (may have 0, 1, or many entries)
    entity_candidates:     List[Tuple[str, float, Tuple[int, int]]]           = field(default_factory=list)
    attribute_candidates:  List[Tuple[str, float, str]]                        = field(default_factory=list)
    value_candidates:      List[Tuple[str, int, float, Tuple[int, int]]]      = field(default_factory=list)
    reference_candidates:  List[Tuple[str, int]]                               = field(default_factory=list)
    
    # Overall parser certainty (aggregate, NOT derived by verifier)
    certainty:             float            = 0.0
    
    # Ambiguity flags raised by the extractor itself (closed set above)
    ambiguity_flags:       Set[AmbiguityFlag] = field(default_factory=set)
    
    # Debugging / audit only
    parser_evidence:       Dict[str, str]   = field(default_factory=dict)


# ============ Verifier output ============

class VerificationStatus(Enum):
    """Three symbolic outcomes of ParseVerifier."""
    ACCEPT              = "ACCEPT"
    PARSE_UNCERTAIN     = "PARSE_UNCERTAIN"
    PARSER_FAILURE      = "PARSER_FAILURE"


@dataclass
class VerificationResult:
    """Verifier returns BOTH a verdict AND the reasons.
    
    If status != ACCEPT, reasons MUST contain at least one AmbiguityFlag.
    This is how we avoid global scores that hide where the protocol broke.
    """
    status:  VerificationStatus
    reasons: Set[AmbiguityFlag] = field(default_factory=set)
    
    # Human-readable debug string (not used for verdict)
    notes:   str                = ""


# ============ Read status extended ============
# v15.1 had: FOUND, NONE_OBJECT, NONE_ATTRIBUTE, PARSER_FAILURE
# v15.2 adds: PARSE_UNCERTAIN (verifier blocked execution)
#
# Distinctions matter. From the spec:
#   PARSER_FAILURE  : extractor couldn't even construct a coherent hypothesis
#   PARSE_UNCERTAIN : extractor constructed something, verifier blocked execution
#   NONE_OBJECT     : parse valid, object absent from memory
#   NONE_ATTRIBUTE  : parse valid, object present, attribute absent
#   FOUND           : parse valid, memory returned a value
READ_STATUS_PARSE_UNCERTAIN = "PARSE_UNCERTAIN"
# The others (FOUND, NONE_OBJECT, NONE_ATTRIBUTE, PARSER_FAILURE) are
# inherited from v15.1 section A and keep the same string values.


print("[v15.2] Step 1: protocol data structures defined")
print(f"         - OpType: {[t.value for t in OpType]}")
print(f"         - AmbiguityFlag: {len(list(AmbiguityFlag))} flags")
print(f"         - ParsePacket: {len(ParsePacket.__dataclass_fields__)} fields")
print(f"         - VerificationStatus: {[s.value for s in VerificationStatus]}")
print(f"         - READ_STATUS_PARSE_UNCERTAIN = {READ_STATUS_PARSE_UNCERTAIN!r}")
# ======================== A2. V15.2 PROTOCOL — STEP 2 ======================
#
# Lexical extractors: produce ParsePacket, no decisions.
#
# Principles (confirmed):
#   - Extractor is LAX: collects all plausible candidates with confidence
#     and evidence span. Never resolves ambiguity itself.
#   - If it cannot extract the lexical minimum: returns ParsePacket with
#     empty candidate lists, low certainty, op_type None-like (READ as
#     default sink). Verifier will convert this into PARSER_FAILURE.
#   - Facts produce WRITE or ANCHOR_DEFINE only. UPDATE is NEVER decided
#     at parse time; it is decided inside v15_2_write_fact after verifier
#     ACCEPT, using bank state.
#   - For fact WRITE: attr_type is inferred from the value token itself
#     (the value is in exactly one typed vocabulary: color/size/loc/state).
#     Extracting the value effectively extracts the attr_type.
#
# Does not modify v15.1 parse_fact / parse_query.
# ===========================================================================

import re as _re


# ------------- Helpers -------------

def _v15_2_find_entity_candidates(text: str, pool: List[str]
                                    ) -> List[Tuple[str, float, Tuple[int, int]]]:
    """Scan text for occurrences of any entity in `pool`, return candidates
    with (entity_id, confidence, char_span).
    
    Confidence heuristic:
      - 1.0 when preceded by article (the / a / an) OR at start of sentence
      - 0.7 when bare occurrence
      - 0.4 when part of a larger word (rejected actually)
    """
    low = text.lower()
    out = []
    for ent in pool:
        e_low = ent.lower()
        start = 0
        while True:
            idx = low.find(e_low, start)
            if idx < 0:
                break
            end = idx + len(e_low)
            # Reject in-word match
            before_ok = (idx == 0) or (not low[idx-1].isalpha())
            after_ok  = (end == len(low)) or (not low[end].isalpha())
            if before_ok and after_ok:
                # Higher confidence if preceded by determiner
                prefix = low[:idx].rstrip()
                conf = 0.7
                if prefix.endswith(("the", " a", " an")) or prefix == "":
                    conf = 1.0
                out.append((canonicalize_entity(ent), conf, (idx, end)))
            start = end
    return out


def _v15_2_find_value_candidates(text: str
                                   ) -> List[Tuple[str, int, float, Tuple[int, int]]]:
    """Scan text for occurrences of any typed value.
    Returns (attr_type, value_idx, confidence, evidence_span).
    
    Because each value string belongs to exactly one attr_type (color/size/
    location/state in v15 vocab), finding the value also identifies the
    attribute. Multiple hits across different attr_types is a legitimate
    signal of ambiguity.
    """
    low = text.lower()
    out = []
    for attr_type, vocab in V15_ATTR_VALUES.items():
        for value_idx, v_str in enumerate(vocab):
            v_low = v_str.lower()
            start = 0
            while True:
                idx = low.find(v_low, start)
                if idx < 0:
                    break
                end = idx + len(v_low)
                before_ok = (idx == 0) or (not low[idx-1].isalpha())
                after_ok  = (end == len(low)) or (not low[end].isalpha())
                if before_ok and after_ok:
                    out.append((attr_type, value_idx, 1.0, (idx, end)))
                start = end
    return out


def _v15_2_find_attr_trigger_candidates(text: str
                                          ) -> List[Tuple[str, float, str]]:
    """Find attribute trigger keywords in QUERY text (e.g. 'color', 'large',
    'feel', 'situated'). Returns (attr_type, confidence, evidence).
    
    Uses V15_1_ATTR_KEYWORDS (explicit triggers) and V15_1_IMPLICIT_QUERY_ATTR
    (implicit triggers like 'measures'->size).
    """
    low = text.lower()
    out = []
    # Explicit attribute nouns/verbs
    for attr_type, keywords in V15_1_ATTR_KEYWORDS.items():
        for kw in keywords:
            kw_low = kw.lower()
            if _re.search(rf"(?<![a-z]){_re.escape(kw_low)}(?![a-z])", low):
                out.append((attr_type, 0.9, kw))
    # Implicit triggers
    for kw, attr_type in V15_1_IMPLICIT_QUERY_ATTR.items():
        if _re.search(rf"(?<![a-z]){_re.escape(kw)}(?![a-z])", low):
            out.append((attr_type, 0.7, kw))
    return out


def _v15_2_detect_op_type_fact(text: str) -> Tuple[OpType, float]:
    """For a FACT (non-question), decide WRITE vs ANCHOR_DEFINE.
    
    ANCHOR_DEFINE pattern: 'A/An X is a/an Y' where Y is a class noun.
    WRITE pattern: 'The/A X is VALUE' where VALUE is a typed attribute value.
    
    Returns (op_type, confidence).
    """
    low = text.lower().strip().rstrip(".")
    # Anchor form: "a <ent> is a <class>" or "<ent> is a <class>"
    if _re.search(r"\bis\s+(a|an)\s+[a-z]+\b", low):
        # Check whether the token after "is a" is a known class noun
        m = _re.search(r"\bis\s+(?:a|an)\s+([a-z]+)\b", low)
        if m:
            class_noun = m.group(1)
            if class_noun in V15_CLASS_KEYWORDS:
                return (OpType.ANCHOR_DEFINE, 0.95)
            # Otherwise treat as uncertain write
            return (OpType.WRITE, 0.6)
    return (OpType.WRITE, 0.85)


def _v15_2_detect_op_type_query(text: str) -> Tuple[OpType, float]:
    """For a QUERY, verify it looks like a READ.
    
    Signals:
      - contains '?'
      - starts with interrogative (what/how/which/where/describe)
    
    Returns (OpType.READ, confidence) or flags OP_TYPE_AMBIGUOUS via
    caller if it also contains a value assertion.
    """
    low = text.lower().strip()
    has_q = "?" in text
    starts_q = bool(_re.match(r"^\s*(what|how|which|where|who|describe|say|tell)\b", low))
    if has_q or starts_q:
        return (OpType.READ, 0.95)
    # Ambiguous: could still be a query without '?' ('The dragon is large.')
    return (OpType.READ, 0.5)


# ------------- Extractors (return ParsePacket) -------------

def v15_2_parse_fact(text: str) -> ParsePacket:
    """Extract lexical signal from a fact sentence.
    
    Returns ParsePacket with:
      - op_type: WRITE or ANCHOR_DEFINE
      - entity_candidates: all plausible entity mentions
      - attribute_candidates: derived from value hits (WRITE) or class hint (ANCHOR_DEFINE)
      - value_candidates: with evidence_span (key for ambiguity detection)
      - reference_candidates: not populated for facts at Stage 1.2
      - ambiguity_flags: raised when extractor sees multi-value or multi-entity
    
    Never decides which candidate is 'correct'. Lists are left complete.
    """
    text = (text or "").strip()
    source = text
    
    # Step 1: op_type
    op_type, op_conf = _v15_2_detect_op_type_fact(source)
    
    # Step 2: entity candidates (scan both train and heldout pools)
    all_entities = [e for e, _ in V15_TRAIN_ENTITIES] + [e for e, _ in V15_HELDOUT_ENTITIES]
    ent_cands = _v15_2_find_entity_candidates(source, all_entities)
    
    # Step 3: value candidates (drive attr extraction for WRITE)
    val_cands = _v15_2_find_value_candidates(source)
    
    # Step 4: attribute candidates derived from value hits
    attr_cands: List[Tuple[str, float, str]] = []
    if op_type == OpType.WRITE:
        seen_attrs = set()
        for (attr_type, v_idx, conf, span) in val_cands:
            key = attr_type
            if key not in seen_attrs:
                attr_cands.append((attr_type, conf, source[span[0]:span[1]]))
                seen_attrs.add(key)
    elif op_type == OpType.ANCHOR_DEFINE:
        # Anchor: find class noun
        m = _re.search(r"\bis\s+(?:a|an)\s+([a-z]+)\b", source.lower())
        if m:
            class_noun = m.group(1)
            if class_noun in V15_CLASS_KEYWORDS:
                # Not a real attribute, but we log evidence
                attr_cands.append(("__class__", 0.95, class_noun))
    
    # Step 5: ambiguity flags from extractor's own view
    flags: Set[AmbiguityFlag] = set()
    
    # Multiple entity candidates with comparable confidence
    if len(ent_cands) > 1:
        top_conf = max(c for _, c, _ in ent_cands)
        close = [c for _, c, _ in ent_cands if c >= top_conf - 0.2]
        if len(close) > 1:
            # Are they same "type" ? In Stage 1.2 we can't type-resolve, so
            # just mark as ambiguous.
            flags.add(AmbiguityFlag.MULTI_ENTITY_SAME_TYPE)
    
    # Multiple value candidates across DIFFERENT attribute types
    # (for facts, seeing both "red" and "large" is structurally ambiguous)
    if op_type == OpType.WRITE:
        attr_types_in_values = {attr for (attr, _, _, _) in val_cands}
        if len(attr_types_in_values) > 1:
            flags.add(AmbiguityFlag.MULTIPLE_ATTR_TRIGGERS)
    
    # WRITE with no value candidate at all → value unclear
    if op_type == OpType.WRITE and len(val_cands) == 0:
        flags.add(AmbiguityFlag.VALUE_MISSING_OR_UNCLEAR)
    
    # Certainty aggregate
    if not ent_cands:
        cert = 0.0
    else:
        e_max = max(c for _, c, _ in ent_cands)
        v_max = max([c for _, _, c, _ in val_cands], default=0.0) if op_type == OpType.WRITE else 1.0
        cert = 0.5 * e_max + 0.5 * (v_max if op_type == OpType.WRITE else op_conf)
    
    return ParsePacket(
        source_text=source,
        source_kind="fact",
        op_type=op_type,
        op_type_confidence=op_conf,
        entity_candidates=ent_cands,
        attribute_candidates=attr_cands,
        value_candidates=val_cands,
        reference_candidates=[],
        certainty=cert,
        ambiguity_flags=flags,
        parser_evidence={"extractor_version": "v15.2.step2", "raw": source},
    )


def v15_2_parse_query(text: str) -> ParsePacket:
    """Extract lexical signal from a query sentence.
    
    Returns ParsePacket with:
      - op_type: READ (always; never decides UPDATE or WRITE for queries)
      - entity_candidates: all plausible entity mentions
      - attribute_candidates: from explicit or implicit trigger keywords
      - value_candidates: populated if a value string also appears (ambiguity signal)
      - reference_candidates: populated for pronouns (Stage 1.2 minimal)
      - ambiguity_flags: MULTIPLE_ATTR_TRIGGERS, REFERENT_AMBIGUOUS, etc.
    """
    text = (text or "").strip()
    source = text
    
    # Step 1: op_type
    op_type, op_conf = _v15_2_detect_op_type_query(source)
    
    # Step 2: entity candidates
    all_entities = [e for e, _ in V15_TRAIN_ENTITIES] + [e for e, _ in V15_HELDOUT_ENTITIES]
    ent_cands = _v15_2_find_entity_candidates(source, all_entities)
    
    # Step 3: attribute trigger candidates
    attr_cands = _v15_2_find_attr_trigger_candidates(source)
    
    # Step 4: value candidates (a query should NOT normally contain a value
    # string; if it does, flag ATTR_VALUE_MISMATCH unless aligned with attr)
    val_cands = _v15_2_find_value_candidates(source)
    
    # Step 5: reference candidates (very minimal: 'it', 'its' → last entity)
    ref_cands: List[Tuple[str, int]] = []
    low = source.lower()
    has_pronoun = bool(_re.search(r"\b(it|its|this|that)\b", low))
    if has_pronoun and ent_cands:
        # Collect unique entities already listed; pronoun could bind to any
        seen = []
        for (eid, _, _) in ent_cands:
            if eid not in seen:
                seen.append(eid)
        for i, eid in enumerate(seen):
            ref_cands.append((eid, i))
    
    # Step 6: ambiguity flags
    flags: Set[AmbiguityFlag] = set()
    
    # Multiple entity candidates
    if len({eid for eid, _, _ in ent_cands}) > 1:
        flags.add(AmbiguityFlag.MULTI_ENTITY_SAME_TYPE)
    
    # Multiple attribute triggers from DIFFERENT attribute types
    attr_types_triggered = {a for (a, _, _) in attr_cands}
    if len(attr_types_triggered) > 1:
        flags.add(AmbiguityFlag.MULTIPLE_ATTR_TRIGGERS)
    
    # Value string appears in a query: ATTR_VALUE_MISMATCH if it doesn't
    # match any triggered attribute type
    if val_cands:
        val_attrs = {a for (a, _, _, _) in val_cands}
        if attr_types_triggered and not (val_attrs & attr_types_triggered):
            flags.add(AmbiguityFlag.ATTR_VALUE_MISMATCH)
    
    # Pronoun with >1 plausible antecedent
    if has_pronoun and len({eid for eid, _ in ref_cands}) > 1:
        flags.add(AmbiguityFlag.REFERENT_AMBIGUOUS)
    
    # Op type ambiguous: query without '?' and without interrogative
    if op_conf < 0.7:
        flags.add(AmbiguityFlag.OP_TYPE_AMBIGUOUS)
    
    # No entity extracted at all → PARSER_FAILURE will be emitted by verifier
    # No attribute trigger at all → TEMPLATE_UNKNOWN
    if not attr_cands:
        flags.add(AmbiguityFlag.TEMPLATE_UNKNOWN)
    
    # Certainty aggregate
    if not ent_cands:
        cert = 0.0
    else:
        e_max = max(c for _, c, _ in ent_cands)
        a_max = max((c for _, c, _ in attr_cands), default=0.0)
        cert = 0.5 * e_max + 0.5 * a_max
    
    return ParsePacket(
        source_text=source,
        source_kind="query",
        op_type=op_type,
        op_type_confidence=op_conf,
        entity_candidates=ent_cands,
        attribute_candidates=attr_cands,
        value_candidates=val_cands,
        reference_candidates=ref_cands,
        certainty=cert,
        ambiguity_flags=flags,
        parser_evidence={"extractor_version": "v15.2.step2", "raw": source,
                           "has_pronoun": str(has_pronoun)},
    )


print("[v15.2] Step 2: lexical extractors defined")
print("         - v15_2_parse_fact  → ParsePacket (WRITE or ANCHOR_DEFINE)")
print("         - v15_2_parse_query → ParsePacket (READ)")
print("         - LAX: collects candidates, never resolves ambiguity")
# ======================== A2. V15.2 PROTOCOL — STEP 3 ======================
#
# ParseVerifier: rule-based, three verifications.
#
# Input:  ParsePacket
# Output: VerificationResult (status + reasons)
#
# Three checks (each may emit one or more AmbiguityFlag into reasons):
#   1. Structural    : declared op_type is compatible with sentence shape
#   2. Referential   : exactly one plausible candidate per required field
#   3. Executability : all fields required by op_type are present
#
# Rule: if ANY check fails → status != ACCEPT.
# Rule: if extractor found no entity at all → PARSER_FAILURE, not UNCERTAIN.
# Rule: otherwise (extractor found something, but not clearly executable)
#       → PARSE_UNCERTAIN with the specific reasons.
# ===========================================================================


class ParseVerifier:
    """Rule-based verifier. No learned parameters. No heuristic thresholds
    beyond the ones documented below.
    """
    
    # Thresholds
    CONFIDENCE_CLOSE_MARGIN = 0.15   # two candidates within this are "too close"
    MIN_CERTAINTY           = 0.50   # below this, packet is not executable
    
    def verify(self, packet: ParsePacket) -> VerificationResult:
        reasons: Set[AmbiguityFlag] = set(packet.ambiguity_flags)  # inherit parser flags
        
        # ---- Phase 0: PARSER_FAILURE shortcut ----
        # Extractor couldn't even find an entity? That's not uncertainty,
        # that's failure to construct a coherent hypothesis.
        if not packet.entity_candidates:
            return VerificationResult(
                status=VerificationStatus.PARSER_FAILURE,
                reasons=set(),
                notes="extractor found no entity candidate",
            )
        
        # ---- Phase 1: Structural check ----
        #
        # Does the declared op_type match what the sentence looks like?
        struct_notes = []
        if packet.source_kind == "fact":
            if packet.op_type not in (OpType.WRITE, OpType.ANCHOR_DEFINE):
                reasons.add(AmbiguityFlag.OP_TYPE_AMBIGUOUS)
                struct_notes.append(f"fact source but op_type={packet.op_type.value}")
        elif packet.source_kind == "query":
            if packet.op_type != OpType.READ:
                reasons.add(AmbiguityFlag.OP_TYPE_AMBIGUOUS)
                struct_notes.append(f"query source but op_type={packet.op_type.value}")
            # If the query also contains a value assertion unrelated to
            # the attribute being asked, that's a semantic conflict.
            if packet.value_candidates and not packet.attribute_candidates:
                reasons.add(AmbiguityFlag.ATTR_VALUE_MISMATCH)
                struct_notes.append("query has value but no attribute trigger")
        
        # Low op_type confidence from extractor → structural ambiguity
        if packet.op_type_confidence < 0.7:
            reasons.add(AmbiguityFlag.OP_TYPE_AMBIGUOUS)
            struct_notes.append(f"op_type_confidence={packet.op_type_confidence:.2f} < 0.7")
        
        # ---- Phase 2: Referential check ----
        #
        # Exactly one clear candidate per required field?
        ref_notes = []
        
        # Entity: one unique entity or at least one dominant
        unique_entities = list({eid for (eid, _, _) in packet.entity_candidates})
        if len(unique_entities) > 1:
            # Check if one has clearly higher confidence
            best_by_id = {}
            for (eid, conf, _) in packet.entity_candidates:
                best_by_id[eid] = max(best_by_id.get(eid, 0.0), conf)
            sorted_confs = sorted(best_by_id.values(), reverse=True)
            if sorted_confs[0] - sorted_confs[1] < self.CONFIDENCE_CLOSE_MARGIN:
                reasons.add(AmbiguityFlag.MULTI_ENTITY_SAME_TYPE)
                ref_notes.append(f"entity tie: {sorted_confs[:2]}")
        
        # Attribute trigger (queries): one unique attr_type or dominant one
        if packet.source_kind == "query":
            unique_attrs = list({a for (a, _, _) in packet.attribute_candidates})
            if len(unique_attrs) > 1:
                best_by_attr = {}
                for (a, conf, _) in packet.attribute_candidates:
                    best_by_attr[a] = max(best_by_attr.get(a, 0.0), conf)
                sorted_confs = sorted(best_by_attr.values(), reverse=True)
                if sorted_confs[0] - sorted_confs[1] < self.CONFIDENCE_CLOSE_MARGIN:
                    reasons.add(AmbiguityFlag.MULTIPLE_ATTR_TRIGGERS)
                    ref_notes.append(f"attr tie: {sorted_confs[:2]}")
        
        # Reference pronoun with >1 plausible antecedent already flagged by extractor
        # but verifier reinforces if reference_candidates has 2+ distinct entity_ids
        if len({eid for (eid, _) in packet.reference_candidates}) > 1:
            reasons.add(AmbiguityFlag.REFERENT_AMBIGUOUS)
            ref_notes.append(f"pronoun binds to {len(packet.reference_candidates)} candidates")
        
        # Value (for WRITE facts): multiple different attr_types in values
        if packet.source_kind == "fact" and packet.op_type == OpType.WRITE:
            attr_types_in_values = {a for (a, _, _, _) in packet.value_candidates}
            if len(attr_types_in_values) > 1:
                reasons.add(AmbiguityFlag.MULTIPLE_ATTR_TRIGGERS)
                ref_notes.append(f"fact has values from {len(attr_types_in_values)} attr types")
            # Multiple DIFFERENT values for SAME attr_type is also ambiguous
            # (e.g. 'The dragon is red and blue' - two colors for one slot)
            value_counts_by_attr = {}
            for (a, _, _, _) in packet.value_candidates:
                value_counts_by_attr[a] = value_counts_by_attr.get(a, 0) + 1
            if any(c > 1 for c in value_counts_by_attr.values()):
                reasons.add(AmbiguityFlag.VALUE_MISSING_OR_UNCLEAR)
                ref_notes.append("multiple values for same attribute slot")
        
        # ---- Phase 3: Executability check ----
        #
        # Does the parse contain everything the declared op_type requires?
        exec_notes = []
        if packet.op_type == OpType.WRITE:
            # Needs: 1 entity, 1 attribute, 1 value
            if not packet.value_candidates:
                reasons.add(AmbiguityFlag.VALUE_MISSING_OR_UNCLEAR)
                exec_notes.append("WRITE without value_candidates")
            if not packet.attribute_candidates:
                reasons.add(AmbiguityFlag.ATTR_VALUE_MISMATCH)
                exec_notes.append("WRITE without attribute_candidates")
        elif packet.op_type == OpType.READ:
            # Needs: 1 entity, 1 attribute
            if not packet.attribute_candidates:
                reasons.add(AmbiguityFlag.TEMPLATE_UNKNOWN)
                exec_notes.append("READ without attribute_candidates")
        elif packet.op_type == OpType.ANCHOR_DEFINE:
            # Needs: 1 entity, 1 class
            has_class = any(a == "__class__" for (a, _, _) in packet.attribute_candidates)
            if not has_class:
                reasons.add(AmbiguityFlag.TEMPLATE_UNKNOWN)
                exec_notes.append("ANCHOR_DEFINE without class noun")
        
        # Overall certainty floor
        if packet.certainty < self.MIN_CERTAINTY:
            reasons.add(AmbiguityFlag.TEMPLATE_UNKNOWN)
            exec_notes.append(f"certainty={packet.certainty:.2f} < {self.MIN_CERTAINTY}")
        
        # ---- Final verdict ----
        if not reasons:
            return VerificationResult(
                status=VerificationStatus.ACCEPT,
                reasons=set(),
                notes=" ; ".join(filter(None, struct_notes + ref_notes + exec_notes)),
            )
        else:
            return VerificationResult(
                status=VerificationStatus.PARSE_UNCERTAIN,
                reasons=reasons,
                notes=" ; ".join(filter(None, struct_notes + ref_notes + exec_notes)),
            )


V15_2_VERIFIER = ParseVerifier()


print("[v15.2] Step 3: ParseVerifier defined")
print(f"         - 3 checks: structural, referential, executability")
print(f"         - CONFIDENCE_CLOSE_MARGIN={ParseVerifier.CONFIDENCE_CLOSE_MARGIN}")
print(f"         - MIN_CERTAINTY={ParseVerifier.MIN_CERTAINTY}")
print(f"         - returns (status, reasons) — reasons empty iff ACCEPT")
# ======================== A2. V15.2 PROTOCOL — STEP 4 ======================
#
# Execution layer: v15_2_write_fact / v15_2_read_query.
#
# Flow:
#   text -> extractor -> ParsePacket -> verifier -> (ACCEPT | PARSE_UNCERTAIN | PARSER_FAILURE)
#   ACCEPT -> execute against DeterministicObjectBank (unchanged from v15.1)
#
# UPDATE decision: made HERE, NOT in the parser.
#   WRITE → if entity exists AND attribute slot is occupied → UPDATE
#         → else new WRITE
#
# Five symbolic outputs for reads:
#   FOUND            : ACCEPTed + bank returned value
#   NONE_OBJECT      : ACCEPTed + entity absent from bank
#   NONE_ATTRIBUTE   : ACCEPTed + entity present, attribute not written
#   PARSE_UNCERTAIN  : verifier blocked execution
#   PARSER_FAILURE   : extractor found no coherent hypothesis
# ===========================================================================


# ------------- Helper: pick top candidate (already ACCEPTed by verifier) -------------

def _top_entity(packet: ParsePacket) -> str:
    """After ACCEPT, the top-confidence entity is the resolved one.
    Verifier already guaranteed no ambiguity.
    """
    best_id, best_conf = None, -1.0
    for (eid, conf, _) in packet.entity_candidates:
        if conf > best_conf:
            best_conf = conf
            best_id = eid
    return best_id


def _top_attribute(packet: ParsePacket) -> Optional[str]:
    """After ACCEPT, the top-confidence attribute trigger."""
    best_attr, best_conf = None, -1.0
    for (attr, conf, _) in packet.attribute_candidates:
        if conf > best_conf:
            best_conf = conf
            best_attr = attr
    return best_attr


def _top_value(packet: ParsePacket) -> Optional[Tuple[str, int]]:
    """After ACCEPT (for WRITE), the single value."""
    if not packet.value_candidates:
        return None
    best = max(packet.value_candidates, key=lambda t: t[2])
    return (best[0], best[1])


# ------------- Write: ACCEPT → WRITE or UPDATE (decided by bank state) -------------

@dataclass
class V15_2_WriteResult:
    """Result of attempting to write a fact."""
    status:          str              # "WRITTEN" | "UPDATED" | "ANCHORED" |
                                        # "PARSE_UNCERTAIN" | "PARSER_FAILURE"
    op_executed:     Optional[OpType] = None    # actual op after UPDATE resolution
    verifier_result: Optional[VerificationResult] = None
    target_entity:   Optional[str]    = None
    target_attr:     Optional[str]    = None
    notes:           str              = ""


def v15_2_write_fact(bank: "DeterministicObjectBank",
                      packet: ParsePacket,
                      entity_emb_fn, class_emb_fn, value_emb_fn,
                      step: int) -> V15_2_WriteResult:
    """Execute a fact ParsePacket against the bank.
    
    If verifier rejects → no write, returns status accordingly.
    If verifier ACCEPTs:
      - ANCHOR_DEFINE: write class binding (no attribute slot)
      - WRITE: check bank state. If entity+attr slot already occupied,
               this is an UPDATE. Otherwise it's a fresh WRITE.
    """
    vr = V15_2_VERIFIER.verify(packet)
    
    if vr.status == VerificationStatus.PARSER_FAILURE:
        return V15_2_WriteResult(status="PARSER_FAILURE", verifier_result=vr)
    if vr.status == VerificationStatus.PARSE_UNCERTAIN:
        return V15_2_WriteResult(status="PARSE_UNCERTAIN", verifier_result=vr,
                                   notes=vr.notes)
    
    # ACCEPT path
    entity_id = _top_entity(packet)
    
    # ANCHOR_DEFINE
    if packet.op_type == OpType.ANCHOR_DEFINE:
        # Look up class noun in parser_evidence
        class_noun = None
        for (attr, _, ev) in packet.attribute_candidates:
            if attr == "__class__":
                class_noun = ev
                break
        ent_emb = entity_emb_fn(entity_id)
        # class_emb_fn expects (class_id, entity_emb); use None -> zero
        cls_emb = class_emb_fn(None, ent_emb)
        existing_slot = bank.find_by_entity_id(entity_id)
        if existing_slot is None:
            bank.allocate_new(entity_id, ent_emb, class_hint=None,
                                class_emb=cls_emb, step=step)
        return V15_2_WriteResult(status="ANCHORED", op_executed=OpType.ANCHOR_DEFINE,
                                   verifier_result=vr, target_entity=entity_id,
                                   target_attr=f"__class__:{class_noun}")
    
    # WRITE / UPDATE
    if packet.op_type == OpType.WRITE:
        attr_type = _top_attribute(packet)
        val_tuple = _top_value(packet)
        if val_tuple is None or attr_type is None:
            return V15_2_WriteResult(status="PARSE_UNCERTAIN", verifier_result=vr,
                                       notes="attr or value missing post-ACCEPT")
        _, value_idx = val_tuple
        
        ent_emb = entity_emb_fn(entity_id)
        cls_emb = class_emb_fn(None, ent_emb)
        val_emb = value_emb_fn(attr_type, value_idx)
        
        existing_slot = bank.find_by_entity_id(entity_id)
        op_actual = OpType.WRITE
        if existing_slot is not None:
            rec = bank.get_record(existing_slot)
            a = rec.attr_slots.get(attr_type)
            if a is not None and a.present:
                op_actual = OpType.UPDATE
        
        if existing_slot is None:
            bank.allocate_new(entity_id, ent_emb, class_hint=None,
                                class_emb=cls_emb, step=step)
        bank.write_attribute(entity_id, attr_type, value_idx, step=step,
                              value_emb=val_emb)
        
        return V15_2_WriteResult(
            status=("UPDATED" if op_actual == OpType.UPDATE else "WRITTEN"),
            op_executed=op_actual,
            verifier_result=vr,
            target_entity=entity_id,
            target_attr=attr_type,
        )
    
    return V15_2_WriteResult(status="PARSE_UNCERTAIN", verifier_result=vr,
                               notes=f"unhandled op_type={packet.op_type.value}")


# ------------- Read: ACCEPT → FOUND / NONE_OBJECT / NONE_ATTRIBUTE -------------

def v15_2_read_query(bank: "DeterministicObjectBank",
                      packet: ParsePacket) -> Tuple[str, Optional[int], VerificationResult]:
    """Execute a query ParsePacket against the bank.
    
    Returns (status, predicted_value_idx, verifier_result):
      FOUND            + value_idx (int)
      NONE_OBJECT      + None
      NONE_ATTRIBUTE   + None
      PARSE_UNCERTAIN  + None
      PARSER_FAILURE   + None
    """
    vr = V15_2_VERIFIER.verify(packet)
    
    if vr.status == VerificationStatus.PARSER_FAILURE:
        return (READ_STATUS_PARSER_FAIL, None, vr)
    if vr.status == VerificationStatus.PARSE_UNCERTAIN:
        return (READ_STATUS_PARSE_UNCERTAIN, None, vr)
    
    # ACCEPT path
    entity_id = _top_entity(packet)
    attr_type = _top_attribute(packet)
    if attr_type is None:
        # Should not happen post-ACCEPT, but fail safe
        return (READ_STATUS_PARSE_UNCERTAIN, None, vr)
    
    status, pred_idx = bank.read_attribute(entity_id, attr_type)
    return (status, pred_idx, vr)


print("[v15.2] Step 4: execution layer defined")
print("         - v15_2_write_fact (WRITE / UPDATE / ANCHORED)")
print("         - v15_2_read_query (5 outputs: FOUND, NONE_OBJECT, NONE_ATTRIBUTE,")
print("           PARSE_UNCERTAIN, PARSER_FAILURE)")
print("         - UPDATE decided from bank state, not from text")
# ======================== B2. V15.2 PROTOCOL VALIDATION ====================
#
# Stage 1.2 validation:
#   Phase 1: memory substrate via v15.2 execution path (substrate preservation)
#   Phase 2: clear probe (fidelity on committed, commit_rate)
#   Phase 3: S1-S4 ambiguity probes (honesty, overcommit)
#   Phase 4: v15.1 substrate benchmark (re-run for full metrics)
#   Phase 5: 6-line verdict assembly
#
# Rules:
#   - Clean case + PARSE_UNCERTAIN = failure (don't reward timidity)
#   - Ambiguous case + commit      = failure (don't reward overcommit)
# ===========================================================================


V15_2_BENCH_CONFIG = {
    "seed_s_probes":   20260701,
    "seed_reinterp":   20260702,
    "n_s_per_probe":   200,
    "n_reinterp":      500,
}


# ---- S1-S4 generators ----

def _v15_2_gen_S1_multi_attr(rng):
    (ent, _) = rng.choice(V15_HELDOUT_ENTITIES)
    attrs = rng.sample(["color", "size", "location", "state"], 2)
    trigger = {"color":"color","size":"size","location":"location","state":"mood"}
    q = f"What is the {trigger[attrs[0]]} and {trigger[attrs[1]]} of the {ent}? The {ent} is"
    return q, {"expected_flags": {"MULTIPLE_ATTR_TRIGGERS"}, "entity": ent}


def _v15_2_gen_S2_semantic_conflict(rng):
    (ent, _) = rng.choice(V15_HELDOUT_ENTITIES)
    q_attr = rng.choice(["color", "size", "location", "state"])
    v_attr = rng.choice([a for a in ["color","size","location","state"] if a != q_attr])
    v_value = rng.choice(V15_ATTR_VALUES[v_attr])
    trigger = {"color":"color","size":"size","location":"location","state":"mood"}
    q = f"What {trigger[q_attr]} is the {v_value} {ent}? The {ent} is"
    return q, {"expected_flags": {"MULTIPLE_ATTR_TRIGGERS","ATTR_VALUE_MISMATCH"},
                 "entity": ent}


def _v15_2_gen_S3_referential(rng):
    picks = rng.sample(V15_HELDOUT_ENTITIES, 2)
    e1, e2 = picks[0][0], picks[1][0]
    q_attr = rng.choice(["color","size","location","state"])
    trigger = {"color":"color","size":"size","location":"location","state":"mood"}
    q = f"The {e1} and the {e2} are here. What {trigger[q_attr]} is it? It is"
    return q, {"expected_flags": {"REFERENT_AMBIGUOUS"}, "entities": [e1, e2]}


def _v15_2_gen_S4_attr_conflict(rng):
    (ent, _) = rng.choice(V15_HELDOUT_ENTITIES)
    attr = rng.choice(["color","size","location","state"])
    vals = rng.sample(V15_ATTR_VALUES[attr], 2)
    fact = f"The {ent} is {vals[0]} and {vals[1]}."
    return fact, {"expected_flags": {"VALUE_MISSING_OR_UNCLEAR"}, "entity": ent,
                    "source_kind": "fact"}


# ---- Scoring ----

def _v15_2_score_query(status, reasons, expected_flags):
    if status == READ_STATUS_PARSER_FAIL:
        return "FAILURE"
    if status == READ_STATUS_PARSE_UNCERTAIN:
        rs = {r.value for r in reasons}
        if rs & expected_flags:
            return "HONEST"
        return "UNCERTAIN_OTHER_REASON"
    return "OVERCOMMIT"


def _v15_2_score_fact(write_res, expected_flags):
    if write_res.status == "PARSER_FAILURE":
        return "FAILURE"
    if write_res.status == "PARSE_UNCERTAIN":
        vr = write_res.verifier_result
        rs = {r.value for r in (vr.reasons if vr else set())}
        if rs & expected_flags:
            return "HONEST"
        return "UNCERTAIN_OTHER_REASON"
    return "OVERCOMMIT"


def v15_2_run_s_probes(bank, ent_fn, cls_fn, val_fn, cfg):
    n = cfg["n_s_per_probe"]
    out = {}
    for name, gen, is_fact in [
        ("S1", _v15_2_gen_S1_multi_attr,       False),
        ("S2", _v15_2_gen_S2_semantic_conflict, False),
        ("S3", _v15_2_gen_S3_referential,       False),
        ("S4", _v15_2_gen_S4_attr_conflict,     True),
    ]:
        rng = random.Random(cfg["seed_s_probes"] + hash(name) % 10000)
        counts = {"HONEST": 0, "OVERCOMMIT": 0, "FAILURE": 0,
                   "UNCERTAIN_OTHER_REASON": 0}
        for i in range(n):
            text, meta = gen(rng)
            bank.reset()
            if is_fact:
                pkt = v15_2_parse_fact(text)
                res = v15_2_write_fact(bank, pkt, ent_fn, cls_fn, val_fn, step=0)
                score = _v15_2_score_fact(res, meta["expected_flags"])
            else:
                pkt = v15_2_parse_query(text)
                status, _, vr = v15_2_read_query(bank, pkt)
                score = _v15_2_score_query(status, vr.reasons, meta["expected_flags"])
            counts[score] += 1
        honesty = counts["HONEST"] / n
        overcommit = counts["OVERCOMMIT"] / n
        out[name] = {
            "n": n,
            "honest":        counts["HONEST"],
            "overcommit":    counts["OVERCOMMIT"],
            "failure":       counts["FAILURE"],
            "uncertain_misc": counts["UNCERTAIN_OTHER_REASON"],
            "honesty_rate":   honesty,
            "overcommit_rate": overcommit,
        }
        print(f"  {name}: honesty={honesty:.1%}  overcommit={overcommit:.1%}  "
              f"failure={counts['FAILURE']/n:.1%}  n={n}")
    return out


# ---- Clear probe (fidelity + commit_rate) ----

def v15_2_run_clear_probe(bank, ent_fn, cls_fn, val_fn, cfg):
    rng = random.Random(cfg["seed_reinterp"])
    n = cfg["n_reinterp"]
    n_committed = n_correct = n_uncertain = n_failure = 0
    for i in range(n):
        ep = v15_generate_episode("single_attr_simple", rng, use_heldout=True)
        bank.reset()
        for step_idx, fact_text in enumerate(ep.facts):
            pkt_f = v15_2_parse_fact(fact_text)
            v15_2_write_fact(bank, pkt_f, ent_fn, cls_fn, val_fn, step=step_idx)
        pkt_q = v15_2_parse_query(ep.query)
        status, pred_idx, vr = v15_2_read_query(bank, pkt_q)
        if status == READ_STATUS_PARSER_FAIL:
            n_failure += 1
        elif status == READ_STATUS_PARSE_UNCERTAIN:
            n_uncertain += 1
        else:
            n_committed += 1
            attr = V15_ATTR_TYPES[ep.query_attr_label]
            vocab = V15_ATTR_VALUES[attr]
            t_tok = int(ep.target_answer_token)
            target_idx = None
            for k, v_str in enumerate(vocab):
                if V15_ANSWER_TOKENS[attr].get(v_str) == t_tok:
                    target_idx = k; break
            if ep.target_is_unknown:
                if status in (READ_STATUS_NONE_OBJECT, READ_STATUS_NONE_ATTRIBUTE):
                    n_correct += 1
            else:
                if status == READ_STATUS_FOUND and pred_idx == target_idx:
                    n_correct += 1
    return {
        "n":                      n,
        "commit_rate":            n_committed / n,
        "uncertain_rate":         n_uncertain / n,
        "failure_rate":           n_failure / n,
        "fidelity_on_committed":  n_correct / max(1, n_committed),
        "coverage":               (n_committed + n_uncertain) / n,
        "n_committed":            n_committed,
        "n_correct":              n_correct,
    }


# ---- Main orchestrator ----

def v15_2_validate_protocol(bank, base_model, v15_1_memory):
    results = {"config": dict(V15_2_BENCH_CONFIG)}
    ent_fn = _make_entity_emb_fn(base_model)
    cls_fn = _make_class_emb_fn(v15_1_memory)
    val_fn = _make_value_emb_fn(base_model)
    
    print()
    print(SEP)
    print("[v15.2 PROTOCOL VALIDATION]")
    print(f"  seed_s_probes = {V15_2_BENCH_CONFIG['seed_s_probes']}")
    print(f"  seed_reinterp = {V15_2_BENCH_CONFIG['seed_reinterp']}")
    print(f"  n_s_per_probe = {V15_2_BENCH_CONFIG['n_s_per_probe']}")
    print(f"  n_reinterp    = {V15_2_BENCH_CONFIG['n_reinterp']}")
    print(SEP)
    
    print()
    print("Phase 1: clear probe via v15.2 path")
    clear = v15_2_run_clear_probe(bank, ent_fn, cls_fn, val_fn, V15_2_BENCH_CONFIG)
    print(f"  commit_rate:           {clear['commit_rate']:.1%}")
    print(f"  fidelity_on_committed: {clear['fidelity_on_committed']:.1%}")
    print(f"  uncertain_rate:        {clear['uncertain_rate']:.1%}  (cowardice on clear)")
    print(f"  failure_rate:          {clear['failure_rate']:.1%}")
    results["clear_probe"] = clear
    
    print()
    print("Phase 2: S1-S4 ambiguity probes")
    s_results = v15_2_run_s_probes(bank, ent_fn, cls_fn, val_fn, V15_2_BENCH_CONFIG)
    avg_honesty = sum(v["honesty_rate"]   for v in s_results.values()) / len(s_results)
    avg_oc      = sum(v["overcommit_rate"] for v in s_results.values()) / len(s_results)
    print(f"  average honesty:    {avg_honesty:.1%}")
    print(f"  average overcommit: {avg_oc:.1%}")
    results["s_probes"]          = s_results
    results["avg_honesty_rate"]   = avg_honesty
    results["avg_overcommit_rate"] = avg_oc
    
    print()
    print("Phase 3: v15.1 substrate benchmark (substrate preservation check)")
    substrate = v15_1_validate_substrate(bank, base_model, v15_1_memory)
    results["v15_1_substrate"] = substrate["_verdicts"]
    
    # ---- Verdict assembly ----
    print()
    print(SEP)
    print("--- V15.2 VERDICTS ---")
    print(SEP)
    
    mem_pass       = substrate["_verdicts"]["memory_substrate"]["pass"]
    coverage_val   = clear["coverage"]
    coverage_pass  = coverage_val >= 0.90
    fidelity_val   = clear["fidelity_on_committed"]
    fidelity_pass  = fidelity_val >= 0.95
    honesty_pass   = avg_honesty >= 0.80
    p5_status      = substrate["_verdicts"]["p5_scaling"]["status"]
    
    blockers = []
    if not mem_pass:       blockers.append("Memory Substrate FAIL")
    if not coverage_pass:  blockers.append(f"Parser Coverage FAIL ({coverage_val:.1%})")
    if not fidelity_pass:  blockers.append(f"Parser Fidelity FAIL ({fidelity_val:.1%})")
    if not honesty_pass:   blockers.append(f"Parser Honesty FAIL ({avg_honesty:.1%})")
    if p5_status != "COMPLETE": blockers.append(f"P5 {p5_status}")
    decision = "READY_FOR_SHADOW" if not blockers else "NOT_READY_FOR_SHADOW"
    
    results["_verdicts"] = {
        "memory_substrate":   {"pass": mem_pass},
        "parser_coverage":    {"pass": coverage_pass, "value": coverage_val,
                                "threshold": 0.90},
        "parser_fidelity":    {"pass": fidelity_pass, "value": fidelity_val,
                                "threshold": 0.95,
                                "n_committed": clear["n_committed"]},
        "parser_honesty":     {"pass": honesty_pass, "value": avg_honesty,
                                "threshold": 0.80,
                                "per_probe": {k: v["honesty_rate"] for k, v in s_results.items()}},
        "p5_scaling":         {"status": p5_status},
        "commit_rate_clear":  clear["commit_rate"],
        "avg_overcommit_s_probes": avg_oc,
        "decision":           decision,
        "decision_blockers":  blockers,
    }
    
    print(f"  Memory Substrate:  {'PASS' if mem_pass else 'FAIL'}")
    print(f"  Parser Coverage:   {'PASS' if coverage_pass else 'FAIL'}  ({coverage_val:.1%})")
    print(f"  Parser Fidelity:   {'PASS' if fidelity_pass else 'FAIL'}  ({fidelity_val:.1%})")
    print(f"  Parser Honesty:    {'PASS' if honesty_pass else 'FAIL'}  ({avg_honesty:.1%})")
    print(f"  P5 scaling:        {p5_status}")
    print(f"  Commit rate clear: {clear['commit_rate']:.1%}")
    print(f"  Overcommit avg:    {avg_oc:.1%}")
    print(f"  Decision:          {decision}")
    if blockers:
        print(f"  Blockers:          {'; '.join(blockers)}")
    print(SEP)
    
    return results


def v15_2_write_memo(results, path):
    v = results["_verdicts"]
    lines = []
    lines.append("# v15.2 Stage 1.2 Internal Memo")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append(f"- **Memory Substrate**: {'PASS' if v['memory_substrate']['pass'] else 'FAIL'}")
    lines.append(f"- **Parser Coverage**: {'PASS' if v['parser_coverage']['pass'] else 'FAIL'}")
    lines.append(f"- **Parser Fidelity**: {'PASS' if v['parser_fidelity']['pass'] else 'FAIL'}")
    lines.append(f"- **Parser Honesty**: {'PASS' if v['parser_honesty']['pass'] else 'FAIL'}")
    lines.append(f"- **P5 scaling**: {v['p5_scaling']['status']}")
    lines.append(f"- **Decision**: {v['decision']}")
    if v["decision_blockers"]:
        lines.append(f"- **Blockers**: {'; '.join(v['decision_blockers'])}")
    lines.append("")
    lines.append("## Metrics")
    lines.append(f"- coverage on clear:     {v['parser_coverage']['value']:.1%}")
    lines.append(f"- fidelity on committed: {v['parser_fidelity']['value']:.1%}")
    lines.append(f"- honesty avg (S1-S4):   {v['parser_honesty']['value']:.1%}")
    lines.append(f"- commit rate on clear:  {v['commit_rate_clear']:.1%}")
    lines.append(f"- overcommit on S1-S4:   {v['avg_overcommit_s_probes']:.1%}")
    lines.append("")
    lines.append("## Per-probe honesty (S1-S4)")
    for k, val in v["parser_honesty"]["per_probe"].items():
        lines.append(f"- {k}: {val:.1%}")
    lines.append("")
    lines.append("## Full results")
    lines.append("```")
    lines.append(json.dumps(results, indent=2, default=str))
    lines.append("```")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


print("[v15.2] Section B2 defined: protocol validation")
print("         - Phase 1: clear probe (fidelity, commit_rate)")
print("         - Phase 2: S1-S4 (honesty, overcommit)")
print("         - Phase 3: v15.1 substrate preservation")
print("         - 6-line Status verdict + Decision")
# ======================== D2. V15.2 SHADOW TRAINING ========================
#
# Shadow training with v15.2 oracle:
#   - Input to shadow: text-derived features (q_pooled, entity_emb, slot_feats)
#   - Target for shadow: output of the FULLY VALIDATED pipeline (v15.2
#     parse -> verifier ACCEPT -> bank read). NEVER raw parser output.
#   - Skip episode entirely if verifier rejects (PARSE_UNCERTAIN or FAILURE).
#   - Trusted episode types only.
#
# This implements distillation from the validated pipeline, not from the
# raw extractor.
# ===========================================================================


V15_2_SHADOW_CONFIG = {
    "n_steps":         2000,
    "batch_episodes":  4,
    "lr":              3e-4,
    "weight_decay":    0.01,
    "betas":           (0.9, 0.95),
    "warmup_steps":    200,
    "grad_clip":       1.0,
    "log_every":       50,
    "seed":            20260802,
    "w_attr":          1.0,
    "w_value":         1.0,
    "w_object":        1.0,
}

# Trusted episode types (SAME as v15.1 SHADOW_EPISODE_TYPES, kept here for clarity)
V15_2_SHADOW_TRUSTED_TYPES = [
    ("single_attr_simple",  0.35),
    ("multi_attr_object",   0.25),
    ("selective_update",    0.20),
    ("no_match",            0.15),
    ("provisional_entity",  0.05),
]


def _v15_2_sample_shadow_type(rng):
    r = rng.random()
    cum = 0.0
    for name, p in V15_2_SHADOW_TRUSTED_TYPES:
        cum += p
        if r < cum:
            return name
    return V15_2_SHADOW_TRUSTED_TYPES[-1][0]


def _v15_2_shadow_lr_at(step, cfg):
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    t = (step - cfg["warmup_steps"]) / max(1, cfg["n_steps"] - cfg["warmup_steps"])
    t = min(1.0, max(0.0, t))
    return cfg["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def _v15_2_compute_shadow_losses(base_model, v15_1_memory, bank,
                                    batch_episodes, cfg, current_step=0):
    """Compute shadow losses via v15.2 validated oracle.
    
    For each episode:
      1. bank.reset()
      2. Write facts via v15.2 (parser -> verifier). Skip facts that fail verifier.
      3. Parse query via v15.2. If verifier REJECTS the query, skip episode.
      4. ACCEPT: use bank (oracle) to produce ground truth (entity_slot, attr, value_idx).
      5. Shadow predicts. Loss on each head.
    """
    device = DEVICE
    shadow = v15_1_memory.shadow
    losses = {
        "attr":   torch.zeros((), device=device),
        "value":  torch.zeros((), device=device),
        "object": torch.zeros((), device=device),
    }
    counts = {k: 0 for k in losses}
    
    entity_emb_fn = _make_entity_emb_fn(base_model)
    class_emb_fn  = _make_class_emb_fn(v15_1_memory)
    value_emb_fn  = _make_value_emb_fn(base_model)
    
    skipped_query_rejected = 0
    
    for ep in batch_episodes:
        bank.reset()
        # Write facts through v15.2 pipeline (includes verifier)
        for step_idx, fact_text in enumerate(ep.facts):
            pkt_f = v15_2_parse_fact(fact_text)
            v15_2_write_fact(bank, pkt_f, entity_emb_fn, class_emb_fn,
                              value_emb_fn, step=step_idx)
            # If verifier rejected or parser failed, write_fact is a no-op.
            # We intentionally allow partial bank state — tests cases like
            # no_match where one of the facts may be filtered out are still
            # valid training signal.
        
        # Query parse + verify
        pkt_q = v15_2_parse_query(ep.query)
        vr_q  = V15_2_VERIFIER.verify(pkt_q)
        
        if vr_q.status != VerificationStatus.ACCEPT:
            # Verifier rejected the query. Do not train on it.
            skipped_query_rejected += 1
            continue
        
        # Resolved fields (ACCEPTed -> no ambiguity by construction)
        entity_id = _top_entity(pkt_q)
        attr_type = _top_attribute(pkt_q)
        
        # Oracle ground truth from bank (validated output)
        slot = bank.find_by_entity_id(entity_id)
        # Oracle attr target: ACCEPTed means parser extracted a concrete attr
        attr_tgt_idx = V15_ATTR_TO_IDX[attr_type]
        
        # ---- Shadow attr_router ----
        q_ids = torch.tensor(ENC.encode(ep.query), dtype=torch.long, device=device)
        q_pooled = base_model.shared_token_emb(q_ids).mean(dim=0)
        attr_logits = shadow.attr_router(q_pooled.unsqueeze(0))
        tgt = torch.tensor([attr_tgt_idx], dtype=torch.long, device=device)
        losses["attr"] = losses["attr"] + F.cross_entropy(attr_logits, tgt)
        counts["attr"] += 1
        
        # ---- Shadow object_resolver ----
        q_entity_emb = entity_emb_fn(entity_id)
        slot_feats = _build_slot_features(bank, q_entity_emb, None, current_step)
        resolver_logits = shadow.object_resolver(q_entity_emb, slot_feats)
        K = slot_feats.shape[0]
        if slot is None:
            obj_tgt = K  # NONE_OBJECT
        else:
            slot_list = bank.occupied_slots()
            obj_tgt = slot_list.index(slot)
        tgt_obj = torch.tensor([obj_tgt], dtype=torch.long, device=device)
        losses["object"] = losses["object"] + F.cross_entropy(
            resolver_logits.unsqueeze(0), tgt_obj)
        counts["object"] += 1
        
        # ---- Shadow value_heads ----
        if slot is not None:
            rec = bank.get_record(slot)
            a = rec.attr_slots.get(attr_type)
            if a is not None and a.present and a.value_emb is not None:
                value_logits = shadow.value_heads(attr_type,
                                                    a.value_emb.unsqueeze(0))
                tgt_v = torch.tensor([a.value_idx], dtype=torch.long, device=device)
                losses["value"] = losses["value"] + F.cross_entropy(value_logits, tgt_v)
                counts["value"] += 1
    
    # Normalize
    total = torch.zeros((), device=device)
    total = total + cfg["w_attr"]   * (losses["attr"]   / max(1, counts["attr"]))
    total = total + cfg["w_value"]  * (losses["value"]  / max(1, counts["value"]))
    total = total + cfg["w_object"] * (losses["object"] / max(1, counts["object"]))
    
    parts = {k: (losses[k] / max(1, counts[k])) for k in losses}
    parts["_skipped_query_rejected"] = skipped_query_rejected
    return total, parts


def v15_2_train_shadow_main(bank, base_model, v15_1_memory):
    """Shadow training via v15.2 validated oracle."""
    cfg = V15_2_SHADOW_CONFIG
    print()
    print(SEP)
    print("[v15.2 SHADOW TRAINING] (via v15.2 validated oracle)")
    print(f"  steps: {cfg['n_steps']}  batch: {cfg['batch_episodes']}")
    print(f"  warmup: {cfg['warmup_steps']}  LR: {cfg['lr']:.0e}")
    print(f"  trusted episode types: {[n for n, _ in V15_2_SHADOW_TRUSTED_TYPES]}")
    print(SEP)
    
    shadow = v15_1_memory.shadow
    shadow.train()
    params = [p for p in shadow.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=cfg["betas"],
                              weight_decay=cfg["weight_decay"])
    
    rng = random.Random(cfg["seed"])
    loss_hist = []
    t0 = time.time()
    skipped_steps = 0
    total_rejected_queries = 0
    
    for step in range(cfg["n_steps"]):
        batch = []
        for _ in range(cfg["batch_episodes"]):
            ep_type = _v15_2_sample_shadow_type(rng)
            batch.append(v15_generate_episode(ep_type, rng, use_heldout=False))
        
        for g in opt.param_groups:
            g["lr"] = _v15_2_shadow_lr_at(step, cfg)
        
        opt.zero_grad(set_to_none=True)
        total, parts = _v15_2_compute_shadow_losses(
            base_model, v15_1_memory, bank, batch, cfg, current_step=step)
        total_rejected_queries += parts.pop("_skipped_query_rejected")
        
        if not total.requires_grad:
            skipped_steps += 1
            continue
        
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        opt.step()
        
        loss_hist.append({
            "step":  step + 1,
            "total": float(total.item()),
            **{k: float(v.detach().item() if v.numel()==1 else v.detach().mean().item())
               for k, v in parts.items()},
        })
        
        if (step + 1) % cfg["log_every"] == 0:
            elapsed = time.time() - t0
            eta_s = (cfg["n_steps"] - step - 1) * (elapsed / max(1, step + 1))
            parts_str = " ".join(f"{k}={float(v.item() if v.numel()==1 else v.mean().item()):.3f}"
                                   for k, v in parts.items())
            print(f"[v15.2 SHADOW] step {step+1}/{cfg['n_steps']} "
                  f"total={float(total.item()):.3f} lr={opt.param_groups[0]['lr']:.2e} "
                  f"ETA={int(eta_s//60)}m{int(eta_s%60)}s", flush=True)
            print(f"             {parts_str}", flush=True)
    
    if skipped_steps > 0:
        print(f"[v15.2 SHADOW] skipped {skipped_steps} steps (no grad path)")
    if total_rejected_queries > 0:
        print(f"[v15.2 SHADOW] rejected {total_rejected_queries} query trials "
              f"(verifier blocked). Skipped from training as designed.")
    
    shadow.eval()
    return {
        "loss_history":           loss_hist,
        "final_total":            float(loss_hist[-1]["total"]) if loss_hist else None,
        "skipped_steps":          skipped_steps,
        "total_rejected_queries": total_rejected_queries,
    }


print("[v15.2] Section D2 defined: shadow training via v15.2 oracle")
print("        - target from bank AFTER verifier ACCEPT (distillation)")
print("        - skip episodes where verifier rejects (PARSE_UNCERTAIN)")
print("        - trusted episode types only")
# ======================== E2. V15.2 SHADOW AUDIT ===========================
#
# A1 audit via v15.2 pipeline, on TRUSTED vs HARD sets, with GPT's brutal
# thresholds.
#
# Three modes:
#   critical_only  : full v15.2 oracle (parse -> verifier -> bank)
#                    ACCEPT on query -> bank.read_attribute
#                    UNCERTAIN/FAILURE -> counted as such
#   shadow_only    : use shadow heads for attr_router + object_resolver + value_heads
#                    Still gates on verifier ACCEPT (to respect protocol)
#   mixed          : verifier ACCEPT + bank object resolution + shadow value head
#
# Two sets:
#   TRUSTED: single_attr_simple + multi_attr_object
#   HARD:    paraphrase + coreference_distant
#
# Thresholds (GPT):
#   critical_trusted_clean:  >= 99.5%
#   shadow_trusted_p_target: >= 85%
#   mixed_trusted_not_worse: >= critical_trusted - 0.5pp
#   no_hidden_shortcut:      mixed_trusted does NOT exceed critical_trusted
#
# Run is INVALIDATED if critical_only deviates from full oracle.
# ===========================================================================


def _v15_2_run_one_mode_trial(bank, base_model, v15_1_memory, ep, mode,
                                entity_emb_fn, class_emb_fn, value_emb_fn):
    """Run one episode in specified mode, return (correct, status, pred_idx)."""
    shadow = v15_1_memory.shadow
    bank.reset()
    
    # Write facts via v15.2 ALWAYS (same for all modes)
    for step_idx, fact_text in enumerate(ep.facts):
        pkt_f = v15_2_parse_fact(fact_text)
        v15_2_write_fact(bank, pkt_f, entity_emb_fn, class_emb_fn,
                          value_emb_fn, step=step_idx)
    
    # Parse query
    pkt_q = v15_2_parse_query(ep.query)
    vr_q  = V15_2_VERIFIER.verify(pkt_q)
    
    # Compute target
    target_idx = None
    target_is_unknown_obj  = False
    target_is_unknown_attr = False
    if ep.target_is_unknown:
        # Determine whether target is NONE_OBJECT or NONE_ATTRIBUTE
        q_ent_str = _top_entity(pkt_q) if pkt_q.entity_candidates else ""
        fact_eids = set()
        for f in ep.facts:
            pf = v15_2_parse_fact(f)
            if pf.entity_candidates:
                fact_eids.add(_top_entity(pf))
        if q_ent_str not in fact_eids:
            target_is_unknown_obj = True
        else:
            target_is_unknown_attr = True
    else:
        attr_type = V15_ATTR_TYPES[ep.query_attr_label]
        vocab = V15_ATTR_VALUES[attr_type]
        t_tok = int(ep.target_answer_token)
        for k, v_str in enumerate(vocab):
            if V15_ANSWER_TOKENS[attr_type].get(v_str) == t_tok:
                target_idx = k; break
    
    # Produce prediction based on mode
    if vr_q.status != VerificationStatus.ACCEPT:
        # All modes respect verifier - if it rejects, result is PARSE_UNCERTAIN/FAILURE
        if vr_q.status == VerificationStatus.PARSER_FAILURE:
            read_status = READ_STATUS_PARSER_FAIL
        else:
            read_status = READ_STATUS_PARSE_UNCERTAIN
        pred = None
    else:
        entity_id = _top_entity(pkt_q)
        attr_type = _top_attribute(pkt_q)
        
        if mode == "critical_only":
            # Pure oracle
            read_status, pred = bank.read_attribute(entity_id, attr_type)
        
        elif mode == "shadow_only":
            # Shadow for attr + object + value
            with torch.no_grad():
                q_ids = torch.tensor(ENC.encode(ep.query), dtype=torch.long, device=DEVICE)
                q_pooled = base_model.shared_token_emb(q_ids).mean(dim=0)
                attr_logits = shadow.attr_router(q_pooled.unsqueeze(0))
                attr_pred_idx = int(attr_logits.argmax(dim=-1).item())
                # Resolve object
                q_entity_emb = entity_emb_fn(entity_id)
                slot_feats = _build_slot_features(bank, q_entity_emb, None, current_step=1000)
                resolver_logits = shadow.object_resolver(q_entity_emb, slot_feats)
                obj_pred = int(resolver_logits.argmax(dim=-1).item())
                K = slot_feats.shape[0]
            if obj_pred == K:
                read_status = READ_STATUS_NONE_OBJECT; pred = None
            elif attr_pred_idx == 4:  # NONE_ATTR
                read_status = READ_STATUS_NONE_ATTRIBUTE; pred = None
            else:
                at = V15_ATTR_TYPES[attr_pred_idx]
                slot_list = bank.occupied_slots()
                rec = bank.get_record(slot_list[obj_pred])
                a = rec.attr_slots.get(at)
                if a is None or not a.present or a.value_emb is None:
                    read_status = READ_STATUS_NONE_ATTRIBUTE; pred = None
                else:
                    with torch.no_grad():
                        vl = shadow.value_heads(at, a.value_emb.unsqueeze(0))
                    pred = int(vl.argmax(dim=-1).item())
                    read_status = READ_STATUS_FOUND
        
        elif mode == "mixed":
            # Critical for object, shadow for value
            slot = bank.find_by_entity_id(entity_id)
            if slot is None:
                read_status = READ_STATUS_NONE_OBJECT; pred = None
            else:
                rec = bank.get_record(slot)
                a = rec.attr_slots.get(attr_type)
                if a is None or not a.present or a.value_emb is None:
                    read_status = READ_STATUS_NONE_ATTRIBUTE; pred = None
                else:
                    with torch.no_grad():
                        vl = shadow.value_heads(attr_type, a.value_emb.unsqueeze(0))
                    pred = int(vl.argmax(dim=-1).item())
                    read_status = READ_STATUS_FOUND
    
    # Score
    if target_is_unknown_obj:
        correct = (read_status == READ_STATUS_NONE_OBJECT)
    elif target_is_unknown_attr:
        correct = (read_status == READ_STATUS_NONE_ATTRIBUTE)
    else:
        correct = (read_status == READ_STATUS_FOUND and pred == target_idx)
    
    return correct, read_status, pred


def _v15_2_audit_A1_on_set(bank, base_model, v15_1_memory, episode_types,
                              n_per_mode, cfg, tag):
    """Run A1 audit on specified episode types, return per-mode accuracy."""
    ent_fn = _make_entity_emb_fn(base_model)
    cls_fn = _make_class_emb_fn(v15_1_memory)
    val_fn = _make_value_emb_fn(base_model)
    
    out = {}
    for mode in ("critical_only", "shadow_only", "mixed"):
        rng = random.Random(hash(tag + mode) % (2**31))
        correct = 0
        total = 0
        status_counts = {}
        for i in range(n_per_mode):
            # Sample episode type
            r = rng.random()
            cum = 0.0
            chosen = episode_types[-1][0]
            for name, p in episode_types:
                cum += p
                if r < cum:
                    chosen = name; break
            ep = v15_generate_episode(chosen, rng, use_heldout=True)
            is_correct, status, _ = _v15_2_run_one_mode_trial(
                bank, base_model, v15_1_memory, ep, mode, ent_fn, cls_fn, val_fn)
            if is_correct:
                correct += 1
            total += 1
            status_counts[status] = status_counts.get(status, 0) + 1
        out[mode] = {
            "n":        total,
            "correct":  correct,
            "accuracy": correct / max(1, total),
            "status_distribution": status_counts,
        }
    return out


def v15_2_run_shadow_audit(bank, base_model, v15_1_memory, cfg=None):
    """Run A1 audit through v15.2 pipeline with trusted/hard split + GPT thresholds."""
    if cfg is None:
        cfg = {"n_per_mode": 200}
    
    trusted_types = [("single_attr_simple", 0.7), ("multi_attr_object", 0.3)]
    hard_types    = [("paraphrase", 0.5),         ("coreference_distant", 0.5)]
    
    print()
    print(SEP)
    print("[v15.2 SHADOW AUDIT] A1 critical_vs_shadow (TRUSTED | HARD)")
    print(SEP)
    
    trusted_results = _v15_2_audit_A1_on_set(
        bank, base_model, v15_1_memory, trusted_types,
        cfg["n_per_mode"], cfg, "trusted")
    hard_results = _v15_2_audit_A1_on_set(
        bank, base_model, v15_1_memory, hard_types,
        cfg["n_per_mode"], cfg, "hard")
    
    print("\n--- TRUSTED set ---")
    for mode in ("critical_only", "shadow_only", "mixed"):
        v = trusted_results[mode]
        print(f"  {mode:15s}: acc={v['accuracy']:.1%}  n={v['n']}")
    print("\n--- HARD set ---")
    for mode in ("critical_only", "shadow_only", "mixed"):
        v = hard_results[mode]
        print(f"  {mode:15s}: acc={v['accuracy']:.1%}  n={v['n']}")
    
    # Interpretation per GPT thresholds
    crit_trusted   = trusted_results["critical_only"]["accuracy"]
    shadow_trusted = trusted_results["shadow_only"]["accuracy"]
    mixed_trusted  = trusted_results["mixed"]["accuracy"]
    crit_hard      = hard_results["critical_only"]["accuracy"]
    shadow_hard    = hard_results["shadow_only"]["accuracy"]
    mixed_hard     = hard_results["mixed"]["accuracy"]
    
    interp = {
        "critical_trusted_clean":   crit_trusted >= 0.995,
        "critical_hard_clean":      crit_hard    >= 0.995,
        "shadow_trusted_target":    shadow_trusted >= 0.85,
        "mixed_trusted_not_worse":  mixed_trusted >= (crit_trusted - 0.005),
        "no_hidden_shortcut":       mixed_trusted <= crit_trusted + 0.001,
    }
    
    print("\n--- Interpretation (GPT thresholds) ---")
    for k, v in interp.items():
        print(f"  {k}: {v}")
    
    # Critical path deviation = RUN INVALIDATED
    run_invalidated = not interp["critical_trusted_clean"]
    
    if run_invalidated:
        shadow_pass = False
        shadow_text = (f"RUN INVALIDATED. Critical path on TRUSTED set deviated "
                       f"from oracle: acc={crit_trusted:.1%} < 99.5% threshold. "
                       f"This means the v15.2 oracle is corrupted in some way "
                       f"during shadow training. Investigate before proceeding.")
    elif not interp["mixed_trusted_not_worse"]:
        shadow_pass = False
        shadow_text = (f"Mixed mode drags TRUSTED below critical-0.5pp: "
                       f"critical={crit_trusted:.1%}, mixed={mixed_trusted:.1%}. "
                       f"Shadow value heads contaminate the oracle path.")
    elif not interp["no_hidden_shortcut"]:
        shadow_pass = False
        shadow_text = f"Hidden shortcut: mixed ({mixed_trusted:.1%}) exceeds critical ({crit_trusted:.1%})."
    elif interp["shadow_trusted_target"]:
        shadow_pass = True
        shadow_text = (f"Shadow reached target on TRUSTED: "
                       f"shadow_only={shadow_trusted:.1%} >= 85%. "
                       f"HARD set reported separately: "
                       f"shadow_only_hard={shadow_hard:.1%}.")
    else:
        shadow_pass = True  # Stage 1 preserved but shadow not at target
        shadow_text = (f"Stage 1 preserved (critical clean, mixed safe, no shortcut), "
                       f"but shadow below 85% target: shadow_only={shadow_trusted:.1%}. "
                       f"More training needed.")
    
    return {
        "trusted": trusted_results,
        "hard":    hard_results,
        "_interpretation": interp,
        "_thresholds": {
            "critical_floor":      0.995,
            "shadow_trusted_min":  0.85,
            "mixed_trusted_slack": 0.005,
        },
        "_shadow_verdict": {
            "pass":            shadow_pass,
            "run_invalidated": run_invalidated,
            "text":            shadow_text,
            "interpretation":  interp,
        },
    }


print("[v15.2] Section E2 defined: A1 audit via v15.2 pipeline")
print("        - TRUSTED vs HARD split")
print("        - critical_only via v15.2 oracle")
print("        - shadow_only via shadow heads (verifier-gated)")
print("        - mixed via bank object + shadow value")
print("        - GPT thresholds: critical >= 99.5%, mixed >= critical - 0.5pp")
print("        - critical path deviation -> RUN INVALIDATED")

# ======================== H. V15.3 HARD DIAGNOSTIC =========================
#
# Pas 1 al Stage 1.3: diagnostic HARD pur, fără fixuri.
#
# Scop: identifică exact unde se pierde acuratețea pe HARD (paraphrase +
# coreference_distant). Raportează metrici la granularitatea necesară pentru
# a alege corect următorul patch arhitectural.
#
# Nu modifică parser-ul. Nu modifică verifier-ul. Nu re-antrenează shadow.
# Doar măsoară.
#
# Metrici raportate:
#   reject_rate_hard_total
#   reject_rate_by_reason (per AmbiguityFlag, evidențiate cele 5 principale)
#   accepted_but_wrong_rate
#   paraphrase_hard_acc
#   coreference_hard_acc
#   shadow_vs_critical_disagreement
#   mixed_vs_shadow_disagreement
#   mixed_vs_critical_disagreement
#   critical_hard_on_accepted_only
#   shadow_hard_on_accepted_only
#   mixed_hard_on_accepted_only
#
# Praguri explicite Stage 1.3 (blocante pentru runs viitoare, nu pentru acest
# diagnostic):
#   CRITICAL_HARD_MIN        = 0.970
#   SHADOW_HARD_MIN_DELTA    = 0.010  (shadow >= critical - 1pp)
#   MIXED_HARD_MIN_DELTA     = 0.010  (mixed >= shadow - 1pp)
#   TRUSTED_CRITICAL_MIN     = 1.000
#   TRUSTED_SHADOW_MIN       = 0.995
# ===========================================================================


V15_3_STAGE_THRESHOLDS = {
    "CRITICAL_HARD_MIN":     0.970,
    "SHADOW_HARD_MIN_DELTA": 0.010,
    "MIXED_HARD_MIN_DELTA":  0.010,
    "TRUSTED_CRITICAL_MIN":  1.000,
    "TRUSTED_SHADOW_MIN":    0.995,
}


V15_3_DIAGNOSTIC_CONFIG = {
    "seed":          20260901,
    "n_paraphrase":  500,
    "n_coreference": 500,
    "hard_types": [("paraphrase", 0.5), ("coreference_distant", 0.5)],
}


# Five reasons flagged explicitly (ordered by operational priority)
V15_3_KEY_REJECT_REASONS = [
    "REFERENT_AMBIGUOUS",
    "TEMPLATE_UNKNOWN",
    "MULTIPLE_ATTR_TRIGGERS",
    "VALUE_MISSING_OR_UNCLEAR",
    "ATTR_VALUE_MISMATCH",
]


def _v15_3_compute_target_idx(ep):
    """Same target logic as A1 audit, extracted."""
    if ep.target_is_unknown:
        return None
    attr_type = V15_ATTR_TYPES[ep.query_attr_label]
    vocab = V15_ATTR_VALUES[attr_type]
    t_tok = int(ep.target_answer_token)
    for k, v_str in enumerate(vocab):
        if V15_ANSWER_TOKENS[attr_type].get(v_str) == t_tok:
            return k
    return None


def _v15_3_classify_target(ep, pkt_q):
    """Return 'FOUND', 'NONE_OBJECT', or 'NONE_ATTRIBUTE' for target spec."""
    if not ep.target_is_unknown:
        return "FOUND"
    # Determine NONE_OBJECT vs NONE_ATTRIBUTE
    q_ent_str = _top_entity(pkt_q) if pkt_q.entity_candidates else ""
    fact_eids = set()
    for f in ep.facts:
        pf = v15_2_parse_fact(f)
        if pf.entity_candidates:
            fact_eids.add(_top_entity(pf))
    if q_ent_str not in fact_eids:
        return "NONE_OBJECT"
    return "NONE_ATTRIBUTE"


def _v15_3_run_one_trial_all_modes(bank, base_model, v15_1_memory, ep,
                                      entity_emb_fn, class_emb_fn, value_emb_fn):
    """Run all 3 modes on the same trial. Returns a dict with per-mode results
    plus verifier reason info.
    """
    shadow = v15_1_memory.shadow
    
    # Write facts (same for all modes)
    bank.reset()
    for step_idx, fact_text in enumerate(ep.facts):
        pkt_f = v15_2_parse_fact(fact_text)
        v15_2_write_fact(bank, pkt_f, entity_emb_fn, class_emb_fn,
                          value_emb_fn, step=step_idx)
    
    # Parse + verify query (single pass, shared across modes)
    pkt_q = v15_2_parse_query(ep.query)
    vr_q  = V15_2_VERIFIER.verify(pkt_q)
    
    target_spec = _v15_3_classify_target(ep, pkt_q)
    target_idx  = _v15_3_compute_target_idx(ep)
    
    result = {
        "target_spec": target_spec,
        "target_idx":  target_idx,
        "verifier_status": vr_q.status.value,
        "verifier_reasons": [r.value for r in vr_q.reasons],
        "accepted": vr_q.status == VerificationStatus.ACCEPT,
    }
    
    if vr_q.status != VerificationStatus.ACCEPT:
        # All modes return the same rejected status; no predictions to compare
        rejected_status = (READ_STATUS_PARSER_FAIL
                            if vr_q.status == VerificationStatus.PARSER_FAILURE
                            else READ_STATUS_PARSE_UNCERTAIN)
        for mode in ("critical_only", "shadow_only", "mixed"):
            result[mode] = {"status": rejected_status, "pred": None, "correct": False}
        return result
    
    # ACCEPT path - run all three modes
    entity_id = _top_entity(pkt_q)
    attr_type = _top_attribute(pkt_q)
    
    def score(status, pred):
        if target_spec == "NONE_OBJECT":
            return status == READ_STATUS_NONE_OBJECT
        if target_spec == "NONE_ATTRIBUTE":
            return status == READ_STATUS_NONE_ATTRIBUTE
        return status == READ_STATUS_FOUND and pred == target_idx
    
    # --- critical_only ---
    status_c, pred_c = bank.read_attribute(entity_id, attr_type)
    result["critical_only"] = {
        "status": status_c, "pred": pred_c, "correct": score(status_c, pred_c),
    }
    
    # --- shadow_only ---
    with torch.no_grad():
        q_ids = torch.tensor(ENC.encode(ep.query), dtype=torch.long, device=DEVICE)
        q_pooled = base_model.shared_token_emb(q_ids).mean(dim=0)
        attr_logits = shadow.attr_router(q_pooled.unsqueeze(0))
        attr_pred_idx = int(attr_logits.argmax(dim=-1).item())
        q_entity_emb = entity_emb_fn(entity_id)
        slot_feats = _build_slot_features(bank, q_entity_emb, None, current_step=1000)
        resolver_logits = shadow.object_resolver(q_entity_emb, slot_feats)
        obj_pred = int(resolver_logits.argmax(dim=-1).item())
        K = slot_feats.shape[0]
    if obj_pred == K:
        status_s, pred_s = READ_STATUS_NONE_OBJECT, None
    elif attr_pred_idx == 4:
        status_s, pred_s = READ_STATUS_NONE_ATTRIBUTE, None
    else:
        at = V15_ATTR_TYPES[attr_pred_idx]
        slot_list = bank.occupied_slots()
        rec = bank.get_record(slot_list[obj_pred])
        a = rec.attr_slots.get(at)
        if a is None or not a.present or a.value_emb is None:
            status_s, pred_s = READ_STATUS_NONE_ATTRIBUTE, None
        else:
            with torch.no_grad():
                vl = shadow.value_heads(at, a.value_emb.unsqueeze(0))
            status_s = READ_STATUS_FOUND
            pred_s = int(vl.argmax(dim=-1).item())
    result["shadow_only"] = {
        "status": status_s, "pred": pred_s, "correct": score(status_s, pred_s),
    }
    
    # --- mixed ---
    slot = bank.find_by_entity_id(entity_id)
    if slot is None:
        status_m, pred_m = READ_STATUS_NONE_OBJECT, None
    else:
        rec = bank.get_record(slot)
        a = rec.attr_slots.get(attr_type)
        if a is None or not a.present or a.value_emb is None:
            status_m, pred_m = READ_STATUS_NONE_ATTRIBUTE, None
        else:
            with torch.no_grad():
                vl = shadow.value_heads(attr_type, a.value_emb.unsqueeze(0))
            status_m = READ_STATUS_FOUND
            pred_m = int(vl.argmax(dim=-1).item())
    result["mixed"] = {
        "status": status_m, "pred": pred_m, "correct": score(status_m, pred_m),
    }
    
    return result


def v15_3_run_hard_diagnostic(bank, base_model, v15_1_memory, cfg=None):
    """Run HARD diagnostic. Pure measurement, no modifications."""
    if cfg is None:
        cfg = V15_3_DIAGNOSTIC_CONFIG
    
    print()
    print(SEP)
    print("[v15.3 HARD DIAGNOSTIC] (Pas 1 - measurement only, no fixes)")
    print(f"  seed:          {cfg['seed']}")
    print(f"  n_paraphrase:  {cfg['n_paraphrase']}")
    print(f"  n_coreference: {cfg['n_coreference']}")
    print(SEP)
    
    ent_fn = _make_entity_emb_fn(base_model)
    cls_fn = _make_class_emb_fn(v15_1_memory)
    val_fn = _make_value_emb_fn(base_model)
    
    rng = random.Random(cfg["seed"])
    trials = []
    
    # Run paraphrase trials
    for i in range(cfg["n_paraphrase"]):
        ep = v15_generate_episode("paraphrase", rng, use_heldout=True)
        r = _v15_3_run_one_trial_all_modes(bank, base_model, v15_1_memory, ep,
                                              ent_fn, cls_fn, val_fn)
        r["hard_type"] = "paraphrase"
        trials.append(r)
    
    # Run coreference_distant trials
    for i in range(cfg["n_coreference"]):
        ep = v15_generate_episode("coreference_distant", rng, use_heldout=True)
        r = _v15_3_run_one_trial_all_modes(bank, base_model, v15_1_memory, ep,
                                              ent_fn, cls_fn, val_fn)
        r["hard_type"] = "coreference_distant"
        trials.append(r)
    
    n = len(trials)
    n_para = cfg["n_paraphrase"]
    n_cor  = cfg["n_coreference"]
    
    # ---- Aggregate metrics ----
    n_rejected = sum(1 for t in trials if not t["accepted"])
    n_accepted = n - n_rejected
    
    # Reject rate by reason (each rejected trial may contribute to multiple reasons)
    reject_by_reason = {}
    for t in trials:
        if not t["accepted"]:
            for r in t["verifier_reasons"]:
                reject_by_reason[r] = reject_by_reason.get(r, 0) + 1
    
    # Accepted-but-wrong: critical disagreed with target despite ACCEPT
    acc_and_wrong = sum(
        1 for t in trials
        if t["accepted"] and not t["critical_only"]["correct"]
    )
    
    # Per-hard-type accuracy (on full set: reject counts as incorrect for FOUND targets,
    # but correct if target is NONE_* and rejection is not the expected NONE signal)
    def overall_acc(ts, mode):
        return sum(1 for t in ts if t[mode]["correct"]) / max(1, len(ts))
    
    paraphrase_trials = [t for t in trials if t["hard_type"] == "paraphrase"]
    coreference_trials = [t for t in trials if t["hard_type"] == "coreference_distant"]
    
    paraphrase_acc = {
        "critical_only": overall_acc(paraphrase_trials, "critical_only"),
        "shadow_only":   overall_acc(paraphrase_trials, "shadow_only"),
        "mixed":         overall_acc(paraphrase_trials, "mixed"),
    }
    coreference_acc = {
        "critical_only": overall_acc(coreference_trials, "critical_only"),
        "shadow_only":   overall_acc(coreference_trials, "shadow_only"),
        "mixed":         overall_acc(coreference_trials, "mixed"),
    }
    
    # Disagreement rates - compare predictions pairwise across all trials
    def disagree_rate(ts, modeA, modeB):
        diffs = 0
        for t in ts:
            s_a = t[modeA]["status"]; p_a = t[modeA]["pred"]
            s_b = t[modeB]["status"]; p_b = t[modeB]["pred"]
            if s_a != s_b or p_a != p_b:
                diffs += 1
        return diffs / max(1, len(ts))
    
    shadow_vs_critical = disagree_rate(trials, "shadow_only",  "critical_only")
    mixed_vs_shadow    = disagree_rate(trials, "mixed",         "shadow_only")
    mixed_vs_critical  = disagree_rate(trials, "mixed",         "critical_only")
    
    # Accepted-only metrics (what does each mode do ON ACCEPT)
    accepted_trials = [t for t in trials if t["accepted"]]
    def accepted_acc(mode):
        if not accepted_trials:
            return 0.0
        return sum(1 for t in accepted_trials if t[mode]["correct"]) / len(accepted_trials)
    
    critical_hard_acc_accepted = accepted_acc("critical_only")
    shadow_hard_acc_accepted   = accepted_acc("shadow_only")
    mixed_hard_acc_accepted    = accepted_acc("mixed")
    
    # Overall HARD accuracy (for reference)
    critical_hard_overall = overall_acc(trials, "critical_only")
    shadow_hard_overall   = overall_acc(trials, "shadow_only")
    mixed_hard_overall    = overall_acc(trials, "mixed")
    
    # ---- Print report ----
    print()
    print("=== Reject profile ===")
    print(f"  total trials:           {n}")
    print(f"  n_accepted:             {n_accepted}  ({n_accepted/n:.1%})")
    print(f"  n_rejected:             {n_rejected}  ({n_rejected/n:.1%})")
    print(f"  reject_rate_hard_total: {n_rejected/n:.3f}")
    
    print()
    print("  Reject by reason (counts over rejected trials, multi-flag possible):")
    for reason in V15_3_KEY_REJECT_REASONS:
        cnt = reject_by_reason.get(reason, 0)
        pct = cnt / max(1, n_rejected)
        print(f"    {reason:30s} {cnt:4d}  ({pct:.1%} of rejected)")
    other_reasons = {k: v for k, v in reject_by_reason.items()
                      if k not in V15_3_KEY_REJECT_REASONS}
    if other_reasons:
        print("  Other flags:")
        for k, v in other_reasons.items():
            pct = v / max(1, n_rejected)
            print(f"    {k:30s} {v:4d}  ({pct:.1%} of rejected)")
    
    print()
    print("=== Accepted-only behavior ===")
    print(f"  accepted_but_wrong (critical): {acc_and_wrong}/{n_accepted} = "
          f"{acc_and_wrong/max(1,n_accepted):.3f}")
    print(f"  critical_hard_on_accepted_only: {critical_hard_acc_accepted:.3f}")
    print(f"  shadow_hard_on_accepted_only:   {shadow_hard_acc_accepted:.3f}")
    print(f"  mixed_hard_on_accepted_only:    {mixed_hard_acc_accepted:.3f}")
    
    print()
    print("=== Overall HARD accuracy (incl. rejections) ===")
    print(f"  critical_hard_overall: {critical_hard_overall:.3f}")
    print(f"  shadow_hard_overall:   {shadow_hard_overall:.3f}")
    print(f"  mixed_hard_overall:    {mixed_hard_overall:.3f}")
    
    print()
    print("=== Per hard-type accuracy ===")
    print(f"  paraphrase  (n={n_para}):")
    for mode in ("critical_only", "shadow_only", "mixed"):
        print(f"    {mode:15s}: {paraphrase_acc[mode]:.3f}")
    print(f"  coreference_distant (n={n_cor}):")
    for mode in ("critical_only", "shadow_only", "mixed"):
        print(f"    {mode:15s}: {coreference_acc[mode]:.3f}")
    
    print()
    print("=== Pairwise disagreement ===")
    print(f"  shadow_vs_critical_disagreement: {shadow_vs_critical:.3f}")
    print(f"  mixed_vs_shadow_disagreement:    {mixed_vs_shadow:.3f}")
    print(f"  mixed_vs_critical_disagreement:  {mixed_vs_critical:.3f}")
    
    # ---- Stage 1.3 threshold check ----
    th = V15_3_STAGE_THRESHOLDS
    stage13_check = {
        "critical_hard_min_met":
            critical_hard_overall >= th["CRITICAL_HARD_MIN"],
        "shadow_hard_delta_met":
            shadow_hard_overall >= (critical_hard_overall - th["SHADOW_HARD_MIN_DELTA"]),
        "mixed_hard_delta_met":
            mixed_hard_overall >= (shadow_hard_overall - th["MIXED_HARD_MIN_DELTA"]),
    }
    
    print()
    print("=== Stage 1.3 threshold check (informative, not blocking in Pas 1) ===")
    print(f"  CRITICAL_HARD_MIN     = {th['CRITICAL_HARD_MIN']:.3f}  "
          f"met={stage13_check['critical_hard_min_met']}  "
          f"(got {critical_hard_overall:.3f})")
    print(f"  SHADOW_HARD_DELTA     = {th['SHADOW_HARD_MIN_DELTA']:.3f}  "
          f"met={stage13_check['shadow_hard_delta_met']}  "
          f"(gap={critical_hard_overall - shadow_hard_overall:+.3f})")
    print(f"  MIXED_HARD_DELTA      = {th['MIXED_HARD_MIN_DELTA']:.3f}  "
          f"met={stage13_check['mixed_hard_delta_met']}  "
          f"(gap={shadow_hard_overall - mixed_hard_overall:+.3f})")
    
    # ---- Diagnostic interpretation helper ----
    print()
    print("=== Diagnostic hint (mechanical, not prescriptive) ===")
    hints = []
    if n_rejected / n > 0.03:
        hints.append(f"reject_rate={n_rejected/n:.1%} > 3%: parser/verifier coverage is a"
                     f" primary source of loss on HARD.")
    if critical_hard_acc_accepted > 0.995:
        hints.append("critical_hard_on_accepted_only > 99.5%: oracle is clean when it "
                     "commits; loss concentrates in reject path.")
    if shadow_vs_critical < 0.02 and mixed_vs_shadow > 0.03:
        hints.append("shadow tracks critical closely, but mixed drifts from both: "
                     "interface mixed is the distinct failure point.")
    if coreference_acc["critical_only"] + 0.05 < paraphrase_acc["critical_only"]:
        hints.append("coreference accuracy noticeably below paraphrase: referential "
                     "resolution is likely the dominant gap.")
    elif paraphrase_acc["critical_only"] + 0.05 < coreference_acc["critical_only"]:
        hints.append("paraphrase accuracy noticeably below coreference: lexical "
                     "extractor and pattern map likely dominant gap.")
    else:
        hints.append("paraphrase and coreference roughly equal: both contribute, patch "
                     "both in one arhitectural step as planned.")
    if not hints:
        hints.append("no dominant breakdown pattern identified; treat as uniform.")
    for h in hints:
        print(f"  - {h}")
    
    print(SEP)
    
    return {
        "config": dict(cfg),
        "thresholds": th,
        "counts": {
            "n_total":     n,
            "n_accepted":  n_accepted,
            "n_rejected":  n_rejected,
            "n_paraphrase": n_para,
            "n_coreference": n_cor,
        },
        "reject_rate_hard_total": n_rejected / n,
        "reject_by_reason_counts": reject_by_reason,
        "reject_by_reason_rates":  {k: v / max(1, n_rejected)
                                     for k, v in reject_by_reason.items()},
        "accepted_but_wrong_rate": acc_and_wrong / max(1, n_accepted),
        "paraphrase_hard_acc":    paraphrase_acc,
        "coreference_hard_acc":   coreference_acc,
        "shadow_vs_critical_disagreement": shadow_vs_critical,
        "mixed_vs_shadow_disagreement":    mixed_vs_shadow,
        "mixed_vs_critical_disagreement":  mixed_vs_critical,
        "critical_hard_on_accepted_only":  critical_hard_acc_accepted,
        "shadow_hard_on_accepted_only":    shadow_hard_acc_accepted,
        "mixed_hard_on_accepted_only":     mixed_hard_acc_accepted,
        "overall": {
            "critical_hard": critical_hard_overall,
            "shadow_hard":   shadow_hard_overall,
            "mixed_hard":    mixed_hard_overall,
        },
        "stage13_threshold_check": stage13_check,
        "diagnostic_hints": hints,
    }


def v15_3_write_diagnostic_memo(diag, path):
    """Write diagnostic report as markdown."""
    lines = []
    lines.append("# v15.3 HARD Diagnostic - Pas 1")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- seed:          {diag['config']['seed']}")
    lines.append(f"- n_paraphrase:  {diag['config']['n_paraphrase']}")
    lines.append(f"- n_coreference: {diag['config']['n_coreference']}")
    lines.append("")
    lines.append("## Reject profile")
    c = diag['counts']
    lines.append(f"- total trials:           {c['n_total']}")
    lines.append(f"- n_accepted:             {c['n_accepted']} ({c['n_accepted']/c['n_total']:.1%})")
    lines.append(f"- n_rejected:             {c['n_rejected']} ({c['n_rejected']/c['n_total']:.1%})")
    lines.append(f"- reject_rate_hard_total: {diag['reject_rate_hard_total']:.3f}")
    lines.append("")
    lines.append("### Reject by reason")
    lines.append("| Reason | Count | % of rejected |")
    lines.append("|---|---:|---:|")
    for reason in V15_3_KEY_REJECT_REASONS:
        cnt = diag['reject_by_reason_counts'].get(reason, 0)
        rate = diag['reject_by_reason_rates'].get(reason, 0.0)
        lines.append(f"| {reason} | {cnt} | {rate:.1%} |")
    other = {k: v for k, v in diag['reject_by_reason_counts'].items()
              if k not in V15_3_KEY_REJECT_REASONS}
    if other:
        lines.append("")
        lines.append("### Other flags (not in key 5)")
        for k, v in other.items():
            rate = diag['reject_by_reason_rates'].get(k, 0.0)
            lines.append(f"- {k}: {v} ({rate:.1%})")
    lines.append("")
    lines.append("## Accepted-only behavior")
    lines.append(f"- accepted_but_wrong_rate:        {diag['accepted_but_wrong_rate']:.3f}")
    lines.append(f"- critical_hard_on_accepted_only: {diag['critical_hard_on_accepted_only']:.3f}")
    lines.append(f"- shadow_hard_on_accepted_only:   {diag['shadow_hard_on_accepted_only']:.3f}")
    lines.append(f"- mixed_hard_on_accepted_only:    {diag['mixed_hard_on_accepted_only']:.3f}")
    lines.append("")
    lines.append("## Overall HARD accuracy (all trials including rejects)")
    lines.append(f"- critical_hard_overall: {diag['overall']['critical_hard']:.3f}")
    lines.append(f"- shadow_hard_overall:   {diag['overall']['shadow_hard']:.3f}")
    lines.append(f"- mixed_hard_overall:    {diag['overall']['mixed_hard']:.3f}")
    lines.append("")
    lines.append("## Per hard-type accuracy")
    for ttype, acc in [("paraphrase", diag['paraphrase_hard_acc']),
                        ("coreference_distant", diag['coreference_hard_acc'])]:
        lines.append(f"### {ttype}")
        for mode in ("critical_only", "shadow_only", "mixed"):
            lines.append(f"- {mode}: {acc[mode]:.3f}")
    lines.append("")
    lines.append("## Pairwise disagreement")
    lines.append(f"- shadow_vs_critical_disagreement: {diag['shadow_vs_critical_disagreement']:.3f}")
    lines.append(f"- mixed_vs_shadow_disagreement:    {diag['mixed_vs_shadow_disagreement']:.3f}")
    lines.append(f"- mixed_vs_critical_disagreement:  {diag['mixed_vs_critical_disagreement']:.3f}")
    lines.append("")
    lines.append("## Stage 1.3 threshold check (informative)")
    th = diag['thresholds']
    chk = diag['stage13_threshold_check']
    lines.append(f"- CRITICAL_HARD_MIN = {th['CRITICAL_HARD_MIN']:.3f}  met={chk['critical_hard_min_met']}")
    lines.append(f"- SHADOW_HARD_DELTA = {th['SHADOW_HARD_MIN_DELTA']:.3f}  met={chk['shadow_hard_delta_met']}")
    lines.append(f"- MIXED_HARD_DELTA  = {th['MIXED_HARD_MIN_DELTA']:.3f}  met={chk['mixed_hard_delta_met']}")
    lines.append("")
    lines.append("## Diagnostic hints")
    for h in diag['diagnostic_hints']:
        lines.append(f"- {h}")
    lines.append("")
    lines.append("## Raw")
    lines.append("```")
    lines.append(json.dumps(diag, indent=2, default=str))
    lines.append("```")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


print("[v15.3] Section H defined: HARD diagnostic (measurement only)")
print("        - reject rate + breakdown by reason")
print("        - accepted-only accuracy per mode")
print("        - paraphrase vs coreference split")
print("        - pairwise disagreement")
print("        - Stage 1.3 thresholds informative")

# ======================== A3. V15.4 LEXICAL EXTRACTOR + VERIFIER ===========
#
# Pas 2 Stage 1.3: extindere parser lexical pentru paraphrase intent
# extraction + verifier cu ATTR_WEAK_SIGNAL.
#
# Principii:
#   - extractorul NU decide, doar scoate candidați cu scor
#   - fiecare candidat atributiv are (confidence, trigger_family, trigger_strength,
#     evidence_span)
#   - verifier-ul respinge unicii-slabi (ATTR_WEAK_SIGNAL)
#   - query patterns au prioritate peste keyword matching plat
#   - bank, write, read, memorie, shadow - NEATINSE
# ===========================================================================


# ------------- A3.1 Trigger families + query patterns -------------

# Trigger words grouped by semantic family per attribute type.
# Each family has `strong` (primary keywords) and `semantic` (synonyms).
V15_4_ATTR_TRIGGER_FAMILIES = {
    "color": {
        "strong":   ["color", "colour", "colored", "coloured"],
        "semantic": ["shade", "hue", "tint", "tone"],
    },
    "size": {
        "strong":   ["size", "sized"],
        "semantic": ["big", "small", "large", "huge", "tiny", "little"],
    },
    "location": {
        "strong":   ["location", "located", "place", "position"],
        "semantic": ["where", "found", "sits", "sitting", "lives", "lived", "remained"],
    },
    "state": {
        "strong":   ["state", "condition", "mood", "feeling", "status"],
        "semantic": ["feel", "feels", "felt", "feeling", "seems", "appears", "looked"],
    },
}

# Query-level patterns that STRONGLY imply attribute intent.
# Evaluated before keyword-level scoring.
# Tuple: (regex, attr_type, confidence, trigger_family_label)
V15_4_QUERY_PATTERNS = [
    # color
    (r"\bwhat\s+(color|colour|shade|hue|tint|tone)\b",            "color",    0.95, "query_pattern"),
    (r"\btell\s+me\s+the\s+(color|colour|shade|hue)\b",           "color",    0.95, "query_pattern"),
    (r"\bhas\s+what\s+(color|colour|shade)\b",                    "color",    0.95, "query_pattern"),
    # size
    (r"\bhow\s+(big|small|large|huge|tiny|little)\b",             "size",     0.95, "query_pattern"),
    (r"\bwhat\s+size\b",                                            "size",     0.95, "query_pattern"),
    (r"\bdescribe\s+the\s+size\b",                                  "size",     0.95, "query_pattern"),
    # location
    (r"\bwhere\s+is\b",                                             "location", 0.95, "query_pattern"),
    (r"\bwhere\s+can\s+the\b",                                      "location", 0.95, "query_pattern"),
    (r"\bin\s+what\s+place\b",                                      "location", 0.95, "query_pattern"),
    (r"\bin\s+the\b.*\bfound\b",                                    "location", 0.80, "query_pattern"),
    # state
    (r"\bwhat\s+state\b",                                           "state",    0.95, "query_pattern"),
    (r"\bhow\s+does\s+.*\s+feel\b",                                 "state",    0.95, "query_pattern"),
    (r"\bin\s+what\s+condition\b",                                  "state",    0.95, "query_pattern"),
    (r"\bdescribe\s+the\s+(mood|feeling|state|condition)\b",        "state",    0.95, "query_pattern"),
]

# Confidence scoring scheme:
#   query_pattern    : 0.95  strength 1.0
#   strong keyword   : 0.85  strength 1.0
#   value-based      : 0.80  strength 0.9   (value uniquely maps to attr_type)
#   semantic keyword : 0.55  strength 0.5
#
# ATTR_WEAK_SIGNAL fires when a UNIQUE attribute candidate survives but
# its max trigger_strength is below V15_4_WEAK_SIGNAL_STRENGTH_MIN, i.e.
# evidence rests only on semantic synonyms without corroboration.
V15_4_CONF_QUERY_PATTERN  = 0.95
V15_4_CONF_STRONG_KW      = 0.85
V15_4_CONF_VALUE_BASED    = 0.80
V15_4_CONF_SEMANTIC_KW    = 0.55

V15_4_STR_QUERY_PATTERN  = 1.0
V15_4_STR_STRONG_KW      = 1.0
V15_4_STR_VALUE_BASED    = 0.9
V15_4_STR_SEMANTIC_KW    = 0.5

V15_4_WEAK_SIGNAL_STRENGTH_MIN = 0.7   # strength below this => weak
V15_4_CONFIDENCE_CLOSE_MARGIN  = 0.15  # same as v15.2

V15_4_MIN_CERTAINTY = 0.50


# Multi-token entity prefix map. The curriculum generator `_gen_paraphrase`
# decodes only the first BPE token of multi-token entities, producing
# truncated surface forms ("gr" for "griffin", "ph" for "phoenix", etc.).
# v15.4 parser remaps these prefixes to full entity ids BEFORE they reach
# the bank. Substrate remains frozen: canonicalize_entity is unchanged; we
# only rewrite what the parser emits.
V15_4_PREFIX_ALIAS_MAP = {
    "basil":  "basilisk",
    "chim":   "chimera",
    "gr":     "griffin",
    "mer":    "mermaid",
    "min":    "minotaur",
    "pe":     "pegasus",
    "ph":     "phoenix",
}


def _v15_4_find_entity_candidates_with_prefix_aliases(source: str
                                                        ) -> List[Tuple[str, float, Tuple[int, int]]]:
    """Find entity candidates, scanning BOTH full entity names and their
    truncated prefix aliases. Prefix hits are remapped to full entity ids.
    """
    all_entities = [e for e, _ in V15_TRAIN_ENTITIES] + [e for e, _ in V15_HELDOUT_ENTITIES]
    # Standard full-name scan
    ent_cands = _v15_2_find_entity_candidates(source, all_entities)
    # Additional: scan prefix aliases as a separate pool
    prefix_keys = list(V15_4_PREFIX_ALIAS_MAP.keys())
    prefix_hits = _v15_2_find_entity_candidates(source, prefix_keys)
    # Remap prefix hits to full entity ids
    for (eid, conf, span) in prefix_hits:
        full = V15_4_PREFIX_ALIAS_MAP.get(eid, eid)
        ent_cands.append((full, conf, span))
    return ent_cands


def _v15_4_remap_entity_candidates(ent_cands):
    """Apply V15_4_PREFIX_ALIAS_MAP to already-found entity candidates."""
    remapped = []
    for (eid, conf, span) in ent_cands:
        if eid in V15_4_PREFIX_ALIAS_MAP:
            remapped.append((V15_4_PREFIX_ALIAS_MAP[eid], conf, span))
        else:
            remapped.append((eid, conf, span))
    return remapped


# ------------- A3.2 V15.4 AmbiguityFlag (extends v15.2 + ATTR_WEAK_SIGNAL +
#                                           v15.4.1 conflict flags) -------------

class V15_4_AmbiguityFlag(Enum):
    MULTIPLE_ATTR_TRIGGERS    = "MULTIPLE_ATTR_TRIGGERS"
    REFERENT_AMBIGUOUS        = "REFERENT_AMBIGUOUS"
    ATTR_VALUE_MISMATCH       = "ATTR_VALUE_MISMATCH"
    TEMPLATE_UNKNOWN          = "TEMPLATE_UNKNOWN"
    MULTI_ENTITY_SAME_TYPE    = "MULTI_ENTITY_SAME_TYPE"
    OP_TYPE_AMBIGUOUS         = "OP_TYPE_AMBIGUOUS"
    VALUE_MISSING_OR_UNCLEAR  = "VALUE_MISSING_OR_UNCLEAR"
    ATTR_WEAK_SIGNAL          = "ATTR_WEAK_SIGNAL"
    # v15.4.1: conflict flags with HIGHER priority than attr dominance
    ATTR_CONFLICT_STRONG      = "ATTR_CONFLICT_STRONG"
    MULTI_FAMILY_COMPETITION  = "MULTI_FAMILY_COMPETITION"


# ------------- A3.3 Helpers -------------

def _v15_4_word_match(keyword: str, text_lower: str) -> Optional[Tuple[int, int]]:
    """Word-boundary match. Returns span or None."""
    m = _re.search(rf"\b{_re.escape(keyword.lower())}\b", text_lower)
    if m:
        return (m.start(), m.end())
    return None


def _v15_4_find_entity_candidates_v2_compat(source, all_entities):
    """Reuse v15.2 implementation."""
    return _v15_2_find_entity_candidates(source, all_entities)


def _v15_4_find_value_candidates_v2_compat(source):
    """Reuse v15.2 value extractor."""
    return _v15_2_find_value_candidates(source)


def _v15_4_score_attribute_evidence(source: str, attr_type: str,
                                       is_query: bool) -> List[Dict]:
    """Collect all evidence for attr_type in source. Each entry:
       {confidence, strength, family_label, evidence_span}
    """
    text_lower = source.lower()
    evidence = []
    
    # 1. Query patterns (only if query)
    if is_query:
        for pattern, p_attr, p_conf, p_fam in V15_4_QUERY_PATTERNS:
            if p_attr != attr_type:
                continue
            m = _re.search(pattern, text_lower)
            if m:
                evidence.append({
                    "confidence":     p_conf,
                    "strength":       V15_4_STR_QUERY_PATTERN,
                    "family_label":   "query_pattern",
                    "evidence_span":  (m.start(), m.end()),
                    "evidence_text":  text_lower[m.start():m.end()],
                })
    
    family = V15_4_ATTR_TRIGGER_FAMILIES[attr_type]
    
    # 2. Strong keywords
    for kw in family["strong"]:
        span = _v15_4_word_match(kw, text_lower)
        if span is not None:
            evidence.append({
                "confidence":    V15_4_CONF_STRONG_KW,
                "strength":      V15_4_STR_STRONG_KW,
                "family_label":  "strong",
                "evidence_span": span,
                "evidence_text": kw,
            })
    
    # 3. Semantic keywords
    for kw in family["semantic"]:
        span = _v15_4_word_match(kw, text_lower)
        if span is not None:
            evidence.append({
                "confidence":    V15_4_CONF_SEMANTIC_KW,
                "strength":      V15_4_STR_SEMANTIC_KW,
                "family_label":  "semantic",
                "evidence_span": span,
                "evidence_text": kw,
            })
    
    # 4. Value-based (only applies to facts; for queries having a value is
    #    ambiguity signal, not attr evidence)
    if not is_query:
        for value in V15_ATTR_VALUES[attr_type]:
            span = _v15_4_word_match(value, text_lower)
            if span is not None:
                evidence.append({
                    "confidence":    V15_4_CONF_VALUE_BASED,
                    "strength":      V15_4_STR_VALUE_BASED,
                    "family_label":  "value_based",
                    "evidence_span": span,
                    "evidence_text": value,
                })
    
    return evidence


def _v15_4_aggregate_evidence(evidence_list: List[Dict]) -> Dict:
    """Given list of evidence dicts, produce single aggregated score."""
    if not evidence_list:
        return {"confidence": 0.0, "max_strength": 0.0, "has_strong": False,
                 "has_semantic_only": False, "top_evidence": None}
    max_conf = max(e["confidence"] for e in evidence_list)
    max_str  = max(e["strength"]   for e in evidence_list)
    has_strong = any(e["family_label"] in ("strong", "query_pattern", "value_based")
                      for e in evidence_list)
    has_semantic_only = all(e["family_label"] == "semantic" for e in evidence_list)
    top = max(evidence_list, key=lambda e: (e["confidence"], e["strength"]))
    return {
        "confidence":         max_conf,
        "max_strength":       max_str,
        "has_strong":         has_strong,
        "has_semantic_only":  has_semantic_only,
        "top_evidence":       top,
    }


# ------------- A3.4 V15.4 Extractors -------------

def v15_4_parse_fact(text: str) -> ParsePacket:
    """V15.4.1 fact parser: v15.2 attr/value logic + prefix-aware entity scan +
    ATTR_CONFLICT_STRONG detection.
    
    Changes vs v15.2:
      - Entity scan also recognizes multi-BPE-token prefixes ("gr" -> "griffin").
      - ATTR_CONFLICT_STRONG raised when WRITE has 2+ values from the SAME
        attribute family (e.g. "The X is red and blue"). This is a STRUCTURAL
        conflict, raised BEFORE any dominance choice. Ensures S4-type fact
        conflicts are never silently committed.
    """
    text = (text or "").strip()
    source = text
    
    # v15.2 op_type, value, attr logic via parse_fact then replace entity list
    pkt = v15_2_parse_fact(text)
    
    # Replace entity candidates with prefix-aware scan
    new_ent_cands = _v15_4_find_entity_candidates_with_prefix_aliases(source)
    pkt.entity_candidates = new_ent_cands
    
    # Recompute certainty with new entity candidates
    if not new_ent_cands:
        pkt.certainty = 0.0
    else:
        e_max = max(c for _, c, _ in new_ent_cands)
        if pkt.op_type == OpType.WRITE:
            v_max = max([c for _, _, c, _ in pkt.value_candidates], default=0.0)
            pkt.certainty = 0.5 * e_max + 0.5 * v_max
        else:
            pkt.certainty = 0.5 * e_max + 0.5 * pkt.op_type_confidence
    
    # Recompute MULTI_ENTITY flag on DISTINCT entity_ids
    pkt.ambiguity_flags.discard(AmbiguityFlag.MULTI_ENTITY_SAME_TYPE)
    if len({eid for eid, _, _ in new_ent_cands}) > 1:
        pkt.ambiguity_flags.add(AmbiguityFlag.MULTI_ENTITY_SAME_TYPE)
    
    # --- V15.4.1 PRECEDENCE: ATTR_CONFLICT_STRONG ---
    # Same-family multi-value fact = structural conflict. Raise BEFORE any
    # dominance logic, regardless of how confident the extractor is.
    if pkt.op_type == OpType.WRITE and len(pkt.value_candidates) >= 2:
        attr_families_in_values = {attr for (attr, _, _, _) in pkt.value_candidates}
        # At least one family with 2+ distinct values
        for fam in attr_families_in_values:
            vals_in_fam = [v for (a, v, _, _) in pkt.value_candidates if a == fam]
            if len(set(vals_in_fam)) >= 2:
                pkt.ambiguity_flags.add(V15_4_AmbiguityFlag.ATTR_CONFLICT_STRONG)
                break
    
    pkt.parser_evidence["extractor_version"] = "v15.4.1_fact_prefix_aware_conflict_precedence"
    return pkt


def v15_4_parse_query(text: str) -> ParsePacket:
    """V15.4 query parser.
    
    Priority order:
      1. Query-level attribute pattern (structural cue)
      2. Strong keyword match
      3. Semantic keyword match
    
    Values appearing IN the query are not attr evidence; they are
    ambiguity signal (ATTR_VALUE_MISMATCH candidate).
    """
    text = (text or "").strip()
    source = text
    
    op_type, op_conf = _v15_2_detect_op_type_query(source)
    ent_cands = _v15_4_find_entity_candidates_with_prefix_aliases(source)
    val_cands = _v15_4_find_value_candidates_v2_compat(source)
    
    # Score every attribute (collect all evidence, do not dominate yet)
    attr_cands: List[Tuple[str, float, str]] = []
    attr_evidence_detail: Dict[str, Dict] = {}
    
    for attr_type in ("color", "size", "location", "state"):
        ev = _v15_4_score_attribute_evidence(source, attr_type, is_query=True)
        agg = _v15_4_aggregate_evidence(ev)
        if agg["confidence"] > 0.0:
            top = agg["top_evidence"]
            attr_cands.append((attr_type, agg["confidence"],
                                 top["evidence_text"]))
            attr_evidence_detail[attr_type] = agg
    
    # --- V15.4.1 PRECEDENCE: conflict detection runs BEFORE family dominance ---
    # If multiple attribute families have strong structural evidence (query pattern
    # or strong keyword), that is a MULTI_FAMILY_COMPETITION regardless of which
    # scores marginally higher.
    strong_attrs = []
    for attr_type, agg in attr_evidence_detail.items():
        if agg.get("has_strong", False):
            strong_attrs.append(attr_type)
    multi_family_competition = len(strong_attrs) >= 2
    
    # Reference candidates (pronouns): inline minimal logic from v15.2
    ref_cands: List[Tuple[str, int]] = []
    low = source.lower()
    has_pronoun = bool(_re.search(r"\b(it|its|this|that)\b", low))
    if has_pronoun and ent_cands:
        seen = []
        for (eid, _, _) in ent_cands:
            if eid not in seen:
                seen.append(eid)
        for i, eid in enumerate(seen):
            ref_cands.append((eid, i))
    
    flags: Set = set()
    
    # v15.4.1: raise MULTI_FAMILY_COMPETITION first (higher priority than dominance)
    if multi_family_competition:
        flags.add(V15_4_AmbiguityFlag.MULTI_FAMILY_COMPETITION)
    
    # Multi-attr triggers with close confidence (retained)
    real_attrs = [a for a in attr_cands]
    if len(real_attrs) > 1:
        top_conf = max(c for _, c, _ in real_attrs)
        close_attrs = [a for a in real_attrs if a[1] >= top_conf - V15_4_CONFIDENCE_CLOSE_MARGIN]
        if len(close_attrs) > 1:
            flags.add(V15_4_AmbiguityFlag.MULTIPLE_ATTR_TRIGGERS)
    
    # Value-in-query: potential ATTR_VALUE_MISMATCH
    if val_cands:
        if attr_cands:
            top_attr = max(attr_cands, key=lambda a: a[1])[0]
            val_attrs = {a for (a, _, _, _) in val_cands}
            if top_attr not in val_attrs:
                flags.add(V15_4_AmbiguityFlag.ATTR_VALUE_MISMATCH)
    
    # Referent ambiguity (only if pronouns AND distinct entity_ids)
    if ref_cands and len({eid for eid, _, _ in ent_cands}) > 1:
        flags.add(V15_4_AmbiguityFlag.REFERENT_AMBIGUOUS)
    
    # Multi-entity flag (only if DISTINCT entity_ids)
    if len({eid for eid, _, _ in ent_cands}) > 1:
        flags.add(V15_4_AmbiguityFlag.MULTI_ENTITY_SAME_TYPE)
    
    # Certainty aggregate
    if not ent_cands:
        cert = 0.0
    elif not attr_cands:
        cert = 0.0
    else:
        e_max = max(c for _, c, _ in ent_cands)
        a_max = max(c for _, c, _ in attr_cands)
        cert = 0.5 * e_max + 0.5 * a_max
    
    return ParsePacket(
        source_text=source,
        source_kind="query",
        op_type=op_type,
        op_type_confidence=op_conf,
        entity_candidates=ent_cands,
        attribute_candidates=attr_cands,
        value_candidates=val_cands,
        reference_candidates=ref_cands,
        certainty=cert,
        ambiguity_flags=flags,
        parser_evidence={
            "extractor_version": "v15.4",
            "raw": source,
            "attr_evidence_detail": {k: {"confidence": agg["confidence"],
                                           "max_strength": agg["max_strength"],
                                           "has_strong": agg["has_strong"],
                                           "has_semantic_only": agg["has_semantic_only"]}
                                        for k, agg in attr_evidence_detail.items()},
        },
    )


# ------------- A3.5 V15.4 Verifier (adds ATTR_WEAK_SIGNAL) -------------

class V15_4_Verifier:
    """Rule-based verifier with ATTR_WEAK_SIGNAL check.
    
    Same three checks as v15.2 (structural, referential, executability),
    plus one new check: UNIQUE attr candidate with evidence resting only
    on semantic synonyms (max strength < WEAK_SIGNAL_STRENGTH_MIN).
    """
    
    def __init__(self):
        self.confidence_close_margin = V15_4_CONFIDENCE_CLOSE_MARGIN
        self.min_certainty           = V15_4_MIN_CERTAINTY
        self.weak_signal_strength_min = V15_4_WEAK_SIGNAL_STRENGTH_MIN
    
    def verify(self, packet: ParsePacket) -> VerificationResult:
        reasons: Set = set()
        
        # Inherited flags from extractor go through to verifier
        for f in packet.ambiguity_flags:
            reasons.add(f)
        
        # ---- Structural checks ----
        # No entity -> TEMPLATE_UNKNOWN
        if not packet.entity_candidates:
            reasons.add(V15_4_AmbiguityFlag.TEMPLATE_UNKNOWN)
        
        # Low certainty -> structural
        if packet.certainty < self.min_certainty:
            reasons.add(V15_4_AmbiguityFlag.TEMPLATE_UNKNOWN)
        
        # No attribute on WRITE/READ -> TEMPLATE_UNKNOWN
        if packet.op_type in (OpType.WRITE, OpType.READ):
            real_attrs = [a for a in packet.attribute_candidates if a[0] != "__class__"]
            if not real_attrs:
                reasons.add(V15_4_AmbiguityFlag.TEMPLATE_UNKNOWN)
        
        # ---- Multi-entity close-confidence -> REFERENT_AMBIGUOUS or MULTI_ENTITY ----
        # (already handled by extractor flags; no duplicate work)
        
        # ---- Value conflict check (from v15.2 logic, preserved) ----
        # Multiple DIFFERENT values for SAME attr_type = VALUE_MISSING_OR_UNCLEAR
        # ("The X is red and blue" = two colors for one slot).
        if packet.source_kind == "fact" and packet.op_type == OpType.WRITE:
            value_counts_by_attr = {}
            for (a, _, _, _) in packet.value_candidates:
                value_counts_by_attr[a] = value_counts_by_attr.get(a, 0) + 1
            if any(c > 1 for c in value_counts_by_attr.values()):
                reasons.add(V15_4_AmbiguityFlag.VALUE_MISSING_OR_UNCLEAR)
        
        # ---- NEW: ATTR_WEAK_SIGNAL (queries only; facts use v15.2 logic) ----
        # Trigger if: source is query AND exactly ONE real attr candidate survives
        # AND that candidate's max evidence strength is below threshold
        # (supported only by semantic synonyms, no strong/pattern/value match).
        if (packet.source_kind == "query"
                and "attr_evidence_detail" in packet.parser_evidence):
            detail = packet.parser_evidence.get("attr_evidence_detail", {})
            real_attrs = [a for a in packet.attribute_candidates if a[0] != "__class__"]
            if len(real_attrs) == 1:
                attr_name = real_attrs[0][0]
                agg = detail.get(attr_name, {})
                max_strength = agg.get("max_strength", 0.0)
                has_strong   = agg.get("has_strong", False)
                if (not has_strong) and (max_strength < self.weak_signal_strength_min):
                    reasons.add(V15_4_AmbiguityFlag.ATTR_WEAK_SIGNAL)
        
        # Final status
        if not reasons:
            return VerificationResult(status=VerificationStatus.ACCEPT, reasons=set())
        
        # PARSER_FAILURE if TEMPLATE_UNKNOWN alone (no extractable signal)
        if V15_4_AmbiguityFlag.TEMPLATE_UNKNOWN in reasons and len(reasons) == 1:
            return VerificationResult(status=VerificationStatus.PARSER_FAILURE,
                                        reasons=reasons)
        
        return VerificationResult(status=VerificationStatus.PARSE_UNCERTAIN,
                                    reasons=reasons)


V15_4_VERIFIER = V15_4_Verifier()


# ------------- A3.6 V15.4 Execution (drop-in for v15.2) -------------

def v15_4_write_fact(bank, packet: ParsePacket, entity_emb_fn, class_emb_fn,
                      value_emb_fn, step: int = 0):
    """Write fact via v15.4 verifier. Returns V15_2_WriteResult."""
    vr = V15_4_VERIFIER.verify(packet)
    
    if vr.status == VerificationStatus.PARSER_FAILURE:
        return V15_2_WriteResult(status="PARSER_FAILURE", verifier_result=vr,
                                   notes="parser failure on fact")
    if vr.status == VerificationStatus.PARSE_UNCERTAIN:
        return V15_2_WriteResult(status="PARSE_UNCERTAIN", verifier_result=vr,
                                   notes="verifier rejected fact")
    
    # ACCEPT path - identical to v15.2
    entity_id = _top_entity(packet)
    
    if packet.op_type == OpType.ANCHOR_DEFINE:
        class_noun = None
        for (attr, _, ev) in packet.attribute_candidates:
            if attr == "__class__":
                class_noun = ev; break
        ent_emb = entity_emb_fn(entity_id)
        cls_emb = class_emb_fn(None, ent_emb)
        existing_slot = bank.find_by_entity_id(entity_id)
        if existing_slot is None:
            bank.allocate_new(entity_id, ent_emb, class_hint=None,
                                class_emb=cls_emb, step=step)
        return V15_2_WriteResult(status="ANCHORED", op_executed=OpType.ANCHOR_DEFINE,
                                   verifier_result=vr, target_entity=entity_id,
                                   target_attr=f"__class__:{class_noun}")
    
    if packet.op_type == OpType.WRITE:
        attr_type = _top_attribute(packet)
        val_tuple = _top_value(packet)
        if val_tuple is None or attr_type is None:
            return V15_2_WriteResult(status="PARSE_UNCERTAIN", verifier_result=vr,
                                       notes="attr or value missing post-ACCEPT")
        _, value_idx = val_tuple
        ent_emb = entity_emb_fn(entity_id)
        cls_emb = class_emb_fn(None, ent_emb)
        val_emb = value_emb_fn(attr_type, value_idx)
        existing_slot = bank.find_by_entity_id(entity_id)
        op_actual = OpType.WRITE
        if existing_slot is not None:
            rec = bank.get_record(existing_slot)
            a = rec.attr_slots.get(attr_type)
            if a is not None and a.present:
                op_actual = OpType.UPDATE
        if existing_slot is None:
            bank.allocate_new(entity_id, ent_emb, class_hint=None,
                                class_emb=cls_emb, step=step)
        bank.write_attribute(entity_id, attr_type, value_idx, step=step,
                              value_emb=val_emb)
        return V15_2_WriteResult(
            status=("UPDATED" if op_actual == OpType.UPDATE else "WRITTEN"),
            op_executed=op_actual, verifier_result=vr,
            target_entity=entity_id, target_attr=attr_type,
        )
    
    return V15_2_WriteResult(status="PARSE_UNCERTAIN", verifier_result=vr,
                               notes=f"unsupported op_type {packet.op_type}")


def v15_4_read_query(bank, packet: ParsePacket):
    """Read via v15.4 verifier. Returns (status, pred_idx, VerificationResult)."""
    vr = V15_4_VERIFIER.verify(packet)
    if vr.status == VerificationStatus.PARSER_FAILURE:
        return (READ_STATUS_PARSER_FAIL, None, vr)
    if vr.status == VerificationStatus.PARSE_UNCERTAIN:
        return (READ_STATUS_PARSE_UNCERTAIN, None, vr)
    
    entity_id = _top_entity(packet)
    attr_type = _top_attribute(packet)
    status, pred_idx = bank.read_attribute(entity_id, attr_type)
    return (status, pred_idx, vr)


print("[v15.4] Section A3 defined: parser + verifier + execution")
print("        - trigger families: color/size/location/state + synonyms")
print("        - query patterns for attribute intent")
print(f"        - V15_4_AmbiguityFlag: {len([f for f in V15_4_AmbiguityFlag])} flags (7 from v15.2 + ATTR_WEAK_SIGNAL)")
print(f"        - ATTR_WEAK_SIGNAL fires on unique semantic-only candidates")
print(f"        - WEAK_STRENGTH_MIN = {V15_4_WEAK_SIGNAL_STRENGTH_MIN}")
print(f"        - v15.4 drop-in: parse_fact, parse_query, VERIFIER, write_fact, read_query")
# ======================== B3. V15.4 PROTOCOL VALIDATION + DIAGNOSTIC =======
#
# Stage 1.3 Pas 2 validation:
#   Phase 0: trusted preservation check (clear probe + S1-S4 via v15.4)
#   Phase 1: HARD diagnostic via v15.4 pipeline (same schema as v15.3)
#   Phase 2: comparative report vs v15.3 baseline (hardcoded)
#
# V15.3 BASELINE (A100, n=1000):
#   critical_hard_overall:         0.910
#   shadow_hard_overall:           0.924
#   mixed_hard_overall:            0.910
#   paraphrase_critical:           0.820
#   coreference_critical:          1.000
#   reject_rate_hard_total:        0.000
#   accepted_but_wrong_rate:       0.090
#   mixed_vs_critical_disagreement: 0.000
#   shadow_vs_critical_disagreement: 0.090
#
# STAGE 1.3 TARGETS:
#   critical_hard >= 0.970
#   paraphrase_critical >= 0.940
#   coreference_critical = 1.000
#   reject_rate_hard NOT exploding artificially
#   overcommit on S-probes ~= 0
#   mixed_vs_critical_disagreement ~= 0
#   shadow_vs_critical_disagreement DECREASE
# ===========================================================================


V15_4_BASELINE = {
    "critical_hard_overall":            0.910,
    "shadow_hard_overall":              0.924,
    "mixed_hard_overall":               0.910,
    "paraphrase_critical":              0.820,
    "coreference_critical":             1.000,
    "reject_rate_hard_total":           0.000,
    "accepted_but_wrong_rate":          0.090,
    "mixed_vs_critical_disagreement":   0.000,
    "shadow_vs_critical_disagreement":  0.090,
    "paraphrase_shadow":                0.848,
    "paraphrase_mixed":                 0.820,
}

V15_4_TARGETS = {
    "critical_hard_min":           0.970,
    "paraphrase_critical_min":     0.940,
    "coreference_critical_fixed":  1.000,
    "reject_hard_max_delta":       0.05,    # no more than +5pp explosion
    "mixed_vs_critical_max":       0.01,    # stay near 0
}


# ------------- V15.4 trusted preservation (clear + S1-S4) -------------

def _v15_4_run_clear_probe(bank, ent_fn, cls_fn, val_fn, cfg):
    """Identical to v15_2_run_clear_probe but through v15.4 pipeline."""
    rng = random.Random(cfg["seed_reinterp"])
    n = cfg["n_reinterp"]
    n_committed = n_correct = n_uncertain = n_failure = 0
    for i in range(n):
        ep = v15_generate_episode("single_attr_simple", rng, use_heldout=True)
        bank.reset()
        for step_idx, fact_text in enumerate(ep.facts):
            pkt_f = v15_4_parse_fact(fact_text)
            v15_4_write_fact(bank, pkt_f, ent_fn, cls_fn, val_fn, step=step_idx)
        pkt_q = v15_4_parse_query(ep.query)
        status, pred_idx, vr = v15_4_read_query(bank, pkt_q)
        if status == READ_STATUS_PARSER_FAIL:
            n_failure += 1
        elif status == READ_STATUS_PARSE_UNCERTAIN:
            n_uncertain += 1
        else:
            n_committed += 1
            attr = V15_ATTR_TYPES[ep.query_attr_label]
            vocab = V15_ATTR_VALUES[attr]
            t_tok = int(ep.target_answer_token)
            target_idx = None
            for k, v_str in enumerate(vocab):
                if V15_ANSWER_TOKENS[attr].get(v_str) == t_tok:
                    target_idx = k; break
            if ep.target_is_unknown:
                if status in (READ_STATUS_NONE_OBJECT, READ_STATUS_NONE_ATTRIBUTE):
                    n_correct += 1
            else:
                if status == READ_STATUS_FOUND and pred_idx == target_idx:
                    n_correct += 1
    return {
        "n":                      n,
        "commit_rate":            n_committed / n,
        "uncertain_rate":         n_uncertain / n,
        "failure_rate":           n_failure / n,
        "fidelity_on_committed":  n_correct / max(1, n_committed),
        "coverage":               (n_committed + n_uncertain) / n,
    }


def _v15_4_score_query(status, reasons, expected_flags):
    if status == READ_STATUS_PARSER_FAIL:
        return "FAILURE"
    if status == READ_STATUS_PARSE_UNCERTAIN:
        rs = {r.value for r in reasons}
        if rs & expected_flags:
            return "HONEST"
        return "UNCERTAIN_OTHER_REASON"
    return "OVERCOMMIT"


def _v15_4_score_fact(write_res, expected_flags):
    if write_res.status == "PARSER_FAILURE":
        return "FAILURE"
    if write_res.status == "PARSE_UNCERTAIN":
        vr = write_res.verifier_result
        rs = {r.value for r in (vr.reasons if vr else set())}
        if rs & expected_flags:
            return "HONEST"
        return "UNCERTAIN_OTHER_REASON"
    return "OVERCOMMIT"


def _v15_4_run_s_probes(bank, ent_fn, cls_fn, val_fn, cfg):
    """Run S1-S4 via v15.4 pipeline. Reuses v15.2 generators."""
    n = cfg["n_s_per_probe"]
    out = {}
    for name, gen, is_fact in [
        ("S1", _v15_2_gen_S1_multi_attr,       False),
        ("S2", _v15_2_gen_S2_semantic_conflict, False),
        ("S3", _v15_2_gen_S3_referential,       False),
        ("S4", _v15_2_gen_S4_attr_conflict,     True),
    ]:
        rng = random.Random(cfg["seed_s_probes"] + hash(name) % 10000)
        counts = {"HONEST": 0, "OVERCOMMIT": 0, "FAILURE": 0,
                   "UNCERTAIN_OTHER_REASON": 0}
        for i in range(n):
            text, meta = gen(rng)
            bank.reset()
            if is_fact:
                pkt = v15_4_parse_fact(text)
                res = v15_4_write_fact(bank, pkt, ent_fn, cls_fn, val_fn, step=0)
                score = _v15_4_score_fact(res, meta["expected_flags"])
            else:
                pkt = v15_4_parse_query(text)
                status, _, vr = v15_4_read_query(bank, pkt)
                score = _v15_4_score_query(status, vr.reasons, meta["expected_flags"])
            counts[score] += 1
        honesty = counts["HONEST"] / n
        overcommit = counts["OVERCOMMIT"] / n
        out[name] = {
            "n": n, "honesty_rate": honesty, "overcommit_rate": overcommit,
            "failure_rate": counts["FAILURE"] / n,
            "counts": counts,
        }
    return out


# ------------- V15.4 HARD diagnostic (one-trial all-modes via v15.4) -------------

def _v15_4_run_trial_all_modes(bank, base_model, v15_1_memory, ep,
                                  ent_fn, cls_fn, val_fn):
    """Same as _v15_3_run_one_trial_all_modes but via v15.4 pipeline."""
    shadow = v15_1_memory.shadow
    
    bank.reset()
    for step_idx, fact_text in enumerate(ep.facts):
        pkt_f = v15_4_parse_fact(fact_text)
        v15_4_write_fact(bank, pkt_f, ent_fn, cls_fn, val_fn, step=step_idx)
    
    pkt_q = v15_4_parse_query(ep.query)
    vr_q  = V15_4_VERIFIER.verify(pkt_q)
    
    # Target classification
    if not ep.target_is_unknown:
        target_spec = "FOUND"
    else:
        q_ent_str = _top_entity(pkt_q) if pkt_q.entity_candidates else ""
        fact_eids = set()
        for f in ep.facts:
            pf = v15_4_parse_fact(f)
            if pf.entity_candidates:
                fact_eids.add(_top_entity(pf))
        target_spec = "NONE_OBJECT" if q_ent_str not in fact_eids else "NONE_ATTRIBUTE"
    
    target_idx = None
    if not ep.target_is_unknown:
        attr_type = V15_ATTR_TYPES[ep.query_attr_label]
        vocab = V15_ATTR_VALUES[attr_type]
        t_tok = int(ep.target_answer_token)
        for k, v_str in enumerate(vocab):
            if V15_ANSWER_TOKENS[attr_type].get(v_str) == t_tok:
                target_idx = k; break
    
    result = {
        "target_spec":     target_spec,
        "target_idx":      target_idx,
        "verifier_status": vr_q.status.value,
        "verifier_reasons": [r.value for r in vr_q.reasons],
        "accepted":        vr_q.status == VerificationStatus.ACCEPT,
    }
    
    if vr_q.status != VerificationStatus.ACCEPT:
        rejected_status = (READ_STATUS_PARSER_FAIL
                            if vr_q.status == VerificationStatus.PARSER_FAILURE
                            else READ_STATUS_PARSE_UNCERTAIN)
        for mode in ("critical_only", "shadow_only", "mixed"):
            result[mode] = {"status": rejected_status, "pred": None, "correct": False}
        return result
    
    entity_id = _top_entity(pkt_q)
    attr_type = _top_attribute(pkt_q)
    
    def score(status, pred):
        if target_spec == "NONE_OBJECT":
            return status == READ_STATUS_NONE_OBJECT
        if target_spec == "NONE_ATTRIBUTE":
            return status == READ_STATUS_NONE_ATTRIBUTE
        return status == READ_STATUS_FOUND and pred == target_idx
    
    # critical_only
    status_c, pred_c = bank.read_attribute(entity_id, attr_type)
    result["critical_only"] = {"status": status_c, "pred": pred_c,
                                  "correct": score(status_c, pred_c)}
    
    # shadow_only
    with torch.no_grad():
        q_ids = torch.tensor(ENC.encode(ep.query), dtype=torch.long, device=DEVICE)
        q_pooled = base_model.shared_token_emb(q_ids).mean(dim=0)
        attr_logits = shadow.attr_router(q_pooled.unsqueeze(0))
        attr_pred_idx = int(attr_logits.argmax(dim=-1).item())
        q_entity_emb = ent_fn(entity_id)
        slot_feats = _build_slot_features(bank, q_entity_emb, None, current_step=1000)
        resolver_logits = shadow.object_resolver(q_entity_emb, slot_feats)
        obj_pred = int(resolver_logits.argmax(dim=-1).item())
        K = slot_feats.shape[0]
    if obj_pred == K:
        status_s, pred_s = READ_STATUS_NONE_OBJECT, None
    elif attr_pred_idx == 4:
        status_s, pred_s = READ_STATUS_NONE_ATTRIBUTE, None
    else:
        at = V15_ATTR_TYPES[attr_pred_idx]
        slot_list = bank.occupied_slots()
        rec = bank.get_record(slot_list[obj_pred])
        a = rec.attr_slots.get(at)
        if a is None or not a.present or a.value_emb is None:
            status_s, pred_s = READ_STATUS_NONE_ATTRIBUTE, None
        else:
            with torch.no_grad():
                vl = shadow.value_heads(at, a.value_emb.unsqueeze(0))
            status_s = READ_STATUS_FOUND
            pred_s = int(vl.argmax(dim=-1).item())
    result["shadow_only"] = {"status": status_s, "pred": pred_s,
                                "correct": score(status_s, pred_s)}
    
    # mixed
    slot = bank.find_by_entity_id(entity_id)
    if slot is None:
        status_m, pred_m = READ_STATUS_NONE_OBJECT, None
    else:
        rec = bank.get_record(slot)
        a = rec.attr_slots.get(attr_type)
        if a is None or not a.present or a.value_emb is None:
            status_m, pred_m = READ_STATUS_NONE_ATTRIBUTE, None
        else:
            with torch.no_grad():
                vl = shadow.value_heads(attr_type, a.value_emb.unsqueeze(0))
            status_m = READ_STATUS_FOUND
            pred_m = int(vl.argmax(dim=-1).item())
    result["mixed"] = {"status": status_m, "pred": pred_m,
                          "correct": score(status_m, pred_m)}
    
    return result


def v15_4_run_hard_diagnostic(bank, base_model, v15_1_memory, cfg=None):
    """HARD diagnostic identical to v15_3, via v15.4 pipeline."""
    if cfg is None:
        cfg = V15_3_DIAGNOSTIC_CONFIG
    
    print()
    print(SEP)
    print("[v15.4 HARD DIAGNOSTIC] (Pas 2 - post-patch measurement)")
    print(f"  seed:          {cfg['seed']}")
    print(f"  n_paraphrase:  {cfg['n_paraphrase']}")
    print(f"  n_coreference: {cfg['n_coreference']}")
    print(SEP)
    
    ent_fn = _make_entity_emb_fn(base_model)
    cls_fn = _make_class_emb_fn(v15_1_memory)
    val_fn = _make_value_emb_fn(base_model)
    
    rng = random.Random(cfg["seed"])
    trials = []
    
    for i in range(cfg["n_paraphrase"]):
        ep = v15_generate_episode("paraphrase", rng, use_heldout=True)
        r = _v15_4_run_trial_all_modes(bank, base_model, v15_1_memory, ep,
                                           ent_fn, cls_fn, val_fn)
        r["hard_type"] = "paraphrase"
        trials.append(r)
    
    for i in range(cfg["n_coreference"]):
        ep = v15_generate_episode("coreference_distant", rng, use_heldout=True)
        r = _v15_4_run_trial_all_modes(bank, base_model, v15_1_memory, ep,
                                           ent_fn, cls_fn, val_fn)
        r["hard_type"] = "coreference_distant"
        trials.append(r)
    
    n = len(trials)
    n_rejected = sum(1 for t in trials if not t["accepted"])
    n_accepted = n - n_rejected
    
    reject_by_reason = {}
    for t in trials:
        if not t["accepted"]:
            for r in t["verifier_reasons"]:
                reject_by_reason[r] = reject_by_reason.get(r, 0) + 1
    
    acc_and_wrong = sum(1 for t in trials
                         if t["accepted"] and not t["critical_only"]["correct"])
    
    def overall_acc(ts, mode):
        return sum(1 for t in ts if t[mode]["correct"]) / max(1, len(ts))
    
    para_trials = [t for t in trials if t["hard_type"] == "paraphrase"]
    cor_trials  = [t for t in trials if t["hard_type"] == "coreference_distant"]
    
    paraphrase_acc = {mode: overall_acc(para_trials, mode)
                      for mode in ("critical_only", "shadow_only", "mixed")}
    coreference_acc = {mode: overall_acc(cor_trials, mode)
                       for mode in ("critical_only", "shadow_only", "mixed")}
    
    def disagree_rate(ts, modeA, modeB):
        diffs = sum(1 for t in ts
                    if (t[modeA]["status"], t[modeA]["pred"]) !=
                       (t[modeB]["status"], t[modeB]["pred"]))
        return diffs / max(1, len(ts))
    
    shadow_vs_critical = disagree_rate(trials, "shadow_only", "critical_only")
    mixed_vs_shadow    = disagree_rate(trials, "mixed",        "shadow_only")
    mixed_vs_critical  = disagree_rate(trials, "mixed",        "critical_only")
    
    accepted_trials = [t for t in trials if t["accepted"]]
    def accepted_acc(mode):
        if not accepted_trials: return 0.0
        return sum(1 for t in accepted_trials if t[mode]["correct"]) / len(accepted_trials)
    
    crit_acc_on_acc   = accepted_acc("critical_only")
    shadow_acc_on_acc = accepted_acc("shadow_only")
    mixed_acc_on_acc  = accepted_acc("mixed")
    
    crit_overall   = overall_acc(trials, "critical_only")
    shadow_overall = overall_acc(trials, "shadow_only")
    mixed_overall  = overall_acc(trials, "mixed")
    
    # ---- Print report ----
    print()
    print("=== Reject profile ===")
    print(f"  n_accepted:             {n_accepted}/{n} = {n_accepted/n:.1%}")
    print(f"  n_rejected:             {n_rejected}/{n} = {n_rejected/n:.1%}")
    print(f"  reject_rate_hard_total: {n_rejected/n:.3f}")
    print()
    print("  Reject by reason:")
    for reason in (V15_3_KEY_REJECT_REASONS + ["ATTR_WEAK_SIGNAL",
                                                  "ATTR_CONFLICT_STRONG",
                                                  "MULTI_FAMILY_COMPETITION"]):
        cnt = reject_by_reason.get(reason, 0)
        pct = cnt / max(1, n_rejected)
        print(f"    {reason:30s} {cnt:5d}  ({pct:.1%})")
    
    print()
    print("=== Accepted-only behavior ===")
    print(f"  accepted_but_wrong_rate:        {acc_and_wrong/max(1,n_accepted):.3f}")
    print(f"  critical_hard_on_accepted_only: {crit_acc_on_acc:.3f}")
    print(f"  shadow_hard_on_accepted_only:   {shadow_acc_on_acc:.3f}")
    print(f"  mixed_hard_on_accepted_only:    {mixed_acc_on_acc:.3f}")
    
    print()
    print("=== Overall HARD accuracy ===")
    print(f"  critical_hard_overall: {crit_overall:.3f}")
    print(f"  shadow_hard_overall:   {shadow_overall:.3f}")
    print(f"  mixed_hard_overall:    {mixed_overall:.3f}")
    
    print()
    print("=== Per hard-type ===")
    print(f"  paraphrase (n={cfg['n_paraphrase']}):")
    for mode in ("critical_only", "shadow_only", "mixed"):
        print(f"    {mode:15s}: {paraphrase_acc[mode]:.3f}")
    print(f"  coreference_distant (n={cfg['n_coreference']}):")
    for mode in ("critical_only", "shadow_only", "mixed"):
        print(f"    {mode:15s}: {coreference_acc[mode]:.3f}")
    
    print()
    print("=== Pairwise disagreement ===")
    print(f"  shadow_vs_critical_disagreement: {shadow_vs_critical:.3f}")
    print(f"  mixed_vs_shadow_disagreement:    {mixed_vs_shadow:.3f}")
    print(f"  mixed_vs_critical_disagreement:  {mixed_vs_critical:.3f}")
    
    return {
        "config": dict(cfg),
        "counts": {
            "n_total": n, "n_accepted": n_accepted, "n_rejected": n_rejected,
            "n_paraphrase": cfg["n_paraphrase"],
            "n_coreference": cfg["n_coreference"],
        },
        "reject_rate_hard_total":          n_rejected / n,
        "reject_by_reason_counts":          reject_by_reason,
        "accepted_but_wrong_rate":          acc_and_wrong / max(1, n_accepted),
        "critical_hard_on_accepted_only":   crit_acc_on_acc,
        "shadow_hard_on_accepted_only":     shadow_acc_on_acc,
        "mixed_hard_on_accepted_only":      mixed_acc_on_acc,
        "overall": {
            "critical_hard": crit_overall,
            "shadow_hard":   shadow_overall,
            "mixed_hard":    mixed_overall,
        },
        "paraphrase_hard_acc":           paraphrase_acc,
        "coreference_hard_acc":          coreference_acc,
        "shadow_vs_critical_disagreement": shadow_vs_critical,
        "mixed_vs_shadow_disagreement":    mixed_vs_shadow,
        "mixed_vs_critical_disagreement":  mixed_vs_critical,
    }


def v15_4_compare_to_baseline(diag, trusted_preservation=None):
    """Produce comparative report v15.4 vs v15.3 baseline + Stage 1.3 targets.
    
    If trusted_preservation is provided, include S4 honesty check as a
    hard acceptance criterion (per GPT: honesty must not regress).
    """
    b = V15_4_BASELINE
    t = V15_4_TARGETS
    
    crit_now = diag["overall"]["critical_hard"]
    shadow_now = diag["overall"]["shadow_hard"]
    mixed_now  = diag["overall"]["mixed_hard"]
    para_crit_now = diag["paraphrase_hard_acc"]["critical_only"]
    cor_crit_now  = diag["coreference_hard_acc"]["critical_only"]
    reject_now = diag["reject_rate_hard_total"]
    mvc_now = diag["mixed_vs_critical_disagreement"]
    svc_now = diag["shadow_vs_critical_disagreement"]
    
    print()
    print(SEP)
    print("=== COMPARATIVE REPORT v15.4 vs v15.3 BASELINE ===")
    print(SEP)
    
    def line(label, baseline_val, new_val, fmt=".3f", arrow_up_good=True):
        delta = new_val - baseline_val
        if arrow_up_good:
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
            good = "✓" if delta > 0 else ("·" if delta == 0 else "✗")
        else:
            arrow = "↓" if delta < 0 else ("↑" if delta > 0 else "=")
            good = "✓" if delta < 0 else ("·" if delta == 0 else "✗")
        print(f"  {label:40s} {baseline_val:7.3f} -> {new_val:7.3f}  "
              f"{arrow} {delta:+.3f}  {good}")
    
    line("critical_hard",                       b["critical_hard_overall"], crit_now)
    line("shadow_hard",                         b["shadow_hard_overall"], shadow_now)
    line("mixed_hard",                          b["mixed_hard_overall"], mixed_now)
    line("paraphrase_critical",                 b["paraphrase_critical"], para_crit_now)
    line("coreference_critical",                b["coreference_critical"], cor_crit_now)
    line("reject_rate_hard_total",              b["reject_rate_hard_total"], reject_now, arrow_up_good=False)
    line("mixed_vs_critical_disagreement",      b["mixed_vs_critical_disagreement"], mvc_now, arrow_up_good=False)
    line("shadow_vs_critical_disagreement",     b["shadow_vs_critical_disagreement"], svc_now, arrow_up_good=False)
    
    print()
    print("=== Stage 1.3 TARGET check ===")
    checks = {
        "critical_hard >= 0.970":            crit_now >= t["critical_hard_min"],
        "paraphrase_critical >= 0.940":      para_crit_now >= t["paraphrase_critical_min"],
        "coreference_critical == 1.000":     cor_crit_now == t["coreference_critical_fixed"],
        "reject_delta <= +0.050":            (reject_now - b["reject_rate_hard_total"]) <= t["reject_hard_max_delta"],
        "mixed_vs_critical <= 0.010":        mvc_now <= t["mixed_vs_critical_max"],
        "shadow_vs_critical decreased":      svc_now < b["shadow_vs_critical_disagreement"],
    }
    # S4 honesty hard check (per GPT acceptance criterion)
    if trusted_preservation is not None and "s_probes" in trusted_preservation:
        s4 = trusted_preservation["s_probes"].get("S4", {})
        s4_honesty   = s4.get("honesty_rate", 0.0)
        s4_overcommit = s4.get("overcommit_rate", 1.0)
        checks["S4 honesty == 1.00"]      = (s4_honesty == 1.0)
        checks["S4 overcommit == 0.00"]   = (s4_overcommit == 0.0)
    all_pass = True
    for k, v in checks.items():
        ico = "✓" if v else "✗"
        print(f"  {ico} {k}")
        if not v:
            all_pass = False
    
    # Failure mode check: critical_hard up but only through reject explosion
    crit_delta = crit_now - b["critical_hard_overall"]
    reject_delta = reject_now - b["reject_rate_hard_total"]
    cowardice_mode = (crit_delta > 0 and reject_delta > 0.02 and
                       para_crit_now < t["paraphrase_critical_min"])
    
    print()
    print(SEP)
    if cowardice_mode:
        print("  VERDICT: COWARDICE MODE — critical_hard urcă prin creșterea")
        print("  reject rate, fără să repari înțelegerea pe paraphrase.")
        print("  Parser-ul NU a fost reparat, doar a devenit mai fricos.")
        print(f"    paraphrase_critical = {para_crit_now:.3f} < {t['paraphrase_critical_min']}")
        print(f"    reject_delta = {reject_delta:+.3f} > 0.02")
        final = "FAILED_COWARDICE"
    elif all_pass:
        print("  VERDICT: PATCH SUCCEEDED — toate pragurile Stage 1.3 sunt atinse.")
        final = "PASSED"
    else:
        print("  VERDICT: PARTIAL — unele praguri rămân neatinse.")
        print(f"    failed: {[k for k, v in checks.items() if not v]}")
        final = "PARTIAL"
    print(SEP)
    
    return {
        "baseline":         b,
        "current":          {
            "critical_hard":                    crit_now,
            "shadow_hard":                      shadow_now,
            "mixed_hard":                       mixed_now,
            "paraphrase_critical":              para_crit_now,
            "coreference_critical":             cor_crit_now,
            "reject_rate_hard_total":           reject_now,
            "mixed_vs_critical_disagreement":   mvc_now,
            "shadow_vs_critical_disagreement":  svc_now,
        },
        "deltas": {
            "critical_hard":                    crit_now - b["critical_hard_overall"],
            "paraphrase_critical":              para_crit_now - b["paraphrase_critical"],
            "reject_rate_hard_total":           reject_now - b["reject_rate_hard_total"],
            "shadow_vs_critical_disagreement":  svc_now - b["shadow_vs_critical_disagreement"],
            "mixed_vs_critical_disagreement":   mvc_now - b["mixed_vs_critical_disagreement"],
        },
        "targets":          t,
        "checks":           checks,
        "cowardice_mode":   cowardice_mode,
        "final_verdict":    final,
    }


def v15_4_write_memo(diag, comparison, trusted_preservation, path):
    """Write Pas 2 memo."""
    lines = []
    lines.append("# v15.4 Stage 1.3 Pas 2 Memo")
    lines.append("")
    lines.append(f"**Final verdict: {comparison['final_verdict']}**")
    lines.append("")
    lines.append("## Trusted preservation (must hold)")
    lines.append(f"- Clear probe commit_rate:   {trusted_preservation['clear']['commit_rate']:.3f}")
    lines.append(f"- Clear probe fidelity:      {trusted_preservation['clear']['fidelity_on_committed']:.3f}")
    for name, sr in trusted_preservation['s_probes'].items():
        lines.append(f"- {name}: honesty={sr['honesty_rate']:.3f} overcommit={sr['overcommit_rate']:.3f}")
    lines.append("")
    lines.append("## HARD diagnostic v15.4 vs v15.3 baseline")
    lines.append("| Metric | Baseline | v15.4 | Delta |")
    lines.append("|---|---:|---:|---:|")
    b = V15_4_BASELINE; c = comparison['current']
    pairs = [
        ("critical_hard_overall", "critical_hard"),
        ("shadow_hard_overall", "shadow_hard"),
        ("mixed_hard_overall", "mixed_hard"),
        ("paraphrase_critical", "paraphrase_critical"),
        ("coreference_critical", "coreference_critical"),
        ("reject_rate_hard_total", "reject_rate_hard_total"),
        ("mixed_vs_critical_disagreement", "mixed_vs_critical_disagreement"),
        ("shadow_vs_critical_disagreement", "shadow_vs_critical_disagreement"),
    ]
    for bk, ck in pairs:
        bv, cv = b[bk], c[ck]
        lines.append(f"| {bk} | {bv:.3f} | {cv:.3f} | {cv-bv:+.3f} |")
    lines.append("")
    lines.append("## Stage 1.3 target checks")
    for k, v in comparison['checks'].items():
        lines.append(f"- {'✓' if v else '✗'} {k}")
    if comparison['cowardice_mode']:
        lines.append("")
        lines.append("## COWARDICE MODE DETECTED")
        lines.append("critical_hard urcă prin reject explosion, nu prin reparare parser.")
    lines.append("")
    lines.append("## Reject breakdown by reason")
    for reason, cnt in diag['reject_by_reason_counts'].items():
        lines.append(f"- {reason}: {cnt}")
    lines.append("")
    lines.append("## Raw diagnostic")
    lines.append("```")
    lines.append(json.dumps(diag, indent=2, default=str))
    lines.append("```")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


print("[v15.4] Section B3 defined: protocol validation + HARD diagnostic via v15.4")
print("        - Phase 0: trusted preservation (clear + S1-S4 via v15.4)")
print("        - Phase 1: HARD diagnostic identical schema to v15.3")
print("        - Phase 2: comparative report vs baseline + cowardice check")

# ======================== external_holdout_generator =======================
#
# EXTERNAL HOLDOUT GENERATOR for Stage 1.3 Pas 3 robustness test.
#
# ZERO contamination rules:
#   - NO import of V15_FACT_TEMPLATES, V15_QUERY_TEMPLATES, V15_4_QUERY_PATTERNS
#   - NO reuse of trigger family words beyond strict minimum needed to express
#     the attribute at all (e.g. we use literal value words like "red" because
#     those ARE the attribute values; we don't use "color/size/location/state"
#     keywords in the orthogonal branches)
#   - NO extension of V15_4_PREFIX_ALIAS_MAP
#   - NO modification of entity pool
#
# The attribute VALUES themselves (red, blue, tiny, forest, angry, etc.) must
# still be present since those are the actual content the system must extract.
# Everything ELSE (query phrasing, fact phrasing, discourse structure) is new.
#
# Families produced:
#   F1 = novel_paraphrase_syntax  : fronted attrs, nested clauses, passive, copula-less
#   F2 = multiword_entities       : "young dragon", "iron sword", etc. (2-3 tokens)
#   F3 = novel_lexical_alias      : synonyms NOT in v15.4 trigger families
#   F4 = discourse_intercalation  : irrelevant sentences between fact and query
#   F5 = novel_query_forms        : tag questions, echo, indirect speech
#
# S-probes (ambiguity honesty on NEW conflict structures):
#   S5 = conflict_intercalated    : contradictory facts separated by distractors
#   S6 = entity_competition_cross : two entities, pronoun-ambiguous cross-sentence
#
# Each family returns V15Episode-compatible objects so the existing pipeline
# (v15_4_parse_fact/query, v15_4_write_fact/read_query, bank, shadow) can run
# end-to-end unchanged.
# ===========================================================================


import random as _rng_module
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set


HOLDOUT_GENERATOR_VERSION = "external_holdout_generator_v1"


# ------------- Entity pools (COPIED but NOT imported — cleaner separation) ----
#
# Same entity strings as V15 pools (we must use the same entities so the
# entity detector can find them; what we change is HOW they are mentioned).
# ---------------------------------------------------------------------------

HOLDOUT_ENTITIES_SINGLE = [
    # creatures (single-token)
    "bear", "dog", "tiger", "fox", "wolf", "bird", "cat", "horse", "deer",
    "rabbit",
    # fantasy (single-token subset — rest go via multiword family)
    "dragon", "unicorn", "hydra",
    # persons
    "teacher", "doctor", "farmer", "pilot", "chef", "judge", "dancer",
    "sailor", "priest", "warrior",
    # objects
    "lantern", "compass", "sword", "mirror", "chest", "crown", "telescope",
    "key", "shield", "scroll",
]

# Multiword entity stems for F2 (SAME canonical head entity, new surface form)
# Attention: "young dragon" must canonicalize to "dragon" in the parser.
# That's the test: will v15.4 find "dragon" even when surrounded by other modifiers?
HOLDOUT_MULTIWORD_PREFIXES = [
    "young", "old", "little", "great", "iron", "wooden", "silent",
    "hungry", "ancient", "small",
]


HOLDOUT_COLORS   = ["red", "blue", "green", "yellow", "black", "white",
                     "brown", "pink", "orange", "purple", "golden", "silver",
                     "crimson", "gray", "violet"]
HOLDOUT_SIZES    = ["tiny", "small", "big", "huge"]
HOLDOUT_LOCATIONS = ["forest", "cave", "castle", "river", "mountain",
                      "garden", "cellar", "tower", "ocean", "desert"]
HOLDOUT_STATES   = ["asleep", "awake", "angry", "calm", "hungry", "tired",
                      "happy", "afraid"]

HOLDOUT_ATTR_VALUES = {
    "color":    HOLDOUT_COLORS,
    "size":     HOLDOUT_SIZES,
    "location": HOLDOUT_LOCATIONS,
    "state":    HOLDOUT_STATES,
}

# These are the ATTR_TYPES used by the target system — must match.
HOLDOUT_ATTR_TYPES = ["color", "size", "location", "state"]


# ------------- HoldoutEpisode dataclass (V15Episode-compatible shape) -------

@dataclass
class HoldoutEpisode:
    """Compatible interface with V15Episode for downstream pipeline."""
    episode_type:             str
    family_tag:               str
    facts:                    List[str]
    fact_entity_tokens:       List[int]
    fact_attr_labels:         List[int]
    fact_answer_tokens:       List[int]
    fact_class_labels:        List[int]
    fact_is_anchor:           List[bool]
    query:                    str
    query_attr_label:         int
    query_entity_token:       int
    target_answer_token:      int
    target_is_unknown:        bool
    target_fact_idx:          int
    target_slot_name:         str
    # For probes: expected ambiguity flags the verifier should raise
    expected_reject_flags:    Set[str] = field(default_factory=set)


# ------------- Helpers -----------------------------------------------------

def _tok_first(enc, text: str) -> int:
    """First BPE token of ' text' (leading space, GPT-2 conventions)."""
    return enc.encode(" " + text.strip())[0]


def _pick_attr_value(rng, attr: str) -> str:
    return rng.choice(HOLDOUT_ATTR_VALUES[attr])


# ===========================================================================
# F1 — novel_paraphrase_syntax
# ===========================================================================
# Syntactic constructions NOT present in V15_FACT_TEMPLATES / V15_QUERY_TEMPLATES
# AND not matched by V15_4_QUERY_PATTERNS regexes.
# ---------------------------------------------------------------------------

F1_FACT_CONSTRUCTIONS = {
    # Each construction takes (entity, value) and produces a fact sentence.
    # Fronted attribute: value-first
    "color": [
        lambda e, v: f"{v.capitalize()} was the thing that defined the {e}.",
        lambda e, v: f"Among the things noticed, the {e} had a {v} tone about it.",
        lambda e, v: f"That {e}, by every account, carried {v} markings.",
        lambda e, v: f"The {e} bore {v} throughout.",
    ],
    "size": [
        lambda e, v: f"{v.capitalize()}, unmistakably, described the {e}.",
        lambda e, v: f"Comparatively, the {e} came across as {v}.",
        lambda e, v: f"In proportion, the {e} matched what one would call {v}.",
        lambda e, v: f"The {e} stood {v} beyond doubt.",
    ],
    "location": [
        lambda e, v: f"It was the {v} where the {e} resided.",
        lambda e, v: f"From within the {v}, the {e} made its presence known.",
        lambda e, v: f"Deep inside the {v}, a {e} could be seen.",
        lambda e, v: f"The {v}, as is known, houses the {e}.",
    ],
    "state": [
        lambda e, v: f"{v.capitalize()} was what the {e} had become.",
        lambda e, v: f"Noticeably, the {e} had turned {v}.",
        lambda e, v: f"The {e}, if one paid attention, grew {v} over time.",
        lambda e, v: f"{v.capitalize()} described the {e} entirely.",
    ],
}

# Query forms that do NOT match v15.4 QUERY_PATTERNS but still ask attribute
F1_QUERY_CONSTRUCTIONS = {
    "color": [
        lambda e: f"As for the {e}, what attribute defined it chromatically? The {e} is",
        lambda e: f"Regarding pigmentation of the {e}, the {e} is",
        lambda e: f"With respect to appearance in the visible spectrum, the {e} is",
        lambda e: f"Concerning chromatic quality, the {e} is",
    ],
    "size": [
        lambda e: f"In terms of proportion, the {e} is",
        lambda e: f"With respect to dimension, the {e} is",
        lambda e: f"Regarding physical scale of the {e}, the {e} is",
        lambda e: f"As a matter of magnitude, the {e} is",
    ],
    "location": [
        lambda e: f"As for whereabouts of the {e}, the {e} is in the",
        lambda e: f"Regarding the dwelling of the {e}, the {e} is in the",
        lambda e: f"With respect to habitat, the {e} is in the",
        lambda e: f"Speaking of surroundings of the {e}, the {e} is in the",
    ],
    "state": [
        lambda e: f"With respect to disposition, the {e} is",
        lambda e: f"As a matter of temperament, the {e} is",
        lambda e: f"Regarding current disposition of the {e}, the {e} is",
        lambda e: f"As for bearing of the {e}, the {e} is",
    ],
}


def gen_F1_novel_paraphrase_syntax(rng, enc, class_map):
    """F1: fact uses unusual syntax (fronted, passive, nested), query uses
    phrasing not covered by v15.4 query patterns.
    """
    ent = rng.choice(HOLDOUT_ENTITIES_SINGLE)
    attr = rng.choice(HOLDOUT_ATTR_TYPES)
    value = _pick_attr_value(rng, attr)
    
    fact = rng.choice(F1_FACT_CONSTRUCTIONS[attr])(ent, value)
    # Sprinkle 1-2 distractor facts with other entities/attrs
    distractor_ents = [e for e in HOLDOUT_ENTITIES_SINGLE if e != ent]
    d_count = rng.choice([1, 2])
    distractors = []
    for _ in range(d_count):
        de = rng.choice(distractor_ents)
        da = rng.choice(HOLDOUT_ATTR_TYPES)
        dv = _pick_attr_value(rng, da)
        distractors.append(rng.choice(F1_FACT_CONSTRUCTIONS[da])(de, dv))
    
    all_facts = [fact] + distractors
    rng.shuffle(all_facts)
    
    query = rng.choice(F1_QUERY_CONSTRUCTIONS[attr])(ent)
    
    target_answer_tok = _tok_first(enc, value)
    return HoldoutEpisode(
        episode_type="external_f1",
        family_tag="F1_novel_paraphrase_syntax",
        facts=all_facts,
        fact_entity_tokens=[_tok_first(enc, ent) for _ in all_facts],
        fact_attr_labels=[HOLDOUT_ATTR_TYPES.index(attr)] * len(all_facts),
        fact_answer_tokens=[target_answer_tok] * len(all_facts),
        fact_class_labels=[class_map.get(ent, 0)] * len(all_facts),
        fact_is_anchor=[False] * len(all_facts),
        query=query,
        query_attr_label=HOLDOUT_ATTR_TYPES.index(attr),
        query_entity_token=_tok_first(enc, ent),
        target_answer_token=target_answer_tok,
        target_is_unknown=False,
        target_fact_idx=-1,
        target_slot_name=attr,
    )


# ===========================================================================
# F2 — multiword_entities
# ===========================================================================
# Entity is mentioned as "young dragon", "iron sword" etc.
# Canonicalization should still resolve to the head noun ("dragon", "sword").
# This tests entity boundary detection across modifiers.
# ---------------------------------------------------------------------------

def gen_F2_multiword_entities(rng, enc, class_map):
    # Pick a head entity that has a clean single-token form
    ent = rng.choice(HOLDOUT_ENTITIES_SINGLE)
    prefix = rng.choice(HOLDOUT_MULTIWORD_PREFIXES)
    multiword_mention = f"{prefix} {ent}"
    attr = rng.choice(HOLDOUT_ATTR_TYPES)
    value = _pick_attr_value(rng, attr)
    
    # Fact uses multiword mention; query may use single-token OR multiword
    fact_template = rng.choice([
        f"The {multiword_mention} is {value}.",
        f"The {multiword_mention} was {value}.",
        f"A {multiword_mention} appeared {value} nearby.",
        f"The {multiword_mention} seemed {value}.",
    ])
    
    # Query uses the SINGLE-TOKEN head to test head-noun resolution
    query_templates = {
        "color":    f"What color is the {ent}? The {ent} is",
        "size":     f"What size is the {ent}? The {ent} is",
        "location": f"Where is the {ent}? The {ent} is in the",
        "state":    f"What state is the {ent} in? The {ent} is",
    }
    query = query_templates[attr]
    
    target_answer_tok = _tok_first(enc, value)
    return HoldoutEpisode(
        episode_type="external_f2",
        family_tag="F2_multiword_entities",
        facts=[fact_template],
        fact_entity_tokens=[_tok_first(enc, ent)],
        fact_attr_labels=[HOLDOUT_ATTR_TYPES.index(attr)],
        fact_answer_tokens=[target_answer_tok],
        fact_class_labels=[class_map.get(ent, 0)],
        fact_is_anchor=[False],
        query=query,
        query_attr_label=HOLDOUT_ATTR_TYPES.index(attr),
        query_entity_token=_tok_first(enc, ent),
        target_answer_token=target_answer_tok,
        target_is_unknown=False,
        target_fact_idx=0,
        target_slot_name=attr,
    )


# ===========================================================================
# F3 — novel_lexical_alias
# ===========================================================================
# Synonyms NOT in v15.4 trigger families (explicitly excluded):
#   color: pigmentation, coloration, dye, wash         (NOT shade/hue/tint/tone)
#   size:  magnitude, girth, scale, proportion          (NOT big/small/large/huge)
#   location: habitat, locale, dwelling, quarters       (NOT where/found/sits)
#   state: demeanor, bearing, disposition, temperament  (NOT feel/seems/appears)
# Queries use these novel synonyms.
# ---------------------------------------------------------------------------

F3_NOVEL_ALIAS_QUERIES = {
    "color":    [
        lambda e: f"The {e} exhibits what pigmentation? The {e} is",
        lambda e: f"What coloration characterizes the {e}? The {e} is",
        lambda e: f"The {e} displays which wash? The {e} is",
        lambda e: f"Which dye marks the {e}? The {e} is",
    ],
    "size":     [
        lambda e: f"What magnitude does the {e} have? The {e} is",
        lambda e: f"The {e} has what girth? The {e} is",
        lambda e: f"Express the scale of the {e}. The {e} is",
        lambda e: f"The {e} holds what proportion? The {e} is",
    ],
    "location": [
        lambda e: f"The habitat of the {e} is the",
        lambda e: f"Identify the locale of the {e}. The {e} is in the",
        lambda e: f"The dwelling of the {e} is the",
        lambda e: f"The quarters of the {e} are in the",
    ],
    "state":    [
        lambda e: f"What demeanor does the {e} carry? The {e} is",
        lambda e: f"The bearing of the {e} is",
        lambda e: f"Describe the disposition of the {e}. The {e} is",
        lambda e: f"The temperament of the {e} is",
    ],
}


def gen_F3_novel_lexical_alias(rng, enc, class_map):
    ent = rng.choice(HOLDOUT_ENTITIES_SINGLE)
    attr = rng.choice(HOLDOUT_ATTR_TYPES)
    value = _pick_attr_value(rng, attr)
    
    # Fact uses standard simple form (so ambiguity is on QUERY side only)
    fact = f"The {ent} is {value}."
    query = rng.choice(F3_NOVEL_ALIAS_QUERIES[attr])(ent)
    
    target_answer_tok = _tok_first(enc, value)
    return HoldoutEpisode(
        episode_type="external_f3",
        family_tag="F3_novel_lexical_alias",
        facts=[fact],
        fact_entity_tokens=[_tok_first(enc, ent)],
        fact_attr_labels=[HOLDOUT_ATTR_TYPES.index(attr)],
        fact_answer_tokens=[target_answer_tok],
        fact_class_labels=[class_map.get(ent, 0)],
        fact_is_anchor=[False],
        query=query,
        query_attr_label=HOLDOUT_ATTR_TYPES.index(attr),
        query_entity_token=_tok_first(enc, ent),
        target_answer_token=target_answer_tok,
        target_is_unknown=False,
        target_fact_idx=0,
        target_slot_name=attr,
    )


# ===========================================================================
# F4 — discourse_intercalation
# ===========================================================================
# Irrelevant sentences injected between fact and query.
# Tests whether bank/parser stay robust to narrative noise.
# ---------------------------------------------------------------------------

F4_DISTRACTOR_SENTENCES = [
    "Meanwhile, a storm gathered on the horizon.",
    "The wind shifted direction without warning.",
    "Nearby, an observer paused for a moment.",
    "Several hours passed uneventfully.",
    "Thunder rolled through the valley.",
    "A distant bell chimed three times.",
    "Leaves scattered across the path.",
    "The sky grew heavier with clouds.",
    "Somewhere, a clock struck the hour.",
    "Time continued its usual passage.",
]


def gen_F4_discourse_intercalation(rng, enc, class_map):
    ent = rng.choice(HOLDOUT_ENTITIES_SINGLE)
    attr = rng.choice(HOLDOUT_ATTR_TYPES)
    value = _pick_attr_value(rng, attr)
    
    fact = f"The {ent} is {value}."
    # Inject 2-4 distractors; they appear in the SAME facts list
    n_dist = rng.randint(2, 4)
    distractors = rng.sample(F4_DISTRACTOR_SENTENCES, n_dist)
    
    # Distractors go BEFORE and AFTER the fact (interleaved)
    split_point = rng.randint(0, n_dist)
    all_facts = distractors[:split_point] + [fact] + distractors[split_point:]
    
    query_templates = {
        "color":    f"What color is the {ent}? The {ent} is",
        "size":     f"What size is the {ent}? The {ent} is",
        "location": f"Where is the {ent}? The {ent} is in the",
        "state":    f"What state is the {ent} in? The {ent} is",
    }
    query = query_templates[attr]
    
    target_answer_tok = _tok_first(enc, value)
    # Compute target fact index in the all_facts list
    target_fact_idx = split_point  # where real fact was inserted
    return HoldoutEpisode(
        episode_type="external_f4",
        family_tag="F4_discourse_intercalation",
        facts=all_facts,
        fact_entity_tokens=[_tok_first(enc, ent)] * len(all_facts),
        fact_attr_labels=[HOLDOUT_ATTR_TYPES.index(attr)] * len(all_facts),
        fact_answer_tokens=[target_answer_tok] * len(all_facts),
        fact_class_labels=[class_map.get(ent, 0)] * len(all_facts),
        fact_is_anchor=[False] * len(all_facts),
        query=query,
        query_attr_label=HOLDOUT_ATTR_TYPES.index(attr),
        query_entity_token=_tok_first(enc, ent),
        target_answer_token=target_answer_tok,
        target_is_unknown=False,
        target_fact_idx=target_fact_idx,
        target_slot_name=attr,
    )


# ===========================================================================
# F5 — novel_query_forms
# ===========================================================================
# Tag questions, echo questions, indirect speech.
# These forms never appear in V15_QUERY_TEMPLATES or v15.4 QUERY_PATTERNS.
# ---------------------------------------------------------------------------

F5_QUERY_FORMS = {
    "color": [
        lambda e, v: f"The {e} is {v}, isn't it? The {e} is",       # tag
        lambda e, v: f"The {e} is {v}? Really? The {e} is",          # echo
        lambda e, v: f"I wonder what color the {e} has. The {e} is",  # indirect
        lambda e, v: f"Tell me whether the {e} is colored. The {e} is",  # indirect
    ],
    "size": [
        lambda e, v: f"The {e} is {v}, correct? The {e} is",
        lambda e, v: f"I'd like to know the dimension of the {e}. The {e} is",
        lambda e, v: f"The {e} is {v}? Surprising. The {e} is",
        lambda e, v: f"I wonder how the {e} measures. The {e} is",
    ],
    "location": [
        lambda e, v: f"The {e} is in the {v}, right? The {e} is in the",
        lambda e, v: f"I'd like to know where the {e} resides. The {e} is in the",
        lambda e, v: f"The {e} is in the {v}? Interesting. The {e} is in the",
        lambda e, v: f"I wonder about the habitat of the {e}. The {e} is in the",
    ],
    "state": [
        lambda e, v: f"The {e} is {v}, isn't it? The {e} is",
        lambda e, v: f"I'd like to know how the {e} is. The {e} is",
        lambda e, v: f"The {e} is {v}? Truly? The {e} is",
        lambda e, v: f"I wonder about the disposition of the {e}. The {e} is",
    ],
}


def gen_F5_novel_query_forms(rng, enc, class_map):
    ent = rng.choice(HOLDOUT_ENTITIES_SINGLE)
    attr = rng.choice(HOLDOUT_ATTR_TYPES)
    value = _pick_attr_value(rng, attr)
    
    # Use a value that MATCHES target for tag/echo questions (they embed the value);
    # a MISMATCHED value would be an S-style ambiguity test, not an F5 test.
    # The ACTUAL target comes from a separate fact.
    fact = f"The {ent} is {value}."
    # Build query with the SAME value embedded (tag/echo/indirect)
    query_builder = rng.choice(F5_QUERY_FORMS[attr])
    query = query_builder(ent, value)
    
    target_answer_tok = _tok_first(enc, value)
    return HoldoutEpisode(
        episode_type="external_f5",
        family_tag="F5_novel_query_forms",
        facts=[fact],
        fact_entity_tokens=[_tok_first(enc, ent)],
        fact_attr_labels=[HOLDOUT_ATTR_TYPES.index(attr)],
        fact_answer_tokens=[target_answer_tok],
        fact_class_labels=[class_map.get(ent, 0)],
        fact_is_anchor=[False],
        query=query,
        query_attr_label=HOLDOUT_ATTR_TYPES.index(attr),
        query_entity_token=_tok_first(enc, ent),
        target_answer_token=target_answer_tok,
        target_is_unknown=False,
        target_fact_idx=0,
        target_slot_name=attr,
    )


# ===========================================================================
# S5 — conflict_intercalated (honesty probe)
# ===========================================================================
# Two contradictory facts about same (entity, attr), separated by distractors.
# Expected: verifier/parser should NOT commit to either value.
# Expected flags: ATTR_CONFLICT_STRONG, VALUE_MISSING_OR_UNCLEAR (after writing
# both, the bank has been given two writes; but the parser on WRITE step 2
# should already be ambiguous OR the read attempt should see both values
# inconsistent).
# ---------------------------------------------------------------------------

def gen_S5_conflict_intercalated(rng, enc, class_map):
    ent = rng.choice(HOLDOUT_ENTITIES_SINGLE)
    attr = rng.choice(HOLDOUT_ATTR_TYPES)
    v1, v2 = rng.sample(HOLDOUT_ATTR_VALUES[attr], 2)
    
    # Two contradictory facts for SAME entity, SAME attribute
    fact_a = f"The {ent} is {v1}."
    fact_b = f"The {ent} is {v2}."
    # Distractors between them
    distractors = rng.sample(F4_DISTRACTOR_SENTENCES, 2)
    all_facts = [fact_a, distractors[0], distractors[1], fact_b]
    
    # Query the conflicted attr
    query_templates = {
        "color":    f"What color is the {ent}? The {ent} is",
        "size":     f"What size is the {ent}? The {ent} is",
        "location": f"Where is the {ent}? The {ent} is in the",
        "state":    f"What state is the {ent} in? The {ent} is",
    }
    query = query_templates[attr]
    
    # Target is ambiguous — system should refuse
    return HoldoutEpisode(
        episode_type="external_s5",
        family_tag="S5_conflict_intercalated",
        facts=all_facts,
        fact_entity_tokens=[_tok_first(enc, ent)] * len(all_facts),
        fact_attr_labels=[HOLDOUT_ATTR_TYPES.index(attr)] * len(all_facts),
        fact_answer_tokens=[0] * len(all_facts),
        fact_class_labels=[class_map.get(ent, 0)] * len(all_facts),
        fact_is_anchor=[False] * len(all_facts),
        query=query,
        query_attr_label=HOLDOUT_ATTR_TYPES.index(attr),
        query_entity_token=_tok_first(enc, ent),
        target_answer_token=0,
        target_is_unknown=True,  # ambiguous; no committed value is correct
        target_fact_idx=-1,
        target_slot_name=attr,
        expected_reject_flags={"ATTR_CONFLICT_STRONG",
                                "VALUE_MISSING_OR_UNCLEAR",
                                "MULTIPLE_ATTR_TRIGGERS"},
    )


# ===========================================================================
# S6 — entity_competition_cross (honesty probe)
# ===========================================================================
# Two different entities in earlier sentences. Query uses pronoun "it"
# with cross-sentence ambiguity over which entity to bind.
# Expected: REFERENT_AMBIGUOUS.
# ---------------------------------------------------------------------------

def gen_S6_entity_competition_cross(rng, enc, class_map):
    e1, e2 = rng.sample(HOLDOUT_ENTITIES_SINGLE, 2)
    attr = rng.choice(HOLDOUT_ATTR_TYPES)
    v1 = _pick_attr_value(rng, attr)
    v2 = _pick_attr_value(rng, attr)
    
    fact_a = f"The {e1} is {v1}."
    fact_b = f"The {e2} is {v2}."
    # Distractor between them
    distractor = rng.choice(F4_DISTRACTOR_SENTENCES)
    all_facts = [fact_a, distractor, fact_b]
    
    # Query uses pronoun — ambiguous across two antecedents
    query_templates = {
        "color":    f"What color is it? It is",
        "size":     f"What size is it? It is",
        "location": f"Where is it? It is in the",
        "state":    f"What state is it in? It is",
    }
    query = query_templates[attr]
    
    return HoldoutEpisode(
        episode_type="external_s6",
        family_tag="S6_entity_competition_cross",
        facts=all_facts,
        fact_entity_tokens=[_tok_first(enc, e1)] * len(all_facts),
        fact_attr_labels=[HOLDOUT_ATTR_TYPES.index(attr)] * len(all_facts),
        fact_answer_tokens=[0] * len(all_facts),
        fact_class_labels=[class_map.get(e1, 0)] * len(all_facts),
        fact_is_anchor=[False] * len(all_facts),
        query=query,
        query_attr_label=HOLDOUT_ATTR_TYPES.index(attr),
        query_entity_token=_tok_first(enc, e1),
        target_answer_token=0,
        target_is_unknown=True,  # ambiguous referent
        target_fact_idx=-1,
        target_slot_name=attr,
        expected_reject_flags={"REFERENT_AMBIGUOUS", "MULTI_ENTITY_SAME_TYPE",
                                "TEMPLATE_UNKNOWN"},
    )


# ===========================================================================
# Registry
# ===========================================================================

EXTERNAL_HOLDOUT_FAMILIES = {
    "F1_novel_paraphrase_syntax":  gen_F1_novel_paraphrase_syntax,
    "F2_multiword_entities":       gen_F2_multiword_entities,
    "F3_novel_lexical_alias":      gen_F3_novel_lexical_alias,
    "F4_discourse_intercalation":  gen_F4_discourse_intercalation,
    "F5_novel_query_forms":        gen_F5_novel_query_forms,
}

EXTERNAL_HOLDOUT_S_PROBES = {
    "S5_conflict_intercalated":       gen_S5_conflict_intercalated,
    "S6_entity_competition_cross":    gen_S6_entity_competition_cross,
}


print(f"[{HOLDOUT_GENERATOR_VERSION}] defined: "
      f"{len(EXTERNAL_HOLDOUT_FAMILIES)} holdout families + "
      f"{len(EXTERNAL_HOLDOUT_S_PROBES)} S-probes")
# ======================== C. V15.5 EXTERNAL HOLDOUT EVALUATOR ==============
#
# Runs every holdout family + S5/S6 through the v15.4 pipeline unchanged.
# Reports per-family: critical / shadow / mixed / reject_rate / reasons /
#                      honesty / overcommit.
#
# Global acceptance criteria (Pas 3):
#   (1) critical >= 85% on at least 4/5 families
#   (2) no family below 70%
#   (3) mean(critical) across 5 families >= 88%
#   (4) honesty(S5/S6) >= 95%
#   (5) overcommit(S5/S6) <= 2%
#   (6) mixed <= critical + 0.5pp  AND  mixed >= shadow - 2pp (on relevant F)
#   (7) trusted regression check obligatoriu: before/after unchanged
#
# Gates:
#   - Before holdout: run v15.4 trusted probe (clear + S1-S4). Record.
#   - After holdout: re-run same trusted probe. Must match byte-identical.
#   - If trusted regression detected -> report invalidates.
# ===========================================================================


V15_5_HOLDOUT_CONFIG = {
    "seed":               20260915,
    "n_per_family":       500,
    "n_per_s_probe":      200,
    "trusted_n_clear":    200,
    "trusted_n_s":        100,
}

V15_5_ACCEPTANCE = {
    "critical_min_per_family":    0.85,
    "critical_floor_any_family":  0.70,
    "critical_mean_over_5":       0.88,
    "s_honesty_min":              0.95,
    "s_overcommit_max":           0.02,
    "mixed_vs_critical_max_gap":  0.005,   # mixed <= critical + 0.5pp
    "mixed_vs_shadow_max_below":  0.02,    # mixed >= shadow - 2pp
    "min_families_passing":       4,
}


# ------------- Build class_map for episode generation ----------------------

def _v15_5_build_class_map():
    """Map entity string -> class_id (creature=0, person=1, object=2)."""
    out = {}
    for (e, cid) in V15_TRAIN_ENTITIES:
        out[e] = cid
    for (e, cid) in V15_HELDOUT_ENTITIES:
        out[e] = cid
    return out


# ------------- Run one episode through v15.4 pipeline, all 3 modes ---------

def _v15_5_run_episode_all_modes(bank, base_model, v15_1_memory, ep,
                                    ent_fn, cls_fn, val_fn):
    """Returns dict with critical/shadow/mixed predictions and correctness."""
    shadow = v15_1_memory.shadow
    bank.reset()
    
    # Write facts
    for step_idx, fact_text in enumerate(ep.facts):
        pkt_f = v15_4_parse_fact(fact_text)
        v15_4_write_fact(bank, pkt_f, ent_fn, cls_fn, val_fn, step=step_idx)
    
    # Parse query
    pkt_q = v15_4_parse_query(ep.query)
    vr_q  = V15_4_VERIFIER.verify(pkt_q)
    
    result = {
        "verifier_status":  vr_q.status.value,
        "verifier_reasons": [r.value for r in vr_q.reasons],
        "accepted":         vr_q.status == VerificationStatus.ACCEPT,
        "target_is_unknown": ep.target_is_unknown,
        "target_token":     ep.target_answer_token,
    }
    
    # Compute target_idx from token
    target_idx = None
    if not ep.target_is_unknown:
        attr_type = HOLDOUT_ATTR_TYPES[ep.query_attr_label]
        vocab = HOLDOUT_ATTR_VALUES[attr_type]
        for k, vstr in enumerate(vocab):
            if V15_ANSWER_TOKENS.get(attr_type, {}).get(vstr) == ep.target_answer_token:
                target_idx = k
                break
    
    def score_status_pred(status, pred):
        if ep.target_is_unknown:
            # For S-probes: correct = refusal (NONE_* or PARSE_UNCERTAIN)
            return status in (READ_STATUS_NONE_OBJECT, READ_STATUS_NONE_ATTRIBUTE,
                                READ_STATUS_PARSE_UNCERTAIN, READ_STATUS_PARSER_FAIL)
        return status == READ_STATUS_FOUND and pred == target_idx
    
    if vr_q.status != VerificationStatus.ACCEPT:
        rejected_status = (READ_STATUS_PARSER_FAIL
                            if vr_q.status == VerificationStatus.PARSER_FAILURE
                            else READ_STATUS_PARSE_UNCERTAIN)
        for mode in ("critical_only", "shadow_only", "mixed"):
            result[mode] = {
                "status":  rejected_status,
                "pred":    None,
                "correct": score_status_pred(rejected_status, None),
            }
        return result
    
    entity_id = _top_entity(pkt_q)
    attr_type = _top_attribute(pkt_q)
    
    # critical_only
    status_c, pred_c = bank.read_attribute(entity_id, attr_type)
    result["critical_only"] = {
        "status": status_c, "pred": pred_c,
        "correct": score_status_pred(status_c, pred_c),
    }
    
    # shadow_only
    with torch.no_grad():
        q_ids = torch.tensor(ENC.encode(ep.query), dtype=torch.long, device=DEVICE)
        q_pooled = base_model.shared_token_emb(q_ids).mean(dim=0)
        attr_logits = shadow.attr_router(q_pooled.unsqueeze(0))
        attr_pred_idx = int(attr_logits.argmax(dim=-1).item())
        q_entity_emb = ent_fn(entity_id)
        slot_feats = _build_slot_features(bank, q_entity_emb, None, current_step=1000)
        resolver_logits = shadow.object_resolver(q_entity_emb, slot_feats)
        obj_pred = int(resolver_logits.argmax(dim=-1).item())
        K = slot_feats.shape[0]
    if obj_pred == K:
        status_s, pred_s = READ_STATUS_NONE_OBJECT, None
    elif attr_pred_idx == 4:
        status_s, pred_s = READ_STATUS_NONE_ATTRIBUTE, None
    else:
        at = V15_ATTR_TYPES[attr_pred_idx]
        slot_list = bank.occupied_slots()
        rec = bank.get_record(slot_list[obj_pred])
        a = rec.attr_slots.get(at)
        if a is None or not a.present or a.value_emb is None:
            status_s, pred_s = READ_STATUS_NONE_ATTRIBUTE, None
        else:
            with torch.no_grad():
                vl = shadow.value_heads(at, a.value_emb.unsqueeze(0))
            status_s = READ_STATUS_FOUND
            pred_s = int(vl.argmax(dim=-1).item())
    result["shadow_only"] = {
        "status": status_s, "pred": pred_s,
        "correct": score_status_pred(status_s, pred_s),
    }
    
    # mixed
    slot = bank.find_by_entity_id(entity_id)
    if slot is None:
        status_m, pred_m = READ_STATUS_NONE_OBJECT, None
    else:
        rec = bank.get_record(slot)
        a = rec.attr_slots.get(attr_type)
        if a is None or not a.present or a.value_emb is None:
            status_m, pred_m = READ_STATUS_NONE_ATTRIBUTE, None
        else:
            with torch.no_grad():
                vl = shadow.value_heads(attr_type, a.value_emb.unsqueeze(0))
            status_m = READ_STATUS_FOUND
            pred_m = int(vl.argmax(dim=-1).item())
    result["mixed"] = {
        "status": status_m, "pred": pred_m,
        "correct": score_status_pred(status_m, pred_m),
    }
    
    return result


# ------------- Run one family: returns aggregate -------------

def _v15_5_run_family(family_name, gen_func, n, seed_offset,
                        bank, base_model, v15_1_memory,
                        ent_fn, cls_fn, val_fn, class_map):
    rng = _rng_module.Random(V15_5_HOLDOUT_CONFIG["seed"] + seed_offset)
    trials = []
    for i in range(n):
        ep = gen_func(rng, ENC, class_map)
        r = _v15_5_run_episode_all_modes(bank, base_model, v15_1_memory, ep,
                                            ent_fn, cls_fn, val_fn)
        r["family"] = family_name
        r["expected_reject_flags"] = ep.expected_reject_flags
        trials.append(r)
    
    n_total = len(trials)
    n_accepted = sum(1 for t in trials if t["accepted"])
    n_rejected = n_total - n_accepted
    
    # Reasons breakdown (ALL flags in rejections)
    reasons_counts = {}
    for t in trials:
        if not t["accepted"]:
            for r in t["verifier_reasons"]:
                reasons_counts[r] = reasons_counts.get(r, 0) + 1
    
    # Accuracy per mode (overall, not just on accepted)
    def acc(mode):
        return sum(1 for t in trials if t[mode]["correct"]) / max(1, n_total)
    
    # Honesty/overcommit (relevant for S-probes only)
    # honesty = % of rejected (rightfully so for unknown targets)
    # overcommit = % that committed + incorrect when target was unknown
    honesty = None
    overcommit = None
    unknown_trials = [t for t in trials if t["target_is_unknown"]]
    if unknown_trials:
        n_unk = len(unknown_trials)
        honesty    = sum(1 for t in unknown_trials
                          if t["verifier_status"] in ("PARSE_UNCERTAIN",
                                                        "PARSER_FAILURE")
                          or t["critical_only"]["status"] in (READ_STATUS_NONE_OBJECT,
                                                                READ_STATUS_NONE_ATTRIBUTE)) / n_unk
        overcommit = sum(1 for t in unknown_trials
                          if t["verifier_status"] == "ACCEPT"
                          and t["critical_only"]["status"] == READ_STATUS_FOUND) / n_unk
    
    return {
        "family":            family_name,
        "n":                 n_total,
        "n_accepted":        n_accepted,
        "n_rejected":        n_rejected,
        "reject_rate":       n_rejected / n_total,
        "reasons_counts":    reasons_counts,
        "critical":          acc("critical_only"),
        "shadow":            acc("shadow_only"),
        "mixed":             acc("mixed"),
        "honesty":           honesty,
        "overcommit":        overcommit,
    }


# ------------- Trusted regression snapshot ---------------------------------

def _v15_5_snapshot_trusted(bank, base_model, v15_1_memory):
    """Take a signature of trusted behavior: clear probe + S1-S4 aggregates."""
    ent_fn = _make_entity_emb_fn(base_model)
    cls_fn = _make_class_emb_fn(v15_1_memory)
    val_fn = _make_value_emb_fn(base_model)
    
    cfg_local = dict(V15_2_BENCH_CONFIG)
    cfg_local["n_reinterp"] = V15_5_HOLDOUT_CONFIG["trusted_n_clear"]
    cfg_local["n_s_per_probe"] = V15_5_HOLDOUT_CONFIG["trusted_n_s"]
    
    clear = _v15_4_run_clear_probe(bank, ent_fn, cls_fn, val_fn, cfg_local)
    s_probes = _v15_4_run_s_probes(bank, ent_fn, cls_fn, val_fn, cfg_local)
    
    return {
        "clear_commit_rate":           clear["commit_rate"],
        "clear_fidelity":              clear["fidelity_on_committed"],
        "clear_uncertain":             clear["uncertain_rate"],
        "s1_honesty":                  s_probes["S1"]["honesty_rate"],
        "s2_honesty":                  s_probes["S2"]["honesty_rate"],
        "s3_honesty":                  s_probes["S3"]["honesty_rate"],
        "s4_honesty":                  s_probes["S4"]["honesty_rate"],
        "s_avg_overcommit":            sum(s_probes[k]["overcommit_rate"]
                                              for k in s_probes) / 4,
    }


def _v15_5_trusted_signatures_match(snap_before, snap_after, tol=1e-6):
    for k, v_before in snap_before.items():
        v_after = snap_after.get(k)
        if v_after is None or abs(v_before - v_after) > tol:
            return False, k, v_before, v_after
    return True, None, None, None


# ------------- Main runner -------------------------------------------------

def v15_5_run_external_holdout(bank, base_model, v15_1_memory):
    """Run the complete external holdout protocol."""
    print()
    print(SEP)
    print("[v15.5 EXTERNAL HOLDOUT] Stage 1.3 Pas 3")
    print(f"  Generator version: {HOLDOUT_GENERATOR_VERSION}")
    print(f"  Families:          {list(EXTERNAL_HOLDOUT_FAMILIES.keys())}")
    print(f"  S-probes:          {list(EXTERNAL_HOLDOUT_S_PROBES.keys())}")
    print(f"  n_per_family:      {V15_5_HOLDOUT_CONFIG['n_per_family']}")
    print(f"  n_per_s_probe:     {V15_5_HOLDOUT_CONFIG['n_per_s_probe']}")
    print(f"  seed:              {V15_5_HOLDOUT_CONFIG['seed']}")
    print(SEP)
    
    ent_fn = _make_entity_emb_fn(base_model)
    cls_fn = _make_class_emb_fn(v15_1_memory)
    val_fn = _make_value_emb_fn(base_model)
    class_map = _v15_5_build_class_map()
    
    # Gate 1: trusted snapshot BEFORE
    print()
    print("[v15.5] Gate 1: trusted snapshot BEFORE holdout")
    snap_before = _v15_5_snapshot_trusted(bank, base_model, v15_1_memory)
    for k, v in snap_before.items():
        print(f"  {k}: {v:.4f}")
    
    # Run 5 holdout families
    family_results = {}
    seed_offset = 1000
    print()
    print("[v15.5] Running 5 holdout families (n={} each)".format(
        V15_5_HOLDOUT_CONFIG["n_per_family"]))
    for fname, gen in EXTERNAL_HOLDOUT_FAMILIES.items():
        print(f"  -> {fname}")
        res = _v15_5_run_family(fname, gen, V15_5_HOLDOUT_CONFIG["n_per_family"],
                                  seed_offset, bank, base_model, v15_1_memory,
                                  ent_fn, cls_fn, val_fn, class_map)
        family_results[fname] = res
        print(f"     critical={res['critical']:.3f} shadow={res['shadow']:.3f} "
              f"mixed={res['mixed']:.3f} reject={res['reject_rate']:.3f}")
        if res["reasons_counts"]:
            top_reasons = sorted(res["reasons_counts"].items(),
                                   key=lambda x: -x[1])[:3]
            print(f"     top rejection reasons: {top_reasons}")
        seed_offset += 1000
    
    # Run 2 S-probes
    s_results = {}
    print()
    print("[v15.5] Running S-probes (n={} each)".format(
        V15_5_HOLDOUT_CONFIG["n_per_s_probe"]))
    for sname, gen in EXTERNAL_HOLDOUT_S_PROBES.items():
        print(f"  -> {sname}")
        res = _v15_5_run_family(sname, gen, V15_5_HOLDOUT_CONFIG["n_per_s_probe"],
                                  seed_offset, bank, base_model, v15_1_memory,
                                  ent_fn, cls_fn, val_fn, class_map)
        s_results[sname] = res
        print(f"     honesty={res['honesty']:.3f} overcommit={res['overcommit']:.3f} "
              f"reject={res['reject_rate']:.3f}")
        if res["reasons_counts"]:
            top_reasons = sorted(res["reasons_counts"].items(),
                                   key=lambda x: -x[1])[:3]
            print(f"     top rejection reasons: {top_reasons}")
        seed_offset += 1000
    
    # Gate 2: trusted snapshot AFTER
    print()
    print("[v15.5] Gate 2: trusted snapshot AFTER holdout")
    snap_after = _v15_5_snapshot_trusted(bank, base_model, v15_1_memory)
    match, bad_k, vb, va = _v15_5_trusted_signatures_match(snap_before, snap_after)
    if match:
        print("  PASS: trusted signature identical before/after")
    else:
        print(f"  FAIL: trusted regression on '{bad_k}': before={vb} after={va}")
    
    # ---- Evaluate acceptance criteria ----
    print()
    print(SEP)
    print("=== ACCEPTANCE CRITERIA ===")
    print(SEP)
    
    crits = {f: r["critical"] for f, r in family_results.items()}
    crit_list = list(crits.values())
    n_fams_passing = sum(1 for v in crit_list if v >= V15_5_ACCEPTANCE["critical_min_per_family"])
    floor_violation = any(v < V15_5_ACCEPTANCE["critical_floor_any_family"]
                            for v in crit_list)
    mean_critical = sum(crit_list) / len(crit_list)
    
    s5_honesty = s_results["S5_conflict_intercalated"]["honesty"] or 0.0
    s6_honesty = s_results["S6_entity_competition_cross"]["honesty"] or 0.0
    s5_overcommit = s_results["S5_conflict_intercalated"]["overcommit"] or 1.0
    s6_overcommit = s_results["S6_entity_competition_cross"]["overcommit"] or 1.0
    
    # Mixed vs critical (max gap)
    mixed_vs_critical_ok_per_fam = {}
    mixed_vs_shadow_ok_per_fam   = {}
    for f, r in family_results.items():
        mixed_vs_critical_ok_per_fam[f] = (r["mixed"] <= r["critical"] +
                                              V15_5_ACCEPTANCE["mixed_vs_critical_max_gap"])
        if r["shadow"] > 0.3:  # only evaluate gap if shadow is relevant
            mixed_vs_shadow_ok_per_fam[f] = (r["mixed"] >= r["shadow"] -
                                                V15_5_ACCEPTANCE["mixed_vs_shadow_max_below"])
        else:
            mixed_vs_shadow_ok_per_fam[f] = True  # n/a
    
    checks = {
        f"critical >= {V15_5_ACCEPTANCE['critical_min_per_family']:.2f} on >= "
        f"{V15_5_ACCEPTANCE['min_families_passing']}/5 families":
            n_fams_passing >= V15_5_ACCEPTANCE["min_families_passing"],
        f"no family below {V15_5_ACCEPTANCE['critical_floor_any_family']:.2f}":
            not floor_violation,
        f"mean(critical) >= {V15_5_ACCEPTANCE['critical_mean_over_5']:.2f}":
            mean_critical >= V15_5_ACCEPTANCE["critical_mean_over_5"],
        f"S5 honesty >= {V15_5_ACCEPTANCE['s_honesty_min']:.2f}":
            s5_honesty >= V15_5_ACCEPTANCE["s_honesty_min"],
        f"S6 honesty >= {V15_5_ACCEPTANCE['s_honesty_min']:.2f}":
            s6_honesty >= V15_5_ACCEPTANCE["s_honesty_min"],
        f"S5 overcommit <= {V15_5_ACCEPTANCE['s_overcommit_max']:.2f}":
            s5_overcommit <= V15_5_ACCEPTANCE["s_overcommit_max"],
        f"S6 overcommit <= {V15_5_ACCEPTANCE['s_overcommit_max']:.2f}":
            s6_overcommit <= V15_5_ACCEPTANCE["s_overcommit_max"],
        "mixed <= critical + 0.5pp on all families":
            all(mixed_vs_critical_ok_per_fam.values()),
        "mixed >= shadow - 2pp on relevant families":
            all(mixed_vs_shadow_ok_per_fam.values()),
        "trusted regression check":
            match,
    }
    
    all_pass = all(checks.values())
    for name, ok in checks.items():
        print(f"  {'✓' if ok else '✗'} {name}")
    print()
    print(f"  n_families_passing: {n_fams_passing}/5")
    print(f"  mean_critical:      {mean_critical:.3f}")
    print(f"  S5 honesty:         {s5_honesty:.3f}   overcommit: {s5_overcommit:.3f}")
    print(f"  S6 honesty:         {s6_honesty:.3f}   overcommit: {s6_overcommit:.3f}")
    print()
    print("-" * 70)
    if all_pass:
        print("VERDICT: PROTOCOL_ROBUSTNESS_VALIDATED")
        final = "PROTOCOL_ROBUSTNESS_VALIDATED"
    else:
        failed = [k for k, v in checks.items() if not v]
        print("VERDICT: BENCHMARK_CLOSURE_ONLY")
        print(f"  Failed: {failed}")
        final = "BENCHMARK_CLOSURE_ONLY"
    print("-" * 70)
    
    # Per-family breakdown
    print()
    print(SEP)
    print("=== PER-FAMILY BREAKDOWN ===")
    print(SEP)
    print(f"  {'family':35s}  {'critical':>10s} {'shadow':>8s} {'mixed':>8s} "
          f"{'reject':>8s}")
    for f, r in family_results.items():
        print(f"  {f:35s}  {r['critical']:>10.3f} {r['shadow']:>8.3f} "
              f"{r['mixed']:>8.3f} {r['reject_rate']:>8.3f}")
    print()
    print(f"  {'S-probe':35s}  {'honesty':>10s} {'overcommit':>10s} {'reject':>8s}")
    for s, r in s_results.items():
        print(f"  {s:35s}  {r['honesty']:>10.3f} {r['overcommit']:>10.3f} "
              f"{r['reject_rate']:>8.3f}")
    print(SEP)
    
    return {
        "generator_version":      HOLDOUT_GENERATOR_VERSION,
        "config":                  dict(V15_5_HOLDOUT_CONFIG),
        "acceptance":              dict(V15_5_ACCEPTANCE),
        "snap_before":             snap_before,
        "snap_after":              snap_after,
        "trusted_regression_ok":   match,
        "trusted_regression_key":  bad_k,
        "family_results":          family_results,
        "s_results":                s_results,
        "checks":                  checks,
        "n_families_passing":      n_fams_passing,
        "mean_critical":           mean_critical,
        "final_verdict":           final,
    }


def v15_5_write_memo(results, path):
    lines = []
    lines.append("# v15.5 External Holdout - Stage 1.3 Pas 3")
    lines.append("")
    lines.append(f"**Verdict: {results['final_verdict']}**")
    lines.append("")
    lines.append(f"- Generator: `{results['generator_version']}`")
    lines.append(f"- Seed: {results['config']['seed']}")
    lines.append(f"- n per family: {results['config']['n_per_family']}")
    lines.append(f"- n per S-probe: {results['config']['n_per_s_probe']}")
    lines.append("")
    lines.append("## Per-family results")
    lines.append("| Family | critical | shadow | mixed | reject |")
    lines.append("|---|---:|---:|---:|---:|")
    for f, r in results["family_results"].items():
        lines.append(f"| {f} | {r['critical']:.3f} | {r['shadow']:.3f} | "
                      f"{r['mixed']:.3f} | {r['reject_rate']:.3f} |")
    lines.append("")
    lines.append("## S-probes")
    lines.append("| Probe | honesty | overcommit | reject |")
    lines.append("|---|---:|---:|---:|")
    for s, r in results["s_results"].items():
        lines.append(f"| {s} | {r['honesty']:.3f} | {r['overcommit']:.3f} | "
                      f"{r['reject_rate']:.3f} |")
    lines.append("")
    lines.append("## Acceptance checks")
    for k, v in results["checks"].items():
        lines.append(f"- {'✓' if v else '✗'} {k}")
    lines.append("")
    lines.append("## Aggregate")
    lines.append(f"- Families passing 85%: {results['n_families_passing']}/5")
    lines.append(f"- Mean critical: {results['mean_critical']:.3f}")
    lines.append(f"- Trusted regression check: {'PASS' if results['trusted_regression_ok'] else 'FAIL'}")
    lines.append("")
    lines.append("## Rejection reason breakdown (per family)")
    for f, r in {**results["family_results"], **results["s_results"]}.items():
        if r["reasons_counts"]:
            lines.append(f"- **{f}**:")
            for reason, cnt in sorted(r["reasons_counts"].items(),
                                         key=lambda x: -x[1]):
                lines.append(f"  - {reason}: {cnt}")
    lines.append("")
    lines.append("## Raw")
    lines.append("```")
    lines.append(json.dumps(results, indent=2, default=str))
    lines.append("```")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


print("[v15.5] Section C defined: external holdout evaluator")
print("        - 5 families x 500 trials")
print("        - 2 S-probes x 200 trials")
print("        - trusted regression gate (before + after)")
print("        - 10 acceptance checks")

# ===========================================================================
# v15.6 INTERNALIZATION LAYER — PAS 1
#   Structural wrapper only. Zero behavioral change vs v15.4.1 baseline.
# ===========================================================================
#
# Scope (Pas 1, per GPT directive):
#   - Define InternalizationPacket dataclass (13 fields per Slide 9)
#   - Build mapper ParsePacket -> InternalizationPacket (lossless, reversible)
#   - Provide v15_6_write_fact_wrapped and v15_6_read_query_wrapped that:
#       1. produce InternalizationPacket from ParsePacket (trace)
#       2. delegate ENTIRELY to v15.4.1 write/read logic
#       3. return the same outputs as v15.4.1
#   - Equivalence test: for N trials, v15.6 wrapped == v15.4 pure on:
#       final bank state (slot-by-slot value_idx identical)
#       read status + predictions identical
#   - v15.4.1 remains the frozen baseline and the functional oracle.
#
# What Pas 1 deliberately does NOT do:
#   - No CommitArbiter (Pas 2)
#   - No ProvisionalMemory (Pas 2)
#   - No new acceptance logic; holdout behavior identical to v15.5 frozen run
#   - No epistemic tags (Pas 5)
#   - No entity span composition (Pas 3)
# ===========================================================================


# --------------------------- I1.1 Data structures ---------------------------


# Commit path enum: 4 possible outcomes from Internalization Layer
# (Pas 1 only uses COMMIT / PARSE_UNCERTAIN / PARSER_FAILURE — delegating to
# v15.4.1 verifier. STORE_PROVISIONAL becomes active in Pas 2.)
class CommitPath(Enum):
    COMMIT            = "COMMIT"
    STORE_PROVISIONAL = "STORE_PROVISIONAL"
    PARSE_UNCERTAIN   = "PARSE_UNCERTAIN"
    PARSER_FAILURE    = "PARSER_FAILURE"


# Epistemic status is a placeholder in Pas 1 (always "observed" for
# compatibility). Becomes meaningful in Pas 5.
class EpistemicStatus(Enum):
    OBSERVED     = "observed"
    INFERRED     = "inferred"
    REPORTED     = "reported"
    UNCERTAIN    = "uncertain"
    CONFLICTUAL  = "conflictual"
    UNKNOWN      = "unknown"


@dataclass
class InternalizationPacket:
    """Rich internal representation of external evidence BEFORE commit.
    
    This is the canonical currency of the Internalization Layer. Pas 1
    constructs this purely from a ParsePacket (lossless mapping). Later
    passes will populate fields that Pas 1 leaves as defaults (epistemic,
    discourse_links, conflict_flags beyond what verifier already reports).
    
    Fields match Slide 9 of v15.6 architecture directive.
    """
    # 1. Operation type (mapped from ParsePacket.op_type)
    op_type:               OpType
    
    # 2. Entity spans (Pas 1: derived from ParsePacket.entity_candidates,
    # each with a pseudo-span. Pas 3 replaces with true character spans.)
    entity_spans:          List[Tuple[int, int]] = field(default_factory=list)
    
    # 3. Entity hypotheses (Pas 1: copy of ParsePacket.entity_candidates)
    entity_hypotheses:     List[Tuple[str, float, Tuple[int, int]]] = field(default_factory=list)
    
    # 4. Attribute hypotheses (Pas 1: copy of ParsePacket.attribute_candidates)
    attribute_hypotheses:  List[Tuple[str, float, str]] = field(default_factory=list)
    
    # 5. Value hypotheses (Pas 1: copy of ParsePacket.value_candidates)
    value_hypotheses:      List[Tuple[str, int, float, Tuple[int, int]]] = field(default_factory=list)
    
    # 6. Discourse links (Pas 1: empty. Pas 6 populates.)
    discourse_links:       List[Dict] = field(default_factory=list)
    
    # 7. Semantic confidence (Pas 1: copy of ParsePacket.certainty)
    semantic_confidence:   float = 0.0
    
    # 8. Epistemic status (Pas 1: OBSERVED for all. Pas 5 populates real values.)
    epistemic_status:      EpistemicStatus = EpistemicStatus.OBSERVED
    
    # 9. Ambiguity flags (Pas 1: copy of ParsePacket.ambiguity_flags as strings)
    ambiguity_flags:       Set[str] = field(default_factory=set)
    
    # 10. Conflict flags (Pas 1: derived from ambiguity_flags intersecting
    # known conflict markers. Pas 4 expands.)
    conflict_flags:        Set[str] = field(default_factory=set)
    
    # 11. Source trace (Pas 1: ParsePacket.source_text + parser version)
    source_trace:          Dict = field(default_factory=dict)
    
    # 12. Scope tag (Pas 1: "default". Reserved for context scoping.)
    scope_tag:             str = "default"
    
    # 13. Time tag (Pas 1: step counter if available; else "unspecified")
    time_tag:              str = "unspecified"
    
    # Provenance back-reference (Pas 1: the originating ParsePacket for
    # reversibility checks). Not part of Slide 9 but useful for equivalence
    # testing.
    source_parse_packet:   Optional["ParsePacket"] = None
    
    # Commit path decision (Pas 1: derived from verifier status 1:1;
    # Pas 2 introduces real arbitration)
    commit_path:           Optional[CommitPath] = None


# Conflict markers recognized in Pas 1 (derived from v15.4.1 flags)
V15_6_CONFLICT_MARKERS = frozenset({
    "ATTR_CONFLICT_STRONG",
    "MULTI_FAMILY_COMPETITION",
    "MULTIPLE_ATTR_TRIGGERS",
    "ATTR_VALUE_MISMATCH",
    "REFERENT_AMBIGUOUS",
})


# --------------------------- I1.2 Mapper ------------------------------------


def parse_packet_to_internalization_packet(pp: "ParsePacket",
                                              step: Optional[int] = None
                                              ) -> InternalizationPacket:
    """Lossless mapping from ParsePacket to InternalizationPacket.
    
    Pas 1 guarantee: every field in ParsePacket that influences downstream
    bank state has a corresponding field in the InternalizationPacket.
    The mapping is purely structural (no semantic reinterpretation).
    """
    # Entity spans from entity_candidates (each candidate carries its own span)
    entity_spans = [span for (_, _, span) in pp.entity_candidates]
    
    # Flags as strings (v15.4 flags are Enum instances, some are v15.2 enum,
    # some are v15.4 extensions; both have .value)
    flags_as_strings = set()
    for f in pp.ambiguity_flags:
        if hasattr(f, "value"):
            flags_as_strings.add(f.value)
        else:
            flags_as_strings.add(str(f))
    
    # Conflict flags = intersection with known conflict markers
    conflict_flags = flags_as_strings & V15_6_CONFLICT_MARKERS
    
    # Source trace
    source_trace = {
        "source_text":       pp.source_text,
        "source_kind":       pp.source_kind,
        "parser_evidence":   dict(pp.parser_evidence) if pp.parser_evidence else {},
        "op_type_confidence": pp.op_type_confidence,
    }
    
    # Time tag
    time_tag = f"step_{step}" if step is not None else "unspecified"
    
    return InternalizationPacket(
        op_type=pp.op_type,
        entity_spans=entity_spans,
        entity_hypotheses=list(pp.entity_candidates),
        attribute_hypotheses=list(pp.attribute_candidates),
        value_hypotheses=list(pp.value_candidates),
        discourse_links=[],  # Pas 6 populates
        semantic_confidence=pp.certainty,
        epistemic_status=EpistemicStatus.OBSERVED,  # Pas 5 refines
        ambiguity_flags=flags_as_strings,
        conflict_flags=conflict_flags,
        source_trace=source_trace,
        scope_tag="default",
        time_tag=time_tag,
        source_parse_packet=pp,
        commit_path=None,  # filled by arbiter below
    )


def attach_pas1_commit_path(ip: InternalizationPacket,
                              verifier_status: "VerificationStatus"
                              ) -> InternalizationPacket:
    """Pas 1 arbiter: 1-to-1 translation from verifier status to commit path.
    
    - ACCEPT            -> COMMIT
    - PARSE_UNCERTAIN   -> PARSE_UNCERTAIN
    - PARSER_FAILURE    -> PARSER_FAILURE
    
    STORE_PROVISIONAL is deliberately unused in Pas 1 (introduced in Pas 2).
    This preserves exact v15.4.1 behavior.
    """
    if verifier_status == VerificationStatus.ACCEPT:
        ip.commit_path = CommitPath.COMMIT
    elif verifier_status == VerificationStatus.PARSE_UNCERTAIN:
        ip.commit_path = CommitPath.PARSE_UNCERTAIN
    elif verifier_status == VerificationStatus.PARSER_FAILURE:
        ip.commit_path = CommitPath.PARSER_FAILURE
    else:
        ip.commit_path = CommitPath.PARSER_FAILURE
    return ip


# --------------------------- I1.3 Wrapped write/read -------------------------
#
# These wrappers produce InternalizationPacket as a side effect for
# traceability but delegate 100% of the decision to v15.4.1 logic.
# ---------------------------------------------------------------------------


def v15_6_write_fact_wrapped(bank: "DeterministicObjectBank",
                                packet: "ParsePacket",
                                entity_emb_fn,
                                class_emb_fn,
                                value_emb_fn,
                                step: int = 0,
                                ip_log: Optional[List[InternalizationPacket]] = None
                                ):
    """Wrapped v15.4.1 write. Produces IP trace. Delegates decision to v15.4.
    
    Returns the exact same WriteResult as v15_4_write_fact.
    If ip_log is provided, the constructed InternalizationPacket is appended.
    """
    # Build IP from packet (before calling v15.4 write)
    ip = parse_packet_to_internalization_packet(packet, step=step)
    
    # v15.4.1 is the oracle. Call it as-is.
    write_result = v15_4_write_fact(bank, packet, entity_emb_fn,
                                       class_emb_fn, value_emb_fn, step=step)
    
    # Attach verifier-derived commit path to IP (for tracing only;
    # no influence on behavior)
    vr = write_result.verifier_result if write_result.verifier_result else None
    if vr is not None:
        ip = attach_pas1_commit_path(ip, vr.status)
    else:
        # verifier_result is None only when extractor did not even try
        ip.commit_path = CommitPath.PARSER_FAILURE
    
    if ip_log is not None:
        ip_log.append(ip)
    
    return write_result


def v15_6_read_query_wrapped(bank: "DeterministicObjectBank",
                                packet: "ParsePacket",
                                ip_log: Optional[List[InternalizationPacket]] = None
                                ):
    """Wrapped v15.4.1 read. Produces IP trace. Delegates to v15.4."""
    ip = parse_packet_to_internalization_packet(packet, step=None)
    
    status, pred, vr = v15_4_read_query(bank, packet)
    
    if vr is not None:
        ip = attach_pas1_commit_path(ip, vr.status)
    else:
        # When read_query returns without verifier (shouldn't happen in v15.4)
        ip.commit_path = (CommitPath.COMMIT
                            if status == READ_STATUS_FOUND
                            else CommitPath.PARSER_FAILURE)
    
    if ip_log is not None:
        ip_log.append(ip)
    
    return status, pred, vr


print("[v15.6] Pas 1 Section I1: InternalizationPacket + structural wrapper")
print("        - 13-field IP dataclass defined (Slide 9 compliant)")
print("        - 4 CommitPath values (only 3 active in Pas 1)")
print("        - parse_packet_to_internalization_packet: lossless mapping")
print("        - v15_6_write_fact_wrapped / v15_6_read_query_wrapped")
print("        - v15.4.1 remains frozen oracle; zero behavioral change")
# --------------------------- I2. EQUIVALENCE TEST --------------------------
#
# Pas 1 acceptance criterion: v15.6 wrapped path produces IDENTICAL bank state
# and IDENTICAL read outputs vs v15.4.1 pure path on every trial.
#
# If even one trial diverges, Pas 1 is broken (the wrapper is not a wrapper).
# ---------------------------------------------------------------------------


def _v15_6_snapshot_bank(bank: "DeterministicObjectBank") -> Dict:
    """Capture a deterministic signature of bank state for diffing."""
    snap = {}
    for slot_id in bank.occupied_slots():
        rec = bank.get_record(slot_id)
        attrs = {}
        for attr_type, slot_obj in rec.attr_slots.items():
            if slot_obj is None or not slot_obj.present:
                continue
            attrs[attr_type] = {
                "value_idx":   slot_obj.value_idx,
                "write_step":  slot_obj.write_step,
            }
        snap[rec.entity_id] = {
            "slot_id":      slot_id,
            "class_id":     rec.class_id,
            "attrs":        attrs,
        }
    return snap


def _v15_6_diff_snapshots(snap_a: Dict, snap_b: Dict) -> List[str]:
    """Return list of human-readable differences between two bank snapshots."""
    diffs = []
    keys_a = set(snap_a.keys())
    keys_b = set(snap_b.keys())
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    if only_a:
        diffs.append(f"only_in_v15_4: {sorted(only_a)}")
    if only_b:
        diffs.append(f"only_in_v15_6: {sorted(only_b)}")
    for k in keys_a & keys_b:
        a, b = snap_a[k], snap_b[k]
        if a["class_id"] != b["class_id"]:
            diffs.append(f"{k}.class_id: {a['class_id']} vs {b['class_id']}")
        if set(a["attrs"].keys()) != set(b["attrs"].keys()):
            diffs.append(f"{k}.attrs keys differ: {set(a['attrs'])} vs {set(b['attrs'])}")
        for attr_name in set(a["attrs"]) & set(b["attrs"]):
            ia, ib = a["attrs"][attr_name]["value_idx"], b["attrs"][attr_name]["value_idx"]
            if ia != ib:
                diffs.append(f"{k}.{attr_name}.value_idx: {ia} vs {ib}")
    return diffs


def v15_6_pas1_equivalence_test(base_model, v15_1_memory, n_trials: int = 500,
                                   seed: int = 20261001):
    """Run N v15-style episodes through both v15.4.1 and v15.6 wrapped path.
    
    Asserts that for every trial:
      - Final bank snapshots are identical (entity-by-entity, attr-by-attr)
      - Read query produces identical (status, pred, verifier reasons)
    
    If any divergence: Pas 1 FAILED.
    """
    print()
    print(SEP)
    print(f"[v15.6 PAS 1 EQUIVALENCE TEST] n_trials={n_trials} seed={seed}")
    print(SEP)
    
    rng = _rng_module.Random(seed)
    ent_fn = _make_entity_emb_fn(base_model)
    cls_fn = _make_class_emb_fn(v15_1_memory)
    val_fn = _make_value_emb_fn(base_model)
    
    # Try a balanced mix of episode types
    episode_types = [
        "single_attr_simple", "multi_attr_object", "selective_update",
        "no_match", "provisional_entity", "paraphrase",
        "coreference_distant", "S1", "S2", "S3", "S4",
    ]
    
    n_divergent_bank   = 0
    n_divergent_read   = 0
    divergent_examples = []
    
    bank_v15_4 = DeterministicObjectBank(capacity=64,
                                            d_model=base_model.config.hidden_dim)
    bank_v15_6 = DeterministicObjectBank(capacity=64,
                                            d_model=base_model.config.hidden_dim)
    
    for trial in range(n_trials):
        ep_type = episode_types[trial % len(episode_types)]
        try:
            ep = v15_generate_episode(ep_type, rng,
                                        use_heldout=(trial % 3 == 0))
        except Exception:
            continue
        
        bank_v15_4.reset()
        bank_v15_6.reset()
        ip_log = []
        
        # Write facts through both paths
        for j, fact_text in enumerate(ep.facts):
            pkt_v154 = v15_4_parse_fact(fact_text)
            pkt_v156 = v15_4_parse_fact(fact_text)  # same parser; wrapper doesn't change parser
            v15_4_write_fact(bank_v15_4, pkt_v154, ent_fn, cls_fn, val_fn, step=j)
            v15_6_write_fact_wrapped(bank_v15_6, pkt_v156, ent_fn, cls_fn,
                                       val_fn, step=j, ip_log=ip_log)
        
        # Read query through both paths
        pkt_q4 = v15_4_parse_query(ep.query)
        pkt_q6 = v15_4_parse_query(ep.query)
        st4, pr4, vr4 = v15_4_read_query(bank_v15_4, pkt_q4)
        st6, pr6, vr6 = v15_6_read_query_wrapped(bank_v15_6, pkt_q6, ip_log=ip_log)
        
        # Compare bank snapshots
        snap4 = _v15_6_snapshot_bank(bank_v15_4)
        snap6 = _v15_6_snapshot_bank(bank_v15_6)
        bank_diffs = _v15_6_diff_snapshots(snap4, snap6)
        if bank_diffs:
            n_divergent_bank += 1
            if len(divergent_examples) < 5:
                divergent_examples.append({
                    "trial":      trial, "ep_type": ep_type,
                    "kind":       "bank",
                    "facts":      ep.facts, "query": ep.query,
                    "diffs":      bank_diffs,
                })
        
        # Compare read outputs
        reasons4 = sorted([r.value for r in (vr4.reasons if vr4 else [])])
        reasons6 = sorted([r.value for r in (vr6.reasons if vr6 else [])])
        if (st4, pr4, reasons4) != (st6, pr6, reasons6):
            n_divergent_read += 1
            if len(divergent_examples) < 5:
                divergent_examples.append({
                    "trial":      trial, "ep_type": ep_type,
                    "kind":       "read",
                    "facts":      ep.facts, "query": ep.query,
                    "v15_4":      (st4, pr4, reasons4),
                    "v15_6":      (st6, pr6, reasons6),
                })
    
    print(f"  Trials run:               {n_trials}")
    print(f"  Bank divergences:         {n_divergent_bank}")
    print(f"  Read divergences:         {n_divergent_read}")
    
    if n_divergent_bank == 0 and n_divergent_read == 0:
        print(f"  VERDICT: PAS 1 EQUIVALENCE PASS")
        return {"pass": True, "n_trials": n_trials,
                "n_bank_div": 0, "n_read_div": 0, "examples": []}
    
    print(f"  VERDICT: PAS 1 EQUIVALENCE FAIL")
    for ex in divergent_examples:
        print(f"  --- Example trial {ex['trial']} ({ex['ep_type']}, {ex['kind']}) ---")
        for f in ex.get("facts", []):
            print(f"      fact: {f}")
        print(f"      query: {ex.get('query')}")
        if ex["kind"] == "bank":
            for d in ex["diffs"]:
                print(f"      diff: {d}")
        else:
            print(f"      v15.4: {ex['v15_4']}")
            print(f"      v15.6: {ex['v15_6']}")
    
    return {"pass": False, "n_trials": n_trials,
            "n_bank_div": n_divergent_bank,
            "n_read_div": n_divergent_read,
            "examples": divergent_examples}


print("[v15.6] Pas 1 Section I2: equivalence test defined")
print("        - n=500 default trials across 11 episode types")
print("        - bank snapshot diff (entity-by-entity, attr-by-attr)")
print("        - read divergence: (status, pred, reasons) tuple identity")
print("        - Pas 1 PASS iff zero divergences on both axes")

# ===========================================================================
# v15.6 PAS 2 — Section P2A: ProvisionalMemory + EpisodeBuffer
# ===========================================================================
# 
# Two new state structures sit ALONGSIDE the deterministic bank, never inside
# it. Bank remains the frozen "raft" — only finalized episodes write into it.
#
# DESIGN INVARIANTS (per GPT directive):
#   - episode_id is EXPLICIT on every write
#   - end_episode(episode_id) is OBLIGATORY before any state becomes stable
#   - writes within episode go to EpisodeBuffer, never directly to bank
#   - committed_stable only exists AFTER end_episode()
#   - Pas 2 does NOT promote provisional to committed (Replay = Pas 7)
# ===========================================================================


@dataclass
class ProvisionalEntry:
    """Single entry in provisional memory.
    
    Provenance: every entry carries the episode_id it came from, the source
    text, and a reference to its InternalizationPacket.
    """
    entity_id:        str
    attr_type:        str
    value_idx:        int
    episode_id:       int
    write_step:       int
    source_text:      str
    internalization_packet_ref: Optional["InternalizationPacket"] = None
    challenge_kind:   str = "empty_slot_conflict"
    # challenge_kind ∈ {"empty_slot_conflict", "challenger_to_stable"}


class ProvisionalMemory:
    """Buffer for facts that the CommitArbiter refused to commit.
    
    NOT a slot-replacement for the bank — a SEPARATE store, indexed by
    (entity_id, attr_type). Multiple entries per slot are normal: that's the
    point. Pas 2 only accumulates here. Promotion is Pas 7.
    """
    
    def __init__(self):
        self.entries: List[ProvisionalEntry] = []
    
    def reset(self):
        self.entries = []
    
    def add(self, entry: ProvisionalEntry):
        self.entries.append(entry)
    
    def query(self, entity_id: str, attr_type: str) -> List[ProvisionalEntry]:
        """Return all provisional entries matching this slot."""
        return [e for e in self.entries
                  if e.entity_id == entity_id and e.attr_type == attr_type]
    
    def has_challenger(self, entity_id: str, attr_type: str) -> bool:
        """True if any challenger exists for this slot."""
        return len(self.query(entity_id, attr_type)) > 0
    
    def values_for(self, entity_id: str, attr_type: str) -> List[int]:
        """Distinct value indices in provisional for this slot."""
        seen = []
        for e in self.query(entity_id, attr_type):
            if e.value_idx not in seen:
                seen.append(e.value_idx)
        return seen
    
    def episodes_for(self, entity_id: str, attr_type: str) -> Set[int]:
        return {e.episode_id for e in self.query(entity_id, attr_type)}


@dataclass
class BufferedWrite:
    """A single fact write captured during an active episode."""
    entity_id:    str
    attr_type:    str
    value_idx:    int
    write_step:   int
    source_text:  str
    parse_packet: "ParsePacket"
    internalization_packet: "InternalizationPacket"
    # Embeddings precomputed at parse time (so end_episode commit doesn't
    # need to re-parse)
    entity_emb_cache: Optional[torch.Tensor] = None
    class_id_cache:   int = -1
    class_emb_cache:  Optional[torch.Tensor] = None
    value_emb_cache:  Optional[torch.Tensor] = None


class EpisodeBuffer:
    """Holds the facts of an active (not-yet-ended) episode.
    
    The bank does not see these writes until end_episode() finalizes them.
    """
    
    def __init__(self):
        self.episode_id: Optional[int] = None
        self.buffered:    List[BufferedWrite] = []
        self.is_active:   bool = False
    
    def begin_episode(self, episode_id: int):
        if self.is_active:
            raise RuntimeError(
                f"EpisodeBuffer.begin_episode called while episode "
                f"{self.episode_id} is still active"
            )
        self.episode_id = episode_id
        self.buffered   = []
        self.is_active  = True
    
    def add_write(self, w: BufferedWrite):
        if not self.is_active:
            raise RuntimeError(
                "EpisodeBuffer.add_write called outside an active episode"
            )
        self.buffered.append(w)
    
    def get_writes(self) -> List[BufferedWrite]:
        return list(self.buffered)
    
    def group_by_slot(self) -> Dict[Tuple[str, str], List[BufferedWrite]]:
        """Group buffered writes by (entity_id, attr_type)."""
        out: Dict[Tuple[str, str], List[BufferedWrite]] = {}
        for w in self.buffered:
            key = (w.entity_id, w.attr_type)
            out.setdefault(key, []).append(w)
        return out
    
    def end_episode(self) -> int:
        """Mark episode as inactive. Returns the episode_id that just ended."""
        if not self.is_active:
            raise RuntimeError("EpisodeBuffer.end_episode called with no active episode")
        ended = self.episode_id
        self.is_active = False
        return ended
    
    def clear(self):
        """Hard reset (used between unrelated runs)."""
        self.episode_id = None
        self.buffered   = []
        self.is_active  = False


# Bank slot stability tracking. We need to know which slots in the bank were
# committed in which episode_id. Pas 2 invariant: a slot is "stable" iff it
# was committed by an end_episode() call from a PREVIOUS episode_id.
class BankStabilityIndex:
    """Tracks which (entity_id, attr_type) slots were committed by which
    episode. Same-episode commits do NOT count as stable until end_episode.
    """
    
    def __init__(self):
        # (entity_id, attr_type) -> episode_id when committed
        self.committed_episode: Dict[Tuple[str, str], int] = {}
    
    def mark_committed(self, entity_id: str, attr_type: str, episode_id: int):
        self.committed_episode[(entity_id, attr_type)] = episode_id
    
    def is_stable(self, entity_id: str, attr_type: str,
                    current_episode_id: int) -> bool:
        """A slot is stable if it was committed in a PRIOR episode."""
        ep_committed = self.committed_episode.get((entity_id, attr_type))
        if ep_committed is None:
            return False
        return ep_committed < current_episode_id
    
    def episode_of(self, entity_id: str, attr_type: str) -> Optional[int]:
        return self.committed_episode.get((entity_id, attr_type))
    
    def reset(self):
        self.committed_episode = {}


print("[v15.6 Pas 2] Section P2A: ProvisionalMemory + EpisodeBuffer + BankStability")
print("        - ProvisionalEntry: 8-field provenance dataclass")
print("        - ProvisionalMemory: indexed by (entity_id, attr_type)")
print("        - BufferedWrite + EpisodeBuffer with begin/add/end protocol")
print("        - BankStabilityIndex: tracks per-slot commit episode")
# ===========================================================================
# v15.6 PAS 2 — Section P2B: CommitArbiter + ReadArbiter
# ===========================================================================
#
# CommitArbiter intercepts every write between v15.4 verifier and bank.
# It NEVER writes directly to bank during an episode. Instead it routes to
# EpisodeBuffer or ProvisionalMemory, depending on verifier result.
#
# At end_episode(), CommitArbiter applies the dual conflict rule:
#   - empty stable slot + same-episode conflict → ALL values to provisional
#   - stable committed slot + same-episode conflict → bank keeps stable,
#     all conflicting buffered values become challengers
#   - cross-episode conflict (different value vs stable) → challenger only
#
# ReadArbiter consults bank, ProvisionalMemory, and BankStabilityIndex to
# decide between FOUND_COMMITTED, FOUND_DISPUTED, NONE_*, PARSE_UNCERTAIN.
# ===========================================================================


# ---- New read status for disputed slots ----
READ_STATUS_FOUND_COMMITTED = "FOUND"           # alias of READ_STATUS_FOUND
READ_STATUS_FOUND_DISPUTED  = "FOUND_DISPUTED"  # bank+provisional disagree


# ---- Arbitrated write outcome ----
@dataclass
class ArbitratedWriteResult:
    """Outcome of arbitrated write, BEFORE end_episode().
    
    During an active episode, CommitArbiter only buffers or rejects; it
    cannot finalize. The actual commit decisions happen in end_episode.
    """
    commit_path:     CommitPath           # COMMIT-PENDING / STORE_PROVISIONAL /
                                            # PARSE_UNCERTAIN / PARSER_FAILURE
    buffered:        bool                 # True if went into EpisodeBuffer
    provisional:     bool                 # True if went into ProvisionalMemory
    rejected:        bool                 # True if PARSE_UNCERTAIN / PARSER_FAILURE
    parse_packet:    Optional["ParsePacket"]
    verifier_result: Optional["VerificationResult"]
    internalization_packet: Optional["InternalizationPacket"]


# ---- Episode finalization outcome ----
@dataclass
class EpisodeFinalizationResult:
    """Returned by end_episode(). Reports per-slot decisions."""
    episode_id:           int
    n_buffered:           int
    n_committed:          int          # slots that got committed to bank
    n_provisional:        int          # entries placed into ProvisionalMemory
    n_empty_slot_conflict: int         # slots with conflict & empty stable
    n_stable_slot_conflict: int        # slots with conflict & existing stable
    decisions_per_slot:   Dict[Tuple[str, str], str]
    # decision ∈ {"committed_clean", "empty_slot_conflict_to_provisional",
    #             "stable_kept_challenger_to_provisional",
    #             "cross_episode_challenger_to_provisional"}


class CommitArbiter:
    """Pre-commit arbitration layer. Runs strictly BEFORE bank writes.
    
    Held INVARIANTS:
      - bank is never touched directly during an active episode
      - every write is routed: EpisodeBuffer | ProvisionalMemory | rejected
      - end_episode() is the only place where bank writes happen
    """
    
    def __init__(self,
                 bank: "DeterministicObjectBank",
                 provisional_memory: ProvisionalMemory,
                 episode_buffer: EpisodeBuffer,
                 stability_index: BankStabilityIndex):
        self.bank = bank
        self.provisional_memory = provisional_memory
        self.episode_buffer     = episode_buffer
        self.stability_index    = stability_index
    
    # -------------------------------------------------------------------
    # Episode lifecycle
    # -------------------------------------------------------------------
    
    def begin_episode(self, episode_id: int):
        self.episode_buffer.begin_episode(episode_id)
    
    def end_episode(self,
                     entity_emb_fn,
                     class_emb_fn,
                     value_emb_fn) -> EpisodeFinalizationResult:
        """Finalize the active episode. Apply conflict rules per slot."""
        episode_id = self.episode_buffer.episode_id
        if episode_id is None:
            raise RuntimeError("end_episode called with no active episode")
        
        groups = self.episode_buffer.group_by_slot()
        
        n_committed             = 0
        n_provisional           = 0
        n_empty_slot_conflict   = 0
        n_stable_slot_conflict  = 0
        decisions_per_slot      = {}
        
        for (entity_id, attr_type), writes in groups.items():
            distinct_values = list({w.value_idx for w in writes})
            
            if len(distinct_values) == 1:
                # No conflict. Apply standard commit: write to bank
                # (or apply UPDATE if cross-episode and same value, no-op)
                w = writes[-1]  # last write wins on identical-value within episode
                self._commit_to_bank(w)
                self.stability_index.mark_committed(entity_id, attr_type, episode_id)
                n_committed += 1
                decisions_per_slot[(entity_id, attr_type)] = "committed_clean"
                continue
            
            # Same-episode CONFLICT (multiple distinct values for same slot)
            is_stable = self.stability_index.is_stable(entity_id, attr_type,
                                                          episode_id)
            
            if not is_stable:
                # Empty stable slot: bank stays empty, ALL values go provisional
                for w in writes:
                    entry = ProvisionalEntry(
                        entity_id=entity_id, attr_type=attr_type,
                        value_idx=w.value_idx, episode_id=episode_id,
                        write_step=w.write_step, source_text=w.source_text,
                        internalization_packet_ref=w.internalization_packet,
                        challenge_kind="empty_slot_conflict",
                    )
                    self.provisional_memory.add(entry)
                    n_provisional += 1
                n_empty_slot_conflict += 1
                decisions_per_slot[(entity_id, attr_type)] = (
                    "empty_slot_conflict_to_provisional"
                )
            else:
                # Stable bank value exists. Bank keeps it; all buffered values
                # for this slot become challengers (skip duplicates of stable).
                for w in writes:
                    entry = ProvisionalEntry(
                        entity_id=entity_id, attr_type=attr_type,
                        value_idx=w.value_idx, episode_id=episode_id,
                        write_step=w.write_step, source_text=w.source_text,
                        internalization_packet_ref=w.internalization_packet,
                        challenge_kind="challenger_to_stable",
                    )
                    self.provisional_memory.add(entry)
                    n_provisional += 1
                n_stable_slot_conflict += 1
                decisions_per_slot[(entity_id, attr_type)] = (
                    "stable_kept_challenger_to_provisional"
                )
        
        ended_id = self.episode_buffer.end_episode()
        
        return EpisodeFinalizationResult(
            episode_id=ended_id,
            n_buffered=sum(len(w) for w in groups.values()),
            n_committed=n_committed,
            n_provisional=n_provisional,
            n_empty_slot_conflict=n_empty_slot_conflict,
            n_stable_slot_conflict=n_stable_slot_conflict,
            decisions_per_slot=decisions_per_slot,
        )
    
    # -------------------------------------------------------------------
    # Single-fact write (during active episode)
    # -------------------------------------------------------------------
    
    def write_fact(self,
                    fact_text: str,
                    entity_emb_fn,
                    class_emb_fn,
                    value_emb_fn,
                    write_step: int = 0) -> ArbitratedWriteResult:
        """Process one fact through the arbiter.
        
        - PARSER_FAILURE / PARSE_UNCERTAIN: rejected, nothing buffered.
        - ACCEPT: extract (entity_id, attr_type, value_idx) and buffer.
        
        Cross-episode conflict (write of different value vs stable slot in
        prior episode) is detected here BEFORE buffering, and routed
        directly to provisional as a challenger.
        """
        if not self.episode_buffer.is_active:
            raise RuntimeError(
                "CommitArbiter.write_fact called outside active episode"
            )
        episode_id = self.episode_buffer.episode_id
        
        pkt = v15_4_parse_fact(fact_text)
        ip  = parse_packet_to_internalization_packet(pkt, step=write_step)
        vr  = V15_4_VERIFIER.verify(pkt)
        
        if vr.status == VerificationStatus.PARSER_FAILURE:
            ip.commit_path = CommitPath.PARSER_FAILURE
            return ArbitratedWriteResult(
                commit_path=CommitPath.PARSER_FAILURE, buffered=False,
                provisional=False, rejected=True, parse_packet=pkt,
                verifier_result=vr, internalization_packet=ip,
            )
        
        if vr.status == VerificationStatus.PARSE_UNCERTAIN:
            ip.commit_path = CommitPath.PARSE_UNCERTAIN
            return ArbitratedWriteResult(
                commit_path=CommitPath.PARSE_UNCERTAIN, buffered=False,
                provisional=False, rejected=True, parse_packet=pkt,
                verifier_result=vr, internalization_packet=ip,
            )
        
        # ACCEPT path: extract slot intent
        entity_id = _top_entity(pkt)
        attr_type = _top_attribute(pkt)
        value_idx = _top_value_for(pkt, attr_type)
        if entity_id is None or attr_type is None or value_idx is None:
            ip.commit_path = CommitPath.PARSER_FAILURE
            return ArbitratedWriteResult(
                commit_path=CommitPath.PARSER_FAILURE, buffered=False,
                provisional=False, rejected=True, parse_packet=pkt,
                verifier_result=vr, internalization_packet=ip,
            )
        
        # Cross-episode conflict check (slot stable from prior episode)
        if self.stability_index.is_stable(entity_id, attr_type, episode_id):
            existing_slot = self.bank.find_by_entity_id(entity_id)
            if existing_slot is not None:
                rec = self.bank.get_record(existing_slot)
                slot = rec.attr_slots.get(attr_type)
                if (slot is not None and slot.present
                        and slot.value_idx != value_idx):
                    # Cross-episode challenger
                    entry = ProvisionalEntry(
                        entity_id=entity_id, attr_type=attr_type,
                        value_idx=value_idx, episode_id=episode_id,
                        write_step=write_step, source_text=fact_text,
                        internalization_packet_ref=ip,
                        challenge_kind="cross_episode_challenger",
                    )
                    self.provisional_memory.add(entry)
                    ip.commit_path = CommitPath.STORE_PROVISIONAL
                    return ArbitratedWriteResult(
                        commit_path=CommitPath.STORE_PROVISIONAL,
                        buffered=False, provisional=True, rejected=False,
                        parse_packet=pkt, verifier_result=vr,
                        internalization_packet=ip,
                    )
        
        # Otherwise buffer for end_episode finalization
        # Precompute embeddings (so end_episode can commit without re-parsing)
        try:
            ent_emb = entity_emb_fn(entity_id)
        except Exception:
            ent_emb = None
        class_id = _entity_class_id(entity_id)
        try:
            cls_emb = (class_emb_fn(class_id, ent_emb)
                          if ent_emb is not None else None)
        except Exception:
            cls_emb = None
        try:
            val_emb = value_emb_fn(attr_type, value_idx)
        except Exception:
            val_emb = None
        
        bw = BufferedWrite(
            entity_id=entity_id, attr_type=attr_type, value_idx=value_idx,
            write_step=write_step, source_text=fact_text,
            parse_packet=pkt,
            internalization_packet=ip,
            entity_emb_cache=ent_emb, class_id_cache=class_id,
            class_emb_cache=cls_emb, value_emb_cache=val_emb,
        )
        self.episode_buffer.add_write(bw)
        ip.commit_path = CommitPath.COMMIT  # tentative; finalized at end_episode
        return ArbitratedWriteResult(
            commit_path=CommitPath.COMMIT, buffered=True, provisional=False,
            rejected=False, parse_packet=pkt, verifier_result=vr,
            internalization_packet=ip,
        )
    
    # -------------------------------------------------------------------
    # Internal: write a buffered (single-value, no-conflict) write to bank
    # -------------------------------------------------------------------
    
    def _commit_to_bank(self, w: BufferedWrite):
        """Apply a single buffered write to the deterministic bank.
        
        Mimics v15_4_write_fact's bank operations but skips re-parsing,
        re-verification (already done in write_fact phase). Uses cached
        embeddings.
        """
        existing_slot = self.bank.find_by_entity_id(w.entity_id)
        if existing_slot is None:
            if w.entity_emb_cache is None:
                return  # cannot allocate without entity embedding
            try:
                self.bank.allocate_new(
                    entity_id=w.entity_id,
                    entity_emb=w.entity_emb_cache,
                    class_hint=w.class_id_cache,
                    class_emb=w.class_emb_cache,
                    step=w.write_step,
                )
            except Exception:
                return  # bank full or other allocation failure
        try:
            self.bank.write_attribute(
                entity_id=w.entity_id,
                attr_type=w.attr_type,
                value_idx=w.value_idx,
                value_emb=w.value_emb_cache,
                step=w.write_step,
            )
        except Exception:
            return  # soft-fail to match v15.4 behavior


# ----- Helpers used by CommitArbiter -----

def _top_value_for(pkt: "ParsePacket", attr_type: str) -> Optional[int]:
    """Highest-confidence value_idx for the given attr_type from the packet."""
    cands = [(idx, conf) for (a, idx, conf, _) in pkt.value_candidates
                if a == attr_type]
    if not cands:
        return None
    cands.sort(key=lambda x: -x[1])
    return cands[0][0]


def _entity_class_id(entity_id: str) -> int:
    """Lookup entity class_id (creature=0, person=1, object=2) from V15 pools."""
    for (e, cid) in V15_TRAIN_ENTITIES:
        if e == entity_id:
            return cid
    for (e, cid) in V15_HELDOUT_ENTITIES:
        if e == entity_id:
            return cid
    return 0  # default


# ===========================================================================
# ReadArbiter — consults bank, provisional_memory, stability_index
# ===========================================================================


@dataclass
class ArbitratedReadResult:
    """Read decision with multi-store visibility."""
    status:                  str
    pred:                    Optional[int]    # primary prediction (committed value)
    disputed_values:         List[int]        # if FOUND_DISPUTED
    parse_packet:            Optional["ParsePacket"]
    verifier_result:         Optional["VerificationResult"]
    internalization_packet:  Optional["InternalizationPacket"]
    source:                  str    # "bank" | "provisional" | "neither"


class ReadArbiter:
    """Read-time arbitration: decide between bank, provisional, refusal."""
    
    def __init__(self,
                 bank: "DeterministicObjectBank",
                 provisional_memory: ProvisionalMemory):
        self.bank = bank
        self.provisional_memory = provisional_memory
    
    def read_query(self, query_text: str) -> ArbitratedReadResult:
        pkt = v15_4_parse_query(query_text)
        ip  = parse_packet_to_internalization_packet(pkt)
        vr  = V15_4_VERIFIER.verify(pkt)
        
        if vr.status == VerificationStatus.PARSER_FAILURE:
            ip.commit_path = CommitPath.PARSER_FAILURE
            return ArbitratedReadResult(
                status=READ_STATUS_PARSER_FAIL, pred=None, disputed_values=[],
                parse_packet=pkt, verifier_result=vr,
                internalization_packet=ip, source="neither",
            )
        if vr.status == VerificationStatus.PARSE_UNCERTAIN:
            ip.commit_path = CommitPath.PARSE_UNCERTAIN
            return ArbitratedReadResult(
                status=READ_STATUS_PARSE_UNCERTAIN, pred=None,
                disputed_values=[], parse_packet=pkt, verifier_result=vr,
                internalization_packet=ip, source="neither",
            )
        
        # Verifier ACCEPT
        entity_id = _top_entity(pkt)
        attr_type = _top_attribute(pkt)
        if entity_id is None or attr_type is None:
            return ArbitratedReadResult(
                status=READ_STATUS_PARSER_FAIL, pred=None, disputed_values=[],
                parse_packet=pkt, verifier_result=vr,
                internalization_packet=ip, source="neither",
            )
        
        # Read bank state
        bank_slot = self.bank.find_by_entity_id(entity_id)
        bank_value: Optional[int] = None
        if bank_slot is not None:
            rec = self.bank.get_record(bank_slot)
            slot = rec.attr_slots.get(attr_type)
            if slot is not None and slot.present:
                bank_value = slot.value_idx
        
        # Read provisional state for this slot
        provisional_values = self.provisional_memory.values_for(entity_id,
                                                                   attr_type)
        
        # Decision tree
        if bank_value is None and not provisional_values:
            # Slot completely empty
            entity_present_at_all = bank_slot is not None
            status = (READ_STATUS_NONE_ATTRIBUTE if entity_present_at_all
                        else READ_STATUS_NONE_OBJECT)
            return ArbitratedReadResult(
                status=status, pred=None, disputed_values=[],
                parse_packet=pkt, verifier_result=vr,
                internalization_packet=ip, source="neither",
            )
        
        if bank_value is None and provisional_values:
            # Empty stable slot, but provisional has entries.
            # If multiple distinct provisional values: DISPUTED
            # If single provisional value: still DISPUTED (since it was placed
            # in provisional, it MEANS arbiter found conflict at write time)
            return ArbitratedReadResult(
                status=READ_STATUS_FOUND_DISPUTED, pred=None,
                disputed_values=provisional_values, parse_packet=pkt,
                verifier_result=vr, internalization_packet=ip,
                source="provisional",
            )
        
        if bank_value is not None and not provisional_values:
            # Clean committed value, no challenger
            return ArbitratedReadResult(
                status=READ_STATUS_FOUND_COMMITTED, pred=bank_value,
                disputed_values=[], parse_packet=pkt, verifier_result=vr,
                internalization_packet=ip, source="bank",
            )
        
        # bank_value is not None AND provisional_values exist
        # Per directive: bank stable wins for read, but report DISPUTED to
        # signal the challenger exists. Disputed values include bank value.
        all_disputed = [bank_value] + [v for v in provisional_values
                                          if v != bank_value]
        if len(set(all_disputed)) == 1:
            # Provisional only echoes bank value (shouldn't normally happen,
            # but if it does, treat as committed clean)
            return ArbitratedReadResult(
                status=READ_STATUS_FOUND_COMMITTED, pred=bank_value,
                disputed_values=[], parse_packet=pkt, verifier_result=vr,
                internalization_packet=ip, source="bank",
            )
        return ArbitratedReadResult(
            status=READ_STATUS_FOUND_DISPUTED, pred=bank_value,
            disputed_values=all_disputed, parse_packet=pkt,
            verifier_result=vr, internalization_packet=ip,
            source="bank+provisional",
        )


print("[v15.6 Pas 2] Section P2B: CommitArbiter + ReadArbiter")
print("        - CommitArbiter: begin/write/end with EpisodeBuffer routing")
print("        - end_episode applies dual conflict rule")
print("        - cross-episode challenger detected at write_fact time")
print("        - ReadArbiter: 6 outcomes incl. FOUND_DISPUTED")
# ===========================================================================
# v15.6 PAS 2 — Section P2C: arbitrated evaluator with new metrics
# ===========================================================================
#
# New per-family/probe metrics (per GPT directive):
#   - commit_correct_rate     : answered correctly via committed bank
#   - provisional_correct_rate: answered DISPUTED with target value among them
#   - uncertain_rate          : returned PARSE_UNCERTAIN / NONE_*
#   - wrong_commit_rate       : committed value but it was wrong  ← Gate 1
#   - parser_failure_rate
#
# For S-probes (target_is_unknown):
#   - honesty                 : did not commit
#   - overcommit              : committed an answer when target was UNKNOWN
# ===========================================================================


@dataclass
class ArbitratedTrialOutcome:
    """One trial's outcome through the arbitrated pipeline."""
    family:                 str
    target_is_unknown:      bool
    target_value_idx:       Optional[int]
    arbitrated_status:      str
    pred_value:             Optional[int]
    disputed_values:        List[int]
    commit_path_at_write:   List[str]
    end_episode_decisions:  Dict[str, str]


def _v15_6_run_arbitrated_episode(arbiter: CommitArbiter,
                                      reader: ReadArbiter,
                                      ep,
                                      episode_id: int,
                                      entity_emb_fn,
                                      class_emb_fn,
                                      value_emb_fn) -> ArbitratedTrialOutcome:
    """Run one episode end-to-end through arbiter + reader."""
    arbiter.begin_episode(episode_id)
    
    write_paths = []
    for j, fact_text in enumerate(ep.facts):
        result = arbiter.write_fact(fact_text, entity_emb_fn, class_emb_fn,
                                       value_emb_fn, write_step=j)
        write_paths.append(result.commit_path.value)
    
    finalize = arbiter.end_episode(entity_emb_fn, class_emb_fn, value_emb_fn)
    end_decisions = {f"{k[0]}::{k[1]}": v
                       for k, v in finalize.decisions_per_slot.items()}
    
    rd = reader.read_query(ep.query)
    
    # Compute target_value_idx
    target_value_idx = None
    if not ep.target_is_unknown:
        attr_type = HOLDOUT_ATTR_TYPES[ep.query_attr_label]
        vocab = HOLDOUT_ATTR_VALUES[attr_type]
        for k, vstr in enumerate(vocab):
            if V15_ANSWER_TOKENS.get(attr_type, {}).get(vstr) == ep.target_answer_token:
                target_value_idx = k
                break
    
    return ArbitratedTrialOutcome(
        family=ep.family_tag,
        target_is_unknown=ep.target_is_unknown,
        target_value_idx=target_value_idx,
        arbitrated_status=rd.status,
        pred_value=rd.pred,
        disputed_values=rd.disputed_values,
        commit_path_at_write=write_paths,
        end_episode_decisions=end_decisions,
    )


def _v15_6_score_family(outcomes: List[ArbitratedTrialOutcome]) -> Dict:
    """Compute the 5 internalization metrics per GPT directive + S-honesty."""
    n = len(outcomes)
    n_committed_correct   = 0
    n_provisional_correct = 0
    n_uncertain           = 0
    n_wrong_commit        = 0
    n_parser_failure      = 0
    n_committed           = 0
    n_disputed            = 0
    
    # S-probe metrics
    n_unk           = 0
    n_unk_honest    = 0
    n_unk_overcommit = 0
    
    for o in outcomes:
        if o.target_is_unknown:
            n_unk += 1
            if o.arbitrated_status == READ_STATUS_FOUND_COMMITTED:
                n_unk_overcommit += 1
            else:
                n_unk_honest += 1
            # S-probes don't contribute to commit_correct etc. on accuracy axis
            continue
        
        # Knowable target
        if o.arbitrated_status == READ_STATUS_FOUND_COMMITTED:
            n_committed += 1
            if o.pred_value == o.target_value_idx:
                n_committed_correct += 1
            else:
                n_wrong_commit += 1
        elif o.arbitrated_status == READ_STATUS_FOUND_DISPUTED:
            n_disputed += 1
            if o.target_value_idx in o.disputed_values:
                n_provisional_correct += 1
            # else: neither correct nor wrong-commit (honest "I see ambiguity")
        elif o.arbitrated_status in (READ_STATUS_PARSE_UNCERTAIN,
                                        READ_STATUS_NONE_OBJECT,
                                        READ_STATUS_NONE_ATTRIBUTE):
            n_uncertain += 1
        elif o.arbitrated_status == READ_STATUS_PARSER_FAIL:
            n_parser_failure += 1
    
    n_knowable = max(1, n - n_unk)
    
    out = {
        "n":                          n,
        "n_knowable":                 n - n_unk,
        "n_unknowable":               n_unk,
        "commit_correct_rate":        n_committed_correct / n_knowable,
        "provisional_correct_rate":   n_provisional_correct / n_knowable,
        "uncertain_rate":             n_uncertain / n_knowable,
        "wrong_commit_rate":          n_wrong_commit / n_knowable,
        "parser_failure_rate":        n_parser_failure / n_knowable,
        "committed_count":            n_committed,
        "disputed_count":             n_disputed,
    }
    if n_unk > 0:
        out["honesty"]    = n_unk_honest / n_unk
        out["overcommit"] = n_unk_overcommit / n_unk
    else:
        out["honesty"]    = None
        out["overcommit"] = None
    return out


# ============== ACCEPTANCE CRITERIA (per GPT, Pas 2) =====================

V15_6_PAS2_ACCEPTANCE = {
    "wrong_commit_max_per_family":           0.02,
    "F2_correct_plus_provisional_min":       0.95,
    "F4_correct_plus_provisional_min":       0.99,
    "F1_F3_F5_correct_plus_provisional_min": 0.85,
    "S5_honesty_min":                        0.95,
    "S5_overcommit_max":                     0.02,
    "S6_honesty_min":                        0.95,
    "S6_overcommit_max":                     0.02,
}


# ============== Main runner =============================================

def v15_6_pas2_run_arbitrated_holdout(bank: "DeterministicObjectBank",
                                          base_model,
                                          v15_1_memory):
    """Run external holdout through CommitArbiter + ReadArbiter pipeline."""
    print()
    print(SEP)
    print("[v15.6 PAS 2 ARBITRATED HOLDOUT]")
    print(SEP)
    
    ent_fn = _make_entity_emb_fn(base_model)
    cls_fn = _make_class_emb_fn(v15_1_memory)
    val_fn = _make_value_emb_fn(base_model)
    class_map = _v15_5_build_class_map()
    
    # Snapshot trusted BEFORE
    print("\n[v15.6 Pas 2] Trusted snapshot BEFORE (regression gate)")
    snap_before = _v15_5_snapshot_trusted(bank, base_model, v15_1_memory)
    for k, v in snap_before.items():
        print(f"  {k}: {v:.4f}")
    
    # Fresh state for arbitrated run
    bank.reset()
    provisional_memory = ProvisionalMemory()
    episode_buffer     = EpisodeBuffer()
    stability_index    = BankStabilityIndex()
    arbiter = CommitArbiter(bank, provisional_memory, episode_buffer,
                              stability_index)
    reader  = ReadArbiter(bank, provisional_memory)
    
    # Run all families and S-probes
    family_results = {}
    seed_offset = 100000
    episode_counter = 1
    
    print("\n[v15.6 Pas 2] Running 5 holdout families (n={} each)".format(
        V15_5_HOLDOUT_CONFIG["n_per_family"]))
    for fname, gen in EXTERNAL_HOLDOUT_FAMILIES.items():
        print(f"  -> {fname}")
        rng = _rng_module.Random(V15_5_HOLDOUT_CONFIG["seed"] + seed_offset)
        outcomes = []
        for trial in range(V15_5_HOLDOUT_CONFIG["n_per_family"]):
            ep = gen(rng, ENC, class_map)
            # Each holdout episode is a fresh episode_id
            # But also reset bank between trials (no cross-trial state)
            bank.reset()
            provisional_memory.reset()
            episode_buffer.clear()
            stability_index.reset()
            o = _v15_6_run_arbitrated_episode(arbiter, reader, ep,
                                                  episode_counter,
                                                  ent_fn, cls_fn, val_fn)
            outcomes.append(o)
            episode_counter += 1
        scored = _v15_6_score_family(outcomes)
        family_results[fname] = scored
        print(f"     commit_correct={scored['commit_correct_rate']:.3f} "
              f"prov_correct={scored['provisional_correct_rate']:.3f} "
              f"unc={scored['uncertain_rate']:.3f} "
              f"wrong_commit={scored['wrong_commit_rate']:.3f} "
              f"parser_fail={scored['parser_failure_rate']:.3f}")
        seed_offset += 1000
    
    s_results = {}
    print("\n[v15.6 Pas 2] Running S-probes (n={} each)".format(
        V15_5_HOLDOUT_CONFIG["n_per_s_probe"]))
    for sname, gen in EXTERNAL_HOLDOUT_S_PROBES.items():
        print(f"  -> {sname}")
        rng = _rng_module.Random(V15_5_HOLDOUT_CONFIG["seed"] + seed_offset)
        outcomes = []
        for trial in range(V15_5_HOLDOUT_CONFIG["n_per_s_probe"]):
            ep = gen(rng, ENC, class_map)
            bank.reset()
            provisional_memory.reset()
            episode_buffer.clear()
            stability_index.reset()
            o = _v15_6_run_arbitrated_episode(arbiter, reader, ep,
                                                  episode_counter,
                                                  ent_fn, cls_fn, val_fn)
            outcomes.append(o)
            episode_counter += 1
        scored = _v15_6_score_family(outcomes)
        s_results[sname] = scored
        print(f"     honesty={scored['honesty']:.3f} "
              f"overcommit={scored['overcommit']:.3f} "
              f"unc={scored['uncertain_rate']:.3f}")
        seed_offset += 1000
    
    # Snapshot trusted AFTER
    print("\n[v15.6 Pas 2] Trusted snapshot AFTER (regression gate)")
    bank.reset()
    snap_after = _v15_5_snapshot_trusted(bank, base_model, v15_1_memory)
    match, bad_k, vb, va = _v15_5_trusted_signatures_match(snap_before,
                                                              snap_after)
    if match:
        print("  PASS: trusted regression check")
    else:
        print(f"  FAIL: regression on '{bad_k}': before={vb} after={va}")
    
    # Acceptance evaluation
    print("\n" + SEP)
    print("=== V15.6 PAS 2 ACCEPTANCE ===")
    print(SEP)
    
    A = V15_6_PAS2_ACCEPTANCE
    checks = {}
    
    # Gate 0: trusted regression
    checks["Gate 0: trusted regression"] = match
    
    # Gate 1: harmful commit on every family
    for fname, r in family_results.items():
        checks[f"Gate 1: {fname} wrong_commit <= {A['wrong_commit_max_per_family']:.2f}"] = (
            r["wrong_commit_rate"] <= A["wrong_commit_max_per_family"]
        )
    
    # Gate 3: per-family targets
    f2 = family_results.get("F2_multiword_entities", {})
    f2_total = f2.get("commit_correct_rate", 0) + f2.get("provisional_correct_rate", 0)
    checks[f"Gate 3: F2 correct+prov >= {A['F2_correct_plus_provisional_min']:.2f}"] = (
        f2_total >= A["F2_correct_plus_provisional_min"]
    )
    
    f4 = family_results.get("F4_discourse_intercalation", {})
    f4_total = f4.get("commit_correct_rate", 0) + f4.get("provisional_correct_rate", 0)
    checks[f"Gate 3: F4 correct+prov >= {A['F4_correct_plus_provisional_min']:.2f}"] = (
        f4_total >= A["F4_correct_plus_provisional_min"]
    )
    
    for fname in ["F1_novel_paraphrase_syntax", "F3_novel_lexical_alias",
                    "F5_novel_query_forms"]:
        r = family_results.get(fname, {})
        total = r.get("commit_correct_rate", 0) + r.get("provisional_correct_rate", 0)
        checks[f"Gate 3: {fname} correct+prov >= {A['F1_F3_F5_correct_plus_provisional_min']:.2f}"] = (
            total >= A["F1_F3_F5_correct_plus_provisional_min"]
        )
    
    # S-probes
    s5 = s_results.get("S5_conflict_intercalated", {})
    checks[f"Gate 3: S5 honesty >= {A['S5_honesty_min']:.2f}"] = (
        (s5.get("honesty") if s5.get("honesty") is not None else 0) >= A["S5_honesty_min"]
    )
    checks[f"Gate 3: S5 overcommit <= {A['S5_overcommit_max']:.2f}"] = (
        (s5.get("overcommit") if s5.get("overcommit") is not None else 1) <= A["S5_overcommit_max"]
    )
    s6 = s_results.get("S6_entity_competition_cross", {})
    checks[f"Gate 3: S6 honesty >= {A['S6_honesty_min']:.2f}"] = (
        (s6.get("honesty") if s6.get("honesty") is not None else 0) >= A["S6_honesty_min"]
    )
    checks[f"Gate 3: S6 overcommit <= {A['S6_overcommit_max']:.2f}"] = (
        (s6.get("overcommit") if s6.get("overcommit") is not None else 1) <= A["S6_overcommit_max"]
    )
    
    all_pass = all(checks.values())
    for name, ok in checks.items():
        print(f"  {'✓' if ok else '✗'} {name}")
    print()
    print(f"  Verdict: {'PAS 2 PASSED' if all_pass else 'PAS 2 PARTIAL — see failures'}")
    
    print()
    print(SEP)
    print("=== PER-FAMILY BREAKDOWN ===")
    print(SEP)
    print(f"  {'family':35s}  {'commit_corr':>11s} {'prov_corr':>10s} "
          f"{'uncertain':>10s} {'wrong_commit':>12s} {'parser_fail':>11s}")
    for f, r in family_results.items():
        print(f"  {f:35s}  {r['commit_correct_rate']:>11.3f} "
              f"{r['provisional_correct_rate']:>10.3f} "
              f"{r['uncertain_rate']:>10.3f} {r['wrong_commit_rate']:>12.3f} "
              f"{r['parser_failure_rate']:>11.3f}")
    print()
    print(f"  {'S-probe':35s}  {'honesty':>10s} {'overcommit':>10s} "
          f"{'uncertain':>10s}")
    for s, r in s_results.items():
        print(f"  {s:35s}  {r['honesty']:>10.3f} {r['overcommit']:>10.3f} "
              f"{r['uncertain_rate']:>10.3f}")
    print(SEP)
    
    return {
        "snap_before":            snap_before,
        "snap_after":             snap_after,
        "trusted_regression_ok":  match,
        "family_results":         family_results,
        "s_results":              s_results,
        "checks":                 checks,
        "all_pass":               all_pass,
    }


print("[v15.6 Pas 2] Section P2C: arbitrated evaluator + acceptance gates")
print("        - 5 metrics per family: commit_correct, prov_correct,")
print("          uncertain, wrong_commit, parser_failure")
print("        - S-probes: honesty + overcommit")
print("        - Gate 0: trusted regression must hold")
print("        - Gate 1: wrong_commit <= 2% per family")
print("        - Gate 3: per-family + S-probe targets")

# ===========================================================================
# v15.6 PAS 3 — Section P3A: EntitySpan + accepted premodifiers
# ===========================================================================
#
# STRICT SCOPE (per GPT directive):
#   - Target: F2 multiword_entities ONLY
#   - Zero touch: v15.4 parser, V15_4_QUERY_PATTERNS, PREFIX_ALIAS_MAP
#   - Zero touch: shadow, bank schema, trusted path
#   - head_entity_id always remains canonical head from the existing pool
#   - "young dragon" never becomes a new entity_id; bank sees only "dragon"
#   - Rule-based only; no training; no embeddings for composition
#
# INVARIANTS:
#   - precision > recall (conservative premodifier list)
#   - if uncertain, return uncomposed and let arbiter honestly PARSE_UNCERTAIN
#   - composer cannot increase wrong_commit_rate
# ===========================================================================


# Conservative premodifier set (per GPT Pas 3 directive).
# Deliberately excludes:
#   - size vocabulary (small, large, huge, tiny, little, big)
#   - color vocabulary (silver, golden, dark, bright)
#   - trait/descriptor words (mighty, strong, weak, brave, wild)
#   - state vocabulary (asleep, awake, angry, calm, tired, happy, afraid,
#     HUNGRY — overlaps HOLDOUT_STATES; removed to keep precision > recall)
#
# Cross-check: every token in this set must satisfy:
#   (a) not in HOLDOUT_COLORS / HOLDOUT_SIZES / HOLDOUT_LOCATIONS /
#       HOLDOUT_STATES
#   (b) not in V15_ATTR_VALUES["color|size|location|state"]
#   (c) not an entity head (not in HOLDOUT_ENTITIES_SINGLE)
#
# The cross-check is runtime-asserted below, not trusted to comments.
V15_6_ACCEPTED_PREMODIFIERS = frozenset({
    "young",
    "old",
    "iron",
    "wooden",
    "silent",
    "ancient",
    "royal",
    "sacred",
    "lost",
})


# Runtime-asserted invariant: premodifier set does not collide with any
# attribute value vocabulary or entity head.
def _v15_6_validate_premodifier_set():
    """Raise if any premodifier would contaminate value/entity spaces."""
    # Attribute value pools (V15 internal + F2 generator external)
    all_values = set()
    for atype in ("color", "size", "location", "state"):
        all_values.update(V15_ATTR_VALUES.get(atype, []))
        all_values.update(HOLDOUT_ATTR_VALUES.get(atype, []))
    
    # Entity heads (both internal and holdout external)
    all_heads = set()
    for (e, _) in V15_TRAIN_ENTITIES:
        all_heads.add(e)
    for (e, _) in V15_HELDOUT_ENTITIES:
        all_heads.add(e)
    all_heads.update(HOLDOUT_ENTITIES_SINGLE)
    
    value_collisions = V15_6_ACCEPTED_PREMODIFIERS & all_values
    head_collisions  = V15_6_ACCEPTED_PREMODIFIERS & all_heads
    
    if value_collisions:
        raise RuntimeError(
            f"[v15.6 Pas 3] Premodifier/value collision: {value_collisions}"
        )
    if head_collisions:
        raise RuntimeError(
            f"[v15.6 Pas 3] Premodifier/head collision: {head_collisions}"
        )


_v15_6_validate_premodifier_set()


# Words that block premodifier expansion (structural boundary markers).
# These stop the backward walk without being consumed as modifiers.
V15_6_EXPANSION_BLOCKERS = frozenset({
    "the", "a", "an",           # determiners (consumed separately by the span)
    "and", "or", "but",         # conjunctions
    "is", "was", "are", "were", "be", "been", "being",  # copula
    "seems", "seemed", "appears", "appeared",
    "looked", "felt", "feels",
    "this", "that", "these", "those",
    "some", "every", "any", "no",
    "of", "in", "on", "at", "by", "for", "from", "to", "with",
})


# Composition classification.
class EntitySpanCompositionKind(Enum):
    BARE_HEAD          = "bare_head"
    PREFIX_MODIFIER_HEAD = "prefix_modifier_head"
    DETERMINER_HEAD    = "determiner_head"
    UNCOMPOSED         = "uncomposed"


@dataclass
class EntitySpan:
    """Compositional span for a single entity mention.
    
    head_entity_id is ALWAYS a canonical head from the existing pool.
    Bank never sees modifier-enriched identifiers.
    """
    head_entity_id:         str
    head_span:              Tuple[int, int]   # char-level (start, end) in source text
    modifiers:              List[str]
    full_span:              Tuple[int, int]   # includes modifiers + optional determiner
    composition_kind:       EntitySpanCompositionKind
    composition_confidence: float
    # Diagnostic-only: did the composer stop because of a blocker (structural)
    # or because it genuinely had no more tokens to consider?
    stop_reason:            str = ""


# New ambiguity flag for span-level ambiguity (does NOT modify v15.4 Verifier;
# added as a string that verifier pass-through mechanism already routes
# through reasons).
V15_6_ENTITY_SPAN_AMBIGUOUS = "ENTITY_SPAN_AMBIGUOUS"


print("[v15.6 Pas 3] Section P3A: EntitySpan + premodifier set")
print(f"        - V15_6_ACCEPTED_PREMODIFIERS: {len(V15_6_ACCEPTED_PREMODIFIERS)} tokens")
print(f"        - validated: no collision with value vocab or entity heads")
print(f"        - V15_6_EXPANSION_BLOCKERS: {len(V15_6_EXPANSION_BLOCKERS)} tokens")
print(f"        - EntitySpanCompositionKind: 4 values")
# ===========================================================================
# v15.6 PAS 3 — Section P3B: EntitySpanComposer (rule-based)
# ===========================================================================
#
# Algorithm:
#   1. Tokenize text into (word_lower, char_start, char_end) tuples.
#      Punctuation is stripped; word boundaries are standard whitespace +
#      punctuation boundaries.
#   2. Locate head candidates: words matching entries in entity pool.
#   3. For each head candidate: walk backward up to 2 tokens, accepting only
#      tokens in V15_6_ACCEPTED_PREMODIFIERS as modifiers. Stop at any
#      blocker, conjunction, attribute-value, or other entity head.
#   4. Absorb determiner ("the"/"a"/"an") immediately before modifier chain
#      into full_span (not modifiers list).
#   5. Detect overlap between spans => mark ENTITY_SPAN_AMBIGUOUS; return
#      spans as-is (composer does NOT pick a winner).
# ===========================================================================


import re as _re_p3


_V15_6_WORD_PATTERN = _re_p3.compile(r"\b([A-Za-z][A-Za-z'\-]*)\b")


def _v15_6_tokenize_for_composer(text: str) -> List[Tuple[str, int, int]]:
    """Return list of (word_lower, char_start, char_end) for every word
    in text. Punctuation separates but is not emitted.
    """
    out = []
    for m in _V15_6_WORD_PATTERN.finditer(text):
        word = m.group(1).lower()
        out.append((word, m.start(), m.end()))
    return out


def _v15_6_entity_head_pool() -> Set[str]:
    """All canonical entity heads that bank might contain.
    
    This IS the union of:
      - HOLDOUT_ENTITIES_SINGLE (external holdout pool)
      - V15_TRAIN_ENTITIES heads
      - V15_HELDOUT_ENTITIES heads
    
    Composer WILL NOT return a head outside this set. Ever.
    """
    heads = set(HOLDOUT_ENTITIES_SINGLE)
    for (e, _) in V15_TRAIN_ENTITIES:
        heads.add(e)
    for (e, _) in V15_HELDOUT_ENTITIES:
        heads.add(e)
    return heads


def _v15_6_attribute_value_pool() -> Set[str]:
    """Union of all attribute value vocabularies. Any word here is NOT
    a valid premodifier — it is part of the attribute assertion pathway.
    """
    pool = set()
    for atype in ("color", "size", "location", "state"):
        pool.update(V15_ATTR_VALUES.get(atype, []))
        pool.update(HOLDOUT_ATTR_VALUES.get(atype, []))
    return pool


class EntitySpanComposer:
    """Rule-based compositional entity span detector.
    
    Public API: compose(text) -> (spans, flags)
    
    Guarantees:
      - head_entity_id is always a canonical head from the known pool
      - if uncertain, returns UNCOMPOSED spans; never invents
      - if two spans overlap, emits V15_6_ENTITY_SPAN_AMBIGUOUS flag
    """
    
    def __init__(self):
        self.heads              = _v15_6_entity_head_pool()
        self.attribute_values   = _v15_6_attribute_value_pool()
        self.premodifiers       = V15_6_ACCEPTED_PREMODIFIERS
        self.blockers           = V15_6_EXPANSION_BLOCKERS
        self.max_modifiers      = 2
        self.determiners        = frozenset({"the", "a", "an"})
    
    def compose(self, text: str) -> Tuple[List[EntitySpan], Set[str]]:
        """Produce list of EntitySpan + any span-level flags."""
        tokens = _v15_6_tokenize_for_composer(text)
        flags: Set[str] = set()
        
        if not tokens:
            return [], flags
        
        # Step 1 — locate head candidate indices
        head_indices = [i for i, (w, _, _) in enumerate(tokens)
                           if w in self.heads]
        if not head_indices:
            return [], flags
        
        spans: List[EntitySpan] = []
        
        # Step 2 — for each head, walk backward up to max_modifiers
        for head_idx in head_indices:
            head_word, head_start, head_end = tokens[head_idx]
            modifiers: List[str] = []
            full_start = head_start
            full_end   = head_end
            stop_reason = "exhausted"
            
            # Backward walk
            look_idx = head_idx - 1
            while look_idx >= 0 and len(modifiers) < self.max_modifiers:
                cand_word, cand_start, cand_end = tokens[look_idx]
                
                # Blocker? stop without consuming
                if cand_word in self.blockers:
                    stop_reason = f"blocker:{cand_word}"
                    break
                # Attribute value? stop without consuming
                if cand_word in self.attribute_values:
                    stop_reason = f"attr_value:{cand_word}"
                    break
                # Another entity head? stop without consuming
                if cand_word in self.heads:
                    stop_reason = f"other_head:{cand_word}"
                    break
                # Accepted premodifier? consume
                if cand_word in self.premodifiers:
                    modifiers.append(cand_word)
                    full_start = cand_start
                    look_idx -= 1
                    continue
                # Unknown token (not blocker, not value, not head, not premod)
                # => conservative halt
                stop_reason = f"unknown_token:{cand_word}"
                break
            
            # Modifiers were accumulated in reverse order; reverse for natural
            # left-to-right order.
            modifiers.reverse()
            
            # Absorb determiner if immediately before modifier chain
            # (or directly before head if no modifiers).
            pre_chain_idx = head_idx - len(modifiers) - 1
            if pre_chain_idx >= 0:
                det_word, det_start, _ = tokens[pre_chain_idx]
                if det_word in self.determiners:
                    full_start = det_start
            
            # Classify
            if modifiers:
                kind = EntitySpanCompositionKind.PREFIX_MODIFIER_HEAD
                confidence = 0.95
            elif full_start < head_start:
                kind = EntitySpanCompositionKind.DETERMINER_HEAD
                confidence = 0.95
            else:
                kind = EntitySpanCompositionKind.BARE_HEAD
                confidence = 0.95
            
            spans.append(EntitySpan(
                head_entity_id=head_word,
                head_span=(head_start, head_end),
                modifiers=modifiers,
                full_span=(full_start, full_end),
                composition_kind=kind,
                composition_confidence=confidence,
                stop_reason=stop_reason,
            ))
        
        # Step 3 — detect overlap between distinct spans
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                a_start, a_end = spans[i].full_span
                b_start, b_end = spans[j].full_span
                if a_start < b_end and b_start < a_end:
                    flags.add(V15_6_ENTITY_SPAN_AMBIGUOUS)
                    # Both spans demoted to UNCOMPOSED; let arbiter decide
                    spans[i].composition_kind = EntitySpanCompositionKind.UNCOMPOSED
                    spans[i].composition_confidence = 0.0
                    spans[j].composition_kind = EntitySpanCompositionKind.UNCOMPOSED
                    spans[j].composition_confidence = 0.0
        
        return spans, flags


# Module-level composer instance for reuse
V15_6_ENTITY_SPAN_COMPOSER = EntitySpanComposer()


print("[v15.6 Pas 3] Section P3B: EntitySpanComposer instantiated")
print(f"        - rule-based, no training, no embeddings")
print(f"        - max_modifiers = 2, conservative fallback")
print(f"        - head pool size: {len(V15_6_ENTITY_SPAN_COMPOSER.heads)}")
print(f"        - attribute value pool size: {len(V15_6_ENTITY_SPAN_COMPOSER.attribute_values)}")
# ===========================================================================
# v15.6 PAS 3 — Section P3C: arbiter integration + F2 breakdown scorer
# ===========================================================================
#
# Integration strategy:
#   - v15_4_parse_fact / v15_4_parse_query: UNCHANGED (they remain frozen)
#   - CommitArbiter.write_fact: we shadow the top-entity selection with a
#     span-aware variant. If composer produces a confident span pointing to
#     the same head that v15.4 parser found → same entity_id, no behavior
#     change. If composer produces ENTITY_SPAN_AMBIGUOUS → flag added, which
#     the existing verifier mechanism treats as PARSE_UNCERTAIN.
#   - Queries are NOT composed in Pas 3 (scope: F2 is about fact-side
#     entity resolution). Query entity detection remains v15.4 path.
#
# F2-specific breakdown adds a new set of sub-labels to each trial outcome
# so we can separate "composer was active and helped" from "composer was
# active but honestly rejected" from "bare head fallback worked".
# ===========================================================================


# Composer activation bookkeeping — a single trial's compose-level outcome
@dataclass
class ComposerTrace:
    """What happened inside the composer for this write."""
    spans_found:                int
    has_modifiers:              bool
    has_ambiguity_flag:         bool
    chose_composed_head:        bool
    top_head_entity:            Optional[str]
    composition_kind:           Optional[str]


def _v15_6_top_entity_span(text: str,
                              parse_entity_candidates: List[Tuple[str, float, Tuple[int, int]]]
                              ) -> Tuple[Optional[str], Set[str], ComposerTrace]:
    """Run composer; produce (chosen_entity_id, extra_flags, trace).
    
    CONSERVATIVE dispatch:
      1. If composer returns no spans → fallback to v15.4 top entity
      2. If composer emits ENTITY_SPAN_AMBIGUOUS → return v15.4 top entity
         BUT add the flag to extra_flags (verifier will PARSE_UNCERTAIN)
      3. If composer has exactly one span with modifiers → use its
         head_entity_id (always a canonical pool head)
      4. If composer has exactly one span without modifiers (bare_head) →
         use v15.4 top entity (no behavior change vs Pas 2)
      5. If composer has multiple non-overlapping spans → delegate to v15.4
         selection (v15.4 already handles multi-entity flag mechanism)
    
    The head_entity_id returned by the composer is guaranteed to be a
    canonical pool head by construction.
    """
    extra_flags: Set[str] = set()
    spans, flags = V15_6_ENTITY_SPAN_COMPOSER.compose(text)
    
    # Trace scaffold
    trace = ComposerTrace(
        spans_found=len(spans),
        has_modifiers=any(len(s.modifiers) > 0 for s in spans),
        has_ambiguity_flag=V15_6_ENTITY_SPAN_AMBIGUOUS in flags,
        chose_composed_head=False,
        top_head_entity=None,
        composition_kind=None,
    )
    
    # Fallback: no spans at all — let v15.4 decide (may produce PARSER_FAILURE)
    if not spans:
        v154_top = parse_entity_candidates[0][0] if parse_entity_candidates else None
        trace.top_head_entity = v154_top
        return v154_top, extra_flags, trace
    
    # Ambiguity: composer refuses to pick; add flag and fallback
    if V15_6_ENTITY_SPAN_AMBIGUOUS in flags:
        extra_flags.add(V15_6_ENTITY_SPAN_AMBIGUOUS)
        v154_top = parse_entity_candidates[0][0] if parse_entity_candidates else None
        trace.top_head_entity = v154_top
        return v154_top, extra_flags, trace
    
    # Multiple non-overlapping spans: v15.4 top-entity mechanism wins
    if len(spans) > 1:
        v154_top = parse_entity_candidates[0][0] if parse_entity_candidates else None
        trace.top_head_entity = v154_top
        return v154_top, extra_flags, trace
    
    # Single span path
    s = spans[0]
    if s.composition_kind == EntitySpanCompositionKind.UNCOMPOSED:
        # Demoted by overlap or other issue; fallback (flag already added)
        v154_top = parse_entity_candidates[0][0] if parse_entity_candidates else None
        trace.top_head_entity = v154_top
        return v154_top, extra_flags, trace
    
    # Active composition win
    trace.chose_composed_head = (s.composition_kind ==
                                    EntitySpanCompositionKind.PREFIX_MODIFIER_HEAD)
    trace.top_head_entity = s.head_entity_id
    trace.composition_kind = s.composition_kind.value
    return s.head_entity_id, extra_flags, trace


# ===========================================================================
# Pas 3 CommitArbiter subclass that uses composer on write_fact
# ===========================================================================

class CommitArbiterPas3(CommitArbiter):
    """Inherits Pas 2 behavior; overrides write_fact to use composer.
    
    Preserves all Pas 2 invariants:
      - episode buffer protocol unchanged
      - end_episode dual conflict rule unchanged
      - cross-episode challenger detection unchanged
      - wrong_commit guarantees unchanged
    """
    
    def __init__(self, *args, composer_trace_log: Optional[List[ComposerTrace]] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.composer_trace_log = composer_trace_log
    
    def write_fact(self,
                    fact_text: str,
                    entity_emb_fn,
                    class_emb_fn,
                    value_emb_fn,
                    write_step: int = 0) -> ArbitratedWriteResult:
        if not self.episode_buffer.is_active:
            raise RuntimeError(
                "CommitArbiterPas3.write_fact called outside active episode"
            )
        episode_id = self.episode_buffer.episode_id
        
        pkt = v15_4_parse_fact(fact_text)
        
        # Pas 3 change: compose entity span BEFORE verifier
        composed_entity_id, extra_flags, trace = _v15_6_top_entity_span(
            fact_text, pkt.entity_candidates
        )
        if self.composer_trace_log is not None:
            self.composer_trace_log.append(trace)
        
        # Inject span-level flags into parse packet so verifier sees them
        if V15_6_ENTITY_SPAN_AMBIGUOUS in extra_flags:
            pkt.ambiguity_flags.add(V15_6_ENTITY_SPAN_AMBIGUOUS)
        
        ip = parse_packet_to_internalization_packet(pkt, step=write_step)
        vr = V15_4_VERIFIER.verify(pkt)
        
        if vr.status == VerificationStatus.PARSER_FAILURE:
            ip.commit_path = CommitPath.PARSER_FAILURE
            return ArbitratedWriteResult(
                commit_path=CommitPath.PARSER_FAILURE, buffered=False,
                provisional=False, rejected=True, parse_packet=pkt,
                verifier_result=vr, internalization_packet=ip,
            )
        
        if vr.status == VerificationStatus.PARSE_UNCERTAIN:
            ip.commit_path = CommitPath.PARSE_UNCERTAIN
            return ArbitratedWriteResult(
                commit_path=CommitPath.PARSE_UNCERTAIN, buffered=False,
                provisional=False, rejected=True, parse_packet=pkt,
                verifier_result=vr, internalization_packet=ip,
            )
        
        # Verifier ACCEPTED v15.6 Pas 3 path: prefer composer's head when it
        # was confidently composed; otherwise fallback to v15.4 top entity.
        if composed_entity_id is not None:
            entity_id = composed_entity_id
        else:
            entity_id = _top_entity(pkt)
        attr_type = _top_attribute(pkt)
        value_idx = _top_value_for(pkt, attr_type) if attr_type else None
        
        if entity_id is None or attr_type is None or value_idx is None:
            ip.commit_path = CommitPath.PARSER_FAILURE
            return ArbitratedWriteResult(
                commit_path=CommitPath.PARSER_FAILURE, buffered=False,
                provisional=False, rejected=True, parse_packet=pkt,
                verifier_result=vr, internalization_packet=ip,
            )
        
        # Cross-episode conflict check (unchanged from Pas 2)
        if self.stability_index.is_stable(entity_id, attr_type, episode_id):
            existing_slot = self.bank.find_by_entity_id(entity_id)
            if existing_slot is not None:
                rec = self.bank.get_record(existing_slot)
                slot = rec.attr_slots.get(attr_type)
                if (slot is not None and slot.present
                        and slot.value_idx != value_idx):
                    entry = ProvisionalEntry(
                        entity_id=entity_id, attr_type=attr_type,
                        value_idx=value_idx, episode_id=episode_id,
                        write_step=write_step, source_text=fact_text,
                        internalization_packet_ref=ip,
                        challenge_kind="cross_episode_challenger",
                    )
                    self.provisional_memory.add(entry)
                    ip.commit_path = CommitPath.STORE_PROVISIONAL
                    return ArbitratedWriteResult(
                        commit_path=CommitPath.STORE_PROVISIONAL,
                        buffered=False, provisional=True, rejected=False,
                        parse_packet=pkt, verifier_result=vr,
                        internalization_packet=ip,
                    )
        
        # Precompute embeddings
        try:
            ent_emb = entity_emb_fn(entity_id)
        except Exception:
            ent_emb = None
        class_id = _entity_class_id(entity_id)
        try:
            cls_emb = (class_emb_fn(class_id, ent_emb)
                          if ent_emb is not None else None)
        except Exception:
            cls_emb = None
        try:
            val_emb = value_emb_fn(attr_type, value_idx)
        except Exception:
            val_emb = None
        
        bw = BufferedWrite(
            entity_id=entity_id, attr_type=attr_type, value_idx=value_idx,
            write_step=write_step, source_text=fact_text,
            parse_packet=pkt,
            internalization_packet=ip,
            entity_emb_cache=ent_emb, class_id_cache=class_id,
            class_emb_cache=cls_emb, value_emb_cache=val_emb,
        )
        self.episode_buffer.add_write(bw)
        ip.commit_path = CommitPath.COMMIT
        return ArbitratedWriteResult(
            commit_path=CommitPath.COMMIT, buffered=True, provisional=False,
            rejected=False, parse_packet=pkt, verifier_result=vr,
            internalization_packet=ip,
        )


# ===========================================================================
# F2 BREAKDOWN SCORER — 8 sub-categories per GPT directive
# ===========================================================================
#
# Sub-categories:
#   1. composed_commit_correct        — composer active, commit, correct
#   2. composed_but_rejected          — composer produced span, verifier
#                                         said PARSE_UNCERTAIN (ambiguous/etc)
#   3. composed_uncertain             — composer active, trial ended
#                                         UNCERTAIN / NONE_* status
#   4. bare_head_commit_correct       — no modifiers, v15.4 path worked
#   5. bare_head_uncertain            — no modifiers, ended uncertain
#   6. span_ambiguous_provisional     — ENTITY_SPAN_AMBIGUOUS → DISPUTED
#   7. head_not_found                 — parser_fail (composer found nothing)
#   8. wrong_commit                   — MUST stay at 0
# ===========================================================================


@dataclass
class F2TrialDetail:
    outcome:          ArbitratedTrialOutcome
    composer_trace:   ComposerTrace


def _v15_6_score_f2_breakdown(details: List[F2TrialDetail]) -> Dict:
    """Compute 8-category F2-specific breakdown."""
    n = len(details)
    buckets = {
        "composed_commit_correct":      0,
        "composed_but_rejected":        0,
        "composed_uncertain":           0,
        "bare_head_commit_correct":     0,
        "bare_head_uncertain":          0,
        "span_ambiguous_provisional":   0,
        "head_not_found":               0,
        "wrong_commit":                 0,
    }
    
    for d in details:
        o = d.outcome
        tr = d.composer_trace
        status = o.arbitrated_status
        is_correct = (o.pred_value == o.target_value_idx
                        and status == READ_STATUS_FOUND_COMMITTED)
        
        if tr.has_ambiguity_flag and status == READ_STATUS_FOUND_DISPUTED:
            buckets["span_ambiguous_provisional"] += 1
            continue
        
        if status == READ_STATUS_FOUND_COMMITTED:
            if o.pred_value == o.target_value_idx:
                if tr.chose_composed_head:
                    buckets["composed_commit_correct"] += 1
                else:
                    buckets["bare_head_commit_correct"] += 1
            else:
                buckets["wrong_commit"] += 1
            continue
        
        if status == READ_STATUS_FOUND_DISPUTED:
            # Target might be in disputed_values (honest)
            if o.target_value_idx in o.disputed_values:
                buckets["span_ambiguous_provisional"] += 1
            else:
                # disputed but wrong set of values → count as uncertain
                if tr.chose_composed_head:
                    buckets["composed_uncertain"] += 1
                else:
                    buckets["bare_head_uncertain"] += 1
            continue
        
        if status == READ_STATUS_PARSE_UNCERTAIN:
            if tr.has_modifiers or tr.chose_composed_head:
                buckets["composed_but_rejected"] += 1
            elif tr.spans_found > 0:
                buckets["bare_head_uncertain"] += 1
            else:
                buckets["head_not_found"] += 1
            continue
        
        if status in (READ_STATUS_NONE_OBJECT, READ_STATUS_NONE_ATTRIBUTE):
            if tr.chose_composed_head:
                buckets["composed_uncertain"] += 1
            elif tr.spans_found > 0:
                buckets["bare_head_uncertain"] += 1
            else:
                buckets["head_not_found"] += 1
            continue
        
        if status == READ_STATUS_PARSER_FAIL:
            buckets["head_not_found"] += 1
            continue
    
    # Normalize
    rates = {k: v / n for k, v in buckets.items()}
    
    # Primary metrics per Pas 3 directive
    safe_resolution = (rates["composed_commit_correct"]
                         + rates["bare_head_commit_correct"]
                         + rates["span_ambiguous_provisional"])
    abstention = (rates["composed_but_rejected"]
                    + rates["composed_uncertain"]
                    + rates["bare_head_uncertain"]
                    + rates["head_not_found"])
    harmful = rates["wrong_commit"]
    
    return {
        "n":                         n,
        "buckets_count":             buckets,
        "buckets_rate":              rates,
        "safe_resolution_rate":      safe_resolution,
        "abstention_rate":           abstention,
        "harmful_commit_rate":       harmful,
    }


print("[v15.6 Pas 3] Section P3C: CommitArbiterPas3 + F2 breakdown scorer")
print(f"        - composer integrated at write_fact; query path untouched")
print(f"        - extra_flags injected pre-verify (ENTITY_SPAN_AMBIGUOUS)")
print(f"        - canonical head enforcement via pool lookup")
print(f"        - 8-bucket F2 breakdown with safe_resolution/abstention/harmful axes")
# ===========================================================================
# v15.6 PAS 3 — Section P3D: evaluator + gate validation
# ===========================================================================


def _v15_6_pas3_run_arbitrated_episode(arbiter: CommitArbiterPas3,
                                           reader: ReadArbiter,
                                           ep,
                                           episode_id: int,
                                           entity_emb_fn,
                                           class_emb_fn,
                                           value_emb_fn,
                                           composer_traces: List[ComposerTrace]
                                           ) -> ArbitratedTrialOutcome:
    """Same as Pas 2 runner, but uses CommitArbiterPas3.
    
    Returns ArbitratedTrialOutcome + populates composer_traces for the
    FACT writes in this episode. (Traces from queries not captured; Pas 3
    does not compose on queries.)
    """
    arbiter.begin_episode(episode_id)
    
    write_paths = []
    # Snapshot trace log size before this episode
    trace_start = len(arbiter.composer_trace_log) if arbiter.composer_trace_log is not None else 0
    
    for j, fact_text in enumerate(ep.facts):
        result = arbiter.write_fact(fact_text, entity_emb_fn, class_emb_fn,
                                        value_emb_fn, write_step=j)
        write_paths.append(result.commit_path.value)
    
    finalize = arbiter.end_episode(entity_emb_fn, class_emb_fn, value_emb_fn)
    end_decisions = {f"{k[0]}::{k[1]}": v
                       for k, v in finalize.decisions_per_slot.items()}
    
    rd = reader.read_query(ep.query)
    
    # For F2 detailed breakdown, we associate the "representative" composer
    # trace with this trial. In F2 each episode has 1 fact → exactly 1 trace
    # is added to the log during this episode.
    trace_end = (len(arbiter.composer_trace_log)
                    if arbiter.composer_trace_log is not None else 0)
    this_ep_traces = (arbiter.composer_trace_log[trace_start:trace_end]
                         if arbiter.composer_trace_log is not None else [])
    # Use the LAST composer trace (corresponding to the target fact).
    # For F2 holdout, each episode has a single fact so last == only.
    representative_trace = (this_ep_traces[-1] if this_ep_traces
                             else ComposerTrace(0, False, False, False, None, None))
    composer_traces.append(representative_trace)
    
    target_value_idx = None
    if not ep.target_is_unknown:
        attr_type = HOLDOUT_ATTR_TYPES[ep.query_attr_label]
        vocab = HOLDOUT_ATTR_VALUES[attr_type]
        for k, vstr in enumerate(vocab):
            if V15_ANSWER_TOKENS.get(attr_type, {}).get(vstr) == ep.target_answer_token:
                target_value_idx = k
                break
    
    return ArbitratedTrialOutcome(
        family=ep.family_tag,
        target_is_unknown=ep.target_is_unknown,
        target_value_idx=target_value_idx,
        arbitrated_status=rd.status,
        pred_value=rd.pred,
        disputed_values=rd.disputed_values,
        commit_path_at_write=write_paths,
        end_episode_decisions=end_decisions,
    )


V15_6_PAS3_ACCEPTANCE = {
    # Primary F2 targets
    "F2_safe_resolution_min":    0.95,
    "F2_harmful_commit_max":     0.02,
    # Regression guards (must match Pas 2 outcomes)
    "S5_honesty_min":            0.95,
    "S5_overcommit_max":         0.02,
    "S6_honesty_min":            0.95,
    "S6_overcommit_max":         0.02,
    "F4_safe_resolution_min":    0.99,
    # Gate on ALL families (no family worse than Pas 2 wrong_commit)
    "wrong_commit_max_per_family": 0.02,
}


def v15_6_pas3_run_full_evaluation(bank, base_model, v15_1_memory,
                                       pas2_baseline: Optional[Dict] = None):
    """Run full Pas 3 evaluation.
    
    If pas2_baseline is provided, include explicit delta reporting for F2.
    """
    print()
    print(SEP)
    print("[v15.6 PAS 3 EVALUATION]")
    print(SEP)
    
    ent_fn = _make_entity_emb_fn(base_model)
    cls_fn = _make_class_emb_fn(v15_1_memory)
    val_fn = _make_value_emb_fn(base_model)
    class_map = _v15_5_build_class_map()
    
    # Trusted snapshot BEFORE
    print("\n[v15.6 Pas 3] Trusted snapshot BEFORE (Gate 0)")
    snap_before = _v15_5_snapshot_trusted(bank, base_model, v15_1_memory)
    for k, v in snap_before.items():
        print(f"  {k}: {v:.4f}")
    
    bank.reset()
    provisional_memory = ProvisionalMemory()
    episode_buffer     = EpisodeBuffer()
    stability_index    = BankStabilityIndex()
    composer_traces_shared: List[ComposerTrace] = []
    arbiter = CommitArbiterPas3(bank, provisional_memory, episode_buffer,
                                   stability_index,
                                   composer_trace_log=composer_traces_shared)
    reader = ReadArbiter(bank, provisional_memory)
    
    family_results = {}
    f2_details: List[F2TrialDetail] = []
    seed_offset = 100000
    episode_counter = 1
    
    print("\n[v15.6 Pas 3] Running 5 holdout families (n={} each)".format(
        V15_5_HOLDOUT_CONFIG["n_per_family"]))
    for fname, gen in EXTERNAL_HOLDOUT_FAMILIES.items():
        print(f"  -> {fname}")
        rng = _rng_module.Random(V15_5_HOLDOUT_CONFIG["seed"] + seed_offset)
        outcomes = []
        per_family_traces: List[ComposerTrace] = []
        for trial in range(V15_5_HOLDOUT_CONFIG["n_per_family"]):
            ep = gen(rng, ENC, class_map)
            bank.reset()
            provisional_memory.reset()
            episode_buffer.clear()
            stability_index.reset()
            composer_traces_shared.clear()
            o = _v15_6_pas3_run_arbitrated_episode(arbiter, reader, ep,
                                                     episode_counter,
                                                     ent_fn, cls_fn, val_fn,
                                                     per_family_traces)
            outcomes.append(o)
            episode_counter += 1
        scored = _v15_6_score_family(outcomes)
        family_results[fname] = scored
        print(f"     commit_correct={scored['commit_correct_rate']:.3f} "
              f"prov_correct={scored['provisional_correct_rate']:.3f} "
              f"unc={scored['uncertain_rate']:.3f} "
              f"wrong_commit={scored['wrong_commit_rate']:.3f} "
              f"parser_fail={scored['parser_failure_rate']:.3f}")
        
        # F2 specific: build detailed breakdown
        if fname == "F2_multiword_entities":
            for o, tr in zip(outcomes, per_family_traces):
                f2_details.append(F2TrialDetail(outcome=o, composer_trace=tr))
        
        seed_offset += 1000
    
    s_results = {}
    print("\n[v15.6 Pas 3] Running S-probes (n={} each)".format(
        V15_5_HOLDOUT_CONFIG["n_per_s_probe"]))
    for sname, gen in EXTERNAL_HOLDOUT_S_PROBES.items():
        print(f"  -> {sname}")
        rng = _rng_module.Random(V15_5_HOLDOUT_CONFIG["seed"] + seed_offset)
        outcomes = []
        per_probe_traces: List[ComposerTrace] = []
        for trial in range(V15_5_HOLDOUT_CONFIG["n_per_s_probe"]):
            ep = gen(rng, ENC, class_map)
            bank.reset()
            provisional_memory.reset()
            episode_buffer.clear()
            stability_index.reset()
            composer_traces_shared.clear()
            o = _v15_6_pas3_run_arbitrated_episode(arbiter, reader, ep,
                                                     episode_counter,
                                                     ent_fn, cls_fn, val_fn,
                                                     per_probe_traces)
            outcomes.append(o)
            episode_counter += 1
        scored = _v15_6_score_family(outcomes)
        s_results[sname] = scored
        print(f"     honesty={scored['honesty']:.3f} "
              f"overcommit={scored['overcommit']:.3f} "
              f"unc={scored['uncertain_rate']:.3f}")
        seed_offset += 1000
    
    # F2 detailed breakdown
    print("\n[v15.6 Pas 3] F2 DETAILED BREAKDOWN (8 buckets)")
    f2_break = _v15_6_score_f2_breakdown(f2_details)
    for bucket, rate in f2_break["buckets_rate"].items():
        print(f"     {bucket:35s}  {rate:.3f}  (n={f2_break['buckets_count'][bucket]})")
    print(f"     --- primary axes ---")
    print(f"     safe_resolution_rate:    {f2_break['safe_resolution_rate']:.3f}")
    print(f"     abstention_rate:         {f2_break['abstention_rate']:.3f}")
    print(f"     harmful_commit_rate:     {f2_break['harmful_commit_rate']:.3f}")
    
    # Trusted snapshot AFTER
    print("\n[v15.6 Pas 3] Trusted snapshot AFTER (Gate 0)")
    bank.reset()
    snap_after = _v15_5_snapshot_trusted(bank, base_model, v15_1_memory)
    match, bad_k, vb, va = _v15_5_trusted_signatures_match(snap_before,
                                                              snap_after)
    if match:
        print("  PASS: trusted regression check")
    else:
        print(f"  FAIL: regression on '{bad_k}': before={vb} after={va}")
    
    # Acceptance gates
    print("\n" + SEP)
    print("=== V15.6 PAS 3 ACCEPTANCE GATES ===")
    print(SEP)
    
    A = V15_6_PAS3_ACCEPTANCE
    checks = {}
    
    # Gate 0
    checks["Gate 0: trusted regression"] = match
    
    # Gate 1: wrong_commit on every family
    for fname, r in family_results.items():
        checks[f"Gate 1: {fname} wrong_commit <= {A['wrong_commit_max_per_family']:.2f}"] = (
            r["wrong_commit_rate"] <= A["wrong_commit_max_per_family"]
        )
    
    # Pas 3 primary: F2
    checks[f"Pas 3: F2 safe_resolution_rate >= {A['F2_safe_resolution_min']:.2f}"] = (
        f2_break["safe_resolution_rate"] >= A["F2_safe_resolution_min"]
    )
    checks[f"Pas 3: F2 harmful_commit_rate <= {A['F2_harmful_commit_max']:.2f}"] = (
        f2_break["harmful_commit_rate"] <= A["F2_harmful_commit_max"]
    )
    
    # Regression guards (S5/S6/F4)
    s5 = s_results.get("S5_conflict_intercalated", {})
    checks[f"Regression: S5 honesty >= {A['S5_honesty_min']:.2f}"] = (
        (s5.get("honesty") if s5.get("honesty") is not None else 0) >= A["S5_honesty_min"]
    )
    checks[f"Regression: S5 overcommit <= {A['S5_overcommit_max']:.2f}"] = (
        (s5.get("overcommit") if s5.get("overcommit") is not None else 1) <= A["S5_overcommit_max"]
    )
    s6 = s_results.get("S6_entity_competition_cross", {})
    checks[f"Regression: S6 honesty >= {A['S6_honesty_min']:.2f}"] = (
        (s6.get("honesty") if s6.get("honesty") is not None else 0) >= A["S6_honesty_min"]
    )
    checks[f"Regression: S6 overcommit <= {A['S6_overcommit_max']:.2f}"] = (
        (s6.get("overcommit") if s6.get("overcommit") is not None else 1) <= A["S6_overcommit_max"]
    )
    f4 = family_results.get("F4_discourse_intercalation", {})
    f4_total = f4.get("commit_correct_rate", 0) + f4.get("provisional_correct_rate", 0)
    checks[f"Regression: F4 safe_resolution >= {A['F4_safe_resolution_min']:.2f}"] = (
        f4_total >= A["F4_safe_resolution_min"]
    )
    
    all_pass = all(checks.values())
    for name, ok in checks.items():
        print(f"  {'✓' if ok else '✗'} {name}")
    
    # Explicit confirmation of F1/F3/F5 untouched-by-design policy
    print()
    print("[Pas 3 scope] F1/F3/F5 deferred to Pas 6 by design — not evaluated as targets.")
    for fname in ("F1_novel_paraphrase_syntax", "F3_novel_lexical_alias",
                    "F5_novel_query_forms"):
        r = family_results.get(fname, {})
        print(f"  {fname}: commit_correct={r.get('commit_correct_rate', 0):.3f} "
              f"wrong_commit={r.get('wrong_commit_rate', 0):.3f} (informational only)")
    
    # Delta vs Pas 2 baseline
    print()
    print(SEP)
    print("=== F2 DELTA vs PAS 2 BASELINE ===")
    print(SEP)
    if pas2_baseline is not None:
        pas2_f2 = pas2_baseline.get("family_results", {}).get(
            "F2_multiword_entities", {})
        if pas2_f2:
            print(f"  commit_correct:     {pas2_f2.get('commit_correct_rate', 0):.3f}"
                    f" -> {family_results['F2_multiword_entities']['commit_correct_rate']:.3f}")
            print(f"  provisional_correct: {pas2_f2.get('provisional_correct_rate', 0):.3f}"
                    f" -> {family_results['F2_multiword_entities']['provisional_correct_rate']:.3f}")
            print(f"  uncertain:          {pas2_f2.get('uncertain_rate', 0):.3f}"
                    f" -> {family_results['F2_multiword_entities']['uncertain_rate']:.3f}")
            print(f"  wrong_commit:       {pas2_f2.get('wrong_commit_rate', 0):.3f}"
                    f" -> {family_results['F2_multiword_entities']['wrong_commit_rate']:.3f}")
            print(f"  parser_fail:        {pas2_f2.get('parser_failure_rate', 0):.3f}"
                    f" -> {family_results['F2_multiword_entities']['parser_failure_rate']:.3f}")
    else:
        print("  (no Pas 2 baseline provided)")
    
    print()
    print(SEP)
    verdict = "PAS 3 PASSED" if all_pass else "PAS 3 PARTIAL"
    print(f"VERDICT: {verdict}")
    print(SEP)
    
    return {
        "snap_before":            snap_before,
        "snap_after":             snap_after,
        "trusted_regression_ok":  match,
        "family_results":         family_results,
        "s_results":              s_results,
        "f2_breakdown":           f2_break,
        "checks":                 checks,
        "all_pass":               all_pass,
        "verdict":                verdict,
    }


print("[v15.6 Pas 3] Section P3D: evaluator + gate validation defined")
print("        - Gate 0: trusted regression")
print("        - Gate 1: wrong_commit per family")
print("        - Pas 3 primary: F2 safe_resolution + harmful_commit")
print("        - Regression guards: S5/S6 honesty, F4 safe_resolution")
print("        - F1/F3/F5 explicit informational-only (Pas 6 scope)")

# ===========================================================================
# v15.6 PAS 3.1a — F2 CAUSAL DIAGNOSIS (offline, no runtime changes)
# ===========================================================================
#
# PURPOSE: Determine whether the residual 21.8% F2 uncertain cases are
# caused by entity internalization asymmetry (write/read mismatch) or by
# attr-side / value-side extraction failures.
#
# METHOD: Offline replay on identical seeds/trials. No new runtime
# structures. No infrastructure. Just measurement.
#
# Per trial, compute 3 counterfactual gains:
#   gain_from_write_internalization_only
#   gain_from_read_internalization_only
#   gain_from_symmetric_internalization
#
# Assign one causal label per failed trial:
#   success / entity_write_failure / entity_read_failure /
#   entity_asymmetry_failure / attr_write_failure / attr_read_failure /
#   value_detection_failure / verifier_other
#
# VERDICT RULE:
#   if gain_from_symmetric >= 50% of all F2 failures:
#       Pas 3.1 JUSTIFIED (build full infrastructure)
#   else:
#       Pas 3.1 FALSIFIED (move directly to Pas 6)
# ===========================================================================


from dataclasses import dataclass as _dc_d, field as _dc_f
from typing import Dict as _D, List as _L, Optional as _O, Set as _S, Tuple as _T


@_dc_d
class F2DiagnosticTrial:
    """Single F2 trial observation set, fully annotated."""
    # 1. Raw observables
    trial_id:                int
    fact_text:               str
    query_text:              str
    target_status:           str
    target_value_idx:        _O[int]
    
    # 2. Write-side (v15.4 parser + verifier, unchanged)
    write_entity_head_found: bool
    write_entity_head:       _O[str]
    write_modifiers_found:   _L[str]
    write_attr_found:        bool
    write_attr_type:         _O[str]
    write_value_found:       bool
    write_value_idx:         _O[int]
    write_verifier_status:   str
    write_reasons:           _L[str]
    
    # 3. Read-side (v15.4 parser + verifier, unchanged)
    read_entity_head_found:  bool
    read_entity_head:        _O[str]
    read_modifiers_found:    _L[str]
    read_attr_found:         bool
    read_attr_type:          _O[str]
    read_verifier_status:    str
    read_reasons:            _L[str]
    
    # 4. Composer-derived counterfactuals
    write_internalized_head:      _O[str]
    write_internalized_modifiers: _L[str]
    read_internalized_head:       _O[str]
    read_internalized_modifiers:  _L[str]
    
    head_match_only:                       bool
    full_internalized_match:               bool
    gain_from_write_internalization_only:  bool
    gain_from_read_internalization_only:   bool
    gain_from_symmetric_internalization:   bool
    
    # 5. Trial outcome via Pas 2 pipeline + final causal label
    pipeline_outcome:        str      # COMMIT_CORRECT / UNCERTAIN / PARSER_FAIL / WRONG_COMMIT
    pipeline_correct:        bool
    causal_label:            str      # one of the 8 labels


def _v15_6_p31a_build_write_observables(fact_text: str):
    """Parse fact through v15.4; extract entity/attr/value observables."""
    pkt = v15_4_parse_fact(fact_text)
    vr  = V15_4_VERIFIER.verify(pkt)
    
    # Entity head (canonical; v15.4 substring-based)
    head = pkt.entity_candidates[0][0] if pkt.entity_candidates else None
    
    # Attr detection: on facts, v15.4 infers attr from value type
    top_attr = _top_attribute(pkt)
    attr_found = top_attr is not None and top_attr != "__class__"
    
    # Value detection
    val_idx = _top_value_for(pkt, top_attr) if attr_found else None
    val_found = val_idx is not None
    
    # Reasons (strings)
    reasons = [r.value if hasattr(r, "value") else str(r)
                  for r in (vr.reasons if vr else [])]
    
    return {
        "pkt":            pkt,
        "head":           head,
        "head_found":     head is not None,
        "top_attr":       top_attr,
        "attr_found":     attr_found,
        "val_idx":        val_idx,
        "val_found":      val_found,
        "verifier_status": vr.status.value if vr else "NO_VERIFIER",
        "reasons":        reasons,
    }


def _v15_6_p31a_build_read_observables(query_text: str):
    """Parse query through v15.4; extract entity/attr observables."""
    pkt = v15_4_parse_query(query_text)
    vr  = V15_4_VERIFIER.verify(pkt)
    
    head = pkt.entity_candidates[0][0] if pkt.entity_candidates else None
    top_attr = _top_attribute(pkt)
    attr_found = top_attr is not None and top_attr != "__class__"
    
    reasons = [r.value if hasattr(r, "value") else str(r)
                  for r in (vr.reasons if vr else [])]
    
    return {
        "pkt":            pkt,
        "head":           head,
        "head_found":     head is not None,
        "top_attr":       top_attr,
        "attr_found":     attr_found,
        "verifier_status": vr.status.value if vr else "NO_VERIFIER",
        "reasons":        reasons,
    }


def _v15_6_p31a_compose_counterfactual(text: str):
    """Run EntitySpanComposer offline; return (head, modifiers)."""
    spans, flags = V15_6_ENTITY_SPAN_COMPOSER.compose(text)
    if not spans:
        return None, []
    # pick first non-uncomposed span
    for s in spans:
        if s.composition_kind != EntitySpanCompositionKind.UNCOMPOSED:
            return s.head_entity_id, list(s.modifiers)
    return spans[0].head_entity_id, list(spans[0].modifiers)


def _v15_6_p31a_compute_gains(w_head, w_mods, r_head, r_mods,
                                 w_ihead, w_imods, r_ihead, r_imods):
    """Compute 3 counterfactual gains based on internalization.
    
    Baseline match: head_match_only = (w_head == r_head) and w_head is not None.
    
    Internalization changes the matching criterion:
      - write-side internalization: adds modifier information to write key
      - read-side internalization:  adds modifier information to read key
      - symmetric: both sides use internalized (head + modifiers) tuples
    
    Gain semantics (counterfactual):
      - gain_X = True iff baseline head_match_only would have SUCCEEDED but
        internalization on X side would preserve/improve match,
        OR baseline would have FAILED but X-side internalization would
        have succeeded.
    
    For F2, baseline matching is already head-based and succeeds when
    heads align. True incremental value appears only when symmetric
    internalization adds a NEW match path that bare heads could not.
    
    We compute 4 match candidates:
      m_bare      = head_match_only
      m_write     = m_bare AND w_ihead is not None AND w_ihead == r_head
      m_read      = m_bare AND r_ihead is not None AND r_ihead == w_head
      m_symmetric = w_ihead == r_ihead AND sorted(w_imods) == sorted(r_imods)
                        (full structural match)
    
    Gain = match via this pathway but NOT via bare baseline, OR provides
    additional disambiguation information.
    
    In the current F2 setup where shelf_key == canonical_head, asymmetric
    paths rarely add new success cases. But we measure all three so the
    data speaks for itself.
    """
    def _safe(x): return x if x is not None else ""
    
    m_bare = (_safe(w_head) == _safe(r_head)) and (w_head is not None)
    
    # Write internalization only: read uses bare head
    m_write_only = ((w_ihead is not None) and (r_head is not None)
                       and (w_ihead == r_head))
    
    # Read internalization only: write uses bare head
    m_read_only = ((r_ihead is not None) and (w_head is not None)
                      and (r_ihead == w_head))
    
    # Symmetric: both internalized with full modifier match
    m_symmetric = False
    if w_ihead is not None and r_ihead is not None and w_ihead == r_ihead:
        m_symmetric = (sorted(w_imods) == sorted(r_imods))
    
    # Gains: did this counterfactual succeed where bare baseline did NOT?
    gain_write = m_write_only and not m_bare
    gain_read  = m_read_only  and not m_bare
    gain_sym   = m_symmetric  and not m_bare
    
    # Alternative gain: symmetric provides strictly MORE information than
    # bare match, even when bare also matched. Asymmetry failure bucket
    # uses a different definition — see causal assignment below.
    
    return {
        "head_match_only":                       m_bare,
        "full_internalized_match":               m_symmetric,
        "gain_from_write_internalization_only":  gain_write,
        "gain_from_read_internalization_only":   gain_read,
        "gain_from_symmetric_internalization":   gain_sym,
    }


def _v15_6_p31a_assign_causal_label(obs: dict) -> str:
    """Assign exactly one causal label per trial.
    
    Priority (first matching rule wins):
      1. pipeline outcome correct   -> success
      2. not w_head_found            -> entity_write_failure
      3. not r_head_found            -> entity_read_failure
      4. both heads exist but differ -> entity_asymmetry_failure
         (AND only if gain_from_symmetric would have fixed it)
      5. write verifier failed pre-commit on attr/value issues
            -> attr_write_failure / value_detection_failure
      6. read verifier failed on attr issues
            -> attr_read_failure
      7. else                        -> verifier_other
    """
    if obs["pipeline_correct"]:
        return "success"
    
    if not obs["write_entity_head_found"]:
        return "entity_write_failure"
    if not obs["read_entity_head_found"]:
        return "entity_read_failure"
    
    # Both heads exist. If they disagree AND internalization would have fixed it:
    heads_agree = (obs["write_entity_head"] == obs["read_entity_head"])
    if not heads_agree:
        if obs["gain_from_symmetric_internalization"]:
            return "entity_asymmetry_failure"
        return "entity_asymmetry_failure"  # still asymmetry, even if gain test does not cover it
    
    # Both heads exist and agree. Check attr/value side.
    if not obs["write_attr_found"]:
        return "attr_write_failure"
    if not obs["write_value_found"]:
        return "value_detection_failure"
    if obs["write_verifier_status"] != "ACCEPT":
        return "attr_write_failure"
    
    if not obs["read_attr_found"]:
        return "attr_read_failure"
    if obs["read_verifier_status"] != "ACCEPT":
        return "attr_read_failure"
    
    return "verifier_other"


def _v15_6_p31a_compute_pipeline_outcome(arbiter, reader, ep, episode_id,
                                             entity_emb_fn, class_emb_fn,
                                             value_emb_fn):
    """Run ep through Pas 3 arbitrated pipeline; return (outcome, correct)."""
    bank = arbiter.bank
    provisional_memory = arbiter.provisional_memory
    episode_buffer     = arbiter.episode_buffer
    stability_index    = arbiter.stability_index
    bank.reset()
    provisional_memory.reset()
    episode_buffer.clear()
    stability_index.reset()
    if arbiter.composer_trace_log is not None:
        arbiter.composer_trace_log.clear()
    
    arbiter.begin_episode(episode_id)
    for j, fact_text in enumerate(ep.facts):
        arbiter.write_fact(fact_text, entity_emb_fn, class_emb_fn,
                             value_emb_fn, write_step=j)
    arbiter.end_episode(entity_emb_fn, class_emb_fn, value_emb_fn)
    rd = reader.read_query(ep.query)
    
    target_value_idx = None
    if not ep.target_is_unknown:
        attr_type = HOLDOUT_ATTR_TYPES[ep.query_attr_label]
        vocab = HOLDOUT_ATTR_VALUES[attr_type]
        for k, vstr in enumerate(vocab):
            if V15_ANSWER_TOKENS.get(attr_type, {}).get(vstr) == ep.target_answer_token:
                target_value_idx = k
                break
    
    if rd.status == READ_STATUS_FOUND_COMMITTED:
        if rd.pred == target_value_idx:
            return "COMMIT_CORRECT", True
        return "WRONG_COMMIT", False
    if rd.status == READ_STATUS_FOUND_DISPUTED:
        if target_value_idx in rd.disputed_values:
            return "DISPUTED_CONTAINS_TARGET", True
        return "DISPUTED_MISSES_TARGET", False
    if rd.status == READ_STATUS_PARSE_UNCERTAIN:
        return "UNCERTAIN", False
    if rd.status in (READ_STATUS_NONE_OBJECT, READ_STATUS_NONE_ATTRIBUTE):
        return "NONE", False
    return "PARSER_FAIL", False


def v15_6_pas3_1a_f2_diagnosis(bank, base_model, v15_1_memory,
                                   n_trials: int = 500):
    """Run Pas 3.1a diagnostic on F2.
    
    Uses EXACT same seeds as Pas 3 evaluation to reproduce identical trials.
    Returns aggregated diagnostic plus per-trial records.
    """
    print()
    print(SEP)
    print(f"[v15.6 Pas 3.1a] F2 CAUSAL DIAGNOSIS (n={n_trials}, offline)")
    print(SEP)
    
    ent_fn = _make_entity_emb_fn(base_model)
    cls_fn = _make_class_emb_fn(v15_1_memory)
    val_fn = _make_value_emb_fn(base_model)
    class_map = _v15_5_build_class_map()
    
    # Initialize Pas 3 arbiter (to get same pipeline outcomes as Pas 3 run)
    provisional_memory = ProvisionalMemory()
    episode_buffer     = EpisodeBuffer()
    stability_index    = BankStabilityIndex()
    composer_traces: _L = []
    arbiter = CommitArbiterPas3(bank, provisional_memory, episode_buffer,
                                    stability_index,
                                    composer_trace_log=composer_traces)
    reader = ReadArbiter(bank, provisional_memory)
    
    # Reproduce F2 seed from Pas 3 runtime
    # Pas 3 evaluator uses V15_5_HOLDOUT_CONFIG["seed"] + seed_offset where
    # seed_offset starts at 100000 and increments by 1000 per family.
    # F2 is index 1 (F1=100000, F2=101000).
    f2_seed = V15_5_HOLDOUT_CONFIG["seed"] + 101000
    rng = _rng_module.Random(f2_seed)
    
    gen_f2 = EXTERNAL_HOLDOUT_FAMILIES["F2_multiword_entities"]
    
    records: _L[F2DiagnosticTrial] = []
    episode_counter = 2_000_000   # separate from Pas 3 evaluator counter
    
    for trial_id in range(n_trials):
        ep = gen_f2(rng, ENC, class_map)
        
        # Observables via v15.4 path
        w_obs = _v15_6_p31a_build_write_observables(ep.facts[0])
        r_obs = _v15_6_p31a_build_read_observables(ep.query)
        
        # Composer counterfactuals
        w_ihead, w_imods = _v15_6_p31a_compose_counterfactual(ep.facts[0])
        r_ihead, r_imods = _v15_6_p31a_compose_counterfactual(ep.query)
        
        gains = _v15_6_p31a_compute_gains(
            w_obs["head"], [],  # v15.4 doesn't extract modifiers; bare head only
            r_obs["head"], [],
            w_ihead, w_imods, r_ihead, r_imods
        )
        
        # Pipeline outcome (using Pas 3 arbiter = matches Pas 3 A100 run)
        outcome, correct = _v15_6_p31a_compute_pipeline_outcome(
            arbiter, reader, ep, episode_counter,
            ent_fn, cls_fn, val_fn
        )
        episode_counter += 1
        
        # Target
        target_value_idx = None
        if not ep.target_is_unknown:
            atype = HOLDOUT_ATTR_TYPES[ep.query_attr_label]
            vocab = HOLDOUT_ATTR_VALUES[atype]
            for k, vstr in enumerate(vocab):
                if V15_ANSWER_TOKENS.get(atype, {}).get(vstr) == ep.target_answer_token:
                    target_value_idx = k
                    break
        
        rec_dict = {
            "trial_id":                trial_id,
            "fact_text":               ep.facts[0],
            "query_text":              ep.query,
            "target_status":           "unknown" if ep.target_is_unknown else "known",
            "target_value_idx":        target_value_idx,
            "write_entity_head_found": w_obs["head_found"],
            "write_entity_head":       w_obs["head"],
            "write_modifiers_found":   [],
            "write_attr_found":        w_obs["attr_found"],
            "write_attr_type":         w_obs["top_attr"],
            "write_value_found":       w_obs["val_found"],
            "write_value_idx":         w_obs["val_idx"],
            "write_verifier_status":   w_obs["verifier_status"],
            "write_reasons":           w_obs["reasons"],
            "read_entity_head_found":  r_obs["head_found"],
            "read_entity_head":        r_obs["head"],
            "read_modifiers_found":    [],
            "read_attr_found":         r_obs["attr_found"],
            "read_attr_type":          r_obs["top_attr"],
            "read_verifier_status":    r_obs["verifier_status"],
            "read_reasons":            r_obs["reasons"],
            "write_internalized_head":      w_ihead,
            "write_internalized_modifiers": w_imods,
            "read_internalized_head":       r_ihead,
            "read_internalized_modifiers":  r_imods,
            **gains,
            "pipeline_outcome":        outcome,
            "pipeline_correct":        correct,
        }
        rec_dict["causal_label"] = _v15_6_p31a_assign_causal_label(rec_dict)
        records.append(F2DiagnosticTrial(**rec_dict))
    
    # ---- Aggregate ----
    n = len(records)
    n_success = sum(1 for r in records if r.pipeline_correct)
    n_failures = n - n_success
    
    from collections import Counter as _Counter
    causal_counts = _Counter(r.causal_label for r in records)
    
    gain_write_count     = sum(1 for r in records
                                  if r.gain_from_write_internalization_only)
    gain_read_count      = sum(1 for r in records
                                  if r.gain_from_read_internalization_only)
    gain_symmetric_count = sum(1 for r in records
                                  if r.gain_from_symmetric_internalization)
    
    # Primary gain metric: gain_symmetric as fraction of failures
    gain_symmetric_of_failures = (gain_symmetric_count / n_failures
                                     if n_failures > 0 else 0.0)
    gain_write_of_failures = (gain_write_count / n_failures
                                 if n_failures > 0 else 0.0)
    gain_read_of_failures  = (gain_read_count / n_failures
                                 if n_failures > 0 else 0.0)
    
    # Buckets by rough origin
    entity_side_failures = (causal_counts.get("entity_write_failure", 0)
                               + causal_counts.get("entity_read_failure", 0)
                               + causal_counts.get("entity_asymmetry_failure", 0))
    attr_side_failures   = (causal_counts.get("attr_write_failure", 0)
                               + causal_counts.get("attr_read_failure", 0))
    value_side_failures  = causal_counts.get("value_detection_failure", 0)
    other_failures       = causal_counts.get("verifier_other", 0)
    
    # Print report
    print(f"\nTotal trials: {n}")
    print(f"  success:   {n_success}")
    print(f"  failures:  {n_failures}")
    print()
    print(f"Causal label distribution:")
    for label in ("success", "entity_write_failure", "entity_read_failure",
                     "entity_asymmetry_failure", "attr_write_failure",
                     "attr_read_failure", "value_detection_failure",
                     "verifier_other"):
        cnt = causal_counts.get(label, 0)
        pct = cnt / n
        print(f"  {label:30s}  {cnt:4d}  ({pct:.3f})")
    
    print()
    print(f"Failure origin summary (of {n_failures} failures):")
    if n_failures > 0:
        print(f"  entity-side:  {entity_side_failures:4d}  ({entity_side_failures/n_failures:.3f})")
        print(f"  attr-side:    {attr_side_failures:4d}  ({attr_side_failures/n_failures:.3f})")
        print(f"  value-side:   {value_side_failures:4d}  ({value_side_failures/n_failures:.3f})")
        print(f"  verifier_other: {other_failures:4d}  ({other_failures/n_failures:.3f})")
    
    print()
    print(f"Counterfactual gains (cases where internalization adds NEW match):")
    print(f"  gain_from_write_only:       {gain_write_count:4d}")
    print(f"  gain_from_read_only:        {gain_read_count:4d}")
    print(f"  gain_from_symmetric:        {gain_symmetric_count:4d}")
    print()
    print(f"  gain_sym / total_failures:  {gain_symmetric_of_failures:.3f}  (threshold 0.50)")
    
    # VERDICT
    print()
    print(SEP)
    if gain_symmetric_of_failures >= 0.50:
        verdict = "PAS_3_1_JUSTIFIED"
        print(f"VERDICT: PAS 3.1 JUSTIFIED")
        print(f"  symmetric internalization would recover {gain_symmetric_of_failures:.1%}")
        print(f"  of F2 failures — exceeds 50% threshold.")
        print(f"  Proceed to full EntityInternalizer implementation.")
    else:
        verdict = "PAS_3_1_FALSIFIED"
        print(f"VERDICT: PAS 3.1 FALSIFIED")
        print(f"  symmetric internalization would recover only {gain_symmetric_of_failures:.1%}")
        print(f"  of F2 failures — below 50% threshold.")
        print(f"  F2 residual is dominantly {('attr-side' if attr_side_failures >= entity_side_failures else 'entity-side')}.")
        print(f"  Move to Pas 6 (SemanticAttributeResolver).")
    print(SEP)
    
    return {
        "n":                     n,
        "n_success":             n_success,
        "n_failures":            n_failures,
        "causal_counts":         dict(causal_counts),
        "entity_side_failures":  entity_side_failures,
        "attr_side_failures":    attr_side_failures,
        "value_side_failures":   value_side_failures,
        "other_failures":        other_failures,
        "gain_from_write_only":  gain_write_count,
        "gain_from_read_only":   gain_read_count,
        "gain_from_symmetric":   gain_symmetric_count,
        "gain_symmetric_of_failures": gain_symmetric_of_failures,
        "gain_write_of_failures":     gain_write_of_failures,
        "gain_read_of_failures":      gain_read_of_failures,
        "verdict":               verdict,
        "records":               [asdict(r) for r in records],
    }


print("[v15.6 Pas 3.1a] F2 causal diagnosis defined")
print("        - offline replay with identical F2 seed")
print("        - 8 causal labels (one per trial)")
print("        - 3 counterfactual gains separately measured")
print("        - verdict threshold: gain_sym >= 50% of failures")

# ===========================================================================
# v15.6 PAS 6 — Role-of-Modifier Resolver (RoMR)
# ===========================================================================
# Section R1: data structures (token-level labels + packet-level conflict)
# ===========================================================================
#
# DESIGN per GPT directive:
#   - Token-level labels: ENTITY_MODIFIER / ATTRIBUTE_VALUE / UNCERTAIN
#   - Packet-level conflict: REAL_CONFLICT (relation between two tokens,
#     not a single-token property)
#   - Span-bounded role assignment: use NP span boundaries, not raw head pos
#   - Conflict precedence BEFORE filtering
#   - Zero destructive overwrite: raw_value_candidates preserved alongside
#     romr_filtered_value_candidates
#   - Fact-only: query path untouched
#
# FROZEN: substrate, bank, shadow, v15.4 parser/verifier, Pas 1/2/3 arbiters.
# ===========================================================================


# Token-level role label (per candidate value).
class RoleLabel(Enum):
    ENTITY_MODIFIER  = "ENTITY_MODIFIER"   # premodifier inside NP span
    ATTRIBUTE_VALUE  = "ATTRIBUTE_VALUE"   # post-copula predicative value
    UNCERTAIN        = "UNCERTAIN"         # cannot decide from structure


# Packet-level conflict flag (raised when two ATTRIBUTE_VALUE candidates
# belong to the same attribute family).
V15_6_PAS6_REAL_CONFLICT = "REAL_CONFLICT"


@dataclass
class RoleClassification:
    """One classification per value candidate in the raw parse packet."""
    attr_type:     str                              # color/size/location/state
    value_idx:     int
    confidence:    float
    span:          Tuple[int, int]                   # char offsets in source
    token_pos:     int                               # word-level token index
    label:         RoleLabel
    reason:        str                               # short diagnostic string


@dataclass
class RoMRResult:
    """Full output of RoMR for a single fact.
    
    Preserves raw input; records filtered output; carries trace for audit.
    """
    fact_text:                        str
    raw_value_candidates:             List[Tuple[str, int, float, Tuple[int, int]]]
    romr_filtered_value_candidates:   List[Tuple[str, int, float, Tuple[int, int]]]
    token_classifications:            List[RoleClassification]
    packet_level_conflict:            bool     # REAL_CONFLICT detected
    conflict_pairs:                   List[Tuple[int, int]]  # indices into token_classifications
    entity_span_used:                 Optional[Tuple[int, int]]
    head_position:                    Optional[int]
    copula_position:                  Optional[int]
    trace_notes:                      List[str]


# Copula / linking verbs that separate NP span from predicative value.
V15_6_PAS6_COPULAS = frozenset({
    "is", "was", "are", "were", "be", "been", "being",
    "seems", "seemed", "appears", "appeared",
    "looked", "looks", "felt", "feels",
    "became", "becomes", "becoming", "grew", "grows", "grown",
    "turned", "turns",
    "remained", "remains", "remaining",
    "stood", "stands", "standing",
    "appeared",
    "bore",    # "The X bore Y throughout" (F1 syntax)
    "carried", "carries",
    "had",     # only when followed by attribute ("had a red tone")
})


print("[v15.6 Pas 6] Section R1: RoleLabel + RoleClassification defined")
print("        - token-level labels: ENTITY_MODIFIER / ATTRIBUTE_VALUE / UNCERTAIN")
print("        - packet-level flag: REAL_CONFLICT (relational)")
print(f"        - V15_6_PAS6_COPULAS: {len(V15_6_PAS6_COPULAS)} linking verbs")
# ===========================================================================
# v15.6 PAS 6 — Section R2: RoleOfModifierResolver core
# ===========================================================================
#
# Algorithm:
#   1. Tokenize fact at word level (same tokenizer as EntitySpanComposer).
#   2. Run EntitySpanComposer to get entity_span = [np_start_char, np_end_char]
#      and head_position (word-level index).
#   3. Find copula position: first copula token at word index >= head_position.
#   4. For each value_candidate (attr, value_idx, confidence, (cs, ce)):
#        map char span to word token index (pos).
#        if pos falls inside [entity_span_start_word, head_position) => ENTITY_MODIFIER
#        elif pos > copula_position => ATTRIBUTE_VALUE
#        elif head_position < pos <= copula_position => ATTRIBUTE_VALUE (rare attributive)
#        elif pos < entity_span_start_word => UNCERTAIN  (outside NP)
#        else => UNCERTAIN
#   5. Conflict precedence (BEFORE filtering): if two ATTRIBUTE_VALUE
#      candidates share the same attr_type => packet_level_conflict = True.
#   6. Filter: keep only ATTRIBUTE_VALUE (and UNCERTAIN, conservative);
#      drop ENTITY_MODIFIER.
#   7. If packet_level_conflict: DO NOT filter — let verifier see both
#      ATTRIBUTE_VALUE entries so ATTR_CONFLICT_STRONG fires correctly.
# ===========================================================================


class RoleOfModifierResolver:
    """Rule-based role classifier for value candidates on the write side.
    
    Invariants:
      - Never mutates the input ParsePacket in place beyond attaching a
        filtered value_candidates field (destructive overwrite forbidden).
      - Never touches entity_candidates.
      - Never runs on queries.
      - Never invents candidates; only relabels/filters extant candidates.
    """
    
    def __init__(self):
        self.copulas = V15_6_PAS6_COPULAS
    
    def classify(self, fact_text: str, pkt: "ParsePacket") -> RoMRResult:
        trace: List[str] = []
        raw = list(pkt.value_candidates)
        
        if not raw:
            return RoMRResult(
                fact_text=fact_text,
                raw_value_candidates=raw,
                romr_filtered_value_candidates=[],
                token_classifications=[],
                packet_level_conflict=False,
                conflict_pairs=[],
                entity_span_used=None,
                head_position=None,
                copula_position=None,
                trace_notes=["no_value_candidates"],
            )
        
        # Tokenize (word-level) with char spans
        tokens = _v15_6_tokenize_for_composer(fact_text)
        if not tokens:
            return RoMRResult(
                fact_text=fact_text,
                raw_value_candidates=raw,
                romr_filtered_value_candidates=list(raw),
                token_classifications=[
                    RoleClassification(a, v, c, s, -1, RoleLabel.UNCERTAIN,
                                          "no_word_tokens")
                    for (a, v, c, s) in raw
                ],
                packet_level_conflict=False,
                conflict_pairs=[],
                entity_span_used=None,
                head_position=None,
                copula_position=None,
                trace_notes=["tokenizer_returned_nothing"],
            )
        
        # Build NP span independently from Pas 3 composer.
        # Pas 3 composer is CONSERVATIVE and only admits whitelisted
        # premodifiers into its span — that is correct for entity composition
        # but WRONG for role labeling: RoMR needs to recognize the NP span
        # inclusively so it can label words like "small", "hungry" as
        # ENTITY_MODIFIER (which is exactly what makes them NOT value-at-this-
        # position).
        #
        # RoMR NP span algorithm:
        #   - find head position via composer (for the head word identity)
        #   - walk backward from head, stopping at a determiner (the/a/an),
        #     a blocker, or beginning of sentence
        #   - entity_span_start_word = first word after the determiner
        #     (or the first word absorbed if no determiner)
        spans, _span_flags = V15_6_ENTITY_SPAN_COMPOSER.compose(fact_text)
        head_word_pos: Optional[int] = None
        if spans:
            # Pick the span with the most modifiers; fallback first non-uncomposed
            best = max(spans, key=lambda s: (len(s.modifiers),
                                                s.composition_confidence))
            if best.composition_kind != EntitySpanCompositionKind.UNCOMPOSED:
                head_word = best.head_entity_id
                for idx, (w, _, _) in enumerate(tokens):
                    if w == head_word:
                        head_word_pos = idx
                        break
        
        # Build RoMR NP span: walk backward from head_word_pos until a
        # determiner, blocker, copula, or another entity head is encountered.
        entity_span_start_word: Optional[int] = None
        entity_span_char: Optional[Tuple[int, int]] = None
        if head_word_pos is not None:
            np_start_word = head_word_pos
            determiners = frozenset({"the", "a", "an"})
            heads_pool = V15_6_ENTITY_SPAN_COMPOSER.heads
            blockers = V15_6_EXPANSION_BLOCKERS
            cop_set = self.copulas
            
            for back_idx in range(head_word_pos - 1, -1, -1):
                w = tokens[back_idx][0]
                if w in determiners:
                    # Determiner is the left edge of NP; absorb it and stop.
                    np_start_word = back_idx
                    break
                if w in cop_set:
                    break
                if w in heads_pool and back_idx != head_word_pos:
                    break
                # Any other word (including attribute values) is a potential
                # NP-internal modifier. Include it.
                np_start_word = back_idx
                # But don't cross additional blockers beyond determiners
                if w in blockers:
                    # e.g. "in the" — "in" is a blocker; stop without absorbing
                    np_start_word = back_idx + 1
                    break
            
            entity_span_start_word = np_start_word
            head_start_char = tokens[head_word_pos][2]  # head end
            np_start_char = tokens[np_start_word][1]
            entity_span_char = (np_start_char, head_start_char)
        
        # Find copula position (first copula at/after head)
        copula_word_pos: Optional[int] = None
        search_from = (head_word_pos + 1) if head_word_pos is not None else 0
        for idx in range(search_from, len(tokens)):
            if tokens[idx][0] in self.copulas:
                copula_word_pos = idx
                break
        
        trace.append(f"entity_span_char={entity_span_char}")
        trace.append(f"entity_span_start_word={entity_span_start_word}")
        trace.append(f"head_word_pos={head_word_pos}")
        trace.append(f"copula_word_pos={copula_word_pos}")
        
        # Classify each candidate
        classifications: List[RoleClassification] = []
        for (attr_type, val_idx, conf, span_char) in raw:
            cs, ce = span_char
            # Map candidate char span -> word token index
            token_pos = -1
            for idx, (_, ts, te) in enumerate(tokens):
                if ts <= cs < te:
                    token_pos = idx
                    break
                if ts >= cs and ts < ce:
                    token_pos = idx
                    break
            
            label: RoleLabel
            reason: str
            
            if token_pos < 0:
                label = RoleLabel.UNCERTAIN
                reason = "token_pos_unmapped"
            elif head_word_pos is None:
                label = RoleLabel.UNCERTAIN
                reason = "no_head_position"
            elif entity_span_start_word is None:
                label = RoleLabel.UNCERTAIN
                reason = "no_entity_span_start"
            else:
                # Position vs NP span + copula
                in_np_pre_head = (entity_span_start_word <= token_pos < head_word_pos)
                pos_after_copula = (copula_word_pos is not None
                                        and token_pos > copula_word_pos)
                pos_between_head_and_copula = (
                    copula_word_pos is not None
                    and head_word_pos < token_pos <= copula_word_pos
                )
                pos_outside_np_pre_head = (token_pos < entity_span_start_word)
                
                if in_np_pre_head:
                    label = RoleLabel.ENTITY_MODIFIER
                    reason = "inside_np_before_head"
                elif pos_after_copula:
                    label = RoleLabel.ATTRIBUTE_VALUE
                    reason = "post_copula"
                elif pos_between_head_and_copula:
                    label = RoleLabel.ATTRIBUTE_VALUE
                    reason = "attributive_between_head_and_copula"
                elif pos_outside_np_pre_head:
                    label = RoleLabel.UNCERTAIN
                    reason = "outside_np_pre_head"
                else:
                    # token_pos >= head_word_pos but no copula found
                    # conservative: uncertain
                    label = RoleLabel.UNCERTAIN
                    reason = "no_copula_post_head"
            
            classifications.append(RoleClassification(
                attr_type=attr_type, value_idx=val_idx, confidence=conf,
                span=span_char, token_pos=token_pos,
                label=label, reason=reason,
            ))
        
        # --- Cross-position same-family REAL_CONFLICT (precedence BEFORE filter) ---
        # Rule per GPT directive: if any attribute family has >= 2 distinct
        # value_idx candidates, regardless of their individual positional labels
        # (pre-head ENTITY_MODIFIER vs post-copula ATTRIBUTE_VALUE), this is a
        # REAL_CONFLICT at packet level. Promote ALL candidates in that family
        # to ATTRIBUTE_VALUE so the verifier downstream raises ATTR_CONFLICT_STRONG.
        #
        # This covers "The small horse is huge" — small would individually
        # be labeled ENTITY_MODIFIER, but since huge (ATTRIBUTE_VALUE) is same
        # family (size) with a DIFFERENT value, we elevate to REAL_CONFLICT.
        raw_by_family: Dict[str, List[int]] = {}
        for ci, c in enumerate(classifications):
            raw_by_family.setdefault(c.attr_type, []).append(ci)
        
        packet_conflict = False
        conflict_pairs: List[Tuple[int, int]] = []
        promoted_indices: Set[int] = set()
        for family, indices in raw_by_family.items():
            distinct_values = {classifications[i].value_idx for i in indices}
            if len(distinct_values) >= 2:
                packet_conflict = True
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        if (classifications[indices[i]].value_idx
                                != classifications[indices[j]].value_idx):
                            conflict_pairs.append((indices[i], indices[j]))
                # Promote all candidates in conflicting family to ATTRIBUTE_VALUE
                for idx in indices:
                    promoted_indices.add(idx)
        
        # Apply promotions (immutable dataclass style: rebuild list)
        if promoted_indices:
            new_classifications: List[RoleClassification] = []
            for ci, c in enumerate(classifications):
                if ci in promoted_indices and c.label != RoleLabel.ATTRIBUTE_VALUE:
                    new_classifications.append(RoleClassification(
                        attr_type=c.attr_type,
                        value_idx=c.value_idx,
                        confidence=c.confidence,
                        span=c.span,
                        token_pos=c.token_pos,
                        label=RoleLabel.ATTRIBUTE_VALUE,
                        reason="real_conflict_promoted_from_" + c.label.value,
                    ))
                else:
                    new_classifications.append(c)
            classifications = new_classifications
        
        # --- Filter stage ---
        # REAL_CONFLICT case: keep all non-ENTITY_MODIFIER candidates (after
        # promotion, conflict family is fully ATTRIBUTE_VALUE; remaining
        # ENTITY_MODIFIER from other families stays dropped as usual).
        # Non-conflict case: keep ATTRIBUTE_VALUE and UNCERTAIN; drop
        # ENTITY_MODIFIER.
        filtered: List[Tuple[str, int, float, Tuple[int, int]]] = []
        for c in classifications:
            if packet_conflict:
                if c.label != RoleLabel.ENTITY_MODIFIER:
                    filtered.append((c.attr_type, c.value_idx, c.confidence, c.span))
            else:
                if c.label in (RoleLabel.ATTRIBUTE_VALUE, RoleLabel.UNCERTAIN):
                    filtered.append((c.attr_type, c.value_idx, c.confidence, c.span))
        
        return RoMRResult(
            fact_text=fact_text,
            raw_value_candidates=raw,
            romr_filtered_value_candidates=filtered,
            token_classifications=classifications,
            packet_level_conflict=packet_conflict,
            conflict_pairs=conflict_pairs,
            entity_span_used=entity_span_char,
            head_position=head_word_pos,
            copula_position=copula_word_pos,
            trace_notes=trace,
        )


# Module-level singleton
V15_6_PAS6_ROMR = RoleOfModifierResolver()


print("[v15.6 Pas 6] Section R2: RoleOfModifierResolver instantiated")
print("        - span-bounded role assignment (NP interior vs post-copula)")
print("        - conflict precedence BEFORE filtering")
print("        - REAL_CONFLICT preserves both ATTRIBUTE_VALUEs for verifier")
print("        - UNCERTAIN kept (conservative, lets verifier reject)")
# ===========================================================================
# v15.6 PAS 6 — Section R3: CommitArbiterPas6 integration
# ===========================================================================
#
# INTEGRATION POLICY:
#   - RoMR runs AFTER v15.4 parser, BEFORE verifier.
#   - ParsePacket is shallow-copied; the copy has value_candidates REPLACED
#     by romr_filtered_value_candidates. Original packet retains raw list.
#   - Verifier runs on the filtered copy.
#   - ParsePacket original is preserved AS-IS in arbiter output for audit.
#   - Read path: UNCHANGED.
#
# This matches the constraint "zero destructive overwrite without audit trail":
# the raw packet is never mutated; a separate filtered packet is constructed
# and passed to the verifier. The RoMR trace is attached to IP as metadata.
# ===========================================================================


# Flag classes per GPT directive:
#   value-dependent flags: invalidated and re-derived after RoMR filtering
#   value-independent flags: preserved unchanged
V15_6_PAS6_VALUE_DEPENDENT_FLAG_VALUES = frozenset({
    "MULTIPLE_ATTR_TRIGGERS",
    "ATTR_CONFLICT_STRONG",
    "ATTR_VALUE_MISMATCH",
    "VALUE_MISSING_OR_UNCLEAR",
})


def _v15_6_pas6_flag_value(flag) -> str:
    """Normalize flag to its string value, regardless of enum subtype."""
    return flag.value if hasattr(flag, "value") else str(flag)


def _v15_6_pas6_recompute_value_flags(filtered_candidates, op_type) -> Set:
    """Derive value-dependent ambiguity flags from filtered candidates only.
    
    Applies same rules as v15.4.1 verifier's pre-check logic:
      - MULTIPLE_ATTR_TRIGGERS: >= 2 distinct attr families in filtered set
      - VALUE_MISSING_OR_UNCLEAR: empty filtered set (only for WRITE)
      - ATTR_CONFLICT_STRONG: same family has >= 2 distinct value_idx
      - ATTR_VALUE_MISMATCH: not recomputable from filtered alone (left out)
    """
    flags: Set = set()
    
    if op_type != OpType.WRITE:
        return flags
    
    # MULTIPLE_ATTR_TRIGGERS
    attr_types_in_filtered = {a for (a, _, _, _) in filtered_candidates}
    if len(attr_types_in_filtered) > 1:
        flags.add(V15_4_AmbiguityFlag.MULTIPLE_ATTR_TRIGGERS)
    
    # ATTR_CONFLICT_STRONG: same family with distinct values
    value_idx_by_attr: Dict[str, Set[int]] = {}
    for (a, v, _, _) in filtered_candidates:
        value_idx_by_attr.setdefault(a, set()).add(v)
    for fam, vals in value_idx_by_attr.items():
        if len(vals) >= 2:
            flags.add(V15_4_AmbiguityFlag.ATTR_CONFLICT_STRONG)
            break
    
    # VALUE_MISSING_OR_UNCLEAR: only when filter produced no usable value
    if len(filtered_candidates) == 0:
        flags.add(V15_4_AmbiguityFlag.VALUE_MISSING_OR_UNCLEAR)
    
    return flags


import copy as _copy_pas6


def _v15_6_pas6_apply_romr_to_packet(pkt: "ParsePacket",
                                         fact_text: str
                                         ) -> Tuple["ParsePacket", RoMRResult]:
    """Run RoMR; return (filtered_packet, romr_result).
    
    Audit trail invariant: BOTH raw_value_candidates AND
    romr_filtered_value_candidates are preserved on the RoMRResult.
    BOTH raw_ambiguity_flags AND romr_recomputed_ambiguity_flags are
    preserved on filtered_pkt.parser_evidence.
    
    Original pkt is NOT mutated.
    """
    romr_result = V15_6_PAS6_ROMR.classify(fact_text, pkt)
    
    filtered_pkt = _copy_pas6.copy(pkt)
    filtered_pkt.value_candidates = romr_result.romr_filtered_value_candidates
    
    # --- Filter attribute_candidates coherently with value_candidates ---
    # v15.4 infers attribute candidates from values (e.g. seeing "small" adds
    # "size" to attribute_candidates). When RoMR drops the value, the inferred
    # attribute becomes spurious and would mislead _top_attribute selection
    # downstream. Keep only attribute types that still have at least one
    # filtered value OR were present in the original attribute_candidates via
    # an explicit attribute word (trusted anchor).
    filtered_attr_types_from_values = {a for (a, _, _, _)
                                            in romr_result.romr_filtered_value_candidates}
    
    # Separate attribute candidates by source: explicit vs inferred.
    # v15.4 stores attribute candidates as (attr_type, confidence, span).
    # We consider an attribute candidate "explicit" if its span corresponds to
    # an explicit attribute trigger word; otherwise it's likely inferred.
    # Without access to a discriminator, the safe policy is: keep attr_types
    # that either have a surviving value OR are anchored by the pkt's
    # attribute trigger evidence.
    raw_attr_candidates = list(pkt.attribute_candidates)
    filtered_attr_candidates = []
    for attr_cand in raw_attr_candidates:
        # attr_cand shape: (attr_type, confidence, span) or similar
        if isinstance(attr_cand, tuple) and len(attr_cand) >= 1:
            a_type = attr_cand[0]
            if a_type in filtered_attr_types_from_values:
                filtered_attr_candidates.append(attr_cand)
            # Drop attribute candidates whose family has no surviving value.
    filtered_pkt.attribute_candidates = filtered_attr_candidates
    
    # --- Flag split: preserve independent, recompute dependent ---
    raw_flags_serialized = sorted(_v15_6_pas6_flag_value(f)
                                      for f in pkt.ambiguity_flags)
    
    # Keep flags that are NOT in the value-dependent class
    preserved_flags: Set = set()
    for flag in pkt.ambiguity_flags:
        flag_val = _v15_6_pas6_flag_value(flag)
        if flag_val not in V15_6_PAS6_VALUE_DEPENDENT_FLAG_VALUES:
            preserved_flags.add(flag)
    
    # Re-derive value-dependent flags from filtered candidates
    recomputed_value_flags = _v15_6_pas6_recompute_value_flags(
        romr_result.romr_filtered_value_candidates,
        pkt.op_type,
    )
    
    filtered_pkt.ambiguity_flags = preserved_flags | recomputed_value_flags
    
    recomputed_flags_serialized = sorted(_v15_6_pas6_flag_value(f)
                                             for f in filtered_pkt.ambiguity_flags)
    
    # --- Audit trail ---
    filtered_pkt.parser_evidence = dict(pkt.parser_evidence)
    filtered_pkt.parser_evidence["romr"] = {
        "packet_level_conflict": romr_result.packet_level_conflict,
        "entity_modifier_count": sum(
            1 for c in romr_result.token_classifications
            if c.label == RoleLabel.ENTITY_MODIFIER
        ),
        "attribute_value_count": sum(
            1 for c in romr_result.token_classifications
            if c.label == RoleLabel.ATTRIBUTE_VALUE
        ),
        "uncertain_count": sum(
            1 for c in romr_result.token_classifications
            if c.label == RoleLabel.UNCERTAIN
        ),
        "entity_span_used":               romr_result.entity_span_used,
        "head_word_pos":                  romr_result.head_position,
        "copula_word_pos":                romr_result.copula_position,
        "raw_value_candidates":           [
            (a, v, c, s) for (a, v, c, s) in romr_result.raw_value_candidates
        ],
        "romr_filtered_value_candidates": [
            (a, v, c, s) for (a, v, c, s) in romr_result.romr_filtered_value_candidates
        ],
        "raw_ambiguity_flags":            raw_flags_serialized,
        "romr_recomputed_ambiguity_flags": recomputed_flags_serialized,
    }
    return filtered_pkt, romr_result


class CommitArbiterPas6(CommitArbiterPas3):
    """Inherits Pas 3 composer logic; adds RoMR filtering of value
    candidates on fact-side BEFORE verifier runs.
    
    Preserves everything frozen from previous passes:
      - episode buffer protocol
      - end_episode dual conflict rule
      - cross-episode challenger detection
      - entity span composer for head selection
      - canonical head enforcement
    """
    
    def __init__(self, *args, romr_trace_log: Optional[List[RoMRResult]] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.romr_trace_log = romr_trace_log
    
    def write_fact(self,
                    fact_text: str,
                    entity_emb_fn,
                    class_emb_fn,
                    value_emb_fn,
                    write_step: int = 0) -> ArbitratedWriteResult:
        if not self.episode_buffer.is_active:
            raise RuntimeError(
                "CommitArbiterPas6.write_fact called outside active episode"
            )
        episode_id = self.episode_buffer.episode_id
        
        # --- v15.4 parse (frozen) ---
        raw_pkt = v15_4_parse_fact(fact_text)
        
        # --- Pas 3 composer (frozen): pick head ---
        composed_entity_id, extra_span_flags, composer_trace = _v15_6_top_entity_span(
            fact_text, raw_pkt.entity_candidates
        )
        if self.composer_trace_log is not None:
            self.composer_trace_log.append(composer_trace)
        
        # --- Pas 6 RoMR: filter value_candidates ---
        filtered_pkt, romr_result = _v15_6_pas6_apply_romr_to_packet(
            raw_pkt, fact_text
        )
        if self.romr_trace_log is not None:
            self.romr_trace_log.append(romr_result)
        
        # Inject composer's span-level flag if any
        if V15_6_ENTITY_SPAN_AMBIGUOUS in extra_span_flags:
            filtered_pkt.ambiguity_flags.add(V15_6_ENTITY_SPAN_AMBIGUOUS)
        
        # --- IP + verifier (on filtered pkt) ---
        ip = parse_packet_to_internalization_packet(filtered_pkt, step=write_step)
        ip.source_trace["romr_packet_conflict"] = romr_result.packet_level_conflict
        vr = V15_4_VERIFIER.verify(filtered_pkt)
        
        if vr.status == VerificationStatus.PARSER_FAILURE:
            ip.commit_path = CommitPath.PARSER_FAILURE
            return ArbitratedWriteResult(
                commit_path=CommitPath.PARSER_FAILURE, buffered=False,
                provisional=False, rejected=True, parse_packet=raw_pkt,
                verifier_result=vr, internalization_packet=ip,
            )
        if vr.status == VerificationStatus.PARSE_UNCERTAIN:
            ip.commit_path = CommitPath.PARSE_UNCERTAIN
            return ArbitratedWriteResult(
                commit_path=CommitPath.PARSE_UNCERTAIN, buffered=False,
                provisional=False, rejected=True, parse_packet=raw_pkt,
                verifier_result=vr, internalization_packet=ip,
            )
        
        # Verifier ACCEPTED on filtered packet
        entity_id = (composed_entity_id
                       if composed_entity_id is not None
                       else _top_entity(filtered_pkt))
        attr_type = _top_attribute(filtered_pkt)
        value_idx = _top_value_for(filtered_pkt, attr_type) if attr_type else None
        
        if entity_id is None or attr_type is None or value_idx is None:
            ip.commit_path = CommitPath.PARSER_FAILURE
            return ArbitratedWriteResult(
                commit_path=CommitPath.PARSER_FAILURE, buffered=False,
                provisional=False, rejected=True, parse_packet=raw_pkt,
                verifier_result=vr, internalization_packet=ip,
            )
        
        # Cross-episode conflict check (unchanged)
        if self.stability_index.is_stable(entity_id, attr_type, episode_id):
            existing_slot = self.bank.find_by_entity_id(entity_id)
            if existing_slot is not None:
                rec = self.bank.get_record(existing_slot)
                slot = rec.attr_slots.get(attr_type)
                if (slot is not None and slot.present
                        and slot.value_idx != value_idx):
                    entry = ProvisionalEntry(
                        entity_id=entity_id, attr_type=attr_type,
                        value_idx=value_idx, episode_id=episode_id,
                        write_step=write_step, source_text=fact_text,
                        internalization_packet_ref=ip,
                        challenge_kind="cross_episode_challenger",
                    )
                    self.provisional_memory.add(entry)
                    ip.commit_path = CommitPath.STORE_PROVISIONAL
                    return ArbitratedWriteResult(
                        commit_path=CommitPath.STORE_PROVISIONAL,
                        buffered=False, provisional=True, rejected=False,
                        parse_packet=raw_pkt, verifier_result=vr,
                        internalization_packet=ip,
                    )
        
        # Buffer for end_episode
        try:
            ent_emb = entity_emb_fn(entity_id)
        except Exception:
            ent_emb = None
        class_id = _entity_class_id(entity_id)
        try:
            cls_emb = (class_emb_fn(class_id, ent_emb)
                          if ent_emb is not None else None)
        except Exception:
            cls_emb = None
        try:
            val_emb = value_emb_fn(attr_type, value_idx)
        except Exception:
            val_emb = None
        
        bw = BufferedWrite(
            entity_id=entity_id, attr_type=attr_type, value_idx=value_idx,
            write_step=write_step, source_text=fact_text,
            parse_packet=raw_pkt,
            internalization_packet=ip,
            entity_emb_cache=ent_emb, class_id_cache=class_id,
            class_emb_cache=cls_emb, value_emb_cache=val_emb,
        )
        self.episode_buffer.add_write(bw)
        ip.commit_path = CommitPath.COMMIT
        return ArbitratedWriteResult(
            commit_path=CommitPath.COMMIT, buffered=True, provisional=False,
            rejected=False, parse_packet=raw_pkt, verifier_result=vr,
            internalization_packet=ip,
        )


print("[v15.6 Pas 6] Section R3: CommitArbiterPas6 integration")
print("        - RoMR runs after v15.4 parse, before verifier")
print("        - Shallow copy of packet; raw preserved; filtered passed to verifier")
print("        - Query path untouched (RoMR fact-only)")
print("        - All Pas 1/2/3 invariants preserved")
# ===========================================================================
# v15.6 PAS 6 — Section R4: evaluator + 7 acceptance gates + F2 re-diagnosis
# ===========================================================================


V15_6_PAS6_ACCEPTANCE = {
    "F2_safe_resolution_min":     0.95,
    "F2_harmful_commit_max":      0.0,      # strict zero
    "wrong_commit_max_per_family": 0.02,
    "S5_honesty_min":             0.95,
    "S5_overcommit_max":          0.02,
    "S6_honesty_min":             0.95,
    "S6_overcommit_max":          0.02,
    "F4_safe_resolution_min":     0.99,
    "F2_attr_write_fail_max":     0.05,     # after RoMR, attr_write_failure
                                               # must drop from 21.8% to <= 5%
}


def _v15_6_pas6_run_arbitrated_episode(arbiter: CommitArbiterPas6,
                                           reader: ReadArbiter,
                                           ep,
                                           episode_id: int,
                                           entity_emb_fn,
                                           class_emb_fn,
                                           value_emb_fn
                                           ) -> ArbitratedTrialOutcome:
    arbiter.begin_episode(episode_id)
    write_paths = []
    for j, fact_text in enumerate(ep.facts):
        result = arbiter.write_fact(fact_text, entity_emb_fn, class_emb_fn,
                                       value_emb_fn, write_step=j)
        write_paths.append(result.commit_path.value)
    finalize = arbiter.end_episode(entity_emb_fn, class_emb_fn, value_emb_fn)
    end_decisions = {f"{k[0]}::{k[1]}": v
                        for k, v in finalize.decisions_per_slot.items()}
    rd = reader.read_query(ep.query)
    
    target_value_idx = None
    if not ep.target_is_unknown:
        attr_type = HOLDOUT_ATTR_TYPES[ep.query_attr_label]
        vocab = HOLDOUT_ATTR_VALUES[attr_type]
        for k, vstr in enumerate(vocab):
            if V15_ANSWER_TOKENS.get(attr_type, {}).get(vstr) == ep.target_answer_token:
                target_value_idx = k
                break
    
    return ArbitratedTrialOutcome(
        family=ep.family_tag,
        target_is_unknown=ep.target_is_unknown,
        target_value_idx=target_value_idx,
        arbitrated_status=rd.status,
        pred_value=rd.pred,
        disputed_values=rd.disputed_values,
        commit_path_at_write=write_paths,
        end_episode_decisions=end_decisions,
    )


def v15_6_pas6_run_full_evaluation(bank, base_model, v15_1_memory,
                                       pas3_baseline: Optional[Dict] = None):
    """Run full Pas 6 evaluation with 7 gates + F2 attr_write_failure check."""
    print()
    print(SEP)
    print("[v15.6 PAS 6 EVALUATION]")
    print(SEP)
    
    ent_fn = _make_entity_emb_fn(base_model)
    cls_fn = _make_class_emb_fn(v15_1_memory)
    val_fn = _make_value_emb_fn(base_model)
    class_map = _v15_5_build_class_map()
    
    # Gate 0: trusted snapshot BEFORE
    print("\n[v15.6 Pas 6] Trusted snapshot BEFORE (Gate 0)")
    snap_before = _v15_5_snapshot_trusted(bank, base_model, v15_1_memory)
    for k, v in snap_before.items():
        print(f"  {k}: {v:.4f}")
    
    bank.reset()
    provisional_memory = ProvisionalMemory()
    episode_buffer     = EpisodeBuffer()
    stability_index    = BankStabilityIndex()
    composer_traces: List[ComposerTrace] = []
    romr_traces: List[RoMRResult] = []
    arbiter = CommitArbiterPas6(bank, provisional_memory, episode_buffer,
                                    stability_index,
                                    composer_trace_log=composer_traces,
                                    romr_trace_log=romr_traces)
    reader = ReadArbiter(bank, provisional_memory)
    
    family_results = {}
    seed_offset = 100000
    episode_counter = 1
    f2_outcomes_for_diag: List[Tuple[ArbitratedTrialOutcome, RoMRResult]] = []
    
    print("\n[v15.6 Pas 6] Running 5 holdout families (n={} each)".format(
        V15_5_HOLDOUT_CONFIG["n_per_family"]))
    for fname, gen in EXTERNAL_HOLDOUT_FAMILIES.items():
        print(f"  -> {fname}")
        rng = _rng_module.Random(V15_5_HOLDOUT_CONFIG["seed"] + seed_offset)
        outcomes = []
        for trial in range(V15_5_HOLDOUT_CONFIG["n_per_family"]):
            ep = gen(rng, ENC, class_map)
            bank.reset()
            provisional_memory.reset()
            episode_buffer.clear()
            stability_index.reset()
            romr_traces_start = len(romr_traces)
            o = _v15_6_pas6_run_arbitrated_episode(arbiter, reader, ep,
                                                     episode_counter,
                                                     ent_fn, cls_fn, val_fn)
            outcomes.append(o)
            if fname == "F2_multiword_entities":
                # Capture the RoMR trace for this trial (last trace added)
                if len(romr_traces) > romr_traces_start:
                    f2_outcomes_for_diag.append((o, romr_traces[romr_traces_start]))
                else:
                    f2_outcomes_for_diag.append((o, None))
            episode_counter += 1
        scored = _v15_6_score_family(outcomes)
        family_results[fname] = scored
        print(f"     commit_correct={scored['commit_correct_rate']:.3f} "
              f"prov_correct={scored['provisional_correct_rate']:.3f} "
              f"unc={scored['uncertain_rate']:.3f} "
              f"wrong_commit={scored['wrong_commit_rate']:.3f} "
              f"parser_fail={scored['parser_failure_rate']:.3f}")
        seed_offset += 1000
    
    s_results = {}
    print("\n[v15.6 Pas 6] Running S-probes (n={} each)".format(
        V15_5_HOLDOUT_CONFIG["n_per_s_probe"]))
    for sname, gen in EXTERNAL_HOLDOUT_S_PROBES.items():
        print(f"  -> {sname}")
        rng = _rng_module.Random(V15_5_HOLDOUT_CONFIG["seed"] + seed_offset)
        outcomes = []
        for trial in range(V15_5_HOLDOUT_CONFIG["n_per_s_probe"]):
            ep = gen(rng, ENC, class_map)
            bank.reset()
            provisional_memory.reset()
            episode_buffer.clear()
            stability_index.reset()
            o = _v15_6_pas6_run_arbitrated_episode(arbiter, reader, ep,
                                                     episode_counter,
                                                     ent_fn, cls_fn, val_fn)
            outcomes.append(o)
            episode_counter += 1
        scored = _v15_6_score_family(outcomes)
        s_results[sname] = scored
        print(f"     honesty={scored['honesty']:.3f} "
              f"overcommit={scored['overcommit']:.3f} "
              f"unc={scored['uncertain_rate']:.3f}")
        seed_offset += 1000
    
    # F2 re-diagnosis: compare attr_write_failure rate before/after RoMR
    print("\n[v15.6 Pas 6] F2 RE-DIAGNOSIS after RoMR")
    n_f2 = len(f2_outcomes_for_diag)
    n_attr_write_fail = 0
    n_romr_packet_conflict = 0
    n_romr_entity_modifier_dropped = 0
    for (o, r) in f2_outcomes_for_diag:
        if (not o.target_is_unknown
                and o.arbitrated_status == READ_STATUS_PARSE_UNCERTAIN):
            # Similar shape to attr_write_failure but measured post-RoMR
            n_attr_write_fail += 1
        if r is not None:
            if r.packet_level_conflict:
                n_romr_packet_conflict += 1
            n_romr_entity_modifier_dropped += sum(
                1 for c in r.token_classifications
                if c.label == RoleLabel.ENTITY_MODIFIER
            )
    
    attr_write_fail_rate = n_attr_write_fail / max(1, n_f2)
    print(f"  n_f2:                                 {n_f2}")
    print(f"  post-RoMR attr_write_fail count:      {n_attr_write_fail}  "
          f"(rate={attr_write_fail_rate:.3f})")
    print(f"  trials with REAL_CONFLICT:            {n_romr_packet_conflict}")
    print(f"  total ENTITY_MODIFIER tokens dropped: {n_romr_entity_modifier_dropped}")
    
    # Gate 0: trusted snapshot AFTER
    print("\n[v15.6 Pas 6] Trusted snapshot AFTER (Gate 0)")
    bank.reset()
    snap_after = _v15_5_snapshot_trusted(bank, base_model, v15_1_memory)
    match, bad_k, vb, va = _v15_5_trusted_signatures_match(snap_before,
                                                              snap_after)
    if match:
        print("  PASS: trusted regression check")
    else:
        print(f"  FAIL: regression on '{bad_k}': before={vb} after={va}")
    
    # ACCEPTANCE GATES (7)
    print("\n" + SEP)
    print("=== V15.6 PAS 6 ACCEPTANCE GATES ===")
    print(SEP)
    
    A = V15_6_PAS6_ACCEPTANCE
    checks = {}
    
    # Gate 0
    checks["Gate 0: trusted regression byte-identical"] = match
    
    # Gate 1: wrong_commit on every family (hard)
    for fname, r in family_results.items():
        checks[f"Gate 1: {fname} wrong_commit <= {A['wrong_commit_max_per_family']:.2f}"] = (
            r["wrong_commit_rate"] <= A["wrong_commit_max_per_family"]
        )
    
    # Gate 2: F2 safe_resolution >= 0.95
    f2 = family_results.get("F2_multiword_entities", {})
    f2_safe = f2.get("commit_correct_rate", 0) + f2.get("provisional_correct_rate", 0)
    checks[f"Gate 2: F2 safe_resolution >= {A['F2_safe_resolution_min']:.2f}"] = (
        f2_safe >= A["F2_safe_resolution_min"]
    )
    
    # Gate 3: F2 wrong_commit = 0
    checks[f"Gate 3: F2 wrong_commit == 0"] = (
        f2.get("wrong_commit_rate", 1.0) <= A["F2_harmful_commit_max"]
    )
    
    # Gate 4: S5/S6 honesty/overcommit
    s5 = s_results.get("S5_conflict_intercalated", {})
    checks[f"Gate 4: S5 honesty >= {A['S5_honesty_min']:.2f}"] = (
        (s5.get("honesty") if s5.get("honesty") is not None else 0) >= A["S5_honesty_min"]
    )
    checks[f"Gate 4: S5 overcommit <= {A['S5_overcommit_max']:.2f}"] = (
        (s5.get("overcommit") if s5.get("overcommit") is not None else 1) <= A["S5_overcommit_max"]
    )
    s6 = s_results.get("S6_entity_competition_cross", {})
    checks[f"Gate 4: S6 honesty >= {A['S6_honesty_min']:.2f}"] = (
        (s6.get("honesty") if s6.get("honesty") is not None else 0) >= A["S6_honesty_min"]
    )
    checks[f"Gate 4: S6 overcommit <= {A['S6_overcommit_max']:.2f}"] = (
        (s6.get("overcommit") if s6.get("overcommit") is not None else 1) <= A["S6_overcommit_max"]
    )
    
    # Gate 5: F4 safe_resolution
    f4 = family_results.get("F4_discourse_intercalation", {})
    f4_safe = f4.get("commit_correct_rate", 0) + f4.get("provisional_correct_rate", 0)
    checks[f"Gate 5: F4 safe_resolution >= {A['F4_safe_resolution_min']:.2f}"] = (
        f4_safe >= A["F4_safe_resolution_min"]
    )
    
    # Gate 6: F2 attr_write_failure rate post-RoMR
    checks[f"Gate 6: F2 attr_write_fail_rate <= {A['F2_attr_write_fail_max']:.2f}"] = (
        attr_write_fail_rate <= A["F2_attr_write_fail_max"]
    )
    
    all_pass = all(checks.values())
    for name, ok in checks.items():
        print(f"  {'✓' if ok else '✗'} {name}")
    
    # F1/F3/F5 informational (collateral only; not a target)
    print()
    print("[Pas 6 scope] F1/F3/F5 informational only — no direct patches")
    for fname in ("F1_novel_paraphrase_syntax", "F3_novel_lexical_alias",
                    "F5_novel_query_forms"):
        r = family_results.get(fname, {})
        print(f"  {fname}: commit_correct={r.get('commit_correct_rate', 0):.3f} "
              f"wrong_commit={r.get('wrong_commit_rate', 0):.3f}")
    
    # Delta vs Pas 3 baseline (F2 focus)
    if pas3_baseline is not None:
        p3_f2 = pas3_baseline.get("family_results", {}).get(
            "F2_multiword_entities", {})
        if p3_f2:
            print()
            print(SEP)
            print("=== F2 DELTA vs PAS 3 BASELINE ===")
            print(SEP)
            print(f"  commit_correct:     {p3_f2.get('commit_correct_rate', 0):.3f}"
                    f" -> {f2.get('commit_correct_rate', 0):.3f}")
            print(f"  provisional_correct: {p3_f2.get('provisional_correct_rate', 0):.3f}"
                    f" -> {f2.get('provisional_correct_rate', 0):.3f}")
            print(f"  uncertain:          {p3_f2.get('uncertain_rate', 0):.3f}"
                    f" -> {f2.get('uncertain_rate', 0):.3f}")
            print(f"  wrong_commit:       {p3_f2.get('wrong_commit_rate', 0):.3f}"
                    f" -> {f2.get('wrong_commit_rate', 0):.3f}")
            print(f"  parser_fail:        {p3_f2.get('parser_failure_rate', 0):.3f}"
                    f" -> {f2.get('parser_failure_rate', 0):.3f}")
    
    print()
    print(SEP)
    verdict = "PAS 6 PASSED" if all_pass else "PAS 6 PARTIAL"
    print(f"VERDICT: {verdict}")
    print(SEP)
    
    return {
        "snap_before":            snap_before,
        "snap_after":             snap_after,
        "trusted_regression_ok":  match,
        "family_results":         family_results,
        "s_results":              s_results,
        "f2_attr_write_fail_rate_post_romr": attr_write_fail_rate,
        "f2_romr_packet_conflict_count":     n_romr_packet_conflict,
        "f2_romr_entity_modifier_dropped":   n_romr_entity_modifier_dropped,
        "checks":                 checks,
        "all_pass":               all_pass,
        "verdict":                verdict,
    }


print("[v15.6 Pas 6] Section R4: evaluator + 7 gates defined")
print("        - Gate 0: trusted regression")
print("        - Gate 1: wrong_commit <= 2% all families")
print("        - Gate 2: F2 safe_resolution >= 0.95")
print("        - Gate 3: F2 wrong_commit == 0 (strict)")
print("        - Gate 4: S5/S6 honesty + overcommit preserved")
print("        - Gate 5: F4 safe_resolution >= 0.99")
print("        - Gate 6: F2 attr_write_fail_rate <= 0.05 post-RoMR")

# ======================== G9. V15.6 PAS 6 RUNTIME DISPATCH ================
V15_6_PAS6_MODE = os.environ.get("V15_6_PAS6_MODE", "pas6_full")
V15_6_DIR         = os.path.join(PROJECT_ROOT, "v15_6")
V15_6_RESULTS_DIR = os.path.join(V15_6_DIR, "results")
os.makedirs(V15_6_RESULTS_DIR, exist_ok=True)
print(f"[v15.6 Pas 6] workspace: {V15_6_DIR}")
print(f"[v15.6 Pas 6] MODE: {V15_6_PAS6_MODE}")
V15_2_SHADOW_CKPT = os.path.join(PROJECT_ROOT, "v15_2", "checkpoints", "shadow_final.pt")
print()
print(SEP)
print("[v15.6 Pas 6] Instantiating base model + v15.1 memory wrapper")
print(SEP)
DEFAULT_CONFIG = DCortexConfig()
base_model_v15_6   = DCortexV2Model(DEFAULT_CONFIG).to(DEVICE)
v15_1_memory_v15_6 = V15_1_Memory(d_model=DEFAULT_CONFIG.hidden_dim, d_sem=64).to(DEVICE)
v15_6_bank         = DeterministicObjectBank(capacity=64, d_model=DEFAULT_CONFIG.hidden_dim)
print()
print(SEP)
print("[v15.6 Pas 6] Phase A: Pas 1 equivalence test")
print(SEP)
eq_result = v15_6_pas1_equivalence_test(base_model_v15_6, v15_1_memory_v15_6, n_trials=500)
if not eq_result["pass"]:
    raise RuntimeError("Pas 1 equivalence failed; refusing Pas 6 run.")
print()
if os.path.exists(V15_2_SHADOW_CKPT):
    print(f"[v15.6 Pas 6] Loading shadow checkpoint: {V15_2_SHADOW_CKPT}")
    ckpt = torch.load(V15_2_SHADOW_CKPT, map_location=DEVICE, weights_only=False)
    v15_1_memory_v15_6.shadow.load_state_dict(ckpt["shadow_state"])
    v15_1_memory_v15_6.shadow.eval()
    print("[v15.6 Pas 6] shadow loaded (frozen)")
else:
    print("[v15.6 Pas 6] Shadow checkpoint missing; regenerating.")
    v15_2_train_shadow_main(v15_6_bank, base_model_v15_6, v15_1_memory_v15_6)
    os.makedirs(os.path.dirname(V15_2_SHADOW_CKPT), exist_ok=True)
    torch.save({"shadow_state": v15_1_memory_v15_6.shadow.state_dict()}, V15_2_SHADOW_CKPT)
    v15_1_memory_v15_6.shadow.eval()
PAS3_BASELINE_PATH = os.path.join(V15_6_RESULTS_DIR, "v15_6_pas3_composer.json")
pas3_baseline = None
if os.path.exists(PAS3_BASELINE_PATH):
    try:
        with open(PAS3_BASELINE_PATH) as f:
            pas3_baseline = json.load(f).get("pas3")
        print(f"\n[v15.6 Pas 6] Pas 3 baseline loaded for delta comparison")
    except Exception as e:
        print(f"\n[v15.6 Pas 6] Pas 3 baseline load failed: {e}")
print()
print(SEP)
print("[v15.6 Pas 6] Phase D: Full evaluation with RoMR active")
print(SEP)
pas6_result = v15_6_pas6_run_full_evaluation(v15_6_bank, base_model_v15_6,
                                                 v15_1_memory_v15_6,
                                                 pas3_baseline=pas3_baseline)
raw_path = os.path.join(V15_6_RESULTS_DIR, "v15_6_pas6_romr.json")
with open(raw_path, "w") as f:
    json.dump({"pas1_equivalence": eq_result, "pas6": pas6_result}, f, indent=2, default=str)
print(f"[v15.6 Pas 6] raw: {raw_path}")
print()
print(SEP)
print(f"FINAL PAS 6 VERDICT: {pas6_result['verdict']}")
print(SEP)
