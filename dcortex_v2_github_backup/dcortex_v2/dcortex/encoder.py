# -*- coding: utf-8 -*-
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
