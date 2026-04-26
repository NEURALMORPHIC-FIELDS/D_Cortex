# -*- coding: utf-8 -*-
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
