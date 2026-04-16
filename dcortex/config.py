# -*- coding: utf-8 -*-
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
    theta_match: float = 0.7
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
