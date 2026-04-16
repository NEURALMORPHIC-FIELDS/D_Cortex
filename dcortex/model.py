# -*- coding: utf-8 -*-
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
            self.writer(h_pool_final, self.updater, self._bank_dict(), step)

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
