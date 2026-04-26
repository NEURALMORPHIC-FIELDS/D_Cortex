# -*- coding: utf-8 -*-
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
