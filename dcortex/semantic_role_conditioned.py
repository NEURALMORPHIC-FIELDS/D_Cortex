# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b-RB2
# Token-level role-conditioned semantic assignment scorer.

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import torch

from dcortex.semantic_role_binder import (
    ASSIGNMENT_ORDER,
    RoleBindingAssignment,
)

ROLE_NONE = 0
ROLE_ENTITY_A = 1
ROLE_VALUE_A = 2
ROLE_ENTITY_B = 3
ROLE_VALUE_B = 4
ROLE_COUNT = 5


@dataclass(frozen=True)
class TokenContextFeatures:
    """Frozen contextual token states with their token and validity masks."""

    hidden: torch.Tensor
    attention_mask: torch.Tensor
    token_ids: torch.Tensor


@dataclass(frozen=True)
class RoleMaskAudit:
    """Auditable mention coverage for the three assignment role masks."""

    phrase_token_counts: Tuple[Tuple[str, int], ...]
    identity_marked_tokens: int
    swapped_marked_tokens: int
    unresolved_marked_tokens: int
    entity_roles_unchanged: bool
    value_roles_swapped: bool

    @property
    def complete(self) -> bool:
        """Return whether every supplied phrase was found and masks are valid."""
        return (
            all(count > 0 for _, count in self.phrase_token_counts)
            and self.identity_marked_tokens > 0
            and self.swapped_marked_tokens > 0
            and self.unresolved_marked_tokens == 0
            and self.entity_roles_unchanged
            and self.value_roles_swapped
        )


class DCortexTokenContextBackend:
    """Return frozen final decoder-standard token states without memory access."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Callable[[str], List[int]],
        max_seq_len: int = 128,
        batch_size: int = 128,
        backend_version: str = "1.0",
    ) -> None:
        if max_seq_len < 1 or batch_size < 1:
            raise ValueError("max_seq_len and batch_size must be positive")
        required = (
            "shared_token_emb",
            "shared_pos_emb",
            "dec_emb_norm",
            "dec_emb_drop",
            "dec_standard_blocks",
        )
        missing = [name for name in required if not hasattr(model, name)]
        if missing:
            raise ValueError(f"contextual model lacks required attributes: {missing}")
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.backend_id = "dcortex_frozen_contextual_token_sequence"
        self.backend_version = backend_version

    @property
    def output_dim(self) -> int:
        """Return the contextual token-state width."""
        return int(self.model.shared_token_emb.embedding_dim)

    def features(self, texts: Sequence[str]) -> TokenContextFeatures:
        """Return padded frozen token states and masks for source texts."""
        if not texts:
            raise ValueError("texts must not be empty")
        encoded = [self.tokenizer(text)[: self.max_seq_len] for text in texts]
        if any(not item for item in encoded):
            raise ValueError("every text must produce at least one token")
        width = max(len(item) for item in encoded)
        device = next(self.model.parameters()).device
        hidden_batches: List[torch.Tensor] = []
        mask_batches: List[torch.Tensor] = []
        id_batches: List[torch.Tensor] = []
        for start in range(0, len(encoded), self.batch_size):
            batch = encoded[start : start + self.batch_size]
            ids = torch.zeros(len(batch), width, dtype=torch.long, device=device)
            mask = torch.zeros(len(batch), width, dtype=torch.bool, device=device)
            for row, tokens in enumerate(batch):
                ids[row, : len(tokens)] = torch.tensor(
                    tokens, dtype=torch.long, device=device
                )
                mask[row, : len(tokens)] = True
            positions = torch.arange(width, device=device).unsqueeze(0)
            with torch.inference_mode():
                hidden = self.model.shared_token_emb(ids) + self.model.shared_pos_emb(
                    positions
                )
                hidden = self.model.dec_emb_drop(self.model.dec_emb_norm(hidden))
                for block in self.model.dec_standard_blocks:
                    hidden = block(hidden, mask.long())
            hidden_batches.append(hidden.detach().float().cpu())
            mask_batches.append(mask.cpu())
            id_batches.append(ids.cpu())
        return TokenContextFeatures(
            hidden=torch.cat(hidden_batches, dim=0),
            attention_mask=torch.cat(mask_batches, dim=0),
            token_ids=torch.cat(id_batches, dim=0),
        )


def _subsequence_spans(
    source: Sequence[int],
    phrase: Sequence[int],
) -> Tuple[Tuple[int, int], ...]:
    if not phrase or len(phrase) > len(source):
        return ()
    return tuple(
        (start, start + len(phrase))
        for start in range(0, len(source) - len(phrase) + 1)
        if tuple(source[start : start + len(phrase)]) == tuple(phrase)
    )


def phrase_token_positions(
    source_token_ids: Sequence[int],
    phrase: str,
    tokenizer: Callable[[str], List[int]],
) -> Tuple[int, ...]:
    """Return every exact token position occupied by a supplied phrase."""
    variants = {
        tuple(tokenizer(phrase)),
        tuple(tokenizer(" " + phrase)),
    }
    spans = {
        span
        for variant in variants
        for span in _subsequence_spans(source_token_ids, variant)
    }
    return tuple(
        sorted({position for start, end in spans for position in range(start, end)})
    )


def build_role_masks(
    source_token_ids: Sequence[int],
    entities: Sequence[str],
    values: Sequence[str],
    tokenizer: Callable[[str], List[int]],
) -> Tuple[torch.Tensor, RoleMaskAudit]:
    """Build identity, swapped, and unresolved candidate role masks."""
    sorted_entities = tuple(sorted(entities))
    sorted_values = tuple(sorted(values))
    if len(sorted_entities) != 2 or len(set(sorted_entities)) != 2:
        raise ValueError("exactly two distinct entities are required")
    if len(sorted_values) != 2 or len(set(sorted_values)) != 2:
        raise ValueError("exactly two distinct values are required")
    positions: Dict[str, Tuple[int, ...]] = {
        phrase: phrase_token_positions(source_token_ids, phrase, tokenizer)
        for phrase in sorted_entities + sorted_values
    }

    def candidate_mask(assignments: Sequence[Tuple[str, int]]) -> torch.Tensor:
        mask = torch.zeros(len(source_token_ids), dtype=torch.long)
        for phrase, role in assignments:
            for position in positions[phrase]:
                existing = int(mask[position])
                if existing not in (ROLE_NONE, role):
                    raise ValueError("overlapping supplied phrases create conflicting roles")
                mask[position] = role
        return mask

    identity = candidate_mask(
        (
            (sorted_entities[0], ROLE_ENTITY_A),
            (sorted_values[0], ROLE_VALUE_A),
            (sorted_entities[1], ROLE_ENTITY_B),
            (sorted_values[1], ROLE_VALUE_B),
        )
    )
    swapped = candidate_mask(
        (
            (sorted_entities[0], ROLE_ENTITY_A),
            (sorted_values[1], ROLE_VALUE_A),
            (sorted_entities[1], ROLE_ENTITY_B),
            (sorted_values[0], ROLE_VALUE_B),
        )
    )
    unresolved = torch.zeros_like(identity)
    entity_positions = tuple(
        sorted(
            set(positions[sorted_entities[0]]).union(positions[sorted_entities[1]])
        )
    )
    value_positions = tuple(
        sorted(set(positions[sorted_values[0]]).union(positions[sorted_values[1]]))
    )
    entity_roles_unchanged = all(
        int(identity[position]) == int(swapped[position]) for position in entity_positions
    )
    value_roles_swapped = all(
        {
            int(identity[position]),
            int(swapped[position]),
        }
        == {ROLE_VALUE_A, ROLE_VALUE_B}
        for position in value_positions
    )
    audit = RoleMaskAudit(
        phrase_token_counts=tuple(
            (phrase, len(positions[phrase]))
            for phrase in sorted_entities + sorted_values
        ),
        identity_marked_tokens=int((identity != ROLE_NONE).sum()),
        swapped_marked_tokens=int((swapped != ROLE_NONE).sum()),
        unresolved_marked_tokens=int((unresolved != ROLE_NONE).sum()),
        entity_roles_unchanged=entity_roles_unchanged,
        value_roles_swapped=value_roles_swapped,
    )
    return torch.stack((identity, swapped, unresolved)), audit


class RoleConditionedSequenceScoringHead(torch.nn.Module):
    """Score complete assignments from frozen token states plus role masks."""

    def __init__(
        self,
        context_dim: int,
        projection_dim: int = 128,
        role_embedding_dim: int = 32,
        recurrent_hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(
            context_dim,
            projection_dim,
            role_embedding_dim,
            recurrent_hidden_dim,
        ) < 1:
            raise ValueError("all dimensions must be positive")
        self.context_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(context_dim),
            torch.nn.Linear(context_dim, projection_dim),
            torch.nn.GELU(),
        )
        self.role_embedding = torch.nn.Embedding(ROLE_COUNT, role_embedding_dim)
        self.sequence_model = torch.nn.GRU(
            input_size=projection_dim + role_embedding_dim,
            hidden_size=recurrent_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        sequence_width = recurrent_hidden_dim * 2
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(sequence_width, recurrent_hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(recurrent_hidden_dim, 1),
        )
        self.output = torch.nn.Sequential(
            torch.nn.LayerNorm(sequence_width),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(sequence_width, 1),
        )

    def forward(
        self,
        contextual_tokens: torch.Tensor,
        role_masks: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return one scalar score per candidate token sequence."""
        if contextual_tokens.ndim != 3:
            raise ValueError("contextual_tokens must have shape [batch, seq, dim]")
        if role_masks.shape != contextual_tokens.shape[:2]:
            raise ValueError("role_masks shape must match token batch and sequence")
        if attention_mask.shape != contextual_tokens.shape[:2]:
            raise ValueError("attention_mask shape must match token batch and sequence")
        projected = self.context_projection(contextual_tokens)
        roles = self.role_embedding(role_masks)
        combined = torch.cat((projected, roles), dim=-1)
        lengths = attention_mask.long().sum(dim=1).cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            combined,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.sequence_model(packed)
        sequence, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=contextual_tokens.shape[1],
        )
        attention_logits = self.attention(sequence).squeeze(-1)
        attention_logits = attention_logits.masked_fill(
            ~attention_mask.bool(), float("-inf")
        )
        weights = torch.softmax(attention_logits, dim=1)
        pooled = (sequence * weights.unsqueeze(-1)).sum(dim=1)
        return self.output(pooled).squeeze(-1)


class RoleConditionedRecordScorer:
    """Compose token extraction, role masks, and the learned sequence head."""

    def __init__(
        self,
        token_backend: DCortexTokenContextBackend,
        head: RoleConditionedSequenceScoringHead,
    ) -> None:
        self.token_backend = token_backend
        self.head = head
        self.backend_id = "dcortex_role_conditioned_sequence_scorer"
        self.backend_version = "1.0"

    def score_record(
        self,
        source_text: str,
        entities: Sequence[str],
        values: Sequence[str],
    ) -> Tuple[torch.Tensor, RoleMaskAudit]:
        """Return identity, swapped, and unresolved scores for one source."""
        context = self.token_backend.features([source_text])
        length = int(context.attention_mask[0].sum())
        role_masks, audit = build_role_masks(
            context.token_ids[0, :length].tolist(),
            entities,
            values,
            self.token_backend.tokenizer,
        )
        device = next(self.head.parameters()).device
        token_states = context.hidden[:, :length].expand(len(ASSIGNMENT_ORDER), -1, -1)
        attention = context.attention_mask[:, :length].expand(
            len(ASSIGNMENT_ORDER), -1
        )
        with torch.inference_mode():
            scores = self.head(
                token_states.to(device),
                role_masks.to(device),
                attention.to(device),
            )
        return scores.detach().float().cpu(), audit
