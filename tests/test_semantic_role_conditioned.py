"""Regression tests for token-level role-conditioned role binding."""

import torch

from dcortex.semantic_role_conditioned import (
    ROLE_ENTITY_A,
    ROLE_ENTITY_B,
    ROLE_NONE,
    ROLE_VALUE_A,
    ROLE_VALUE_B,
    RoleConditionedSequenceScoringHead,
    build_role_masks,
)


def character_tokenizer(text: str) -> list[int]:
    return [ord(character) for character in text]


def test_role_masks_mark_mentions_without_truth_input() -> None:
    text = "key is blue while wolf is red"
    token_ids = character_tokenizer(text)
    masks, audit = build_role_masks(
        token_ids,
        ("wolf", "key"),
        ("red", "blue"),
        character_tokenizer,
    )
    assert masks.shape == (3, len(token_ids))
    assert audit.complete
    assert int((masks[2] != ROLE_NONE).sum()) == 0
    assert set(masks[0].tolist()) >= {
        ROLE_NONE,
        ROLE_ENTITY_A,
        ROLE_ENTITY_B,
        ROLE_VALUE_A,
        ROLE_VALUE_B,
    }


def test_sequence_head_scores_three_candidate_sequences() -> None:
    head = RoleConditionedSequenceScoringHead(
        context_dim=16,
        projection_dim=8,
        role_embedding_dim=4,
        recurrent_hidden_dim=6,
        dropout=0.0,
    ).eval()
    context = torch.randn(3, 9, 16)
    roles = torch.zeros(3, 9, dtype=torch.long)
    attention = torch.ones(3, 9, dtype=torch.bool)
    scores = head(context, roles, attention)
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
