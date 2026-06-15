"""Regression tests for the frozen token-level role-conditioned verdict."""

import tiktoken
import torch

from dcortex.semantic_role_conditioned import build_role_masks
from scripts.semantic_role_binder_verdict import load_rb0_sample
from scripts.semantic_role_binding_benchmark import REPO_ROOT


def test_all_sealed_rb0_mentions_produce_complete_role_masks() -> None:
    tokenizer = tiktoken.get_encoding("gpt2").encode_ordinary
    records = load_rb0_sample(
        REPO_ROOT / "runs" / "semantic_role_binding_benchmark" / "results" / "sample.json"
    )
    for record in records:
        token_ids = tokenizer(record.text)
        masks, audit = build_role_masks(
            token_ids,
            record.entities,
            record.values,
            tokenizer,
        )
        assert masks.shape == (3, len(token_ids))
        assert audit.complete
        assert not torch.any(masks[2])
