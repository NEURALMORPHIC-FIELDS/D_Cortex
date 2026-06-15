"""Regression tests for the frozen learned role-binder verdict helpers."""

import torch

from dcortex.semantic_role_binder import ASSIGNMENT_ORDER
from scripts.semantic_role_binder_verdict import (
    QUERY_SIDE_SEALS,
    load_rb0_sample,
    select_margin_threshold,
    sha256_file,
    split_records,
)
from scripts.semantic_role_binding_benchmark import REPO_ROOT


def test_frozen_rb0_split_is_disjoint_and_complete() -> None:
    records = load_rb0_sample(
        REPO_ROOT / "runs" / "semantic_role_binding_benchmark" / "results" / "sample.json"
    )
    splits = split_records(records)
    text_sets = {
        name: {record.text for record in subset} for name, subset in splits.items()
    }
    assert sum(len(items) for items in splits.values()) == len(records)
    assert not text_sets["train"].intersection(text_sets["validation"])
    assert not text_sets["train"].intersection(text_sets["test"])
    assert not text_sets["validation"].intersection(text_sets["test"])


def test_margin_calibration_uses_smallest_honest_grid_value() -> None:
    records = load_rb0_sample(
        REPO_ROOT / "runs" / "semantic_role_binding_benchmark" / "results" / "sample.json"
    )
    ambiguous = tuple(record for record in records if record.ambiguous)[:10]
    logits = torch.zeros(len(ambiguous), len(ASSIGNMENT_ORDER))
    logits[:, 2] = 2.0
    calibration = select_margin_threshold(ambiguous, logits)
    assert calibration["target_met"]
    assert calibration["threshold"] == 0.0


def test_query_side_expected_seal_hashes_match_sources() -> None:
    assert all(
        sha256_file(REPO_ROOT / path) == expected
        for path, expected in QUERY_SIDE_SEALS.items()
    )
