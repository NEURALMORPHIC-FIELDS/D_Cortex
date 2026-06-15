"""Regression tests for the conservative learned semantic role binder."""

from typing import Sequence

import torch

from dcortex.semantic_adapter import (
    ConservativeSemanticAdapter,
    DecisionStatus,
    RequestedDestination,
)
from dcortex.semantic_role_binder import (
    ConservativeLearnedRoleBinder,
    RoleBindingAssignment,
    RoleBindingScoringBackend,
    assignment_facts,
    candidate_views,
    expected_assignment,
)


class FixedRoleBindingBackend(RoleBindingScoringBackend):
    """Test-only fixed assignment scorer."""

    backend_id = "fixed"
    backend_version = "test"

    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = torch.tensor(scores, dtype=torch.float32)

    def score(self, views: Sequence[str]) -> torch.Tensor:
        assert len(views) == 3
        return self.scores


def test_assignment_contract_and_candidate_views_are_complete() -> None:
    identity = assignment_facts("color", ("wolf", "key"), ("red", "blue"), RoleBindingAssignment.IDENTITY)
    swapped = assignment_facts("color", ("wolf", "key"), ("red", "blue"), RoleBindingAssignment.SWAPPED)
    assert identity != swapped
    assert expected_assignment("color", ("wolf", "key"), ("red", "blue"), identity) == RoleBindingAssignment.IDENTITY
    assert expected_assignment("color", ("wolf", "key"), ("red", "blue"), swapped) == RoleBindingAssignment.SWAPPED
    assert expected_assignment("color", ("wolf", "key"), ("red", "blue"), ()) == RoleBindingAssignment.UNRESOLVED
    views = candidate_views("The wolf is red and the key is blue.", "color", ("wolf", "key"), ("red", "blue"))
    assert len(views) == 3
    assert all("Source statement:" in view for view in views)


def test_supported_mapping_emits_exactly_two_provisional_facts() -> None:
    adapter = ConservativeSemanticAdapter()
    binder = ConservativeLearnedRoleBinder(
        FixedRoleBindingBackend((6.0, 0.0, -2.0)),
        adapter,
        margin_threshold=0.20,
    )
    result = binder.produce(
        "binding",
        7,
        "The key is blue, while the wolf is red.",
        "color",
        ("wolf", "key"),
        ("red", "blue"),
        provenance=("test:identity",),
    )
    assert result.emitted
    assert len(result.facts) == 2
    assert len(result.hypotheses) == 2
    assert all(
        item.requested_destination == RequestedDestination.PROVISIONAL_ONLY
        for item in result.hypotheses
    )
    assert all(
        item.status == DecisionStatus.ACCEPT_PROVISIONAL
        for item in result.adapter_decisions
    )


def test_unresolved_and_low_margin_predictions_abstain() -> None:
    unresolved = ConservativeLearnedRoleBinder(
        FixedRoleBindingBackend((0.0, 0.0, 6.0)),
        ConservativeSemanticAdapter(),
        margin_threshold=0.20,
    ).produce(
        "unresolved",
        8,
        "Either the key or wolf is blue; the other is red.",
        "color",
        ("wolf", "key"),
        ("red", "blue"),
        provenance=("test:unresolved",),
    )
    low_margin = ConservativeLearnedRoleBinder(
        FixedRoleBindingBackend((1.0, 0.99, -2.0)),
        ConservativeSemanticAdapter(),
        margin_threshold=0.20,
    ).produce(
        "low-margin",
        9,
        "The key is blue, while the wolf is red.",
        "color",
        ("wolf", "key"),
        ("red", "blue"),
        provenance=("test:low-margin",),
    )
    assert not unresolved.emitted and not unresolved.hypotheses
    assert not low_margin.emitted and not low_margin.hypotheses


def test_binder_exposes_no_direct_commit_path() -> None:
    forbidden = ("commit", "write", "consolidate", "promote")
    assert not any(hasattr(ConservativeLearnedRoleBinder, name) for name in forbidden)
