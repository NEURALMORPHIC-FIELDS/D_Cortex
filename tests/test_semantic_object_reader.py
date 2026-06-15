"""Regression tests for direct semantic-coordinate object-memory reads."""

from dataclasses import FrozenInstanceError

import pytest

from dcortex.semantic_adapter import (
    ConservativeSemanticAdapter,
    HypothesisMode,
    RequestedDestination,
    SemanticHypothesis,
)
from dcortex.semantic_object_reader import (
    DirectSemanticObjectReader,
    ObjectMemorySnapshot,
    ObjectReadStatus,
)


def approved_query(entity: str = "dragon", attribute: str = "color"):
    """Build one adapter-approved query-only semantic coordinate."""
    hypothesis = SemanticHypothesis(
        hypothesis_id=f"query-{entity}-{attribute}",
        episode_id=1,
        mode=HypothesisMode.QUERY,
        source_text="source text remains audit-only",
        producer="test",
        producer_version="1.0",
        provenance=("test",),
        confidence=0.99,
        uncertainty=0.01,
        requested_destination=RequestedDestination.QUERY_ONLY,
        entity_id=entity,
        attr_type=attribute,
    )
    return hypothesis, ConservativeSemanticAdapter().submit(hypothesis)


def test_committed_coordinate_read_is_direct_and_immutable() -> None:
    snapshot = ObjectMemorySnapshot(
        known_entities=("dragon",),
        committed=(("dragon", "color", "red"),),
    )
    hypothesis, decision = approved_query()
    before = snapshot.fingerprint
    result = DirectSemanticObjectReader().read(snapshot, hypothesis, decision)
    assert result.status == ObjectReadStatus.FOUND_COMMITTED
    assert result.pred_value == "red"
    assert snapshot.fingerprint == before == result.snapshot_fingerprint


def test_provisional_only_slot_is_disputed_without_prediction() -> None:
    snapshot = ObjectMemorySnapshot(
        provisional=(
            ("dragon", "color", "red"),
            ("dragon", "color", "blue"),
        )
    )
    hypothesis, decision = approved_query()
    result = DirectSemanticObjectReader().read(snapshot, hypothesis, decision)
    assert result.status == ObjectReadStatus.FOUND_DISPUTED
    assert result.pred_value is None
    assert result.disputed_values == ("blue", "red")


def test_committed_with_challenger_reports_dispute() -> None:
    snapshot = ObjectMemorySnapshot(
        committed=(("dragon", "color", "red"),),
        provisional=(("dragon", "color", "blue"),),
    )
    hypothesis, decision = approved_query()
    result = DirectSemanticObjectReader().read(snapshot, hypothesis, decision)
    assert result.status == ObjectReadStatus.FOUND_DISPUTED
    assert result.pred_value == "red"
    assert result.disputed_values == ("blue", "red")


def test_none_object_and_none_attribute_are_distinct() -> None:
    snapshot = ObjectMemorySnapshot(known_entities=("dragon",))
    reader = DirectSemanticObjectReader()
    dragon, dragon_decision = approved_query()
    wolf, wolf_decision = approved_query("wolf")
    assert (
        reader.read(snapshot, dragon, dragon_decision).status
        == ObjectReadStatus.NONE_ATTRIBUTE
    )
    assert reader.read(snapshot, wolf, wolf_decision).status == ObjectReadStatus.NONE_OBJECT


def test_missing_approval_is_refused() -> None:
    snapshot = ObjectMemorySnapshot(
        committed=(("dragon", "color", "red"),),
    )
    hypothesis, _ = approved_query()
    result = DirectSemanticObjectReader().read(snapshot, hypothesis, None)
    assert result.status == ObjectReadStatus.REFUSED_INPUT
    assert result.pred_value is None


def test_snapshot_rejects_multiple_committed_values_for_one_slot() -> None:
    with pytest.raises(ValueError):
        ObjectMemorySnapshot(
            committed=(
                ("dragon", "color", "red"),
                ("dragon", "color", "blue"),
            )
        )


def test_snapshot_is_frozen() -> None:
    snapshot = ObjectMemorySnapshot(known_entities=("dragon",))
    with pytest.raises(FrozenInstanceError):
        snapshot.known_entities = ("wolf",)  # type: ignore[misc]
