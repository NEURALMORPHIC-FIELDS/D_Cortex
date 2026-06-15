"""Regression tests for explicit-referent grounded object reads."""

from dcortex.semantic_adapter import (
    ConservativeSemanticAdapter,
    HypothesisMode,
    RequestedDestination,
    SemanticHypothesis,
)
from dcortex.semantic_grounded_reader import (
    ExplicitReferentGroundingGate,
    GroundedSemanticObjectReader,
    ReferentGroundingStatus,
)
from dcortex.semantic_object_reader import ObjectMemorySnapshot, ObjectReadStatus


def approved_query(source_text: str, entity: str = "dragon"):
    """Build one approved semantic query hypothesis."""
    hypothesis = SemanticHypothesis(
        hypothesis_id="grounded-query",
        episode_id=1,
        mode=HypothesisMode.QUERY,
        source_text=source_text,
        producer="test",
        producer_version="1.0",
        provenance=("test",),
        confidence=0.99,
        uncertainty=0.01,
        requested_destination=RequestedDestination.QUERY_ONLY,
        entity_id=entity,
        attr_type="color",
    )
    return hypothesis, ConservativeSemanticAdapter().submit(hypothesis)


def test_explicit_referent_is_grounded_and_read() -> None:
    hypothesis, decision = approved_query("What color is the dragon?")
    snapshot = ObjectMemorySnapshot(committed=(("dragon", "color", "red"),))
    result = GroundedSemanticObjectReader().read(snapshot, hypothesis, decision)
    assert result.grounding.status == ReferentGroundingStatus.GROUNDED
    assert result.grounding.matched_span is not None
    assert result.read.status == ObjectReadStatus.FOUND_COMMITTED
    assert result.read.pred_value == "red"


def test_pronoun_only_query_is_rejected_before_read() -> None:
    hypothesis, decision = approved_query("What color is it?")
    snapshot = ObjectMemorySnapshot(committed=(("dragon", "color", "red"),))
    result = GroundedSemanticObjectReader().read(snapshot, hypothesis, decision)
    assert result.grounding.status == ReferentGroundingStatus.REJECTED
    assert result.read.status == ObjectReadStatus.REFUSED_INPUT
    assert result.read.pred_value is None


def test_multiword_entity_requires_contiguous_exact_tokens() -> None:
    gate = ExplicitReferentGroundingGate()
    grounded, grounded_decision = approved_query(
        "Where is the silver dragon?", "silver dragon"
    )
    separated, separated_decision = approved_query(
        "The silver object was near a dragon.", "silver dragon"
    )
    assert (
        gate.ground(grounded, grounded_decision).status
        == ReferentGroundingStatus.GROUNDED
    )
    assert (
        gate.ground(separated, separated_decision).status
        == ReferentGroundingStatus.REJECTED
    )


def test_grounded_read_preserves_snapshot() -> None:
    hypothesis, decision = approved_query("What color is the dragon?")
    snapshot = ObjectMemorySnapshot(committed=(("dragon", "color", "red"),))
    before = snapshot.fingerprint
    result = GroundedSemanticObjectReader().read(snapshot, hypothesis, decision)
    assert before == snapshot.fingerprint == result.read.snapshot_fingerprint
