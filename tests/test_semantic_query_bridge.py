"""Regression tests for the v15.7b-R read-only semantic query bridge."""

from dcortex.semantic_adapter import (
    AdapterDecision,
    ConservativeSemanticAdapter,
    DecisionStatus,
    HypothesisMode,
    RequestedDestination,
    SemanticHypothesis,
)
from dcortex.semantic_query_bridge import (
    QueryRouteStatus,
    ReadOnlySemanticQueryBridge,
)


def make_query(
    hypothesis_id: str = "query-1",
    attr_type: str = "color",
) -> SemanticHypothesis:
    """Build one valid query hypothesis."""
    return SemanticHypothesis(
        hypothesis_id=hypothesis_id,
        episode_id=1,
        mode=HypothesisMode.QUERY,
        source_text="Which dye marks the dragon?",
        producer="test-producer",
        producer_version="1.0",
        provenance=("test:query",),
        confidence=0.9,
        uncertainty=0.1,
        requested_destination=RequestedDestination.QUERY_ONLY,
        entity_id="dragon",
        attr_type=attr_type,
    )


def accepted(hypothesis: SemanticHypothesis) -> AdapterDecision:
    """Submit a query through the conservative adapter."""
    return ConservativeSemanticAdapter().submit(hypothesis)


def test_accepted_query_routes_to_canonical_read() -> None:
    hypothesis = make_query()
    route = ReadOnlySemanticQueryBridge().route(
        hypothesis.source_text, hypothesis, accepted(hypothesis)
    )
    assert route.status == QueryRouteStatus.ROUTED
    assert route.routed_query == "What color is the dragon? The dragon is"
    assert route.original_query == hypothesis.source_text


def test_missing_decision_preserves_exact_fallback() -> None:
    original = "  Preserve this query exactly.  "
    route = ReadOnlySemanticQueryBridge().route(original, make_query(), None)
    assert route.status == QueryRouteStatus.FALLBACK
    assert route.routed_query == original


def test_rejected_decision_preserves_exact_fallback() -> None:
    hypothesis = make_query()
    decision = AdapterDecision(
        status=DecisionStatus.REJECT,
        hypothesis_id=hypothesis.hypothesis_id,
        reason_codes=("TEST_REJECT",),
        audit_sequence=1,
    )
    route = ReadOnlySemanticQueryBridge().route(
        hypothesis.source_text, hypothesis, decision
    )
    assert route.status == QueryRouteStatus.FALLBACK
    assert route.routed_query == hypothesis.source_text


def test_mismatched_decision_cannot_route() -> None:
    hypothesis = make_query()
    decision = AdapterDecision(
        status=DecisionStatus.ACCEPT_QUERY,
        hypothesis_id="different",
        reason_codes=("ACCEPTED_QUERY",),
        audit_sequence=1,
    )
    route = ReadOnlySemanticQueryBridge().route(
        hypothesis.source_text, hypothesis, decision
    )
    assert route.status == QueryRouteStatus.FALLBACK
    assert ReadOnlySemanticQueryBridge.REASON_ID_MISMATCH in route.reason_codes


def test_unsupported_attribute_cannot_route() -> None:
    hypothesis = make_query(attr_type="temperature")
    route = ReadOnlySemanticQueryBridge().route(
        hypothesis.source_text, hypothesis, accepted(hypothesis)
    )
    assert route.status == QueryRouteStatus.FALLBACK
    assert (
        ReadOnlySemanticQueryBridge.REASON_UNSUPPORTED_ATTRIBUTE
        in route.reason_codes
    )


def test_route_is_deterministic() -> None:
    def run() -> str:
        hypothesis = make_query()
        return ReadOnlySemanticQueryBridge().route(
            hypothesis.source_text, hypothesis, accepted(hypothesis)
        ).to_json()

    assert run() == run()


def test_bridge_runtime_state_contains_no_mutation_dependency() -> None:
    bridge = ReadOnlySemanticQueryBridge()
    forbidden = (
        "model",
        "memory",
        "bank",
        "reader",
        "writer",
        "commit",
        "provisional",
        "consolidator",
    )
    assert not any(hasattr(bridge, name) for name in forbidden)
