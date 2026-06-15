"""Regression tests for the v15.7b conservative semantic adapter."""

import json

from dcortex.semantic_adapter import (
    ConservativeSemanticAdapter,
    DecisionStatus,
    HypothesisMode,
    RequestedDestination,
    SemanticHypothesis,
)


def make_fact(
    hypothesis_id: str = "h1",
    episode_id: int = 1,
    value_id: str = "blue",
    destination: RequestedDestination = RequestedDestination.PROVISIONAL_ONLY,
    provenance: tuple[str, ...] = ("source:1",),
    confidence: float = 0.8,
    uncertainty: float = 0.2,
) -> SemanticHypothesis:
    """Build one valid fact hypothesis."""
    return SemanticHypothesis(
        hypothesis_id=hypothesis_id,
        episode_id=episode_id,
        mode=HypothesisMode.FACT,
        source_text="The dragon appears blue.",
        producer="test-producer",
        producer_version="1.0",
        provenance=provenance,
        confidence=confidence,
        uncertainty=uncertainty,
        requested_destination=destination,
        entity_id="dragon",
        attr_type="color",
        value_id=value_id,
    )


def make_query(
    hypothesis_id: str = "q1",
    destination: RequestedDestination = RequestedDestination.QUERY_ONLY,
    value_id: str | None = None,
) -> SemanticHypothesis:
    """Build one query hypothesis."""
    return SemanticHypothesis(
        hypothesis_id=hypothesis_id,
        episode_id=1,
        mode=HypothesisMode.QUERY,
        source_text="What hue does the dragon have?",
        producer="test-producer",
        producer_version="1.0",
        provenance=("query:1",),
        confidence=0.75,
        uncertainty=0.25,
        requested_destination=destination,
        entity_id="dragon",
        attr_type="color",
        value_id=value_id,
    )


def test_valid_fact_is_provisional_only() -> None:
    adapter = ConservativeSemanticAdapter()
    decision = adapter.submit(make_fact())
    assert decision.status == DecisionStatus.ACCEPT_PROVISIONAL
    assert decision.candidate is not None
    assert adapter.confirmation_count("dragon", "color", "blue") == 1


def test_direct_commit_is_rejected() -> None:
    adapter = ConservativeSemanticAdapter()
    decision = adapter.submit(
        make_fact(destination=RequestedDestination.COMMITTED_DIRECT)
    )
    assert decision.status == DecisionStatus.REJECT
    assert ConservativeSemanticAdapter.REASON_DIRECT_COMMIT_FORBIDDEN in decision.reason_codes
    assert adapter.confirmation_count("dragon", "color", "blue") == 0


def test_missing_provenance_is_rejected() -> None:
    adapter = ConservativeSemanticAdapter()
    decision = adapter.submit(make_fact(provenance=()))
    assert decision.status == DecisionStatus.REJECT
    assert ConservativeSemanticAdapter.REASON_PROVENANCE_REQUIRED in decision.reason_codes


def test_query_is_read_only() -> None:
    adapter = ConservativeSemanticAdapter()
    accepted = adapter.submit(make_query())
    rejected = adapter.submit(make_query("q2", value_id="blue"))
    assert accepted.status == DecisionStatus.ACCEPT_QUERY
    assert rejected.status == DecisionStatus.REJECT
    assert ConservativeSemanticAdapter.REASON_QUERY_READ_ONLY in rejected.reason_codes
    assert len(adapter.accepted_queries()) == 1


def test_same_episode_does_not_inflate_confirmation() -> None:
    adapter = ConservativeSemanticAdapter()
    adapter.submit(make_fact("h1", episode_id=1))
    adapter.submit(make_fact("h2", episode_id=1))
    assert adapter.confirmation_count("dragon", "color", "blue") == 1


def test_distinct_episodes_confirm_longitudinally() -> None:
    adapter = ConservativeSemanticAdapter()
    adapter.submit(make_fact("h1", episode_id=1))
    adapter.submit(make_fact("h2", episode_id=2))
    assert adapter.confirmation_count("dragon", "color", "blue") == 2


def test_conflicting_values_remain_separate() -> None:
    adapter = ConservativeSemanticAdapter()
    adapter.submit(make_fact("h1", episode_id=1, value_id="blue"))
    adapter.submit(make_fact("h2", episode_id=2, value_id="red"))
    candidates = adapter.candidates_for_slot("dragon", "color")
    assert {candidate.value_id for candidate in candidates} == {"blue", "red"}


def test_audit_is_deterministic() -> None:
    def run() -> str:
        adapter = ConservativeSemanticAdapter()
        adapter.submit(make_fact())
        adapter.submit(make_query())
        adapter.submit(make_fact("h3", destination=RequestedDestination.COMMITTED_DIRECT))
        return adapter.audit_json()

    assert run() == run()


def test_hypothesis_roundtrip() -> None:
    hypothesis = make_fact()
    rebuilt = SemanticHypothesis.from_dict(
        json.loads(json.dumps(hypothesis.to_dict(), ensure_ascii=False))
    )
    assert rebuilt == hypothesis


def test_invalid_ranges_are_rejected() -> None:
    adapter = ConservativeSemanticAdapter()
    high = adapter.submit(make_fact("high", confidence=1.1))
    low = adapter.submit(make_fact("low", uncertainty=-0.1))
    assert high.status == DecisionStatus.REJECT
    assert low.status == DecisionStatus.REJECT
    assert ConservativeSemanticAdapter.REASON_INVALID_RANGE in high.reason_codes
    assert ConservativeSemanticAdapter.REASON_INVALID_RANGE in low.reason_codes
