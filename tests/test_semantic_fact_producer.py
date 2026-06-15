"""Regression tests for the v15.7b-F provisional semantic fact producer."""

from typing import Dict, Sequence, Tuple

import torch

from dcortex.semantic_adapter import ConservativeSemanticAdapter, DecisionStatus
from dcortex.semantic_fact_producer import (
    ConservativeAttributeConditionedFactProducer,
    ConservativeTrainedFactProducer,
    SemanticFactClassificationBackend,
)


class StaticFactBackend(SemanticFactClassificationBackend):
    """Deterministic fact backend for contract tests."""

    backend_id = "static-fact"
    backend_version = "1.0"
    entity_ids = ("dragon", "wolf", "UNKNOWN_ENTITY")
    attribute_ids = ("color", "state", "UNKNOWN")
    value_ids = ("color:blue", "state:awake", "UNKNOWN_VALUE")
    unknown_entity_id = "UNKNOWN_ENTITY"
    unknown_attribute_id = "UNKNOWN"
    unknown_value_id = "UNKNOWN_VALUE"

    def __init__(
        self,
        values: Dict[
            str,
            Tuple[Sequence[float], Sequence[float], Sequence[float]],
        ],
    ) -> None:
        self.values = values

    def classify(
        self, texts: Sequence[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor([self.values[text][0] for text in texts]),
            torch.tensor([self.values[text][1] for text in texts]),
            torch.tensor([self.values[text][2] for text in texts]),
        )


VALUES = {
    "valid": ([0.95, 0.03, 0.02], [0.96, 0.02, 0.02], [0.97, 0.02, 0.01]),
    "ambiguous": ([0.34, 0.33, 0.33], [0.34, 0.33, 0.33], [0.34, 0.33, 0.33]),
    "mismatch": ([0.95, 0.03, 0.02], [0.96, 0.02, 0.02], [0.20, 0.79, 0.01]),
}


def producer(adapter: ConservativeSemanticAdapter) -> ConservativeTrainedFactProducer:
    """Build one frozen-threshold test producer."""
    return ConservativeTrainedFactProducer(
        StaticFactBackend(VALUES),
        adapter,
        entity_margin_threshold=0.0,
        attribute_margin_threshold=0.4,
        value_margin_threshold=0.4,
    )


def test_valid_fact_is_adapter_accepted_provisional() -> None:
    adapter = ConservativeSemanticAdapter()
    result = producer(adapter).produce("fact-1", 1, "valid", provenance=("test",))
    assert result.emitted
    assert result.hypothesis is not None
    assert result.hypothesis.value_id == "blue"
    assert result.adapter_decision is not None
    assert result.adapter_decision.status == DecisionStatus.ACCEPT_PROVISIONAL
    assert adapter.confirmation_count("dragon", "color", "blue") == 1


def test_ambiguous_fact_abstains() -> None:
    result = producer(ConservativeSemanticAdapter()).produce(
        "fact-2", 1, "ambiguous", provenance=("test",)
    )
    assert not result.emitted
    assert result.adapter_decision is None


def test_value_attribute_mismatch_abstains() -> None:
    result = producer(ConservativeSemanticAdapter()).produce(
        "fact-3", 1, "mismatch", provenance=("test",)
    )
    assert not result.emitted
    assert (
        ConservativeTrainedFactProducer.REASON_VALUE_ATTRIBUTE_MISMATCH
        in result.reason_codes
    )


def test_attribute_conditioning_recovers_matching_value() -> None:
    adapter = ConservativeSemanticAdapter()
    instance = ConservativeAttributeConditionedFactProducer(
        StaticFactBackend(VALUES),
        adapter,
        entity_margin_threshold=0.0,
        attribute_margin_threshold=0.4,
        value_margin_threshold=0.4,
    )
    result = instance.produce("fact-conditioned", 1, "mismatch", provenance=("test",))
    assert result.emitted
    assert result.hypothesis is not None
    assert result.hypothesis.attr_type == "color"
    assert result.hypothesis.value_id == "blue"


def test_same_episode_does_not_inflate_confirmation() -> None:
    adapter = ConservativeSemanticAdapter()
    instance = producer(adapter)
    instance.produce("fact-4", 1, "valid", provenance=("test",))
    instance.produce("fact-5", 1, "valid", provenance=("test",))
    assert adapter.confirmation_count("dragon", "color", "blue") == 1


def test_distinct_episodes_remain_distinct() -> None:
    adapter = ConservativeSemanticAdapter()
    instance = producer(adapter)
    instance.produce("fact-6", 1, "valid", provenance=("test",))
    instance.produce("fact-7", 2, "valid", provenance=("test",))
    assert adapter.confirmation_count("dragon", "color", "blue") == 2
