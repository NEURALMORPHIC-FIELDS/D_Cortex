"""Regression tests for the v15.7b conservative semantic producer."""

from typing import Dict, Sequence, Tuple

import torch

from dcortex.semantic_adapter import (
    ConservativeSemanticAdapter,
    DecisionStatus,
    HypothesisMode,
    RequestedDestination,
)
from dcortex.semantic_producer import (
    CandidateScoringBackend,
    ConservativeTrainedQueryProducer,
    ConservativeLikelihoodQueryProducer,
    ConservativeMultiViewLikelihoodQueryProducer,
    ConservativePrototypeProducer,
    DCortexContextualFeatureBackend,
    EmbeddingBackend,
    SemanticClassificationBackend,
)


class StaticBackend(EmbeddingBackend):
    """Deterministic embedding backend for producer contract tests."""

    backend_id = "static-test"
    backend_version = "1.0"

    def __init__(self, vectors: Dict[str, Sequence[float]]) -> None:
        self.vectors = vectors

    def embed(self, texts: Sequence[str]) -> torch.Tensor:
        return torch.tensor([self.vectors[text] for text in texts], dtype=torch.float32)


class StaticScoringBackend(CandidateScoringBackend):
    """Deterministic candidate-scoring backend for likelihood tests."""

    backend_id = "static-scoring-test"
    backend_version = "1.0"

    def __init__(self, scores: Dict[Tuple[str, str], Sequence[float]]) -> None:
        self.scores = scores

    def score(
        self,
        source_text: str,
        prompt_template: str,
        candidates: Sequence[str],
    ) -> torch.Tensor:
        del candidates
        return torch.tensor(self.scores[(source_text, prompt_template)], dtype=torch.float32)


class StaticClassificationBackend(SemanticClassificationBackend):
    """Deterministic trained-classification backend for producer tests."""

    backend_id = "static-classification-test"
    backend_version = "1.0"
    entity_ids = ("dragon", "phoenix", "UNKNOWN_ENTITY")
    attribute_ids = ("color", "size", "UNKNOWN")
    unknown_entity_id = "UNKNOWN_ENTITY"
    unknown_attribute_id = "UNKNOWN"

    def __init__(self, values: Dict[str, Tuple[Sequence[float], Sequence[float]]]) -> None:
        self.values = values

    def classify(self, texts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        entity = [self.values[text][0] for text in texts]
        attribute = [self.values[text][1] for text in texts]
        return torch.tensor(entity), torch.tensor(attribute)


class IdentityContextBlock(torch.nn.Module):
    """Minimal order-preserving block for contextual backend shape tests."""

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        del attention_mask
        return hidden


class TinyContextModel(torch.nn.Module):
    """Minimal model exposing the contextual backend contract."""

    def __init__(self) -> None:
        super().__init__()
        self.shared_token_emb = torch.nn.Embedding(16, 4)
        self.shared_pos_emb = torch.nn.Embedding(16, 4)
        self.dec_emb_norm = torch.nn.LayerNorm(4)
        self.dec_emb_drop = torch.nn.Dropout(0.0)
        self.dec_standard_blocks = torch.nn.ModuleList([IdentityContextBlock()])


VECTORS = {
    "fact": [1, 0, 1, 0, 1, 0],
    "query": [1, 0, 1, 0, 0, 0],
    "ambiguous": [1, 1, 1, 0, 0, 0],
    "unrelated": [-1, -1, -1, -1, -1, -1],
    "dragon": [1, 0, 0, 0, 0, 0],
    "phoenix": [0, 1, 0, 0, 0, 0],
    "color": [0, 0, 1, 0, 0, 0],
    "size": [0, 0, 0, 1, 0, 0],
    "blue": [0, 0, 0, 0, 1, 0],
    "red": [0, 0, 0, 0, 0, 1],
}
ENTITIES = {"dragon": ("dragon",), "phoenix": ("phoenix",)}
ATTRS = {"color": ("color",), "size": ("size",)}
VALUES = {"blue": ("blue",), "red": ("red",)}


def producer(
    similarity_threshold: float = 0.7, margin_threshold: float = 0.1
) -> ConservativePrototypeProducer:
    """Build the deterministic test producer."""
    return ConservativePrototypeProducer(
        StaticBackend(VECTORS),
        ConservativeSemanticAdapter(),
        similarity_threshold=similarity_threshold,
        margin_threshold=margin_threshold,
    )


def test_fact_emits_only_through_adapter() -> None:
    result = producer().produce(
        "fact-1",
        1,
        HypothesisMode.FACT,
        "fact",
        ENTITIES,
        ATTRS,
        VALUES,
        ("source:1",),
    )
    assert result.emitted
    assert result.hypothesis is not None
    assert result.hypothesis.requested_destination == RequestedDestination.PROVISIONAL_ONLY
    assert result.adapter_decision is not None
    assert result.adapter_decision.status == DecisionStatus.ACCEPT_PROVISIONAL


def test_query_emits_read_only() -> None:
    result = producer().produce(
        "query-1",
        1,
        HypothesisMode.QUERY,
        "query",
        ENTITIES,
        ATTRS,
        provenance=("query:1",),
    )
    assert result.emitted
    assert result.hypothesis is not None
    assert result.hypothesis.requested_destination == RequestedDestination.QUERY_ONLY
    assert result.hypothesis.value_id is None
    assert result.adapter_decision is not None
    assert result.adapter_decision.status == DecisionStatus.ACCEPT_QUERY


def test_below_threshold_abstains() -> None:
    result = producer(similarity_threshold=0.95).produce(
        "low-1",
        1,
        HypothesisMode.QUERY,
        "query",
        ENTITIES,
        ATTRS,
        provenance=("query:1",),
    )
    assert not result.emitted
    assert result.adapter_decision is None
    assert any(reason.startswith("BELOW_THRESHOLD") for reason in result.reason_codes)


def test_ambiguous_margin_abstains() -> None:
    result = producer().produce(
        "ambiguous-1",
        1,
        HypothesisMode.QUERY,
        "ambiguous",
        ENTITIES,
        ATTRS,
        provenance=("query:1",),
    )
    assert not result.emitted
    assert any(reason.startswith("MARGIN_TOO_SMALL") for reason in result.reason_codes)


def test_producer_is_deterministic() -> None:
    def run() -> dict:
        result = producer().produce(
            "query-1",
            1,
            HypothesisMode.QUERY,
            "query",
            ENTITIES,
            ATTRS,
            provenance=("query:1",),
        )
        return result.to_dict()

    assert run() == run()


def likelihood_producer(
    entity_margin: float = 0.1,
    attribute_margin: float = 0.1,
) -> ConservativeLikelihoodQueryProducer:
    """Build the deterministic likelihood test producer."""
    scores = {
        ("clear", "entity:{source_text}"): (4.0, 0.0),
        ("clear", "attribute:{source_text}"): (4.0, 0.0),
        ("ambiguous", "entity:{source_text}"): (4.0, 0.0),
        ("ambiguous", "attribute:{source_text}"): (1.0, 1.0),
    }
    return ConservativeLikelihoodQueryProducer(
        StaticScoringBackend(scores),
        ConservativeSemanticAdapter(),
        entity_prompt="entity:{source_text}",
        attribute_prompt="attribute:{source_text}",
        entity_margin_threshold=entity_margin,
        attribute_margin_threshold=attribute_margin,
    )


def test_likelihood_query_emits_only_through_adapter() -> None:
    result = likelihood_producer().produce(
        "likelihood-1",
        1,
        "clear",
        {"dragon": "dragon", "phoenix": "phoenix"},
        {"color": "color", "size": "size"},
        provenance=("query:likelihood",),
    )
    assert result.emitted
    assert result.hypothesis is not None
    assert result.hypothesis.requested_destination == RequestedDestination.QUERY_ONLY
    assert result.adapter_decision is not None
    assert result.adapter_decision.status == DecisionStatus.ACCEPT_QUERY


def test_likelihood_query_abstains_on_small_margin() -> None:
    result = likelihood_producer().produce(
        "likelihood-2",
        1,
        "ambiguous",
        {"dragon": "dragon", "phoenix": "phoenix"},
        {"color": "color", "size": "size"},
        provenance=("query:likelihood",),
    )
    assert not result.emitted
    assert "MARGIN_TOO_SMALL:attribute" in result.reason_codes


def test_likelihood_query_is_deterministic() -> None:
    def run() -> dict:
        return likelihood_producer().produce(
            "likelihood-3",
            1,
            "clear",
            {"dragon": "dragon", "phoenix": "phoenix"},
            {"color": "color", "size": "size"},
            provenance=("query:likelihood",),
        ).to_dict()

    assert run() == run()


def multiview_producer(
    attribute_margin: float = 0.1,
    attribute_consensus: int = 2,
) -> ConservativeMultiViewLikelihoodQueryProducer:
    """Build the deterministic multi-view likelihood test producer."""
    scores = {
        ("clear", "entity-a:{source_text}"): (4.0, 0.0),
        ("clear", "entity-b:{source_text}"): (3.0, 0.0),
        ("clear", "attribute-a:{source_text}"): (4.0, 0.0),
        ("clear", "attribute-b:{source_text}"): (3.0, 0.0),
        ("ambiguous", "entity-a:{source_text}"): (4.0, 0.0),
        ("ambiguous", "entity-b:{source_text}"): (3.0, 0.0),
        ("ambiguous", "attribute-a:{source_text}"): (1.0, 1.0),
        ("ambiguous", "attribute-b:{source_text}"): (1.0, 1.0),
    }
    return ConservativeMultiViewLikelihoodQueryProducer(
        StaticScoringBackend(scores),
        ConservativeSemanticAdapter(),
        entity_prompts=("entity-a:{source_text}", "entity-b:{source_text}"),
        attribute_prompts=("attribute-a:{source_text}", "attribute-b:{source_text}"),
        entity_margin_threshold=0.1,
        attribute_margin_threshold=attribute_margin,
        entity_minimum_consensus=2,
        attribute_minimum_consensus=attribute_consensus,
    )


def test_multiview_query_emits_with_consensus() -> None:
    result = multiview_producer().produce(
        "multiview-1",
        1,
        "clear",
        {"dragon": "dragon", "phoenix": "phoenix"},
        {"color": "color", "size": "size"},
        provenance=("query:multiview",),
    )
    assert result.emitted
    assert all(score.consensus_count == 2 for score in result.scores)
    assert result.adapter_decision is not None
    assert result.adapter_decision.status == DecisionStatus.ACCEPT_QUERY


def test_multiview_query_abstains_without_margin() -> None:
    result = multiview_producer().produce(
        "multiview-2",
        1,
        "ambiguous",
        {"dragon": "dragon", "phoenix": "phoenix"},
        {"color": "color", "size": "size"},
        provenance=("query:multiview",),
    )
    assert not result.emitted
    assert "MARGIN_TOO_SMALL:attribute" in result.reason_codes


def trained_producer() -> ConservativeTrainedQueryProducer:
    """Build the deterministic trained-classifier test producer."""
    backend = StaticClassificationBackend(
        {
            "clear": ((0.9, 0.05, 0.05), (0.9, 0.05, 0.05)),
            "unknown": ((0.9, 0.05, 0.05), (0.05, 0.05, 0.9)),
            "small-margin": ((0.9, 0.05, 0.05), (0.51, 0.49, 0.0)),
        }
    )
    return ConservativeTrainedQueryProducer(
        backend,
        ConservativeSemanticAdapter(),
        entity_margin_threshold=0.1,
        attribute_margin_threshold=0.4,
    )


def test_trained_query_emits_only_through_adapter() -> None:
    result = trained_producer().produce(
        "trained-1",
        1,
        "clear",
        provenance=("query:trained",),
    )
    assert result.emitted
    assert result.hypothesis is not None
    assert result.hypothesis.requested_destination == RequestedDestination.QUERY_ONLY
    assert result.adapter_decision is not None
    assert result.adapter_decision.status == DecisionStatus.ACCEPT_QUERY


def test_trained_query_abstains_on_unknown() -> None:
    result = trained_producer().produce(
        "trained-2",
        1,
        "unknown",
        provenance=("query:trained",),
    )
    assert not result.emitted
    assert "UNKNOWN_SELECTED:attribute" in result.reason_codes


def test_trained_query_abstains_on_small_margin() -> None:
    result = trained_producer().produce(
        "trained-3",
        1,
        "small-margin",
        provenance=("query:trained",),
    )
    assert not result.emitted
    assert "MARGIN_TOO_SMALL:attribute" in result.reason_codes


def test_contextual_feature_backend_is_deterministic_and_finite() -> None:
    backend = DCortexContextualFeatureBackend(
        TinyContextModel(),
        tokenizer=lambda text: [ord(char) % 16 for char in text],
        max_seq_len=8,
        batch_size=2,
    )
    first = backend.features(("ab", "cde"))
    second = backend.features(("ab", "cde"))
    assert first.shape == (2, 20)
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
