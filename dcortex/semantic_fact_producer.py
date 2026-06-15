# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b-F
# Conservative trained semantic fact producer, provisional-only.

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch

from dcortex.semantic_adapter import (
    AdapterDecision,
    ConservativeSemanticAdapter,
    DecisionStatus,
    HypothesisMode,
    RequestedDestination,
    SemanticHypothesis,
)
from dcortex.semantic_producer import SemanticFeatureBackend


class SemanticFactClassificationBackend(ABC):
    """Abstract entity/attribute/value fact-classification backend."""

    backend_id: str
    backend_version: str
    entity_ids: Tuple[str, ...]
    attribute_ids: Tuple[str, ...]
    value_ids: Tuple[str, ...]
    unknown_entity_id: str
    unknown_attribute_id: str
    unknown_value_id: str

    @abstractmethod
    def classify(
        self, texts: Sequence[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return entity, attribute, and value probability tensors."""


class SemanticFactHead(torch.nn.Module):
    """Separate entity, attribute, and value MLPs over frozen features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        entity_classes: int,
        attribute_classes: int,
        value_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(
            input_dim,
            hidden_dim,
            entity_classes,
            attribute_classes,
            value_classes,
        ) < 1:
            raise ValueError("all dimensions and class counts must be positive")

        def network(output_dim: int) -> torch.nn.Sequential:
            return torch.nn.Sequential(
                torch.nn.LayerNorm(input_dim),
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, output_dim),
            )

        self.entity_network = network(entity_classes)
        self.attribute_network = network(attribute_classes)
        self.value_network = network(value_classes)

    def forward(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return entity, attribute, and value logits."""
        return (
            self.entity_network(features),
            self.attribute_network(features),
            self.value_network(features),
        )


class PooledSemanticFactClassificationBackend(SemanticFactClassificationBackend):
    """Trained fact classifier over a frozen semantic feature backend."""

    def __init__(
        self,
        feature_backend: SemanticFeatureBackend,
        head: SemanticFactHead,
        entity_ids: Sequence[str],
        attribute_ids: Sequence[str],
        value_ids: Sequence[str],
        unknown_entity_id: str,
        unknown_attribute_id: str,
        unknown_value_id: str,
        backend_version: str = "1.0",
    ) -> None:
        if len(entity_ids) != head.entity_network[-1].out_features:
            raise ValueError("entity_ids count must match entity head")
        if len(attribute_ids) != head.attribute_network[-1].out_features:
            raise ValueError("attribute_ids count must match attribute head")
        if len(value_ids) != head.value_network[-1].out_features:
            raise ValueError("value_ids count must match value head")
        if unknown_entity_id not in entity_ids:
            raise ValueError("unknown_entity_id must be present")
        if unknown_attribute_id not in attribute_ids:
            raise ValueError("unknown_attribute_id must be present")
        if unknown_value_id not in value_ids:
            raise ValueError("unknown_value_id must be present")
        self.feature_backend = feature_backend
        self.head = head
        self.entity_ids = tuple(entity_ids)
        self.attribute_ids = tuple(attribute_ids)
        self.value_ids = tuple(value_ids)
        self.unknown_entity_id = unknown_entity_id
        self.unknown_attribute_id = unknown_attribute_id
        self.unknown_value_id = unknown_value_id
        self.backend_id = "dcortex_trained_contextual_semantic_fact_classifier"
        self.backend_version = backend_version

    def classify(
        self, texts: Sequence[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return deterministic entity, attribute, and value probabilities."""
        features = self.feature_backend.features(texts)
        device = next(self.head.parameters()).device
        with torch.inference_mode():
            entity, attribute, value = self.head(features.to(device))
        return (
            torch.softmax(entity.float(), dim=-1).cpu(),
            torch.softmax(attribute.float(), dim=-1).cpu(),
            torch.softmax(value.float(), dim=-1).cpu(),
        )


@dataclass(frozen=True)
class FactClassificationAxisScore:
    """Ranked probabilities for one semantic fact axis."""

    axis: str
    selected_id: str
    top_probability: float
    second_probability: float
    margin: float
    margin_threshold: float
    unknown_selected: bool
    passed: bool
    candidate_probabilities: Tuple[Tuple[str, float], ...]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the axis score."""
        data = asdict(self)
        data["candidate_probabilities"] = {
            candidate_id: probability
            for candidate_id, probability in self.candidate_probabilities
        }
        return data


@dataclass(frozen=True)
class FactClassificationProducerResult:
    """Conservative fact-producer result, including abstentions."""

    emitted: bool
    reason_codes: Tuple[str, ...]
    scores: Tuple[FactClassificationAxisScore, ...]
    hypothesis: SemanticHypothesis | None
    adapter_decision: AdapterDecision | None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the fact-producer result."""
        return {
            "emitted": self.emitted,
            "reason_codes": list(self.reason_codes),
            "scores": [score.to_dict() for score in self.scores],
            "hypothesis": None if self.hypothesis is None else self.hypothesis.to_dict(),
            "adapter_decision": (
                None if self.adapter_decision is None else self.adapter_decision.to_dict()
            ),
        }


class ConservativeTrainedFactProducer:
    """Emit only adapter-approved provisional semantic fact hypotheses."""

    REASON_EMITTED = "EMITTED_PROVISIONAL"
    REASON_UNKNOWN_SELECTED = "UNKNOWN_SELECTED"
    REASON_MARGIN_TOO_SMALL = "MARGIN_TOO_SMALL"
    REASON_NONFINITE_SCORE = "NONFINITE_SCORE"
    REASON_VALUE_ATTRIBUTE_MISMATCH = "VALUE_ATTRIBUTE_MISMATCH"

    def __init__(
        self,
        backend: SemanticFactClassificationBackend,
        adapter: ConservativeSemanticAdapter,
        entity_margin_threshold: float,
        attribute_margin_threshold: float,
        value_margin_threshold: float,
    ) -> None:
        for name, threshold in (
            ("entity_margin_threshold", entity_margin_threshold),
            ("attribute_margin_threshold", attribute_margin_threshold),
            ("value_margin_threshold", value_margin_threshold),
        ):
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self.backend = backend
        self.adapter = adapter
        self.entity_margin_threshold = entity_margin_threshold
        self.attribute_margin_threshold = attribute_margin_threshold
        self.value_margin_threshold = value_margin_threshold

    @staticmethod
    def _score_axis(
        axis: str,
        probabilities: torch.Tensor,
        candidate_ids: Sequence[str],
        unknown_id: str,
        margin_threshold: float,
    ) -> FactClassificationAxisScore:
        if probabilities.shape != (len(candidate_ids),) or not torch.isfinite(
            probabilities
        ).all():
            probabilities = torch.zeros(len(candidate_ids), dtype=torch.float32)
            probabilities[candidate_ids.index(unknown_id)] = 1.0
        values = probabilities.detach().cpu().float().tolist()
        ranked = sorted(
            zip(candidate_ids, values), key=lambda item: (-item[1], item[0])
        )
        selected_id, top_probability = ranked[0]
        second_probability = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_probability - second_probability
        unknown_selected = selected_id == unknown_id
        return FactClassificationAxisScore(
            axis=axis,
            selected_id=selected_id,
            top_probability=round(float(top_probability), 6),
            second_probability=round(float(second_probability), 6),
            margin=round(float(margin), 6),
            margin_threshold=margin_threshold,
            unknown_selected=unknown_selected,
            passed=(not unknown_selected and margin >= margin_threshold),
            candidate_probabilities=tuple(
                (candidate_id, round(float(probability), 6))
                for candidate_id, probability in sorted(
                    zip(candidate_ids, values), key=lambda item: item[0]
                )
            ),
        )

    def produce(
        self,
        hypothesis_id: str,
        episode_id: int,
        source_text: str,
        provenance: Tuple[str, ...] = (),
    ) -> FactClassificationProducerResult:
        """Produce and submit one provisional-only semantic fact hypothesis."""
        entity_probabilities, attribute_probabilities, value_probabilities = (
            self.backend.classify([source_text])
        )
        scores = (
            self._score_axis(
                "entity",
                entity_probabilities[0],
                self.backend.entity_ids,
                self.backend.unknown_entity_id,
                self.entity_margin_threshold,
            ),
            self._score_axis(
                "attribute",
                attribute_probabilities[0],
                self.backend.attribute_ids,
                self.backend.unknown_attribute_id,
                self.attribute_margin_threshold,
            ),
            self._score_axis(
                "value",
                value_probabilities[0],
                self.backend.value_ids,
                self.backend.unknown_value_id,
                self.value_margin_threshold,
            ),
        )
        reasons: List[str] = []
        for score in scores:
            if score.unknown_selected:
                reasons.append(f"{self.REASON_UNKNOWN_SELECTED}:{score.axis}")
            elif score.margin < score.margin_threshold:
                reasons.append(f"{self.REASON_MARGIN_TOO_SMALL}:{score.axis}")
            if not score.candidate_probabilities:
                reasons.append(f"{self.REASON_NONFINITE_SCORE}:{score.axis}")
        selected = {score.axis: score.selected_id for score in scores}
        value_parts = selected["value"].split(":", 1)
        if (
            len(value_parts) != 2
            or value_parts[0] != selected["attribute"]
        ):
            reasons.append(self.REASON_VALUE_ATTRIBUTE_MISMATCH)
        if reasons:
            return FactClassificationProducerResult(
                emitted=False,
                reason_codes=tuple(sorted(set(reasons))),
                scores=scores,
                hypothesis=None,
                adapter_decision=None,
            )

        confidence = min(score.top_probability for score in scores)
        hypothesis = SemanticHypothesis(
            hypothesis_id=hypothesis_id,
            episode_id=episode_id,
            mode=HypothesisMode.FACT,
            source_text=source_text,
            producer="conservative_trained_fact_producer",
            producer_version="1.0",
            provenance=tuple(provenance)
            + (f"backend:{self.backend.backend_id}:{self.backend.backend_version}",),
            confidence=confidence,
            uncertainty=1.0 - confidence,
            requested_destination=RequestedDestination.PROVISIONAL_ONLY,
            entity_id=selected["entity"],
            attr_type=selected["attribute"],
            value_id=value_parts[1],
        )
        decision = self.adapter.submit(hypothesis)
        emitted = decision.status == DecisionStatus.ACCEPT_PROVISIONAL
        return FactClassificationProducerResult(
            emitted=emitted,
            reason_codes=(self.REASON_EMITTED,) if emitted else decision.reason_codes,
            scores=scores,
            hypothesis=hypothesis if emitted else None,
            adapter_decision=decision,
        )


class ConservativeAttributeConditionedFactProducer:
    """Decode values only within the independently accepted attribute."""

    def __init__(
        self,
        backend: SemanticFactClassificationBackend,
        adapter: ConservativeSemanticAdapter,
        entity_margin_threshold: float,
        attribute_margin_threshold: float,
        value_margin_threshold: float,
    ) -> None:
        validator = ConservativeTrainedFactProducer(
            backend,
            adapter,
            entity_margin_threshold,
            attribute_margin_threshold,
            value_margin_threshold,
        )
        self.backend = validator.backend
        self.adapter = validator.adapter
        self.entity_margin_threshold = validator.entity_margin_threshold
        self.attribute_margin_threshold = validator.attribute_margin_threshold
        self.value_margin_threshold = validator.value_margin_threshold

    def produce(
        self,
        hypothesis_id: str,
        episode_id: int,
        source_text: str,
        provenance: Tuple[str, ...] = (),
    ) -> FactClassificationProducerResult:
        """Produce one attribute-conditioned provisional-only fact hypothesis."""
        entity_probabilities, attribute_probabilities, value_probabilities = (
            self.backend.classify([source_text])
        )
        entity_score = ConservativeTrainedFactProducer._score_axis(
            "entity",
            entity_probabilities[0],
            self.backend.entity_ids,
            self.backend.unknown_entity_id,
            self.entity_margin_threshold,
        )
        attribute_score = ConservativeTrainedFactProducer._score_axis(
            "attribute",
            attribute_probabilities[0],
            self.backend.attribute_ids,
            self.backend.unknown_attribute_id,
            self.attribute_margin_threshold,
        )
        reasons: List[str] = []
        for score in (entity_score, attribute_score):
            if score.unknown_selected:
                reasons.append(
                    f"{ConservativeTrainedFactProducer.REASON_UNKNOWN_SELECTED}:{score.axis}"
                )
            elif score.margin < score.margin_threshold:
                reasons.append(
                    f"{ConservativeTrainedFactProducer.REASON_MARGIN_TOO_SMALL}:{score.axis}"
                )
        allowed = [
            index
            for index, candidate_id in enumerate(self.backend.value_ids)
            if candidate_id == self.backend.unknown_value_id
            or candidate_id.startswith(f"{attribute_score.selected_id}:")
        ]
        conditioned_ids = tuple(self.backend.value_ids[index] for index in allowed)
        conditioned_probabilities = value_probabilities[0, allowed].float()
        conditioned_probabilities = conditioned_probabilities / conditioned_probabilities.sum().clamp_min(
            1e-12
        )
        value_score = ConservativeTrainedFactProducer._score_axis(
            "value",
            conditioned_probabilities,
            conditioned_ids,
            self.backend.unknown_value_id,
            self.value_margin_threshold,
        )
        if value_score.unknown_selected:
            reasons.append(
                f"{ConservativeTrainedFactProducer.REASON_UNKNOWN_SELECTED}:value"
            )
        elif value_score.margin < value_score.margin_threshold:
            reasons.append(
                f"{ConservativeTrainedFactProducer.REASON_MARGIN_TOO_SMALL}:value"
            )
        scores = (entity_score, attribute_score, value_score)
        value_parts = value_score.selected_id.split(":", 1)
        if (
            len(value_parts) != 2
            or value_parts[0] != attribute_score.selected_id
        ):
            reasons.append(
                ConservativeTrainedFactProducer.REASON_VALUE_ATTRIBUTE_MISMATCH
            )
        if reasons:
            return FactClassificationProducerResult(
                emitted=False,
                reason_codes=tuple(sorted(set(reasons))),
                scores=scores,
                hypothesis=None,
                adapter_decision=None,
            )
        confidence = min(score.top_probability for score in scores)
        hypothesis = SemanticHypothesis(
            hypothesis_id=hypothesis_id,
            episode_id=episode_id,
            mode=HypothesisMode.FACT,
            source_text=source_text,
            producer="conservative_attribute_conditioned_fact_producer",
            producer_version="1.0",
            provenance=tuple(provenance)
            + (f"backend:{self.backend.backend_id}:{self.backend.backend_version}",),
            confidence=confidence,
            uncertainty=1.0 - confidence,
            requested_destination=RequestedDestination.PROVISIONAL_ONLY,
            entity_id=entity_score.selected_id,
            attr_type=attribute_score.selected_id,
            value_id=value_parts[1],
        )
        decision = self.adapter.submit(hypothesis)
        emitted = decision.status == DecisionStatus.ACCEPT_PROVISIONAL
        return FactClassificationProducerResult(
            emitted=emitted,
            reason_codes=(
                (ConservativeTrainedFactProducer.REASON_EMITTED,)
                if emitted
                else decision.reason_codes
            ),
            scores=scores,
            hypothesis=hypothesis if emitted else None,
            adapter_decision=decision,
        )
