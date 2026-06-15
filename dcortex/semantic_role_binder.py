# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b-RB1
# Conservative learned semantic role binder, provisional-only.

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
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

TupleFact = Tuple[str, str, str]


class RoleBindingAssignment(str, Enum):
    """Complete candidate assignment selected by the role binder."""

    IDENTITY = "IDENTITY"
    SWAPPED = "SWAPPED"
    UNRESOLVED = "UNRESOLVED"


ASSIGNMENT_ORDER = (
    RoleBindingAssignment.IDENTITY,
    RoleBindingAssignment.SWAPPED,
    RoleBindingAssignment.UNRESOLVED,
)


def assignment_facts(
    attribute: str,
    entities: Sequence[str],
    values: Sequence[str],
    assignment: RoleBindingAssignment,
) -> Tuple[TupleFact, ...]:
    """Return the complete one-to-one facts for one candidate assignment."""
    sorted_entities = tuple(sorted(entities))
    sorted_values = tuple(sorted(values))
    if len(sorted_entities) != 2 or len(set(sorted_entities)) != 2:
        raise ValueError("exactly two distinct entities are required")
    if len(sorted_values) != 2 or len(set(sorted_values)) != 2:
        raise ValueError("exactly two distinct values are required")
    if not attribute.strip():
        raise ValueError("attribute must not be empty")
    if assignment == RoleBindingAssignment.UNRESOLVED:
        return ()
    assigned_values = (
        sorted_values
        if assignment == RoleBindingAssignment.IDENTITY
        else tuple(reversed(sorted_values))
    )
    return tuple(
        sorted(
            (entity, attribute, value)
            for entity, value in zip(sorted_entities, assigned_values)
        )
    )


def expected_assignment(
    attribute: str,
    entities: Sequence[str],
    values: Sequence[str],
    expected: Sequence[TupleFact],
) -> RoleBindingAssignment:
    """Return which fixed candidate assignment matches the expected facts."""
    expected_tuple = tuple(sorted(tuple(item) for item in expected))
    if not expected_tuple:
        return RoleBindingAssignment.UNRESOLVED
    for assignment in (
        RoleBindingAssignment.IDENTITY,
        RoleBindingAssignment.SWAPPED,
    ):
        if assignment_facts(attribute, entities, values, assignment) == expected_tuple:
            return assignment
    raise ValueError("expected facts do not match identity, swapped, or unresolved")


def candidate_views(
    source_text: str,
    attribute: str,
    entities: Sequence[str],
    values: Sequence[str],
) -> Tuple[str, str, str]:
    """Build the three fixed complete-assignment scoring views."""
    identity = assignment_facts(
        attribute, entities, values, RoleBindingAssignment.IDENTITY
    )
    swapped = assignment_facts(
        attribute, entities, values, RoleBindingAssignment.SWAPPED
    )

    def mapping_text(facts: Sequence[TupleFact]) -> str:
        return "; ".join(
            f"{entity} has {attribute} value {value}"
            for entity, _, value in facts
        )

    prefix = (
        f"Source statement: {source_text}\n"
        f"Task: judge the complete one-to-one {attribute} assignment.\n"
    )
    return (
        prefix + f"Candidate assignment: {mapping_text(identity)}.",
        prefix + f"Candidate assignment: {mapping_text(swapped)}.",
        prefix
        + "Candidate assignment: unresolved because the source does not "
        + "identify which entity has which value.",
    )


class RoleBindingScoringBackend(ABC):
    """Abstract scalar scorer for complete role-binding candidate views."""

    backend_id: str
    backend_version: str

    @abstractmethod
    def score(self, views: Sequence[str]) -> torch.Tensor:
        """Return one finite scalar score per candidate view."""


class RoleBindingScoringHead(torch.nn.Module):
    """Shared scalar assignment scorer over frozen contextual features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.network = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one scalar score per feature row."""
        return self.network(features).squeeze(-1)


class ContextualRoleBindingScoringBackend(RoleBindingScoringBackend):
    """Learned scalar role-binding scorer over a frozen feature backend."""

    def __init__(
        self,
        feature_backend: SemanticFeatureBackend,
        head: RoleBindingScoringHead,
        backend_version: str = "1.0",
    ) -> None:
        self.feature_backend = feature_backend
        self.head = head
        self.backend_id = "dcortex_contextual_role_binding_scorer"
        self.backend_version = backend_version

    def score(self, views: Sequence[str]) -> torch.Tensor:
        """Return deterministic scalar assignment scores."""
        features = self.feature_backend.features(views)
        device = next(self.head.parameters()).device
        with torch.inference_mode():
            scores = self.head(features.to(device))
        return scores.detach().float().cpu()


@dataclass(frozen=True)
class RoleBindingScore:
    """Auditable ranked score distribution for one binding decision."""

    selected_assignment: RoleBindingAssignment
    top_probability: float
    second_probability: float
    margin: float
    margin_threshold: float
    candidate_probabilities: Tuple[Tuple[str, float], ...]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the ranked score distribution."""
        data = asdict(self)
        data["selected_assignment"] = self.selected_assignment.value
        data["candidate_probabilities"] = dict(self.candidate_probabilities)
        return data


@dataclass(frozen=True)
class RoleBindingResult:
    """Conservative complete-assignment decision and adapter evidence."""

    emitted: bool
    reason_codes: Tuple[str, ...]
    score: RoleBindingScore
    facts: Tuple[TupleFact, ...]
    hypotheses: Tuple[SemanticHypothesis, ...]
    adapter_decisions: Tuple[AdapterDecision, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the complete role-binding decision."""
        return {
            "emitted": self.emitted,
            "reason_codes": list(self.reason_codes),
            "score": self.score.to_dict(),
            "facts": [list(item) for item in self.facts],
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "adapter_decisions": [item.to_dict() for item in self.adapter_decisions],
        }


class ConservativeLearnedRoleBinder:
    """Emit a complete provisional mapping or abstain without memory mutation."""

    REASON_EMITTED = "EMITTED_COMPLETE_PROVISIONAL_MAPPING"
    REASON_UNRESOLVED = "UNRESOLVED_SELECTED"
    REASON_MARGIN_TOO_SMALL = "MARGIN_TOO_SMALL"
    REASON_NONFINITE_SCORE = "NONFINITE_SCORE"
    REASON_ADAPTER_REJECTED = "ADAPTER_REJECTED"

    def __init__(
        self,
        backend: RoleBindingScoringBackend,
        adapter: ConservativeSemanticAdapter,
        margin_threshold: float,
    ) -> None:
        if not 0.0 <= margin_threshold <= 1.0:
            raise ValueError("margin_threshold must be in [0, 1]")
        self.backend = backend
        self.adapter = adapter
        self.margin_threshold = margin_threshold

    def _rank(
        self,
        source_text: str,
        attribute: str,
        entities: Sequence[str],
        values: Sequence[str],
    ) -> RoleBindingScore:
        views = candidate_views(source_text, attribute, entities, values)
        raw_scores = self.backend.score(views).float()
        if raw_scores.shape != (len(ASSIGNMENT_ORDER),) or not torch.isfinite(
            raw_scores
        ).all():
            probabilities = torch.zeros(len(ASSIGNMENT_ORDER), dtype=torch.float32)
            probabilities[ASSIGNMENT_ORDER.index(RoleBindingAssignment.UNRESOLVED)] = 1.0
        else:
            probabilities = torch.softmax(raw_scores, dim=0)
        ranked = sorted(
            zip(ASSIGNMENT_ORDER, probabilities.tolist()),
            key=lambda item: (-item[1], item[0].value),
        )
        selected, top_probability = ranked[0]
        second_probability = ranked[1][1]
        return RoleBindingScore(
            selected_assignment=selected,
            top_probability=round(float(top_probability), 6),
            second_probability=round(float(second_probability), 6),
            margin=round(float(top_probability - second_probability), 6),
            margin_threshold=self.margin_threshold,
            candidate_probabilities=tuple(
                (assignment.value, round(float(probability), 6))
                for assignment, probability in zip(
                    ASSIGNMENT_ORDER, probabilities.tolist()
                )
            ),
        )

    def produce(
        self,
        hypothesis_prefix: str,
        episode_id: int,
        source_text: str,
        attribute: str,
        entities: Sequence[str],
        values: Sequence[str],
        provenance: Tuple[str, ...] = (),
    ) -> RoleBindingResult:
        """Emit exactly two provisional facts for a supported complete mapping."""
        score = self._rank(source_text, attribute, entities, values)
        if score.selected_assignment == RoleBindingAssignment.UNRESOLVED:
            return RoleBindingResult(
                emitted=False,
                reason_codes=(self.REASON_UNRESOLVED,),
                score=score,
                facts=(),
                hypotheses=(),
                adapter_decisions=(),
            )
        if score.margin < score.margin_threshold:
            return RoleBindingResult(
                emitted=False,
                reason_codes=(self.REASON_MARGIN_TOO_SMALL,),
                score=score,
                facts=(),
                hypotheses=(),
                adapter_decisions=(),
            )
        facts = assignment_facts(
            attribute, entities, values, score.selected_assignment
        )
        hypotheses: List[SemanticHypothesis] = []
        decisions: List[AdapterDecision] = []
        for index, (entity, attr_type, value) in enumerate(facts):
            hypothesis = SemanticHypothesis(
                hypothesis_id=f"{hypothesis_prefix}-{index}",
                episode_id=episode_id,
                mode=HypothesisMode.FACT,
                source_text=source_text,
                producer="conservative_learned_role_binder",
                producer_version="1.0",
                provenance=tuple(provenance)
                + (
                    f"backend:{self.backend.backend_id}:{self.backend.backend_version}",
                    f"assignment:{score.selected_assignment.value}",
                ),
                confidence=score.top_probability,
                uncertainty=1.0 - score.top_probability,
                requested_destination=RequestedDestination.PROVISIONAL_ONLY,
                entity_id=entity,
                attr_type=attr_type,
                value_id=value,
            )
            hypotheses.append(hypothesis)
            decisions.append(self.adapter.submit(hypothesis))
        accepted = all(
            decision.status == DecisionStatus.ACCEPT_PROVISIONAL
            for decision in decisions
        )
        return RoleBindingResult(
            emitted=accepted,
            reason_codes=(
                (self.REASON_EMITTED,)
                if accepted
                else (self.REASON_ADAPTER_REJECTED,)
            ),
            score=score,
            facts=facts if accepted else (),
            hypotheses=tuple(hypotheses),
            adapter_decisions=tuple(decisions),
        )
