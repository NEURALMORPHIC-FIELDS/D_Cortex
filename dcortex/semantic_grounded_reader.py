# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b-G
# Explicit-referent grounding guard for direct semantic object reads.

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

from dcortex.semantic_adapter import (
    AdapterDecision,
    DecisionStatus,
    HypothesisMode,
    RequestedDestination,
    SemanticHypothesis,
)
from dcortex.semantic_object_reader import (
    DirectSemanticObjectReader,
    ObjectMemorySnapshot,
    ObjectReadStatus,
    SemanticObjectReadResult,
)


class ReferentGroundingStatus(str, Enum):
    """Outcome of exact explicit-referent grounding."""

    GROUNDED = "GROUNDED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class ReferentGroundingResult:
    """Auditable exact-evidence result for one semantic query coordinate."""

    status: ReferentGroundingStatus
    entity_id: Optional[str]
    entity_tokens: Tuple[str, ...]
    source_tokens: Tuple[str, ...]
    matched_span: Optional[Tuple[int, int]]
    reason_codes: Tuple[str, ...]
    evidence_fingerprint: str

    def to_dict(self) -> Dict[str, object]:
        """Serialize the grounding result."""
        data = asdict(self)
        data["status"] = self.status.value
        data["entity_tokens"] = list(self.entity_tokens)
        data["source_tokens"] = list(self.source_tokens)
        data["matched_span"] = (
            None if self.matched_span is None else list(self.matched_span)
        )
        data["reason_codes"] = list(self.reason_codes)
        return data

    def to_json(self) -> str:
        """Return deterministic compact JSON."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class GroundedSemanticObjectReadResult:
    """Combined grounding evidence and direct object-read result."""

    grounding: ReferentGroundingResult
    read: SemanticObjectReadResult

    def to_dict(self) -> Dict[str, object]:
        """Serialize the grounded read result."""
        return {
            "grounding": self.grounding.to_dict(),
            "read": self.read.to_dict(),
        }

    def to_json(self) -> str:
        """Return deterministic compact JSON."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


class ExplicitReferentGroundingGate:
    """Require exact entity-token evidence already present in source text."""

    REASON_GROUNDED = "EXPLICIT_REFERENT_GROUNDED"
    REASON_HYPOTHESIS_REQUIRED = "HYPOTHESIS_REQUIRED"
    REASON_DECISION_REQUIRED = "DECISION_REQUIRED"
    REASON_NOT_ACCEPT_QUERY = "DECISION_NOT_ACCEPT_QUERY"
    REASON_NOT_QUERY_MODE = "NOT_QUERY_MODE"
    REASON_NOT_QUERY_ONLY = "NOT_QUERY_ONLY"
    REASON_ENTITY_REQUIRED = "ENTITY_REQUIRED"
    REASON_REFERENT_NOT_EXPLICIT = "REFERENT_NOT_EXPLICIT"

    @staticmethod
    def _tokens(text: str) -> Tuple[str, ...]:
        return tuple(re.findall(r"[a-z0-9]+", text.lower()))

    @staticmethod
    def _fingerprint(payload: Dict[str, object]) -> str:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    @classmethod
    def _result(
        cls,
        status: ReferentGroundingStatus,
        entity_id: Optional[str],
        entity_tokens: Tuple[str, ...],
        source_tokens: Tuple[str, ...],
        matched_span: Optional[Tuple[int, int]],
        reason_codes: Tuple[str, ...],
    ) -> ReferentGroundingResult:
        payload: Dict[str, object] = {
            "status": status.value,
            "entity_id": entity_id,
            "entity_tokens": list(entity_tokens),
            "source_tokens": list(source_tokens),
            "matched_span": (
                None if matched_span is None else list(matched_span)
            ),
            "reason_codes": list(reason_codes),
        }
        return ReferentGroundingResult(
            status=status,
            entity_id=entity_id,
            entity_tokens=entity_tokens,
            source_tokens=source_tokens,
            matched_span=matched_span,
            reason_codes=reason_codes,
            evidence_fingerprint=cls._fingerprint(payload),
        )

    def ground(
        self,
        hypothesis: Optional[SemanticHypothesis],
        decision: Optional[AdapterDecision],
    ) -> ReferentGroundingResult:
        """Accept only exact contiguous entity-token evidence."""
        reasons = []
        if hypothesis is None:
            reasons.append(self.REASON_HYPOTHESIS_REQUIRED)
        if decision is None:
            reasons.append(self.REASON_DECISION_REQUIRED)
        if decision is not None and decision.status != DecisionStatus.ACCEPT_QUERY:
            reasons.append(self.REASON_NOT_ACCEPT_QUERY)
        if hypothesis is not None and hypothesis.mode != HypothesisMode.QUERY:
            reasons.append(self.REASON_NOT_QUERY_MODE)
        if (
            hypothesis is not None
            and hypothesis.requested_destination != RequestedDestination.QUERY_ONLY
        ):
            reasons.append(self.REASON_NOT_QUERY_ONLY)
        entity_id = (
            hypothesis.entity_id
            if hypothesis is not None and isinstance(hypothesis.entity_id, str)
            else None
        )
        if entity_id is None or not entity_id.strip():
            reasons.append(self.REASON_ENTITY_REQUIRED)
        source_text = "" if hypothesis is None else hypothesis.source_text
        entity_tokens = self._tokens("" if entity_id is None else entity_id)
        source_tokens = self._tokens(source_text)
        if reasons:
            return self._result(
                ReferentGroundingStatus.REJECTED,
                entity_id,
                entity_tokens,
                source_tokens,
                None,
                tuple(sorted(set(reasons))),
            )
        matched_span = None
        width = len(entity_tokens)
        for start in range(0, len(source_tokens) - width + 1):
            if source_tokens[start : start + width] == entity_tokens:
                matched_span = (start, start + width)
                break
        if matched_span is None:
            return self._result(
                ReferentGroundingStatus.REJECTED,
                entity_id,
                entity_tokens,
                source_tokens,
                None,
                (self.REASON_REFERENT_NOT_EXPLICIT,),
            )
        return self._result(
            ReferentGroundingStatus.GROUNDED,
            entity_id,
            entity_tokens,
            source_tokens,
            matched_span,
            (self.REASON_GROUNDED,),
        )


class GroundedSemanticObjectReader:
    """Ground an approved coordinate before direct immutable object reading."""

    def __init__(self) -> None:
        self.grounding_gate = ExplicitReferentGroundingGate()
        self.object_reader = DirectSemanticObjectReader()

    def read(
        self,
        snapshot: ObjectMemorySnapshot,
        hypothesis: Optional[SemanticHypothesis],
        decision: Optional[AdapterDecision],
    ) -> GroundedSemanticObjectReadResult:
        """Ground the referent, then read only when explicit evidence exists."""
        grounding = self.grounding_gate.ground(hypothesis, decision)
        if grounding.status == ReferentGroundingStatus.GROUNDED:
            return GroundedSemanticObjectReadResult(
                grounding=grounding,
                read=self.object_reader.read(snapshot, hypothesis, decision),
            )
        read = SemanticObjectReadResult(
            status=ObjectReadStatus.REFUSED_INPUT,
            entity_id=grounding.entity_id,
            attr_type=(
                hypothesis.attr_type
                if hypothesis is not None and isinstance(hypothesis.attr_type, str)
                else None
            ),
            pred_value=None,
            disputed_values=(),
            source="grounding",
            reason_codes=grounding.reason_codes,
            snapshot_fingerprint=snapshot.fingerprint,
            adapter_audit_sequence=(
                None if decision is None else decision.audit_sequence
            ),
        )
        return GroundedSemanticObjectReadResult(grounding=grounding, read=read)
