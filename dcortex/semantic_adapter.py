# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b
# Conservative adapter for semantic hypotheses.

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class HypothesisMode(str, Enum):
    """Semantic hypothesis operating mode."""

    FACT = "FACT"
    QUERY = "QUERY"


class RequestedDestination(str, Enum):
    """Destination requested by an untrusted semantic producer."""

    PROVISIONAL_ONLY = "PROVISIONAL_ONLY"
    QUERY_ONLY = "QUERY_ONLY"
    COMMITTED_DIRECT = "COMMITTED_DIRECT"


class DecisionStatus(str, Enum):
    """Adapter decision status."""

    ACCEPT_PROVISIONAL = "ACCEPT_PROVISIONAL"
    ACCEPT_QUERY = "ACCEPT_QUERY"
    REJECT = "REJECT"


@dataclass(frozen=True)
class SemanticHypothesis:
    """Untrusted interpretation proposed by a semantic producer."""

    hypothesis_id: str
    episode_id: int
    mode: HypothesisMode
    source_text: str
    producer: str
    producer_version: str
    provenance: Tuple[str, ...]
    confidence: float
    uncertainty: float
    requested_destination: RequestedDestination
    entity_id: Optional[str] = None
    attr_type: Optional[str] = None
    value_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the hypothesis into JSON-compatible data."""
        data = asdict(self)
        data["mode"] = self.mode.value
        data["requested_destination"] = self.requested_destination.value
        data["provenance"] = list(self.provenance)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticHypothesis":
        """Reconstruct a hypothesis from JSON-compatible data."""
        return cls(
            hypothesis_id=str(data["hypothesis_id"]),
            episode_id=int(data["episode_id"]),
            mode=HypothesisMode(data["mode"]),
            source_text=str(data["source_text"]),
            producer=str(data["producer"]),
            producer_version=str(data["producer_version"]),
            provenance=tuple(str(item) for item in data["provenance"]),
            confidence=float(data["confidence"]),
            uncertainty=float(data["uncertainty"]),
            requested_destination=RequestedDestination(data["requested_destination"]),
            entity_id=data.get("entity_id"),
            attr_type=data.get("attr_type"),
            value_id=data.get("value_id"),
        )


@dataclass(frozen=True)
class ProvisionalCandidate:
    """Validated fact evidence safe for provisional-memory ingestion."""

    hypothesis_id: str
    episode_id: int
    entity_id: str
    attr_type: str
    value_id: str
    confidence: float
    uncertainty: float
    producer: str
    producer_version: str
    provenance: Tuple[str, ...]
    source_fingerprint: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the candidate into JSON-compatible data."""
        data = asdict(self)
        data["provenance"] = list(self.provenance)
        return data


@dataclass(frozen=True)
class AuditRecord:
    """Deterministic audit record for one adapter decision."""

    sequence: int
    hypothesis_id: str
    episode_id: int
    mode: str
    requested_destination: str
    status: str
    reason_codes: Tuple[str, ...]
    source_fingerprint: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the audit record into JSON-compatible data."""
        data = asdict(self)
        data["reason_codes"] = list(self.reason_codes)
        return data


@dataclass(frozen=True)
class AdapterDecision:
    """Result returned for one submitted semantic hypothesis."""

    status: DecisionStatus
    hypothesis_id: str
    reason_codes: Tuple[str, ...]
    audit_sequence: int
    candidate: Optional[ProvisionalCandidate] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the decision into JSON-compatible data."""
        return {
            "status": self.status.value,
            "hypothesis_id": self.hypothesis_id,
            "reason_codes": list(self.reason_codes),
            "audit_sequence": self.audit_sequence,
            "candidate": None if self.candidate is None else self.candidate.to_dict(),
        }


class ConservativeSemanticAdapter:
    """Validate semantic hypotheses without permitting direct memory commit.

    Fact hypotheses can become provisional candidates. Query hypotheses can
    become read-only interpretations. Every submission produces an audit
    record, including rejected submissions.
    """

    REASON_ACCEPTED_PROVISIONAL = "ACCEPTED_PROVISIONAL"
    REASON_ACCEPTED_QUERY = "ACCEPTED_QUERY"
    REASON_DIRECT_COMMIT_FORBIDDEN = "DIRECT_COMMIT_FORBIDDEN"
    REASON_PROVENANCE_REQUIRED = "PROVENANCE_REQUIRED"
    REASON_QUERY_READ_ONLY = "QUERY_READ_ONLY"
    REASON_FACT_FIELDS_REQUIRED = "FACT_FIELDS_REQUIRED"
    REASON_QUERY_FIELDS_REQUIRED = "QUERY_FIELDS_REQUIRED"
    REASON_IDENTITY_REQUIRED = "IDENTITY_REQUIRED"
    REASON_PRODUCER_REQUIRED = "PRODUCER_REQUIRED"
    REASON_SOURCE_REQUIRED = "SOURCE_REQUIRED"
    REASON_INVALID_RANGE = "INVALID_RANGE"
    REASON_INVALID_EPISODE = "INVALID_EPISODE"
    REASON_WRONG_DESTINATION = "WRONG_DESTINATION"

    def __init__(self) -> None:
        self._audit: List[AuditRecord] = []
        self._candidates: Dict[
            Tuple[str, str], Dict[str, Dict[int, ProvisionalCandidate]]
        ] = {}
        self._queries: Dict[str, SemanticHypothesis] = {}

    @staticmethod
    def fingerprint(hypothesis: SemanticHypothesis) -> str:
        """Return a stable SHA-256 fingerprint of a hypothesis."""
        payload = json.dumps(
            hypothesis.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _present(value: Optional[str]) -> bool:
        return isinstance(value, str) and bool(value.strip())

    def _validate_common(self, hypothesis: SemanticHypothesis) -> List[str]:
        reasons: List[str] = []
        if not self._present(hypothesis.hypothesis_id):
            reasons.append(self.REASON_IDENTITY_REQUIRED)
        if hypothesis.episode_id < 0:
            reasons.append(self.REASON_INVALID_EPISODE)
        if not self._present(hypothesis.source_text):
            reasons.append(self.REASON_SOURCE_REQUIRED)
        if not self._present(hypothesis.producer) or not self._present(
            hypothesis.producer_version
        ):
            reasons.append(self.REASON_PRODUCER_REQUIRED)
        if not hypothesis.provenance or any(not item.strip() for item in hypothesis.provenance):
            reasons.append(self.REASON_PROVENANCE_REQUIRED)
        if not 0.0 <= hypothesis.confidence <= 1.0:
            reasons.append(self.REASON_INVALID_RANGE)
        if not 0.0 <= hypothesis.uncertainty <= 1.0:
            reasons.append(self.REASON_INVALID_RANGE)
        if hypothesis.requested_destination == RequestedDestination.COMMITTED_DIRECT:
            reasons.append(self.REASON_DIRECT_COMMIT_FORBIDDEN)
        return reasons

    def _validate_mode(self, hypothesis: SemanticHypothesis) -> List[str]:
        reasons: List[str] = []
        if hypothesis.mode == HypothesisMode.FACT:
            if hypothesis.requested_destination != RequestedDestination.PROVISIONAL_ONLY:
                reasons.append(self.REASON_WRONG_DESTINATION)
            if not all(
                self._present(value)
                for value in (hypothesis.entity_id, hypothesis.attr_type, hypothesis.value_id)
            ):
                reasons.append(self.REASON_FACT_FIELDS_REQUIRED)
        elif hypothesis.mode == HypothesisMode.QUERY:
            if hypothesis.requested_destination != RequestedDestination.QUERY_ONLY:
                reasons.append(self.REASON_QUERY_READ_ONLY)
            if not all(
                self._present(value) for value in (hypothesis.entity_id, hypothesis.attr_type)
            ):
                reasons.append(self.REASON_QUERY_FIELDS_REQUIRED)
            if self._present(hypothesis.value_id):
                reasons.append(self.REASON_QUERY_READ_ONLY)
        return reasons

    def _record(
        self,
        hypothesis: SemanticHypothesis,
        status: DecisionStatus,
        reasons: Tuple[str, ...],
    ) -> int:
        sequence = len(self._audit) + 1
        self._audit.append(
            AuditRecord(
                sequence=sequence,
                hypothesis_id=hypothesis.hypothesis_id,
                episode_id=hypothesis.episode_id,
                mode=hypothesis.mode.value,
                requested_destination=hypothesis.requested_destination.value,
                status=status.value,
                reason_codes=reasons,
                source_fingerprint=self.fingerprint(hypothesis),
            )
        )
        return sequence

    def submit(self, hypothesis: SemanticHypothesis) -> AdapterDecision:
        """Validate and route one semantic hypothesis conservatively."""
        reasons = self._validate_common(hypothesis) + self._validate_mode(hypothesis)
        reasons = sorted(set(reasons))
        if reasons:
            reason_tuple = tuple(reasons)
            sequence = self._record(hypothesis, DecisionStatus.REJECT, reason_tuple)
            return AdapterDecision(
                status=DecisionStatus.REJECT,
                hypothesis_id=hypothesis.hypothesis_id,
                reason_codes=reason_tuple,
                audit_sequence=sequence,
            )

        if hypothesis.mode == HypothesisMode.QUERY:
            self._queries[hypothesis.hypothesis_id] = hypothesis
            reasons_ok = (self.REASON_ACCEPTED_QUERY,)
            sequence = self._record(hypothesis, DecisionStatus.ACCEPT_QUERY, reasons_ok)
            return AdapterDecision(
                status=DecisionStatus.ACCEPT_QUERY,
                hypothesis_id=hypothesis.hypothesis_id,
                reason_codes=reasons_ok,
                audit_sequence=sequence,
            )

        candidate = ProvisionalCandidate(
            hypothesis_id=hypothesis.hypothesis_id,
            episode_id=hypothesis.episode_id,
            entity_id=str(hypothesis.entity_id),
            attr_type=str(hypothesis.attr_type),
            value_id=str(hypothesis.value_id),
            confidence=hypothesis.confidence,
            uncertainty=hypothesis.uncertainty,
            producer=hypothesis.producer,
            producer_version=hypothesis.producer_version,
            provenance=hypothesis.provenance,
            source_fingerprint=self.fingerprint(hypothesis),
        )
        slot = (candidate.entity_id, candidate.attr_type)
        by_value = self._candidates.setdefault(slot, {})
        by_episode = by_value.setdefault(candidate.value_id, {})
        by_episode[candidate.episode_id] = candidate
        reasons_ok = (self.REASON_ACCEPTED_PROVISIONAL,)
        sequence = self._record(hypothesis, DecisionStatus.ACCEPT_PROVISIONAL, reasons_ok)
        return AdapterDecision(
            status=DecisionStatus.ACCEPT_PROVISIONAL,
            hypothesis_id=hypothesis.hypothesis_id,
            reason_codes=reasons_ok,
            audit_sequence=sequence,
            candidate=candidate,
        )

    def confirmation_count(self, entity_id: str, attr_type: str, value_id: str) -> int:
        """Return distinct-episode confirmation count for one candidate value."""
        return len(
            self._candidates.get((entity_id, attr_type), {}).get(value_id, {})
        )

    def candidates_for_slot(
        self, entity_id: str, attr_type: str
    ) -> Tuple[ProvisionalCandidate, ...]:
        """Return all provisional candidates for a slot in deterministic order."""
        result: List[ProvisionalCandidate] = []
        by_value = self._candidates.get((entity_id, attr_type), {})
        for value_id in sorted(by_value):
            for episode_id in sorted(by_value[value_id]):
                result.append(by_value[value_id][episode_id])
        return tuple(result)

    def accepted_queries(self) -> Tuple[SemanticHypothesis, ...]:
        """Return accepted read-only queries in deterministic order."""
        return tuple(self._queries[key] for key in sorted(self._queries))

    def audit_records(self) -> Tuple[AuditRecord, ...]:
        """Return immutable audit records in submission order."""
        return tuple(self._audit)

    def audit_json(self) -> str:
        """Return deterministic compact JSON for the complete audit trail."""
        return json.dumps(
            [record.to_dict() for record in self._audit],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
