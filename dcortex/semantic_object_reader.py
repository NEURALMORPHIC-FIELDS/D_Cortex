# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b-O
# Direct read-only semantic-coordinate object-memory contract.

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, Optional, Sequence, Tuple

from dcortex.semantic_adapter import (
    AdapterDecision,
    DecisionStatus,
    HypothesisMode,
    RequestedDestination,
    SemanticHypothesis,
)


class ObjectReadStatus(str, Enum):
    """Read outcomes for an immutable epistemic object snapshot."""

    REFUSED_INPUT = "REFUSED_INPUT"
    FOUND_COMMITTED = "FOUND_COMMITTED"
    FOUND_DISPUTED = "FOUND_DISPUTED"
    NONE_OBJECT = "NONE_OBJECT"
    NONE_ATTRIBUTE = "NONE_ATTRIBUTE"


@dataclass(frozen=True)
class ObjectMemorySnapshot:
    """Immutable object-memory state expressed as semantic coordinates."""

    known_entities: Tuple[str, ...] = ()
    committed: Tuple[Tuple[str, str, str], ...] = ()
    provisional: Tuple[Tuple[str, str, str], ...] = ()

    def __post_init__(self) -> None:
        known_entities = tuple(
            sorted({self._clean_id(item, "known entity") for item in self.known_entities})
        )
        committed = tuple(
            sorted(
                {
                    (
                        self._clean_id(entity, "committed entity"),
                        self._clean_id(attribute, "committed attribute"),
                        self._clean_id(value, "committed value"),
                    )
                    for entity, attribute, value in self.committed
                }
            )
        )
        provisional = tuple(
            sorted(
                {
                    (
                        self._clean_id(entity, "provisional entity"),
                        self._clean_id(attribute, "provisional attribute"),
                        self._clean_id(value, "provisional value"),
                    )
                    for entity, attribute, value in self.provisional
                }
            )
        )
        committed_slots = [(entity, attribute) for entity, attribute, _ in committed]
        if len(committed_slots) != len(set(committed_slots)):
            raise ValueError("committed snapshot contains multiple values for one slot")
        all_entities = {
            entity for entity, _, _ in committed + provisional
        } | set(known_entities)
        object.__setattr__(self, "known_entities", tuple(sorted(all_entities)))
        object.__setattr__(self, "committed", committed)
        object.__setattr__(self, "provisional", provisional)

    @staticmethod
    def _clean_id(value: str, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a non-empty string")
        return value.strip()

    def to_dict(self) -> Dict[str, object]:
        """Serialize the immutable snapshot."""
        return {
            "known_entities": list(self.known_entities),
            "committed": [list(item) for item in self.committed],
            "provisional": [list(item) for item in self.provisional],
        }

    @property
    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 fingerprint of the complete state."""
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def committed_value(self, entity_id: str, attr_type: str) -> Optional[str]:
        """Return the exact committed value for one slot, if present."""
        for entity, attribute, value in self.committed:
            if entity == entity_id and attribute == attr_type:
                return value
        return None

    def provisional_values(self, entity_id: str, attr_type: str) -> Tuple[str, ...]:
        """Return sorted distinct provisional values for one slot."""
        return tuple(
            value
            for entity, attribute, value in self.provisional
            if entity == entity_id and attribute == attr_type
        )


@dataclass(frozen=True)
class SemanticObjectReadResult:
    """Auditable direct semantic-coordinate object-memory read."""

    status: ObjectReadStatus
    entity_id: Optional[str]
    attr_type: Optional[str]
    pred_value: Optional[str]
    disputed_values: Tuple[str, ...]
    source: str
    reason_codes: Tuple[str, ...]
    snapshot_fingerprint: str
    adapter_audit_sequence: Optional[int]

    def to_dict(self) -> Dict[str, object]:
        """Serialize the read result."""
        data = asdict(self)
        data["status"] = self.status.value
        data["disputed_values"] = list(self.disputed_values)
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


class DirectSemanticObjectReader:
    """Read an immutable object snapshot from an approved semantic coordinate."""

    REASON_DECISION_REQUIRED = "DECISION_REQUIRED"
    REASON_HYPOTHESIS_REQUIRED = "HYPOTHESIS_REQUIRED"
    REASON_DECISION_NOT_ACCEPT_QUERY = "DECISION_NOT_ACCEPT_QUERY"
    REASON_ID_MISMATCH = "DECISION_HYPOTHESIS_ID_MISMATCH"
    REASON_NOT_QUERY_MODE = "NOT_QUERY_MODE"
    REASON_NOT_QUERY_ONLY = "NOT_QUERY_ONLY"
    REASON_COORDINATE_REQUIRED = "SEMANTIC_COORDINATE_REQUIRED"
    REASON_FOUND_COMMITTED = "FOUND_COMMITTED"
    REASON_FOUND_DISPUTED = "FOUND_DISPUTED"
    REASON_NONE_OBJECT = "NONE_OBJECT"
    REASON_NONE_ATTRIBUTE = "NONE_ATTRIBUTE"

    @classmethod
    def _invalid_reasons(
        cls,
        hypothesis: Optional[SemanticHypothesis],
        decision: Optional[AdapterDecision],
    ) -> Tuple[str, ...]:
        reasons = []
        if hypothesis is None:
            reasons.append(cls.REASON_HYPOTHESIS_REQUIRED)
        if decision is None:
            reasons.append(cls.REASON_DECISION_REQUIRED)
        if decision is not None and decision.status != DecisionStatus.ACCEPT_QUERY:
            reasons.append(cls.REASON_DECISION_NOT_ACCEPT_QUERY)
        if (
            hypothesis is not None
            and decision is not None
            and hypothesis.hypothesis_id != decision.hypothesis_id
        ):
            reasons.append(cls.REASON_ID_MISMATCH)
        if hypothesis is not None and hypothesis.mode != HypothesisMode.QUERY:
            reasons.append(cls.REASON_NOT_QUERY_MODE)
        if (
            hypothesis is not None
            and hypothesis.requested_destination != RequestedDestination.QUERY_ONLY
        ):
            reasons.append(cls.REASON_NOT_QUERY_ONLY)
        if hypothesis is not None and (
            not isinstance(hypothesis.entity_id, str)
            or not hypothesis.entity_id.strip()
            or not isinstance(hypothesis.attr_type, str)
            or not hypothesis.attr_type.strip()
        ):
            reasons.append(cls.REASON_COORDINATE_REQUIRED)
        return tuple(sorted(set(reasons)))

    @staticmethod
    def _result(
        snapshot: ObjectMemorySnapshot,
        status: ObjectReadStatus,
        entity_id: Optional[str],
        attr_type: Optional[str],
        pred_value: Optional[str],
        disputed_values: Sequence[str],
        source: str,
        reason_codes: Tuple[str, ...],
        decision: Optional[AdapterDecision],
    ) -> SemanticObjectReadResult:
        return SemanticObjectReadResult(
            status=status,
            entity_id=entity_id,
            attr_type=attr_type,
            pred_value=pred_value,
            disputed_values=tuple(sorted(set(disputed_values))),
            source=source,
            reason_codes=reason_codes,
            snapshot_fingerprint=snapshot.fingerprint,
            adapter_audit_sequence=(
                None if decision is None else decision.audit_sequence
            ),
        )

    def read(
        self,
        snapshot: ObjectMemorySnapshot,
        hypothesis: Optional[SemanticHypothesis],
        decision: Optional[AdapterDecision],
    ) -> SemanticObjectReadResult:
        """Read by semantic coordinate without raw text, parsing, or mutation."""
        if not isinstance(snapshot, ObjectMemorySnapshot):
            raise TypeError("snapshot must be an ObjectMemorySnapshot")
        reasons = self._invalid_reasons(hypothesis, decision)
        if reasons:
            return self._result(
                snapshot,
                ObjectReadStatus.REFUSED_INPUT,
                None,
                None,
                None,
                (),
                "input",
                reasons,
                decision,
            )

        assert hypothesis is not None
        entity_id = str(hypothesis.entity_id).strip()
        attr_type = str(hypothesis.attr_type).strip()
        committed = snapshot.committed_value(entity_id, attr_type)
        provisional = snapshot.provisional_values(entity_id, attr_type)

        if entity_id not in snapshot.known_entities:
            return self._result(
                snapshot,
                ObjectReadStatus.NONE_OBJECT,
                entity_id,
                attr_type,
                None,
                (),
                "neither",
                (self.REASON_NONE_OBJECT,),
                decision,
            )
        if committed is None and not provisional:
            return self._result(
                snapshot,
                ObjectReadStatus.NONE_ATTRIBUTE,
                entity_id,
                attr_type,
                None,
                (),
                "neither",
                (self.REASON_NONE_ATTRIBUTE,),
                decision,
            )
        if committed is None:
            return self._result(
                snapshot,
                ObjectReadStatus.FOUND_DISPUTED,
                entity_id,
                attr_type,
                None,
                provisional,
                "provisional",
                (self.REASON_FOUND_DISPUTED,),
                decision,
            )
        challengers = tuple(value for value in provisional if value != committed)
        if not challengers:
            return self._result(
                snapshot,
                ObjectReadStatus.FOUND_COMMITTED,
                entity_id,
                attr_type,
                committed,
                (),
                "committed",
                (self.REASON_FOUND_COMMITTED,),
                decision,
            )
        return self._result(
            snapshot,
            ObjectReadStatus.FOUND_DISPUTED,
            entity_id,
            attr_type,
            committed,
            (committed,) + challengers,
            "committed+provisional",
            (self.REASON_FOUND_DISPUTED,),
            decision,
        )
