# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b-R
# Pure read-only routing bridge for adapter-approved semantic queries.

import hashlib
import json
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


class QueryRouteStatus(str, Enum):
    """Read-only bridge routing status."""

    ROUTED = "ROUTED"
    FALLBACK = "FALLBACK"


@dataclass(frozen=True)
class ReadOnlyQueryRoute:
    """Immutable routed or exact-fallback query result."""

    status: QueryRouteStatus
    hypothesis_id: Optional[str]
    original_query: str
    routed_query: str
    entity_id: Optional[str]
    attr_type: Optional[str]
    adapter_audit_sequence: Optional[int]
    reason_codes: Tuple[str, ...]
    route_fingerprint: str

    def to_dict(self) -> Dict[str, object]:
        """Serialize the route into deterministic JSON-compatible data."""
        data = asdict(self)
        data["status"] = self.status.value
        data["reason_codes"] = list(self.reason_codes)
        return data

    def to_json(self) -> str:
        """Return deterministic compact JSON for this route."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


class ReadOnlySemanticQueryBridge:
    """Convert approved semantic query hypotheses into canonical read routes.

    The bridge is pure routing. It owns no model, memory, reader, writer,
    provisional store, commit path, or consolidator.
    """

    REASON_ROUTED = "ROUTED_ACCEPTED_QUERY"
    REASON_DECISION_REQUIRED = "DECISION_REQUIRED"
    REASON_HYPOTHESIS_REQUIRED = "HYPOTHESIS_REQUIRED"
    REASON_DECISION_NOT_ACCEPT_QUERY = "DECISION_NOT_ACCEPT_QUERY"
    REASON_ID_MISMATCH = "DECISION_HYPOTHESIS_ID_MISMATCH"
    REASON_NOT_QUERY_MODE = "NOT_QUERY_MODE"
    REASON_NOT_QUERY_ONLY = "NOT_QUERY_ONLY"
    REASON_FIELDS_REQUIRED = "QUERY_FIELDS_REQUIRED"
    REASON_UNSUPPORTED_ATTRIBUTE = "UNSUPPORTED_ATTRIBUTE"

    CANONICAL_READ_TEMPLATES: Dict[str, str] = {
        "color": "What color is the {entity}? The {entity} is",
        "size": "What size is the {entity}? The {entity} is",
        "location": "Where is the {entity}? The {entity} is in the",
        "state": "What state is the {entity} in? The {entity} is",
    }

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
    def _build_route(
        cls,
        status: QueryRouteStatus,
        original_query: str,
        routed_query: str,
        hypothesis: Optional[SemanticHypothesis],
        decision: Optional[AdapterDecision],
        reason_codes: Tuple[str, ...],
    ) -> ReadOnlyQueryRoute:
        payload: Dict[str, object] = {
            "status": status.value,
            "hypothesis_id": None if hypothesis is None else hypothesis.hypothesis_id,
            "original_query": original_query,
            "routed_query": routed_query,
            "entity_id": None if hypothesis is None else hypothesis.entity_id,
            "attr_type": None if hypothesis is None else hypothesis.attr_type,
            "adapter_audit_sequence": (
                None if decision is None else decision.audit_sequence
            ),
            "reason_codes": list(reason_codes),
        }
        return ReadOnlyQueryRoute(
            status=status,
            hypothesis_id=payload["hypothesis_id"],
            original_query=original_query,
            routed_query=routed_query,
            entity_id=payload["entity_id"],
            attr_type=payload["attr_type"],
            adapter_audit_sequence=payload["adapter_audit_sequence"],
            reason_codes=reason_codes,
            route_fingerprint=cls._fingerprint(payload),
        )

    def route(
        self,
        original_query: str,
        hypothesis: Optional[SemanticHypothesis],
        decision: Optional[AdapterDecision],
    ) -> ReadOnlyQueryRoute:
        """Return a canonical read route or the exact original fallback."""
        reasons = []
        if hypothesis is None:
            reasons.append(self.REASON_HYPOTHESIS_REQUIRED)
        if decision is None:
            reasons.append(self.REASON_DECISION_REQUIRED)
        if decision is not None and decision.status != DecisionStatus.ACCEPT_QUERY:
            reasons.append(self.REASON_DECISION_NOT_ACCEPT_QUERY)
        if (
            hypothesis is not None
            and decision is not None
            and hypothesis.hypothesis_id != decision.hypothesis_id
        ):
            reasons.append(self.REASON_ID_MISMATCH)
        if hypothesis is not None and hypothesis.mode != HypothesisMode.QUERY:
            reasons.append(self.REASON_NOT_QUERY_MODE)
        if (
            hypothesis is not None
            and hypothesis.requested_destination != RequestedDestination.QUERY_ONLY
        ):
            reasons.append(self.REASON_NOT_QUERY_ONLY)
        if hypothesis is not None and (
            not isinstance(hypothesis.entity_id, str)
            or not hypothesis.entity_id.strip()
            or not isinstance(hypothesis.attr_type, str)
            or not hypothesis.attr_type.strip()
        ):
            reasons.append(self.REASON_FIELDS_REQUIRED)
        if (
            hypothesis is not None
            and isinstance(hypothesis.attr_type, str)
            and hypothesis.attr_type not in self.CANONICAL_READ_TEMPLATES
        ):
            reasons.append(self.REASON_UNSUPPORTED_ATTRIBUTE)

        if reasons:
            return self._build_route(
                QueryRouteStatus.FALLBACK,
                original_query,
                original_query,
                hypothesis,
                decision,
                tuple(sorted(set(reasons))),
            )

        assert hypothesis is not None
        assert decision is not None
        routed_query = self.CANONICAL_READ_TEMPLATES[str(hypothesis.attr_type)].format(
            entity=str(hypothesis.entity_id)
        )
        return self._build_route(
            QueryRouteStatus.ROUTED,
            original_query,
            routed_query,
            hypothesis,
            decision,
            (self.REASON_ROUTED,),
        )
