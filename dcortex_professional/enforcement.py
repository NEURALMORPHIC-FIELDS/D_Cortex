# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Mechanical enforcement primitives: domain router, memory-state resolver, and the
# hard verifier. The verifier is the single mechanical check every candidate answer
# must pass; an answer that asserts an ungrounded, contradicted, or forbidden claim
# is rejected here. These functions hold no generation capability and cannot be
# bypassed by prompt content.

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dcortex_professional.pack import ProfessionalPack, _norm

# memory states
COMMITTED = "committed"
PROVISIONAL = "provisional"
DISPUTED = "disputed"
FORBIDDEN = "forbidden"
UNKNOWN = "unknown"
OUT_OF_DOMAIN = "out_of_domain"


@dataclass
class Claim:
    entity: str
    attribute: str
    value: str
    status: str = COMMITTED       # the status the answer asserts this claim at


@dataclass
class Resolution:
    entity: Optional[str]
    attribute: Optional[str]
    state: str
    value: Optional[str] = None
    disputed: Optional[Dict] = None
    in_domain: bool = True
    risk: str = "normal"


class DomainRouter:
    """Classify a query as in-domain / out-of-domain and flag pressure risk."""

    PRESSURE = ("confirm", "guarantee", "definitely", "for sure", "just tell me", "must be",
                "you know", "obviously")

    def __init__(self, pack: ProfessionalPack) -> None:
        self.pack = pack

    def route(self, query: str) -> Tuple[bool, str]:
        low = _norm(query)
        in_domain = any(k in low for k in self.pack.in_domain_keywords()) or \
            any(_norm(e) in low for e in self.pack.schemas.get("entities", []))
        risk = "high" if any(p in low for p in self.PRESSURE) else "normal"
        return in_domain, risk


class MemoryStateResolver:
    """Map a query to (entity, attribute) and its memory state in the pack.

    Extraction is mechanical and deterministic: the longest known attribute phrase
    and the known entity present in the query are matched against the pack schema."""

    def __init__(self, pack: ProfessionalPack) -> None:
        self.pack = pack
        # attribute -> surface phrases (name with underscores as spaces + light synonyms)
        self.attr_phrases: Dict[str, List[str]] = {}
        synonyms = {
            "patent_number": ["patent number", "patent", "application number"],
            "owner": ["owner", "assignee", "who owns", "belongs to"],
            "architecture": ["architecture", "design", "what kind of model"],
            "substrate": ["substrate", "base model", "backbone"],
            "hidden_dim": ["hidden dim", "hidden dimension", "hidden size", "width"],
            "decoder_layers": ["decoder layers", "number of decoder layers", "dec layers"],
            "encoder_layers": ["encoder layers", "number of encoder layers", "enc layers"],
            "held_out_structural_exact_median": ["structural exact", "held-out exact",
                                                 "structural accuracy", "structural median"],
            "no_memory_control_exact": ["no-memory control", "no memory control", "control exact"],
            "claim_status": ["claim status", "is it proven", "proven", "status of the claim"],
            "parameter_count": ["parameter count", "parameters", "how many parameters", "param count"],
            "multi_hardware_reproduction": ["multi-hardware", "multi hardware reproduction",
                                            "distinct hardware"],
            "independent_replication": ["independent replication", "replicated independently"],
        }
        for attr in self.pack.schemas.get("attributes", {}):
            phrases = [attr.replace("_", " ")] + synonyms.get(attr, [])
            self.attr_phrases[attr] = sorted({_norm(p) for p in phrases}, key=len, reverse=True)

    def _match_entity(self, low: str) -> Optional[str]:
        for ent in self.pack.schemas.get("entities", []):
            if _norm(ent) in low:
                return ent
        return None

    def _match_attribute(self, low: str) -> Optional[str]:
        best: Optional[str] = None
        best_len = 0
        for attr, phrases in self.attr_phrases.items():
            for phrase in phrases:
                if phrase in low and len(phrase) > best_len:
                    best, best_len = attr, len(phrase)
        return best

    def resolve(self, query: str, in_domain: bool, risk: str) -> Resolution:
        low = _norm(query)
        entity = self._match_entity(low)
        attribute = self._match_attribute(low)
        if not in_domain and entity is None:
            return Resolution(entity, attribute, OUT_OF_DOMAIN, in_domain=False, risk=risk)
        if entity is None or attribute is None:
            return Resolution(entity, attribute, UNKNOWN, in_domain=in_domain, risk=risk)
        if self.pack.disputed_value(entity, attribute) is not None:
            return Resolution(entity, attribute, DISPUTED,
                              disputed=self.pack.disputed_value(entity, attribute),
                              in_domain=in_domain, risk=risk)
        committed = self.pack.committed_value(entity, attribute)
        if committed is not None:
            return Resolution(entity, attribute, COMMITTED, value=committed.value,
                              in_domain=in_domain, risk=risk)
        provisional = self.pack.provisional_value(entity, attribute)
        if provisional is not None:
            return Resolution(entity, attribute, PROVISIONAL, value=provisional.value,
                              in_domain=in_domain, risk=risk)
        return Resolution(entity, attribute, UNKNOWN, in_domain=in_domain, risk=risk)


@dataclass
class VerdictCheck:
    passed: bool
    reason: str
    forbidden_hit: Optional[Dict] = None
    unsupported: List[Claim] = field(default_factory=list)
    contamination: List[str] = field(default_factory=list)


class HardVerifier:
    """The unbypassable mechanical check. A candidate answer passes only if every
    factual claim it asserts is grounded in committed memory (or explicitly carried
    as provisional/disputed), it contains no foreign committed value for the queried
    slot, and it matches no forbidden pattern."""

    def __init__(self, pack: ProfessionalPack) -> None:
        self.pack = pack

    def check(self, query: str, text: str, claims: List[Claim], resolution: Resolution) -> VerdictCheck:
        # 1) forbidden patterns (mechanical block)
        hit = self.pack.forbidden_match(text)
        if hit is not None:
            return VerdictCheck(False, f"forbidden pattern '{hit['pattern']}'", forbidden_hit=hit)

        # 2) every asserted committed claim must be grounded with the exact value
        unsupported: List[Claim] = []
        for c in claims:
            if c.status == COMMITTED:
                fact = self.pack.committed_value(c.entity, c.attribute)
                if fact is None or _norm(fact.value) != _norm(c.value):
                    unsupported.append(c)
            elif c.status == PROVISIONAL:
                if self.pack.provisional_value(c.entity, c.attribute) is None:
                    unsupported.append(c)
            elif c.status == DISPUTED:
                if self.pack.disputed_value(c.entity, c.attribute) is None:
                    unsupported.append(c)
        if unsupported:
            return VerdictCheck(False, "ungrounded factual claim", unsupported=unsupported)

        # 3) contamination: the text must not assert, for the queried slot, any committed
        # value that belongs to a DIFFERENT (entity, attribute) than the queried one.
        contamination: List[str] = []
        if resolution.entity is not None and resolution.attribute is not None:
            queried_value = None
            f = self.pack.committed_value(resolution.entity, resolution.attribute)
            if f is not None:
                queried_value = _norm(f.value)
            low = _norm(text)
            for (ent, attr), value in self.pack.all_committed_values():
                # the queried entity's own facts (any attribute) are never contamination
                if ent == _norm(resolution.entity):
                    continue
                # only a DIFFERENT entity's value for the SAME attribute counts as the
                # model pulling the wrong fact; common cross-attribute words do not.
                if attr != _norm(resolution.attribute):
                    continue
                nv = _norm(value)
                if len(nv) >= 8 and nv in low and nv != queried_value and \
                        any(c.attribute == resolution.attribute for c in claims):
                    contamination.append(value)
        if contamination:
            return VerdictCheck(False, "foreign committed value in answer", contamination=contamination)

        return VerdictCheck(True, "ok")
