# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Thin wrapper over the SEALED symbolic organ (DeterministicObjectBank +
# CommitArbiterPas7a). It does NOT change the organ: it constructs the sealed
# components, renders a clean (entity, attribute, value) triple into the parser's
# native canonical sentence, writes through the arbiter (RoMR applies at write),
# runs the Pas7a consolidator at end_episode (N_promote=2, M_retrograde=2,
# K_promote_age=2, K_prune_stale=3), and reads back a committed value with the
# sealed status taxonomy plus a trace. All grounded truth comes from the organ;
# this wrapper never fabricates a value.

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from integration.sealed_loader import load_sealed_substrate

# status taxonomy (the organ emits FOUND / NONE_OBJECT / NONE_ATTRIBUTE natively;
# FOUND is refined to COMMITTED vs DISPUTED here using the provisional store).
FOUND_COMMITTED = "FOUND_COMMITTED"
FOUND_DISPUTED = "FOUND_DISPUTED"
NONE_OBJECT = "NONE_OBJECT"
NONE_ATTRIBUTE = "NONE_ATTRIBUTE"


@dataclass
class OrganReply:
    value: Optional[str]
    status: str
    trace: Dict[str, object]


def _det_emb(text: str) -> torch.Tensor:
    """Deterministic per-string unit vector (distinct strings -> near orthogonal),
    so the symbolic bank's entity/value identity matching is stable and reproducible."""
    seed = int(hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:15], 16)
    gen = torch.Generator().manual_seed(seed)
    v = torch.randn(768, generator=gen)
    return v / v.norm()


class OrganClient:
    """Sealed symbolic organ wrapper: write_fact / end_episode / query."""

    def __init__(self) -> None:
        ns = load_sealed_substrate()
        self._ns = ns
        self.attr_values: Dict[str, List[str]] = {k: list(v) for k, v in ns["V15_ATTR_VALUES"].items()}
        self.known_entities: List[str] = list(ns["HOLDOUT_ENTITIES_SINGLE"])
        self._bank = ns["DeterministicObjectBank"](capacity=64, d_model=768)
        self._prov = ns["ProvisionalMemory"]()
        self._ep = ns["EpisodeBuffer"]()
        self._stab = ns["BankStabilityIndex"]()
        self._audit: List[object] = []
        self._arbiter = ns["CommitArbiterPas7a"](self._bank, self._prov, self._ep, self._stab,
                                                 consolidation_audit_log=self._audit)
        self._episode = 0

    # ---- vocabulary helpers ----
    def is_attribute(self, attribute: str) -> bool:
        return attribute in self.attr_values

    def is_value(self, attribute: str, value: str) -> bool:
        return attribute in self.attr_values and value in self.attr_values[attribute]

    def _canonical(self, entity: str, attribute: str, value: str) -> str:
        # parse_fact's native, always-parseable forms (so extraction, not parsing,
        # is what is under test). location uses the prepositional form.
        if attribute == "location":
            return f"The {entity} is in the {value}."
        return f"The {entity} is {value}."

    # ---- episode lifecycle ----
    def begin_episode(self) -> int:
        self._episode += 1
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            self._arbiter.begin_episode(self._episode)
        return self._episode

    def end_episode(self) -> object:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            return self._arbiter.end_episode(_det_emb, _det_emb, _det_emb)

    # ---- write a gold-validated triple ----
    def write_fact(self, entity: str, attribute: str, value: str) -> Dict[str, object]:
        """Write a vocabulary-valid triple through the sealed arbiter. Returns the
        arbitrated write result summary (commit path) for attribution. Out-of-vocab
        triples are rejected here and never reach the organ."""
        if not self.is_attribute(attribute) or not self.is_value(attribute, value):
            return {"written": False, "reason": "out_of_vocabulary",
                    "entity": entity, "attribute": attribute, "value": value}
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            result = self._arbiter.write_fact(self._canonical(entity, attribute, value),
                                              _det_emb, _det_emb, _det_emb, write_step=self._episode)
        return {"written": True, "entity": entity, "attribute": attribute, "value": value,
                "commit_path": getattr(result, "commit_path", None),
                "rejected": getattr(result, "rejected", None)}

    # ---- read a committed value with sealed status taxonomy ----
    def query(self, entity: str, attribute: str, resolution_cos: float = 1.0) -> OrganReply:
        status, value_idx = self._bank.read_attribute(entity, attribute)   # FOUND/NONE_OBJECT/NONE_ATTRIBUTE
        slot = self._bank.find_by_entity_id(entity)
        value: Optional[str] = None
        if status == "FOUND":
            value = self.attr_values.get(attribute, [None] * (value_idx + 1))[value_idx] \
                if value_idx is not None and value_idx < len(self.attr_values.get(attribute, [])) else None
            disputed = False
            try:
                disputed = bool(self._prov.has_challenger(entity, attribute))
            except Exception:  # noqa: BLE001
                disputed = len(self._prov.values_for(entity, attribute) or []) > 0
            status = FOUND_DISPUTED if disputed else FOUND_COMMITTED
        trace = {"value": value, "status": status, "bank": "committed",
                 "slot_idx": slot, "cos_sim": round(float(resolution_cos), 4)}
        return OrganReply(value=value, status=status, trace=trace)

    def standalone_recall(self, gold_triples: List[Tuple[str, str, str]]) -> float:
        """Organ recall when fed gold triples directly (no LLM extractor)."""
        ok = total = 0
        for e, a, v in gold_triples:
            total += 1
            reply = self.query(e, a)
            ok += int(reply.status == FOUND_COMMITTED and reply.value == v)
        return ok / max(1, total)
