# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# VerbalizerControl: the single exit point. answer(question) extracts the query with
# the Qwen IngestAdapter, reads the sealed organ via OrganClient, then verbalizes with
# Qwen under a constraint: FOUND_COMMITTED is masked to the committed value (the model
# verbalizes but cannot alter it), FOUND_DISPUTED emits the values with a dispute flag,
# and every NONE_* / extraction failure is masked to an abstain template. No code path
# returns text that has not passed this veto, so no ungrounded assertion reaches the user.

from dataclasses import dataclass, field
from typing import Dict, Optional

from integration.ingest_adapter import IngestAdapter, ExtractError, FactTriple
from integration.organ_client import (OrganClient, FOUND_COMMITTED, FOUND_DISPUTED,
                                       NONE_OBJECT, NONE_ATTRIBUTE)

PARSER_FAILURE = "PARSER_FAILURE"
PARSE_UNCERTAIN = "PARSE_UNCERTAIN"


@dataclass
class Answer:
    text: str
    status: str
    grounded: bool
    value: Optional[str]
    trace: Dict[str, object] = field(default_factory=dict)
    extraction_ok: bool = True
    extracted_entity: Optional[str] = None
    extracted_attribute: Optional[str] = None
    source_path: str = "verifier"


def _abstain(message: str) -> str:
    return f"[ABSTAIN] {message}"


class VerbalizerControl:
    def __init__(self, qwen, ingest: IngestAdapter, organ: OrganClient) -> None:
        self.qwen = qwen
        self.ingest = ingest
        self.organ = organ

    def answer(self, question: str) -> Answer:
        ext = self.ingest.extract_query(question)
        if isinstance(ext, ExtractError):
            status = PARSER_FAILURE if ext.reason in ("malformed_json", "missing_field") else PARSE_UNCERTAIN
            return self._finalize(_abstain("Could not extract a grounded query."), status, None,
                                  {"value": None, "status": status, "bank": None, "slot_idx": None,
                                   "cos_sim": 0.0}, extraction_ok=False)
        reply = self.organ.query(ext.entity, ext.attribute, ext.resolution_cos)
        status, value, trace = reply.status, reply.value, reply.trace
        if status == FOUND_COMMITTED:
            prompt = f"The {ext.entity}'s {ext.attribute} is"
            cr = self.qwen.generate_constrained(prompt, value)   # value tokens forced
            text = f"{prompt} {cr.text}.".replace(f"is {value}", f"is {value}")
            if value.lower() not in text.lower():
                text = f"The {ext.entity}'s {ext.attribute} is {value}."
            return self._finalize(text, status, value, trace, grounded=True,
                                  entity=ext.entity, attribute=ext.attribute)
        if status == FOUND_DISPUTED:
            return self._finalize(f"[DISPUTED] {ext.entity} {ext.attribute}: {value} (disputed).",
                                  status, value, trace, entity=ext.entity, attribute=ext.attribute)
        # NONE_OBJECT / NONE_ATTRIBUTE
        return self._finalize(_abstain(f"Not grounded in memory ({status})."), status, None, trace,
                              entity=ext.entity, attribute=ext.attribute)

    # ---- the single return path: enforce that grounded text carries the committed value ----
    def _finalize(self, text: str, status: str, value: Optional[str], trace: Dict[str, object],
                  grounded: bool = False, extraction_ok: bool = True,
                  entity: Optional[str] = None, attribute: Optional[str] = None) -> Answer:
        if grounded:
            if status != FOUND_COMMITTED or value is None or value.lower() not in text.lower():
                # veto: a grounded claim that does not carry the committed value is refused
                text, grounded, status = _abstain("Verification failed; refusing ungrounded assertion."), False, status
        source = ("organ+constrained_verbalize" if grounded else
                  "verifier" if status in (NONE_OBJECT, NONE_ATTRIBUTE, PARSER_FAILURE, PARSE_UNCERTAIN)
                  else "organ")
        return Answer(text=text, status=status, grounded=grounded, value=value, trace=trace,
                      extraction_ok=extraction_ok, extracted_entity=entity, extracted_attribute=attribute,
                      source_path=source)
