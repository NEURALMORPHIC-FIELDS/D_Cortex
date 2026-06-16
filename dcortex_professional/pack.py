# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Professional pack loader. A pack is a directory of committed / provisional /
# disputed / forbidden facts plus sources, abstain rules, and schemas. The pack is
# the only source of grounded truth the control layer may use.

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _norm(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


@dataclass(frozen=True)
class Fact:
    entity: str
    attribute: str
    value: str
    provenance: Dict[str, str]
    status: str = "committed"


@dataclass
class ProfessionalPack:
    name: str
    committed: Dict[Tuple[str, str], Fact]
    provisional: Dict[Tuple[str, str], Fact]
    disputed: Dict[Tuple[str, str], Dict]
    forbidden: List[Dict]
    sources: Dict[str, Dict]
    abstain_rules: Dict
    schemas: Dict

    @staticmethod
    def load(pack_dir: str) -> "ProfessionalPack":
        d = Path(pack_dir)
        if not d.exists():
            raise FileNotFoundError(f"pack directory not found: {d}")

        def read_jsonl(name: str) -> List[Dict]:
            p = d / name
            if not p.exists():
                return []
            return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

        def read_json(name: str) -> Dict:
            p = d / name
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

        committed = {(_norm(r["entity"]), _norm(r["attribute"])):
                     Fact(r["entity"], r["attribute"], r["value"], r.get("provenance", {}), "committed")
                     for r in read_jsonl("committed.jsonl")}
        provisional = {(_norm(r["entity"]), _norm(r["attribute"])):
                       Fact(r["entity"], r["attribute"], r["value"], r.get("provenance", {}), "provisional")
                       for r in read_jsonl("provisional.jsonl")}
        disputed = {(_norm(r["entity"]), _norm(r["attribute"])): r for r in read_jsonl("disputed.jsonl")}
        forbidden = read_jsonl("forbidden.jsonl")
        return ProfessionalPack(
            name=d.name, committed=committed, provisional=provisional, disputed=disputed,
            forbidden=forbidden, sources=read_json("sources.json"),
            abstain_rules=read_json("abstain_rules.json"), schemas=read_json("schemas.json"))

    # --- lookups (the only grounded-truth access) ---
    def committed_value(self, entity: str, attribute: str) -> Optional[Fact]:
        return self.committed.get((_norm(entity), _norm(attribute)))

    def provisional_value(self, entity: str, attribute: str) -> Optional[Fact]:
        return self.provisional.get((_norm(entity), _norm(attribute)))

    def disputed_value(self, entity: str, attribute: str) -> Optional[Dict]:
        return self.disputed.get((_norm(entity), _norm(attribute)))

    def all_committed_values(self) -> List[Tuple[Tuple[str, str], str]]:
        return [(k, f.value) for k, f in self.committed.items()]

    def forbidden_match(self, text: str) -> Optional[Dict]:
        low = _norm(text)
        for rule in self.forbidden:
            if _norm(rule.get("pattern", "")) in low:
                return rule
        return None

    def is_known_entity(self, entity: str) -> bool:
        ents = {_norm(e) for e in self.schemas.get("entities", [])}
        return _norm(entity) in ents

    def is_known_attribute(self, attribute: str) -> bool:
        attrs = {_norm(a) for a in self.schemas.get("attributes", {})}
        return _norm(attribute) in attrs

    def in_domain_keywords(self) -> List[str]:
        return [_norm(k) for k in self.schemas.get("in_domain_keywords", [])]
