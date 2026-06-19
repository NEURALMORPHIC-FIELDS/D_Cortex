# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Shared, inspectable memory store - the place where the user and the model meet in the same
# explicit memory. SCOPE: this is the SUBSTRATE track (scalable, value-based, honest STORAGE +
# ADMINISTRATION). It does NOT add reasoning - Stage C showed the model does not yet reason over
# memory (that is Stage 5, the separate frontier). v1 = an exact, honest, auditable, self-revising
# store the model grounds its answers in; NOT "the model reasons over your domain".
#
# Two write paths with DIFFERENT exactness, by design:
#   write_canonical : the user GIVES the value, so we KNOW it -> assign the token DIRECTLY from the
#                     value registry. EXACT (no internalization drift). status=committed, authoritative.
#   write_extracted : a model-internalized value -> tokenize(w_value) via the codebook (carries the
#                     ~8% internalization-consistency floor; the extraction path = Stage I, not v1).

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class MemoryObject:
    memory_token: int
    entity: str
    attribute: str
    value: str
    status: str = "committed"            # committed | provisional | disputed | uncertain
    provenance: str = "user_canonical"   # user_canonical | model_internalized
    trust: float = 1.0
    payload: Optional[torch.Tensor] = None   # 768-d internalized content (for future reasoning); None for canonical
    support: Dict = field(default_factory=dict)


class SharedMemoryStore:
    """Inspectable, editable object store. Canonical = exact direct token; extracted = codebook token."""

    def __init__(self, tokenizer=None) -> None:
        self.tokenizer = tokenizer                       # MemoryTokenizer (for the extracted path)
        self._objects: Dict[tuple, MemoryObject] = {}    # (entity, attribute) -> object
        self._registry: Dict[str, int] = {}              # value -> canonical token (exact, assigned on first sight)

    # ---- canonical token: exact, derived from the KNOWN value (not from internalization) ----
    def canonical_token(self, value: str) -> int:
        if value not in self._registry:
            self._registry[value] = len(self._registry)
        return self._registry[value]

    # ---- write paths ----
    def write_canonical(self, entity: str, attribute: str, value: str, rule: Optional[Dict] = None) -> MemoryObject:
        obj = MemoryObject(memory_token=self.canonical_token(value), entity=entity, attribute=attribute,
                           value=value, status="committed", provenance="user_canonical", trust=1.0,
                           support={"rule": rule} if rule else {})
        self._objects[(entity, attribute)] = obj
        return obj

    def write_extracted(self, entity: str, attribute: str, w_value: torch.Tensor) -> MemoryObject:
        if self.tokenizer is None:
            raise RuntimeError("extracted path needs a fitted tokenizer")
        token = self.tokenizer.tokenize(w_value)
        value = self.tokenizer.decode(token)
        obj = MemoryObject(memory_token=token, entity=entity, attribute=attribute, value=value,
                           status="provisional", provenance="model_internalized", trust=0.6, payload=w_value)
        self._objects[(entity, attribute)] = obj
        return obj

    # ---- read / inspect / edit (the "shared" property: the user sees and corrects) ----
    def get(self, entity: str, attribute: str) -> Optional[MemoryObject]:
        return self._objects.get((entity, attribute))

    def list(self) -> List[MemoryObject]:
        return list(self._objects.values())

    def edit(self, entity: str, attribute: str, new_value: str) -> Optional[MemoryObject]:
        key = (entity, attribute)
        if key not in self._objects:
            return None
        o = self._objects[key]
        o.value = new_value
        o.memory_token = self.canonical_token(new_value)   # re-tokenize on edit (canonical exact)
        o.provenance = "user_canonical"; o.status = "committed"; o.trust = 1.0; o.payload = None
        return o

    def dump(self) -> str:
        lines = []
        for (e, a), o in sorted(self._objects.items()):
            lines.append(f"{e}.{a} = {o.value!r} [tok={o.memory_token} {o.status} {o.provenance} trust={o.trust}]")
        return "\n".join(lines)
