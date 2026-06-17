# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# DomainAdapter: the value-adapter mandated by Part 0. The sealed organ stores a value as
# an INTEGER INDEX into a frozen closed vocabulary (verified empirically: write_fact rejects
# out-of-vocab values, read_attribute returns an int), so it cannot hold an open string
# natively. This adapter makes the organ hold a REFERENCE TOKEN per open value: each open
# string is assigned a position in the mapped organ attribute's reference vocabulary, the
# organ commits that reference word through its real arbiter (RoMR at write, Pas7a at
# end_episode) exactly as for synthetic facts, and the adapter inverts (organ_attr,
# value_idx) -> open string on read. It touches NOTHING under dcortex/ or steps/13 and does
# not edit organ_client.py: it drives the public OrganClient API and keeps the open<->token
# tables in integration/. Capacity is bounded (4 organ attributes, len(vocab) tokens each):
# on overflow the adapter ABSTAINS (returns not-written) rather than colliding two open
# strings onto one token, which would corrupt the store.

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from integration.organ_client import OrganClient, FOUND_COMMITTED, NONE_OBJECT
from integration import patent_schema


@dataclass
class WriteOutcome:
    written: bool
    reason: str                      # committed | capacity_overflow | entity_overflow | organ_rejected
    patent_id: str
    attribute: str
    open_value: str
    organ_attr: Optional[str] = None
    reference_token: Optional[str] = None
    value_idx: Optional[int] = None


@dataclass
class ReadOutcome:
    status: str
    open_value: Optional[str]
    organ_attr: Optional[str]
    trace: Dict[str, object] = field(default_factory=dict)


class DomainAdapter:
    """Reference-token indirection over the sealed organ for open-string domain values."""

    def __init__(self, organ: OrganClient, attr_map: Optional[Dict[str, str]] = None) -> None:
        self.organ = organ
        self.attr_map: Dict[str, str] = dict(attr_map or patent_schema.ORGAN_ATTR_MAP)
        # reference vocabulary per organ attribute, taken from the LIVE sealed vocab copy
        # (organ.attr_values is a per-instance copy of V15_ATTR_VALUES; we only READ it).
        self.ref_words: Dict[str, List[str]] = {
            oa: list(organ.attr_values[oa]) for oa in set(self.attr_map.values())
        }
        # per patent-attribute open-string table; index into this list == reference token
        # position == value index the organ stores. Injective by construction.
        self.value_table: Dict[str, List[str]] = {pa: [] for pa in self.attr_map}
        # entity indirection: patent slug -> a known single-token organ entity
        self.entity_pool: List[str] = list(organ.known_entities)
        self.entity_map: Dict[str, str] = {}
        self.overflow_log: List[Dict[str, object]] = []

    # ---- capacity ----
    def capacity(self, patent_attr: str) -> int:
        return len(self.ref_words[self.attr_map[patent_attr]])

    def capacity_report(self) -> Dict[str, Dict[str, int]]:
        rep = {}
        for pa in self.attr_map:
            used = len(self.value_table[pa])
            overflow = sum(1 for e in self.overflow_log
                           if e["attribute"] == pa and e["reason"] == "capacity_overflow")
            rep[pa] = {"organ_attr": self.attr_map[pa], "capacity": self.capacity(pa),
                       "distinct_used": used, "capacity_overflow_writes": overflow}
        return rep

    # ---- entity indirection ----
    def map_entity(self, patent_id: str) -> Optional[str]:
        if patent_id not in self.entity_map:
            if len(self.entity_map) >= len(self.entity_pool):
                return None
            self.entity_map[patent_id] = self.entity_pool[len(self.entity_map)]
        return self.entity_map[patent_id]

    # ---- value indirection (allocate a reference token for an open string) ----
    def _alloc_value_idx(self, patent_attr: str, open_value: str) -> Optional[int]:
        table = self.value_table[patent_attr]
        if open_value in table:
            return table.index(open_value)
        if len(table) >= self.capacity(patent_attr):
            return None                                  # capacity overflow -> abstain
        table.append(open_value)
        return len(table) - 1

    # ---- write one open (patent_id, attribute, open_value) through the real organ ----
    def write_open_fact(self, patent_id: str, attribute: str, open_value: str) -> WriteOutcome:
        organ_attr = self.attr_map[attribute]
        ent = self.map_entity(patent_id)
        if ent is None:
            self.overflow_log.append({"patent_id": patent_id, "attribute": attribute,
                                      "reason": "entity_overflow"})
            return WriteOutcome(False, "entity_overflow", patent_id, attribute, open_value, organ_attr)
        idx = self._alloc_value_idx(attribute, open_value)
        if idx is None:
            self.overflow_log.append({"patent_id": patent_id, "attribute": attribute,
                                      "reason": "capacity_overflow"})
            return WriteOutcome(False, "capacity_overflow", patent_id, attribute, open_value, organ_attr)
        ref_word = self.ref_words[organ_attr][idx]
        res = self.organ.write_fact(ent, organ_attr, ref_word)   # real arbiter (RoMR at write)
        if not res.get("written"):
            return WriteOutcome(False, "organ_rejected", patent_id, attribute, open_value,
                                organ_attr, ref_word, idx)
        return WriteOutcome(True, "committed", patent_id, attribute, open_value,
                            organ_attr, ref_word, idx)

    # ---- read one open value back, decoding the reference token ----
    def read_open(self, patent_id: str, attribute: str) -> ReadOutcome:
        organ_attr = self.attr_map[attribute]
        ent = self.entity_map.get(patent_id)
        if ent is None:
            return ReadOutcome(NONE_OBJECT, None, organ_attr, {"reason": "entity_not_mapped"})
        reply = self.organ.query(ent, organ_attr)
        if reply.status != FOUND_COMMITTED or reply.value is None:
            return ReadOutcome(reply.status, None, organ_attr, reply.trace)
        # reply.value is the reference word; recover its index, then the open string
        ref_list = self.ref_words[organ_attr]
        try:
            idx = ref_list.index(reply.value)
        except ValueError:
            return ReadOutcome(reply.status, None, organ_attr, reply.trace)
        table = self.value_table[attribute]
        open_value = table[idx] if idx < len(table) else None
        return ReadOutcome(reply.status, open_value, organ_attr, reply.trace)
