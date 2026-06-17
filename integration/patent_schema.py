# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Patent domain schema for the real-facts feasibility campaign (step_33). It declares
# BOTH a closed-categorical part (constrained classification applies) and an open-valued
# part (free extraction, the hard path), plus the mapping from each patent attribute onto
# one of the sealed organ's four closed attributes (color/size/location/state). The organ
# stores a VALUE INDEX into a frozen per-attribute vocabulary (verified empirically in
# Part 0), so an open string can only be stored via a reference-token indirection held in
# integration/ (see domain_adapter.DomainAdapter). The mapping below is 1:1 onto the four
# organ attributes, which is exactly the capacity ceiling: at most four attributes per
# entity, and at most len(vocab) distinct values per mapped organ attribute.

from typing import Dict, List

# Closed-categorical patent attributes: extracted by logit-masked classification.
CLOSED_ATTRIBUTES: Dict[str, List[str]] = {
    "legal_status": ["granted", "pending", "refused", "withdrawn"],
    "ipc_section": ["A", "B", "C", "D", "E", "F", "G", "H"],
}

# Open-valued patent attributes: free extraction, no closed value set (the hard path).
OPEN_ATTRIBUTES: List[str] = ["patent_number", "filing_date", "applicant", "title_keyword"]

# Mapping patent attribute -> sealed organ attribute. The organ has exactly four
# attributes; their vocabulary sizes (color=15, location=10, state=8, size=4) are the
# reference-token budget per attribute. Only four patent attributes can be stored at once;
# filing_date and title_keyword have NO remaining organ slot and are therefore
# extraction-only (a measured capacity blocker, not a bug).
ORGAN_ATTR_MAP: Dict[str, str] = {
    "patent_number": "color",     # 15 reference tokens -> fits exactly 15 distinct numbers
    "applicant": "location",      # 10 reference tokens -> OVERFLOW beyond 10 distinct applicants
    "ipc_section": "state",       # 8 reference tokens  -> 7 distinct sections used (A-H minus E)
    "legal_status": "size",       # 4 reference tokens  -> granted/pending (lifecycle update)
}

# Patent attributes that have NO organ slot under ORGAN_ATTR_MAP (only four organ
# attributes exist). They are still extracted and scored, but cannot be committed.
EXTRACTION_ONLY_ATTRIBUTES: List[str] = [a for a in OPEN_ATTRIBUTES if a not in ORGAN_ATTR_MAP]

# All schema attributes (closed first, then open), the full extraction target set.
ALL_ATTRIBUTES: List[str] = list(CLOSED_ATTRIBUTES) + list(OPEN_ATTRIBUTES)


def attribute_kind(attribute: str) -> str:
    """Return 'closed' or 'open' for a schema attribute."""
    return "closed" if attribute in CLOSED_ATTRIBUTES else "open"
