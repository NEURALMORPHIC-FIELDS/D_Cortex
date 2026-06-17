# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Real, source-pinned patent corpus for the domain-extension feasibility campaign
# (step_33). Every record was fetched live from Google Patents (patents.google.com,
# the per-record source_url) and only fields read directly from that page are marked
# verified. NO value is fabricated: where a field could not be confirmed (e.g. the IPC
# section of US7479949B2 did not render), it is left empty and excluded from gold.
#
# The bibliographic source_text per record is a sentence ASSEMBLED from the verified
# fields (not a verbatim copy of the patent abstract); the campaign tests extraction of
# the verified facts from that text, then storage of those facts through the sealed organ
# via the reference-token adapter. The owner patent EP25216372.0 is intentionally absent:
# as of the fetch it returned HTTP 404 and no public bibliographic record exists (normal
# for an EP application filed in late 2025, unpublished until ~18 months after priority).

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PatentRecord:
    slug: str                       # canonical entity id within this corpus
    patent_number: str
    title: str
    title_keyword: str              # a short keyword substring of the title
    legal_status: str               # granted | pending | refused | withdrawn
    ipc_section: str                # single letter A-H, or "" if unverifiable
    filing_date: str                # ISO date
    applicant: str                  # canonical short assignee name (distinctness key)
    applicant_full: str             # full assignee string as shown on the page
    source_url: str
    verified_fields: List[str] = field(default_factory=list)
    unverifiable_fields: List[str] = field(default_factory=list)


# 15 real patents fetched from Google Patents (see source_url). applicant is the
# canonical short assignee used as the distinctness key; applicant_full is the page string.
PATENTS: List[PatentRecord] = [
    PatentRecord("pat01", "US6285999B1", "Method for node ranking in a linked database",
                 "node ranking", "granted", "G", "1998-01-09", "Stanford",
                 "Leland Stanford Junior University", "https://patents.google.com/patent/US6285999B1/en",
                 ["patent_number", "title", "legal_status", "ipc_section", "filing_date", "applicant"]),
    PatentRecord("pat02", "US5960411A", "Method and system for placing a purchase order via a communications network",
                 "purchase order", "granted", "G", "1997-09-12", "Amazon",
                 "Amazon.com Inc", "https://patents.google.com/patent/US5960411A/en",
                 ["patent_number", "title", "legal_status", "ipc_section", "filing_date", "applicant"]),
    PatentRecord("pat03", "US4237224A", "Process for producing biologically functional molecular chimeras",
                 "molecular chimeras", "granted", "C", "1979-01-04", "Stanford",
                 "Leland Stanford Junior University", "https://patents.google.com/patent/US4237224A/en",
                 ["patent_number", "title", "legal_status", "ipc_section", "filing_date", "applicant"]),
    PatentRecord("pat04", "US6004596A", "Sealed crustless sandwich",
                 "crustless sandwich", "granted", "A", "1997-12-08", "Menusaver",
                 "Menusaver Inc", "https://patents.google.com/patent/US6004596A/en",
                 ["patent_number", "title", "legal_status", "ipc_section", "filing_date", "applicant"]),
    # ipc_section unverifiable on the page -> empty, excluded from gold (abstention case).
    PatentRecord("pat05", "US7479949B2", "Touch screen device method and graphical user interface",
                 "touch screen", "granted", "", "2008-04-11", "Apple",
                 "Apple Inc", "https://patents.google.com/patent/US7479949B2/en",
                 ["patent_number", "title", "legal_status", "filing_date", "applicant"], ["ipc_section"]),
    PatentRecord("pat06", "US5934507A", "Refueling machine having no electrically actuated means in an explosive area",
                 "refueling machine", "granted", "B", "1997-04-09", "Hoechst",
                 "Hoechst Japan Limited", "https://patents.google.com/patent/US5934507A/en",
                 ["patent_number", "title", "legal_status", "ipc_section", "filing_date", "applicant"]),
    PatentRecord("pat07", "US3005282A", "Toy building brick",
                 "building brick", "granted", "A", "1958-07-28", "Interlego",
                 "Interlego AG", "https://patents.google.com/patent/US3005282A/en",
                 ["patent_number", "title", "legal_status", "ipc_section", "filing_date", "applicant"]),
    PatentRecord("pat08", "US4399216A", "Processes for inserting DNA into eucaryotic cells",
                 "inserting DNA", "granted", "C", "1980-02-25", "Columbia",
                 "Columbia University in the City of New York", "https://patents.google.com/patent/US4399216A/en",
                 ["patent_number", "title", "legal_status", "ipc_section", "filing_date", "applicant"]),
    PatentRecord("pat09", "US6097812A", "Cryptographic system",
                 "cryptographic system", "granted", "H", "1933-07-25", "NSA",
                 "National Security Agency", "https://patents.google.com/patent/US6097812A/en",
                 ["patent_number", "title", "legal_status", "ipc_section", "filing_date", "applicant"]),
    PatentRecord("pat10", "US2297691A", "Electrophotography",
                 "electrophotography", "granted", "G", "1939-04-04", "Carlson",
                 "Chester F. Carlson", "https://patents.google.com/patent/US2297691A/en",
                 ["patent_number", "title", "legal_status", "ipc_section", "filing_date", "applicant"]),
    PatentRecord("pat11", "US821393A", "Flying-machine",
                 "flying machine", "granted", "B", "1903-03-23", "Wright",
                 "Orville Wright and Wilbur Wright", "https://patents.google.com/patent/US821393A/en",
                 ["patent_number", "title", "legal_status", "ipc_section", "filing_date", "applicant"]),
    PatentRecord("pat12", "US4733665A", "Expandable intraluminal graft",
                 "intraluminal graft", "granted", "A", "1985-11-07", "ExpandableGrafts",
                 "Expandable Grafts Partnership", "https://patents.google.com/patent/US4733665A/en",
                 ["patent_number", "title", "legal_status", "ipc_section", "filing_date", "applicant"]),
    PatentRecord("pat13", "US2130948A", "Synthetic fiber",
                 "synthetic fiber", "granted", "D", "1937-04-09", "DuPont",
                 "E.I. Du Pont de Nemours and Co", "https://patents.google.com/patent/US2130948A/en",
                 ["patent_number", "title", "legal_status", "ipc_section", "filing_date", "applicant"]),
    PatentRecord("pat14", "US1781541A", "Refrigeration",
                 "refrigeration", "granted", "F", "1927-12-16", "Electrolux",
                 "Electrolux Servel Corporation", "https://patents.google.com/patent/US1781541A/en",
                 ["patent_number", "title", "legal_status", "ipc_section", "filing_date", "applicant"]),
    PatentRecord("pat15", "US4476343A", "Olefin oligomerization with tantalum halide oxide metal oxide catalysts",
                 "olefin oligomerization", "granted", "C", "1983-09-23", "Shell",
                 "Shell Oil Co", "https://patents.google.com/patent/US4476343A/en",
                 ["patent_number", "title", "legal_status", "ipc_section", "filing_date", "applicant"]),
]


def source_text(rec: PatentRecord) -> str:
    """Bibliographic sentence assembled from the verified fields (source_url).
    The IPC clause is omitted when the section is unverifiable, so nothing unconfirmed
    is asserted in the text the extractor reads."""
    parts = [
        f"United States Patent {rec.patent_number}, titled \"{rec.title}\", "
        f"was filed on {rec.filing_date} and assigned to {rec.applicant_full}.",
        f"It is a {rec.legal_status} patent.",
    ]
    if rec.ipc_section:
        parts.append(f"The invention is classified in IPC section {rec.ipc_section}.")
    return " ".join(parts)


def gold_triples(rec: PatentRecord) -> Dict[str, str]:
    """Gold (attribute -> value) for one patent. Unverifiable fields are omitted, so a
    missing key means 'no gold; the system should abstain', not 'wrong'."""
    g = {
        "patent_number": rec.patent_number,
        "filing_date": rec.filing_date,
        "applicant": rec.applicant,
        "title_keyword": rec.title_keyword,
        "legal_status": rec.legal_status,
    }
    if rec.ipc_section:
        g["ipc_section"] = rec.ipc_section
    return g


# Pas7a lifecycle update case (G_ORGAN_REAL): a real granted patent was 'pending' before
# grant. pat05 (US7479949B2, Apple) is filed 2008 and granted 2009 -> pending then granted
# is its true legal history (not fabricated). Used to exercise the cross-episode challenger
# + Pas7a promote/retrograde on legal_status.
LIFECYCLE_UPDATE = {"slug": "pat05", "attribute": "legal_status",
                    "from_value": "pending", "to_value": "granted",
                    "source_url": "https://patents.google.com/patent/US7479949B2/en"}

# Owner patent: no public record at fetch time (HTTP 404). Recorded for transparency.
OWNER_PATENT_ABSENT = {"patent_number": "EP25216372.0", "reason": "no_public_record_http_404",
                       "note": "EP application filed late 2025, unpublished until ~18 months after priority"}
