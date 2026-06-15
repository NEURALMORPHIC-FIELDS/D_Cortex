# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b-RB4 independent role-binding corpus constructor.
#
# Completes the RB4 corpus acquisition: it reads the validated, pinned, license
# clean country/capital facts (an external source independent of the RB0-RB3
# training families) and constructs two-entity two-value role-binding records in
# the exact schema the frozen RB4 audit (scripts/independent_role_corpus_audit.py)
# consumes. The RB4 gate logic, the role-binding model, and the Pas 7a seals are
# NOT touched. Surface order is crossed on purpose and every known record is
# filtered with the audit's own position baselines so the corpus is not solvable
# by lexical or positional heuristics. This is data construction only; it does
# not train or run the model.

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import independent_role_corpus_audit as audit  # noqa: E402

SEP = "=" * 70
ATTRIBUTE = "capital"
SEED = 42
PINNED_DIR = REPO_ROOT / "data" / "rb4" / "source"
DEFAULT_OUT = REPO_ROOT / "data" / "rb4" / "independent_role_corpus.jsonl"

# Provenance points only at the external upstream source, never the local rb4
# path (the audit rejects provenance containing reserved RB family tokens).
SOURCE_PROVENANCE = {
    "source": "samayo/country-json",
    "commit": "41d4084bc1ccf9614dab45255a41ba3a5473be74",
    "file": "src/country-by-capital-city.json",
    "url": (
        "https://raw.githubusercontent.com/samayo/country-json/"
        "41d4084bc1ccf9614dab45255a41ba3a5473be74/src/country-by-capital-city.json"
    ),
    "license": "MIT",
    "pinned_sha256": "807c7db9907bffd7f1e469d166224ad330a52b773ab44f905a9435b5e0dbf026",
}

KnownTemplate = Callable[[str, str, str, str], str]
AmbiguousTemplate = Callable[[str, str, str, str], str]

# Known templates per split. Surface order is CapB, A, B, CapA with the true
# binding A->CapA, B->CapB, so first-occurrence and minimum-distance pairings are
# crossed (verified per record). Wording differs per split so masked structural
# signatures stay disjoint across splits.
# Every template emits the surface order CapB, A, B, CapA with the true binding
# A->CapA, B->CapB, so first-occurrence pairing is crossed by construction; every
# candidate is still filtered against both position baselines before acceptance.
KNOWN_TEMPLATES: Dict[str, List[Tuple[str, KnownTemplate]]] = {
    "train": [
        ("gazetteer_proximity", lambda a, b, ca, cb:
         f"Gazetteer note: {cb}, although geographically close to {a}, is the "
         f"capital of {b}; the capital of {a} is {ca}."),
        ("gazetteer_recorded", lambda a, b, ca, cb:
         f"Gazetteer record: {cb} is recorded near {a}; yet {cb} is the capital "
         f"of {b}, and {a} pairs with {ca}."),
        ("gazetteer_index", lambda a, b, ca, cb:
         f"Gazetteer index: {cb} is wrongly linked to {a}; in fact {b} is "
         f"governed from {cb}, while {a} is governed from {ca}."),
    ],
    "validation": [
        ("atlas_concession", lambda a, b, ca, cb:
         f"Atlas entry: {cb}, listed beside {a}, serves {b} as capital; {ca} "
         f"serves {a}."),
        ("atlas_directive", lambda a, b, ca, cb:
         f"Atlas brief: do not bind {a} to {cb}; rather {cb} pairs with {b}, "
         f"while {ca} pairs with {a}."),
        ("atlas_seat", lambda a, b, ca, cb:
         f"Atlas memo: {cb} appears next to {a}; it is administered for {b}, "
         f"whereas {a} is administered from {ca}."),
    ],
    "evaluation": [
        ("dossier_former", lambda a, b, ca, cb:
         f"Dossier: {cb} is cited with {a}, but the former is the capital of {b}; "
         f"{a} keeps {ca} as its capital."),
        ("dossier_consequence", lambda a, b, ca, cb:
         f"Dossier report: {cb} appears before {a}; {cb} answers to {b}, and "
         f"therefore {a} answers to {ca}."),
        ("dossier_elimination", lambda a, b, ca, cb:
         f"Dossier record: {cb} is grouped with {a}; the capital of {b} is {cb}, "
         f"so {a} holds {ca}."),
    ],
}

AMBIGUOUS_TEMPLATES: Dict[str, Tuple[str, AmbiguousTemplate]] = {
    "train": ("gazetteer_redacted", lambda a, b, ca, cb:
              f"Gazetteer fragment: the four names {a}, {b}, {ca}, and {cb} appear "
              f"together, but which capital belongs to which nation is not stated."),
    "validation": ("atlas_omitted", lambda a, b, ca, cb:
                   f"Atlas fragment: {a}, {b}, {ca}, and {cb} are co-listed; the "
                   f"capital relation pairing was omitted from this entry."),
    "evaluation": ("dossier_unspecified", lambda a, b, ca, cb:
                   f"Dossier fragment: among {a}, {b}, {ca}, and {cb}, the nation "
                   f"to capital binding is left unspecified here."),
}

KNOWN_PER_SPLIT = 51       # divisible by 3 templates
AMBIGUOUS_PER_SPLIT = 16


def load_clean_facts(pinned_path: Path) -> List[Tuple[str, str]]:
    """Read the pinned country/capital source and keep tokenization-clean facts."""
    payload = json.loads(pinned_path.read_text(encoding="utf-8"))
    facts: List[Tuple[str, str]] = []
    seen = set()
    for entry in payload:
        country = str(entry.get("country", "")).strip()
        capital = str(entry.get("capital", "")).strip()
        if not country or not capital:
            continue
        if not _clean_phrase(country) or not _clean_phrase(capital):
            continue
        if country.casefold() == capital.casefold():
            continue
        key = (country.casefold(), capital.casefold())
        if key in seen:
            continue
        seen.add(key)
        facts.append((country, capital))
    return facts


def _clean_phrase(text: str) -> bool:
    """Accept only letter/space phrases that tokenize to non-empty tokens."""
    if any(ch for ch in text if not (ch.isalpha() or ch == " ")):
        return False
    return bool(audit.tokens(text))


def _phrases_well_separated(phrases: Sequence[str]) -> bool:
    """No phrase shares a token with another, and none is a subsequence."""
    token_sets = [set(audit.tokens(p)) for p in phrases]
    for i in range(len(phrases)):
        if not token_sets[i]:
            return False
        for j in range(len(phrases)):
            if i != j and token_sets[i] & token_sets[j]:
                return False
    return True


def _make_record(record_id: str, split: str, family: str, source_text: str,
                 a: str, b: str, ca: str, cb: str,
                 ambiguous: bool) -> audit.IndependentRoleRecord:
    """Build one IndependentRoleRecord with provenance citing the external fact."""
    provenance = dict(SOURCE_PROVENANCE)
    provenance["facts"] = sorted([f"{a}={ca}", f"{b}={cb}"])
    expected = () if ambiguous else audit.facts(ATTRIBUTE, ((a, ca), (b, cb)))
    return audit.IndependentRoleRecord(
        record_id=record_id,
        split=split,
        construction_family=family,
        source_text=source_text,
        attribute=ATTRIBUTE,
        entities=(a, b),
        values=(ca, cb),
        expected=expected,
        ambiguous=ambiguous,
        provenance=audit.coerce_provenance(provenance),
    )


def _known_is_nontrivial(record: audit.IndependentRoleRecord) -> bool:
    """Known record must defeat both position baselines and be inventory-complete."""
    if not audit.inventory_present(record):
        return False
    if not audit.expected_is_one_to_one(record):
        return False
    if audit.ordered_first_occurrence(record) == record.expected:
        return False
    if audit.minimum_distance(record) == record.expected:
        return False
    return audit.provenance_is_auditable(record.provenance)


def build_corpus(facts: List[Tuple[str, str]]) -> List[audit.IndependentRoleRecord]:
    """Construct the full split-separated role-binding corpus."""
    rng = random.Random(SEED)
    pairs: List[Tuple[Tuple[str, str], Tuple[str, str]]] = []
    for i in range(len(facts)):
        for j in range(i + 1, len(facts)):
            pairs.append((facts[i], facts[j]))
    rng.shuffle(pairs)

    records: List[audit.IndependentRoleRecord] = []
    used_pairs = set()
    pair_iter = iter(pairs)
    counters = {"known": 0, "ambiguous": 0, "skipped": 0}

    def next_usable_pair() -> Tuple[str, str, str, str]:
        for (ca_fact, cb_fact) in pair_iter:
            (a, capa), (b, capb) = ca_fact, cb_fact
            key = tuple(sorted((a.casefold(), b.casefold())))
            if key in used_pairs:
                continue
            if not _phrases_well_separated((a, b, capa, capb)):
                continue
            used_pairs.add(key)
            return a, b, capa, capb
        raise RuntimeError("ran out of usable country pairs for RB4 construction")

    for split in ("train", "validation", "evaluation"):
        templates = KNOWN_TEMPLATES[split]
        produced = 0
        attempts = 0
        target = KNOWN_PER_SPLIT
        # Rotate templates by ATTEMPT (not by produced count) so a template that
        # cannot cross for a given pair never blocks the loop.
        while produced < target and attempts < target * 50:
            family, template = templates[attempts % len(templates)]
            attempts += 1
            a, b, capa, capb = next_usable_pair()
            rid = f"rec-{split}-known-{produced:04d}"
            rec = _make_record(rid, split, family, template(a, b, capa, capb),
                               a, b, capa, capb, ambiguous=False)
            if _known_is_nontrivial(rec):
                records.append(rec)
                produced += 1
                counters["known"] += 1
            else:
                counters["skipped"] += 1
        if produced < target:
            raise RuntimeError(f"could not build {target} known records for {split} "
                               f"(only {produced})")

        amb_family, amb_template = AMBIGUOUS_TEMPLATES[split]
        for k in range(AMBIGUOUS_PER_SPLIT):
            a, b, capa, capb = next_usable_pair()
            rid = f"rec-{split}-amb-{k:04d}"
            rec = _make_record(rid, split, amb_family, amb_template(a, b, capa, capb),
                               a, b, capa, capb, ambiguous=True)
            if not (audit.inventory_present(rec) and audit.provenance_is_auditable(rec.provenance)):
                counters["skipped"] += 1
                continue
            records.append(rec)
            counters["ambiguous"] += 1
    print(f"[INFO] Built known={counters['known']} ambiguous={counters['ambiguous']} "
          f"(skipped {counters['skipped']} non-conforming candidates)", flush=True)
    return records


def write_corpus(records: Sequence[audit.IndependentRoleRecord], out_path: Path) -> str:
    """Write the corpus as JSONL and return its SHA-256."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
             for record in records]
    body = "\n".join(lines) + "\n"
    # write_bytes keeps LF only (no platform newline translation) so the pinned
    # corpus SHA is deterministic and matches the audit's file hash.
    encoded = body.encode("utf-8")
    out_path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the RB4 independent role-binding corpus")
    parser.add_argument("--pinned", default=str(
        PINNED_DIR / ("country_capitals_samayo_country_json_pinned_"
                      "807c7db9907bffd7f1e469d166224ad330a52b773ab44f905a9435b5e0dbf026.json")))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    pinned_path = Path(args.pinned)
    print(SEP, flush=True)
    print("[INFO] RB4 independent role-binding corpus construction", flush=True)
    if not pinned_path.exists():
        print(f"[ERROR] Pinned source not found: {pinned_path}. Run "
              "scripts/acquire_rb4_independent_corpus.py first.", flush=True)
        return 2
    facts = load_clean_facts(pinned_path)
    print(f"[INFO] Clean independent facts: {len(facts)} (from pinned source "
          f"{pinned_path.name})", flush=True)
    records = build_corpus(facts)
    out_path = Path(args.out)
    sha = write_corpus(records, out_path)
    by_split = {s: sum(r.split == s for r in records)
                for s in ("train", "validation", "evaluation")}
    known = sum(not r.ambiguous for r in records)
    ambiguous = sum(r.ambiguous for r in records)
    print(f"✓ Wrote {len(records)} records -> {out_path}", flush=True)
    print(f"✓ Per split: {by_split} | known={known} ambiguous={ambiguous}", flush=True)
    print(f"✓ Corpus SHA-256: {sha}", flush=True)
    print(SEP, flush=True)
    print("CORPUS_SHA256 " + sha, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
