# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex role-binding STRUCTURAL corpus builder (vnext3). Builds a multi-relation,
# multi-construction, non-trivial role-binding corpus that separates two distinct
# generalization axes:
#   (a) cue-predicate generalization: held-out predicates (concessive, relative)
#       in a LOCAL structure (same surface structure class as training);
#   (b) STRUCTURAL generalization: held-out constructions that differ on a genuine
#       structural axis (long filler-gap distance, clause embedding depth), which is
#       what actually stresses a position-blind, content-addressed binder.
# Plus a compositional holdout (seen relation x seen predicate, unseen pairing), an
# entire held-out relation family, disjoint entity pools, and a separate (entity-
# disjoint) calibration split used only by the pilot. The non-triviality crossing
# order (entity-A before B, value-VB before VA, true binding A->VA, B->VB) is
# preserved in EVERY construction so lexical/position baselines stay near zero. The
# samayo/country-json sources are pinned by commit + SHA and payload-validated per
# source. The build is deterministic (build-SHA == file-SHA).

import argparse
import dataclasses
import hashlib
import json
import random
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for extra in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import independent_role_corpus_audit as audit  # noqa: E402
import build_rb4_role_corpus as rb4  # noqa: E402

SEP = "=" * 70
SEED = 23
COMMIT = "41d4084bc1ccf9614dab45255a41ba3a5473be74"
BASE = f"https://raw.githubusercontent.com/samayo/country-json/{COMMIT}/src/"
SOURCE_DIR = REPO_ROOT / "data" / "role_struct" / "source"
DEFAULT_OUT = REPO_ROOT / "data" / "role_struct" / "role_struct_corpus.jsonl"
DEFAULT_CALIB = REPO_ROOT / "data" / "role_struct" / "role_struct_calibration.jsonl"

# relation key -> (samayo file, value field, relation noun used in the cue)
RELATIONS = {
    "capital": ("country-by-capital-city.json", "city", "capital"),
    "abbreviation": ("country-by-abbreviation.json", "abbreviation", "abbreviation"),
    "currency_code": ("country-by-currency-code.json", "currency_code", "currency code"),
}
TRAIN_RELATIONS = ("capital", "abbreviation")
HELDOUT_RELATION = "currency_code"   # entire relation family held out for eval

Template = Callable[[str, str, str, str, str], str]


def _t(prefix: str, mid: str, tail: str) -> Template:
    """Crossing-order template: VB, A, B, VA with binding A->VA, B->VB.

    The first-occurrence order is entity-A (in mid) before entity-B, and value-VB
    (the opener) before value-VA (in tail); the true binding is the crossed
    A->VA / B->VB, so ordered/minimum-distance baselines stay near zero.
    mid binds VB->B and mentions A then B; tail binds A->VA. {n}=relation noun."""
    return lambda a, b, va, vb, n: (
        f"{prefix}: {vb}, {mid.format(a=a, b=b, va=va, vb=vb, n=n)}; "
        f"{tail.format(a=a, b=b, va=va, vb=vb, n=n)}.")


# Long, name-free filler used to blow up the entity->value token distance for the
# STRUCTURAL long-gap constructions without adding any entity/value tokens (so the
# inventory and the crossing order are untouched).
_FILL_A = ("after a long stretch of intervening archival notes, survey remarks, and "
           "clerical annotations that name no party and merely pad the surrounding "
           "record at considerable length")
_FILL_B = ("following an equally long passage of unrelated marginalia, transcription "
           "errata, and bookkeeping asides that mention no place at all and serve only "
           "to separate the two halves of the entry")

# construction type -> structural_class -> list of (variant_name, template)
# structural_class in {local, long_gap, embedded}; local constructions are short and
# share one surface-structure class, long_gap/embedded are structurally distinct.
COPULAR = [
    ("copular_v1", _t("Ledger", "near {a}, is the {n} of {b}", "the {n} of {a} is {va}")),
    ("copular_v2", _t("Index", "beside {a}, is the {n} of {b}", "the {n} of {a} is {va}")),
    ("copular_v3", _t("Roster", "next to {a}, is the {n} of {b}", "the {n} of {a} is {va}")),
]
POSSESSIVE = [
    ("possessive_v1", _t("Sheet", "filed by {a}, is {b}'s {n}", "{a}'s {n} is {va}")),
    ("possessive_v2", _t("Tab", "shown with {a}, is {b}'s {n}", "{a}'s {n} is {va}")),
    ("possessive_v3", _t("Card", "set by {a}, is {b}'s {n}", "{a}'s {n} is {va}")),
]
APPOSITIVE = [   # seen predicate; used for the compositional holdout across relations
    ("appositive_v1", _t("Folio", "by {a}, the {n} of {b}", "{a}, with {va} as its {n}")),
    ("appositive_v2", _t("Plate", "by {a}, the noted {n} of {b}", "{a}, with {va} for its {n}")),
]
PASSIVE = [   # held-out predicate (validation, early-stop on generalization)
    ("passive_v1", _t("Memo", "logged with {a}, is used by {b} as its {n}", "{a} is paired with {va}")),
    ("passive_v2", _t("Brief", "stored with {a}, is used by {b} as its {n}", "{a} is matched with {va}")),
]
CONCESSIVE = [   # held-out predicate (eval, LOCAL structure -> cue-predicate axis)
    ("concessive_v1", _t("Notice", "although filed under {a}, answers to {b} as its {n}", "{a} answers to {va}")),
    ("concessive_v2", _t("Dispatch", "although placed under {a}, answers to {b} as its {n}", "{a} answers to {va}")),
]
RELATIVE = [   # held-out predicate (eval, LOCAL structure -> cue-predicate axis)
    ("relative_v1", _t("Atlas", "beside {a}, is the value that {b} holds as its {n}", "{a} holds {va}")),
    ("relative_v2", _t("Gazette", "past {a}, is the value that {b} owns as its {n}", "{a} owns {va}")),
]
LONG_GAP = [   # held-out STRUCTURE: large entity->value filler-gap distance
    ("long_gap_v1", _t("Register", "near {a}, " + _FILL_A + ", is the {n} of {b}",
                       "the {n} of {a}, " + _FILL_B + ", is {va}")),
    ("long_gap_v2", _t("Bulletin", "by {a}, " + _FILL_B + ", stands as the {n} of {b}",
                       "the {n} held by {a}, " + _FILL_A + ", is {va}")),
]
EMBEDDED = [   # held-out STRUCTURE: nested clause embedding (depth >= 2)
    ("embedded_v1", _t("Register",
                       "near {a}, is the {n} that {b} (which the nested clerks' records, that "
                       "auditors in turn later reviewed, jointly confirm) maintains",
                       "the {n} of {a}, which the same embedded sources that earlier scribes "
                       "compiled and that reviewers re-checked together attest, is {va}")),
    ("embedded_v2", _t("Codex",
                       "by {a}, is the {n} that {b} (which the layered minutes, that a second "
                       "panel that convened afterward endorsed, separately record) keeps",
                       "the {n} of {a}, which those same twice-nested notes that one committee "
                       "drafted and that another committee that met later ratified report, is {va}")),
]

AMBIGUOUS = {
    "train": ("ambig_train", lambda a, b, va, vb, n:
              f"Ledger fragment: the four entries {a}, {b}, {va}, and {vb} appear together, "
              f"but which {n} belongs to which is not stated."),
    "validation": ("ambig_val", lambda a, b, va, vb, n:
                   f"Roster fragment: {a}, {b}, {va}, and {vb} are co-listed; the {n} pairing "
                   f"was omitted from this entry."),
    "calibration": ("ambig_cal", lambda a, b, va, vb, n:
                    f"Pilot fragment: {a}, {b}, {va}, and {vb} are grouped; the {n} assignment "
                    f"is not recorded in this draft."),
    "evaluation": ("ambig_eval", lambda a, b, va, vb, n:
                   f"Register fragment: among {a}, {b}, {va}, and {vb}, the {n} binding is left "
                   f"unspecified here."),
}

# A cell: (split, axis, structural_class, variants, relations, pool_split)
#   split        -> the dataset split (train/validation/calibration/evaluation)
#   axis         -> the generalization axis the cell probes (used to group gates)
#   structural_class -> local / long_gap / embedded (structure signature class)
#   pool_split   -> which disjoint entity-pool slice to draw from
Cell = Tuple[str, str, str, List[Tuple[str, Template]], Tuple[str, ...], str]
CELLS: List[Cell] = [
    # --- training (seen relation x seen LOCAL predicate) ---
    ("train", "seen", "local", COPULAR, TRAIN_RELATIONS, "train"),
    ("train", "seen", "local", POSSESSIVE, TRAIN_RELATIONS, "train"),
    ("train", "seen", "local", APPOSITIVE, ("capital",), "train"),   # appositive seen on capital only
    # --- validation (held-out predicate, LOCAL) for generalization early-stop ---
    ("validation", "val", "local", PASSIVE, TRAIN_RELATIONS, "validation"),
    # --- calibration pilot split (entity-disjoint), measures relative + compositional ---
    ("calibration", "cue_predicate", "local", RELATIVE, TRAIN_RELATIONS, "calibration"),
    ("calibration", "compositional", "local", APPOSITIVE, ("abbreviation",), "calibration"),
    # --- evaluation: compositional (seen relation + seen predicate, unseen pairing) ---
    ("evaluation", "compositional", "local", APPOSITIVE, ("abbreviation",), "evaluation"),
    # --- evaluation: cue-predicate generalization (LOCAL structure, unseen predicate) ---
    ("evaluation", "cue_predicate", "local", CONCESSIVE, TRAIN_RELATIONS, "evaluation"),
    ("evaluation", "cue_predicate", "local", RELATIVE, TRAIN_RELATIONS, "evaluation"),
    # --- evaluation: STRUCTURAL generalization (genuinely distinct structure) ---
    ("evaluation", "structural", "long_gap", LONG_GAP, TRAIN_RELATIONS, "evaluation"),
    ("evaluation", "structural", "embedded", EMBEDDED, TRAIN_RELATIONS, "evaluation"),
    # --- evaluation: entire held-out relation family across constructions ---
    ("evaluation", "relation", "local", COPULAR, (HELDOUT_RELATION,), "evaluation"),
    ("evaluation", "relation", "local", RELATIVE, (HELDOUT_RELATION,), "evaluation"),
    ("evaluation", "relation", "long_gap", LONG_GAP, (HELDOUT_RELATION,), "evaluation"),
    ("evaluation", "relation", "embedded", EMBEDDED, (HELDOUT_RELATION,), "evaluation"),
]

PER_VARIANT_PER_RELATION = 55
AMBIGUOUS_PER_SPLIT = 70


def ctype_of(construction_family: str) -> str:
    """copular_v1 -> copular ; long_gap_v1 -> long_gap ; embedded_v2 -> embedded."""
    return construction_family.rsplit("_", 1)[0]


def structural_class_of(construction_family: str) -> str:
    ct = ctype_of(construction_family)
    return ct if ct in ("long_gap", "embedded") else "local"


HELDOUT_RELATION_NOUN = RELATIONS[HELDOUT_RELATION][2]      # "currency code"
COMPOSITIONAL_RELATION_NOUN = RELATIONS["abbreviation"][2]  # "abbreviation"


def axis_of(construction_family: str, attribute: str, split: str) -> str:
    """Generalization axis a held-out cell probes, derived from family+attribute+split.

    `attribute` is the relation noun stored on the record. Kept consistent with the
    CELLS table so the certifier can group eval records by axis without relying on
    provenance string parsing."""
    ct = ctype_of(construction_family)
    if construction_family.startswith("ambig"):
        return "ambiguous"
    if split == "train":
        return "seen"
    if split == "validation":
        return "val"
    if attribute == HELDOUT_RELATION_NOUN:
        return "relation"
    if ct == "appositive" and attribute == COMPOSITIONAL_RELATION_NOUN:
        return "compositional"
    if ct in ("concessive", "relative"):
        return "cue_predicate"
    if ct in ("long_gap", "embedded"):
        return "structural"
    return "seen"


def fetch_validate_pin(rel_key: str) -> List[Tuple[str, str]]:
    """Fetch one relation source, payload-validate, pin by SHA, return clean facts."""
    fname, vfield, _noun = RELATIONS[rel_key]
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(BASE + fname, timeout=30).read()
    if len(raw) <= 5120:
        raise RuntimeError(f"{rel_key}: payload {len(raw)} bytes <= 5 KB (rejected)")
    head = raw[:64].lstrip()[:1]
    if head not in (b"[", b"{"):
        raise RuntimeError(f"{rel_key}: payload does not start as JSON (redirect/error body)")
    payload = json.loads(raw)
    if not isinstance(payload, list) or len(payload) < 180:
        raise RuntimeError(f"{rel_key}: not a list of >= 180 entries")
    facts: List[Tuple[str, str]] = []
    seen = set()
    for entry in payload:
        country = str(entry.get("country", "")).strip()
        value = str(entry.get(vfield, "")).strip()
        if not country or not value:
            continue
        if not rb4._clean_phrase(country) or not rb4._clean_phrase(value):
            continue
        if country.casefold() == value.casefold():
            continue
        key = (country.casefold(), value.casefold())
        if key in seen:
            continue
        seen.add(key)
        facts.append((country, value))
    if len(facts) < 150:
        raise RuntimeError(f"{rel_key}: only {len(facts)} clean facts (< 150)")
    sha = hashlib.sha256(raw).hexdigest()
    pinned = SOURCE_DIR / f"{rel_key}_samayo_{COMMIT[:8]}_{sha[:16]}.json"
    pinned.write_bytes(raw)
    print(f"[INFO] {rel_key}: {len(facts)} clean facts; pinned {pinned.name} sha {sha[:16]}",
          flush=True)
    return facts


SOURCE_PROVENANCE = {
    "source": "samayo/country-json", "commit": COMMIT, "license": "MIT",
    "url": f"https://github.com/samayo/country-json/tree/{COMMIT}/src",
}


def make_record(rid: str, split: str, family: str, axis: str, structural_class: str,
                text: str, a: str, b: str, va: str, vb: str, noun: str, ambiguous: bool,
                relation: str) -> audit.IndependentRoleRecord:
    prov = dict(SOURCE_PROVENANCE)
    prov["relation"] = relation
    prov["facts"] = sorted([f"{a}={va}", f"{b}={vb}"])
    expected = () if ambiguous else audit.facts(noun, ((a, va), (b, vb)))
    return audit.IndependentRoleRecord(
        record_id=rid, split=split, construction_family=family, source_text=text,
        attribute=noun, entities=(a, b), values=(va, vb), expected=expected,
        ambiguous=ambiguous, provenance=audit.coerce_provenance(prov))


def pair_stream(pool: Sequence[Tuple[str, str]], rng: random.Random):
    pairs = [(pool[i], pool[j]) for i in range(len(pool)) for j in range(i + 1, len(pool))]
    rng.shuffle(pairs)
    used = set()
    for (a, va), (b, vb) in pairs:
        key = tuple(sorted((a.casefold(), b.casefold())))
        if key in used or not rb4._phrases_well_separated((a, b, va, vb)):
            continue
        used.add(key)
        yield a, b, va, vb


def build() -> List[audit.IndependentRoleRecord]:
    rng = random.Random(SEED)
    relation_facts = {k: fetch_validate_pin(k) for k in RELATIONS}
    # GLOBALLY disjoint entity pools: partition the union of all countries (entities)
    # into four disjoint groups, then assign each relation's facts by their country's
    # group. This guarantees no entity (country) appears in two splits for ANY
    # relation, so there is no entity leakage across train/validation/calibration/eval.
    all_countries = sorted({country for facts in relation_facts.values() for country, _ in facts})
    rng.shuffle(all_countries)
    n = len(all_countries)
    group_of: Dict[str, str] = {}
    for i, country in enumerate(all_countries):
        frac = i / n
        group_of[country] = ("train" if frac < 0.45 else
                             "validation" if frac < 0.57 else
                             "calibration" if frac < 0.70 else "evaluation")
    pools: Dict[str, Dict[str, List]] = {}
    for rel, facts in relation_facts.items():
        pools[rel] = {s: [(c, v) for c, v in facts if group_of[c] == s]
                      for s in ("train", "validation", "calibration", "evaluation")}
    print(f"[INFO] global entity partition: {n} countries -> "
          + ", ".join(f"{s}={sum(1 for g in group_of.values() if g == s)}"
                      for s in ("train", "validation", "calibration", "evaluation")), flush=True)

    records: List[audit.IndependentRoleRecord] = []
    counters = {"known": 0, "skipped": 0}
    cell_counts: Dict[str, int] = {}
    for split, axis, structural_class, variants, relations, pool_split in CELLS:
        for family, template in variants:
            for relation in relations:
                noun = RELATIONS[relation][2]
                stream = pair_stream(pools[relation][pool_split], rng)
                made = 0
                for a, b, va, vb in stream:
                    if made >= PER_VARIANT_PER_RELATION:
                        break
                    rid = f"{split}-{family}-{relation}-{made:04d}"
                    rec = make_record(rid, split, family, axis, structural_class,
                                      template(a, b, va, vb, noun),
                                      a, b, va, vb, noun, False, relation)
                    if rb4._known_is_nontrivial(rec):
                        if rng.random() < 0.5:
                            rec = dataclasses.replace(rec, values=(rec.values[1], rec.values[0]))
                        records.append(rec)
                        made += 1
                        counters["known"] += 1
                    else:
                        counters["skipped"] += 1
                cell_counts[f"{split}:{family}:{relation}"] = made
    # ambiguous per split (drawn from that split's own pools / train relations)
    for split in ("train", "validation", "calibration", "evaluation"):
        family, template = AMBIGUOUS[split]
        rels = TRAIN_RELATIONS if split != "evaluation" else tuple(RELATIONS)
        made = 0
        for relation in rels:
            noun = RELATIONS[relation][2]
            for a, b, va, vb in pair_stream(pools[relation][split], rng):
                if made >= AMBIGUOUS_PER_SPLIT:
                    break
                rid = f"{split}-{family}-{relation}-{made:04d}"
                rec = make_record(rid, split, family, "ambiguous", "local",
                                  template(a, b, va, vb, noun), a, b, va, vb, noun, True, relation)
                if audit.inventory_present(rec) and audit.provenance_is_auditable(rec.provenance):
                    records.append(rec)
                    made += 1
    thin = {k: v for k, v in cell_counts.items() if v < PER_VARIANT_PER_RELATION}
    print(f"[INFO] known={counters['known']} ambiguous={sum(r.ambiguous for r in records)} "
          f"skipped={counters['skipped']}", flush=True)
    if thin:
        print(f"[WARN] thin cells (made < {PER_VARIANT_PER_RELATION}): {thin}", flush=True)
    return records


def write_corpus(records: Sequence[audit.IndependentRoleRecord], out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=True)
                     for r in records) + "\n"
    encoded = body.encode("utf-8")
    out_path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the role-binding STRUCTURAL corpus")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--calib-out", default=str(DEFAULT_CALIB))
    args = parser.parse_args()
    print(SEP, flush=True)
    print("[INFO] Role-binding STRUCTURAL corpus (cue-predicate vs structural holdout)", flush=True)
    records = build()
    # main certification corpus = train/validation/evaluation; calibration is a
    # separate, entity-disjoint pilot file (kept out of the certification set).
    main_records = [r for r in records if r.split != "calibration"]
    calib_records = [r for r in records if r.split == "calibration"]
    out_path, calib_path = Path(args.out), Path(args.calib_out)
    sha = write_corpus(main_records, out_path)
    calib_sha = write_corpus(calib_records, calib_path)
    for split in ("train", "validation", "calibration", "evaluation"):
        subset = [r for r in records if r.split == split]
        rels = sorted({r.attribute for r in subset if not r.ambiguous})
        axes = sorted({axis_of(r.construction_family, r.attribute, r.split)
                       for r in subset if not r.ambiguous})
        sclasses = sorted({structural_class_of(r.construction_family)
                           for r in subset if not r.ambiguous})
        print(f"[INFO] {split}: n={len(subset)} known={sum(not r.ambiguous for r in subset)} "
              f"ambiguous={sum(r.ambiguous for r in subset)} relations={rels} axes={axes} "
              f"structural_classes={sclasses}", flush=True)
    print(f"✓ Wrote {len(main_records)} main records -> {out_path}", flush=True)
    print(f"✓ Wrote {len(calib_records)} calibration records -> {calib_path}", flush=True)
    print(f"✓ Main corpus SHA-256: {sha}", flush=True)
    print(f"✓ Calibration SHA-256: {calib_sha}", flush=True)
    print(SEP, flush=True)
    print("CORPUS_SHA256 " + sha, flush=True)
    print("CALIB_SHA256 " + calib_sha, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
