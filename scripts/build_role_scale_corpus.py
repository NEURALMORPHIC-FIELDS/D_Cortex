# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex role-binding SCALE corpus builder. Builds a large, multi-relation,
# multi-construction, non-trivial role-binding corpus for legal-grade
# certification of the unchanged ContentAddressedRoleBinder. Holds out entire
# construction variants (including relative-clause variants), an entire relation
# family, and entire entity pools. Sources are pinned (samayo/country-json at a
# fixed commit) and payload-validated; the build is deterministic
# (build-SHA == file-SHA). Data construction only; substrate/model untouched.

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
SEED = 17
COMMIT = "41d4084bc1ccf9614dab45255a41ba3a5473be74"
BASE = f"https://raw.githubusercontent.com/samayo/country-json/{COMMIT}/src/"
SOURCE_DIR = REPO_ROOT / "data" / "role_scale" / "source"
DEFAULT_OUT = REPO_ROOT / "data" / "role_scale" / "role_scale_corpus.jsonl"

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

    prefix opens; mid binds VB->B; tail binds A->VA. {n}=relation noun."""
    return lambda a, b, va, vb, n: (
        f"{prefix}: {vb}, {mid.format(a=a, b=b, va=va, vb=vb, n=n)}; "
        f"{tail.format(a=a, b=b, va=va, vb=vb, n=n)}.")


# ENTIRE predicate types are held out (not one-word variants): training sees the
# copular and possessive predicates; the evaluation predicates (concessive,
# relative) are NEVER seen in training, so the gate tests generalization to
# genuinely unseen cue structures. The validation predicate (passive) is also
# held out, so early stopping is on generalization, not on a seen predicate.
CONSTRUCTIONS: Dict[str, Dict[str, List[Tuple[str, Template]]]] = {
    "copular": {  # SEEN predicate (train only)
        "train": [
            ("copular_v1", _t("Ledger", "near {a}, is the {n} of {b}", "the {n} of {a} is {va}")),
            ("copular_v2", _t("Index", "beside {a}, is the {n} of {b}", "the {n} of {a} is {va}")),
            ("copular_v3", _t("Roster", "next to {a}, is the {n} of {b}", "the {n} of {a} is {va}")),
        ],
    },
    "possessive": {  # SEEN predicate (train only)
        "train": [
            ("possessive_v1", _t("Sheet", "filed by {a}, is {b}'s {n}", "{a}'s {n} is {va}")),
            ("possessive_v2", _t("Tab", "shown with {a}, is {b}'s {n}", "{a}'s {n} is {va}")),
            ("possessive_v3", _t("Card", "set by {a}, is {b}'s {n}", "{a}'s {n} is {va}")),
        ],
    },
    "passive": {  # HELD-OUT predicate (validation, for generalization early-stop)
        "validation": [
            ("passive_v1", _t("Memo", "logged with {a}, is used by {b} as its {n}", "{a} is paired with {va}")),
            ("passive_v2", _t("Brief", "stored with {a}, is used by {b} as its {n}", "{a} is matched with {va}")),
        ],
    },
    "concessive": {  # HELD-OUT predicate (eval only)
        "evaluation": [
            ("concessive_v1", _t("Notice", "although filed under {a}, answers to {b} as its {n}", "{a} answers to {va}")),
            ("concessive_v2", _t("Dispatch", "although placed under {a}, answers to {b} as its {n}", "{a} answers to {va}")),
        ],
    },
    "relative": {  # HELD-OUT predicate (eval only) - the prior weak spot
        "evaluation": [
            ("relative_v1", _t("Atlas", "beside {a}, is the value that {b} holds as its {n}", "{a} holds {va}")),
            ("relative_v2", _t("Gazette", "past {a}, is the value that {b} owns as its {n}", "{a} owns {va}")),
            ("relative_v3", _t("Digest", "above {a}, is the value that {b} retains as its {n}", "{a} retains {va}")),
        ],
    },
}
AMBIGUOUS = {
    "train": ("ambig_train", lambda a, b, va, vb, n:
              f"Ledger fragment: the four entries {a}, {b}, {va}, and {vb} appear together, "
              f"but which {n} belongs to which is not stated."),
    "validation": ("ambig_val", lambda a, b, va, vb, n:
                   f"Roster fragment: {a}, {b}, {va}, and {vb} are co-listed; the {n} pairing "
                   f"was omitted from this entry."),
    "evaluation": ("ambig_eval", lambda a, b, va, vb, n:
                   f"Register fragment: among {a}, {b}, {va}, and {vb}, the {n} binding is left "
                   f"unspecified here."),
}

PER_VARIANT_PER_RELATION = 60   # records per (variant, relation) -> thousands total
AMBIGUOUS_PER_SPLIT = 70


def fetch_validate_pin(rel_key: str) -> List[Tuple[str, str]]:
    """Fetch one relation source, payload-validate, pin by SHA, return facts."""
    fname, vfield, _noun = RELATIONS[rel_key]
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(BASE + fname, timeout=30).read()
    if len(raw) <= 5120:
        raise RuntimeError(f"{rel_key}: payload {len(raw)} bytes <= 5 KB (rejected)")
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


def make_record(rid: str, split: str, family: str, text: str, a: str, b: str,
                va: str, vb: str, noun: str, ambiguous: bool, relation: str
                ) -> audit.IndependentRoleRecord:
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
    # disjoint entity pools per split, per relation
    pools: Dict[str, Dict[str, List]] = {}
    for rel, facts in relation_facts.items():
        f = facts[:]
        rng.shuffle(f)
        n = len(f)
        pools[rel] = {"train": f[: int(n * 0.55)],
                      "validation": f[int(n * 0.55): int(n * 0.70)],
                      "evaluation": f[int(n * 0.70):]}

    records: List[audit.IndependentRoleRecord] = []
    counters = {"known": 0, "skipped": 0}
    for ctype, by_split in CONSTRUCTIONS.items():
        for split, variants in by_split.items():
            for family, template in variants:
                # which relations appear in this split's records
                rels = TRAIN_RELATIONS if split != "evaluation" else tuple(RELATIONS)
                for relation in rels:
                    if relation == HELDOUT_RELATION and split != "evaluation":
                        continue
                    noun = RELATIONS[relation][2]
                    stream = pair_stream(pools[relation][split], rng)
                    made = 0
                    for a, b, va, vb in stream:
                        if made >= PER_VARIANT_PER_RELATION:
                            break
                        rid = f"{split}-{family}-{relation}-{made:04d}"
                        rec = make_record(rid, split, family, template(a, b, va, vb, noun),
                                          a, b, va, vb, noun, False, relation)
                        if rb4._known_is_nontrivial(rec):
                            if rng.random() < 0.5:
                                rec = dataclasses.replace(rec, values=(rec.values[1], rec.values[0]))
                            records.append(rec)
                            made += 1
                            counters["known"] += 1
                        else:
                            counters["skipped"] += 1
    # ambiguous per split (own relations/pools)
    for split in ("train", "validation", "evaluation"):
        family, template = AMBIGUOUS[split]
        rels = TRAIN_RELATIONS if split != "evaluation" else tuple(RELATIONS)
        made = 0
        for relation in rels:
            noun = RELATIONS[relation][2]
            for a, b, va, vb in pair_stream(pools[relation][split], rng):
                if made >= AMBIGUOUS_PER_SPLIT:
                    break
                rid = f"{split}-{family}-{relation}-{made:04d}"
                rec = make_record(rid, split, family, template(a, b, va, vb, noun),
                                  a, b, va, vb, noun, True, relation)
                if audit.inventory_present(rec) and audit.provenance_is_auditable(rec.provenance):
                    records.append(rec)
                    made += 1
    print(f"[INFO] known={counters['known']} ambiguous={sum(r.ambiguous for r in records)} "
          f"skipped={counters['skipped']}", flush=True)
    return records


def write_corpus(records: Sequence[audit.IndependentRoleRecord], out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=True)
                     for r in records) + "\n"
    encoded = body.encode("utf-8")
    out_path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the role-binding scale corpus")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    print(SEP, flush=True)
    print("[INFO] Role-binding SCALE corpus (multi-relation, multi-construction holdout)", flush=True)
    records = build()
    out_path = Path(args.out)
    sha = write_corpus(records, out_path)
    for split in ("train", "validation", "evaluation"):
        subset = [r for r in records if r.split == split]
        rels = sorted({r.attribute for r in subset if not r.ambiguous})
        types = sorted({r.construction_family.rsplit("_", 1)[0] for r in subset if not r.ambiguous})
        print(f"[INFO] {split}: n={len(subset)} known={sum(not r.ambiguous for r in subset)} "
              f"ambiguous={sum(r.ambiguous for r in subset)} relations={rels} ctypes={types}",
              flush=True)
    print(f"✓ Wrote {len(records)} records -> {out_path}", flush=True)
    print(f"✓ Corpus SHA-256: {sha}", flush=True)
    print(SEP, flush=True)
    print("CORPUS_SHA256 " + sha, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
