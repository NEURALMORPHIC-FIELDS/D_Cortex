# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex role-binding evolution corpus builder.
#
# Builds a non-trivial, multi-syntax-family role-binding corpus for the
# systematic-generalization evaluation. Entire syntactic families AND entire
# entity pools are held out per split (train families and entities are disjoint
# from validation, which are disjoint from evaluation), so the eval split tests
# unseen syntax on unseen entities, not record-level interpolation. Each train
# fact is paraphrased across every train family so surface form carries no
# information. Every record is filtered with the RB4 audit's own position
# baselines so lexical and positional heuristics cannot solve it. This is data
# construction only; the frozen substrate and the model are not touched.

import argparse
import dataclasses
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
for extra in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import independent_role_corpus_audit as audit  # noqa: E402
import build_rb4_role_corpus as rb4  # noqa: E402

SEP = "=" * 70
ATTRIBUTE = "capital"
SEED = 7
DEFAULT_OUT = REPO_ROOT / "data" / "role_evolution" / "role_evolution_corpus.jsonl"

Template = Callable[[str, str, str, str], str]

# Each family emits the surface order CapB, A, B, CapA (true binding A->CapA,
# B->CapB), crossed against first-occurrence and minimum-distance pairings.
# Families are partitioned across splits, so a held-out split sees only unseen
# syntactic frames. Wording differs per family to keep masked structural
# signatures disjoint across splits.
# Each split uses a DISTINCT binding-cue construction (not just different
# wording): train = copular "is the capital of", validation = possessive
# "X's capital", evaluation = relative-clause / cleft / concessive. All keep the
# CapB, A, B, CapA surface order so positional baselines stay crossed (filtered).
TRAIN_FAMILIES: List[Tuple[str, Template]] = [
    ("train_copular_a", lambda a, b, ca, cb:
     f"Field note: {cb}, although geographically close to {a}, is the capital of "
     f"{b}; the capital of {a} is {ca}."),
    ("train_copular_b", lambda a, b, ca, cb:
     f"Field record: {cb} is recorded near {a}; here {cb} is the capital of {b}, "
     f"and the capital of {a} is {ca}."),
    ("train_copular_c", lambda a, b, ca, cb:
     f"Field index: noted beside {a}, the city {cb} is the capital of {b}; the "
     f"capital of {a} is {ca}."),
    ("train_copular_d", lambda a, b, ca, cb:
     f"Field ledger: {cb} appears next to {a}; {cb} is the capital of {b}, and "
     f"the capital of {a} is {ca}."),
]
VALIDATION_FAMILIES: List[Tuple[str, Template]] = [
    ("val_possessive_a", lambda a, b, ca, cb:
     f"Survey entry: {cb} sits beside {a} on the page; {cb} is {b}'s capital, "
     f"whereas {a}'s capital is {ca}."),
    ("val_possessive_b", lambda a, b, ca, cb:
     f"Survey brief: listed with {a}, the city {cb} is {b}'s seat of government, "
     f"while {a}'s seat of government is {ca}."),
]
# Distinct cue PREDICATES (relative "holds", passive "used by", concessive
# "answers to") versus train (copular "is the capital of") and validation
# (possessive). Crossing order (CapB, A, B, CapA) is kept because the lexical
# baseline must stay crossed; the content-addressed head is position-blind, so
# the genuine generalization test is the unseen cue predicate, not word order.
EVALUATION_FAMILIES: List[Tuple[str, Template]] = [
    ("eval_relative", lambda a, b, ca, cb:
     f"Bulletin: {cb}, situated beside {a}, is the city that {b} holds as its "
     f"capital; {a} holds {ca}."),
    ("eval_passive", lambda a, b, ca, cb:
     f"Bulletin report: {cb}, recorded with {a}, is used by {b} as its capital; "
     f"{a} is paired with {ca}."),
    ("eval_concessive", lambda a, b, ca, cb:
     f"Bulletin record: although {cb} is filed under {a}, it answers to {b} as "
     f"its capital; {a} answers to {ca}."),
]
AMBIGUOUS_FAMILIES: Dict[str, Template] = {
    "train": lambda a, b, ca, cb:
        f"Field fragment: the four names {a}, {b}, {ca}, and {cb} appear together, "
        f"but which capital belongs to which nation is not stated.",
    "validation": lambda a, b, ca, cb:
        f"Survey fragment: {a}, {b}, {ca}, and {cb} are co-listed; the capital "
        f"relation pairing was omitted from this entry.",
    "evaluation": lambda a, b, ca, cb:
        f"Bulletin fragment: among {a}, {b}, {ca}, and {cb}, the nation to capital "
        f"binding is left unspecified here.",
}

SPLIT_PLAN = {
    "train": {"families": TRAIN_FAMILIES, "facts_frac": 0.55, "facts_used": 44,
              "ambiguous": 40},
    "validation": {"families": VALIDATION_FAMILIES, "facts_frac": 0.15,
                   "facts_used": 28, "ambiguous": 22},
    "evaluation": {"families": EVALUATION_FAMILIES, "facts_frac": 0.30,
                   "facts_used": 34, "ambiguous": 26},
}


def split_fact_pools(facts: List[Tuple[str, str]]) -> Dict[str, List[Tuple[str, str]]]:
    """Partition facts into disjoint train/validation/evaluation entity pools."""
    rng = random.Random(SEED)
    shuffled = facts[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * SPLIT_PLAN["train"]["facts_frac"])
    n_val = int(n * SPLIT_PLAN["validation"]["facts_frac"])
    return {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train:n_train + n_val],
        "evaluation": shuffled[n_train + n_val:],
    }


def pair_stream(pool: Sequence[Tuple[str, str]], rng: random.Random):
    """Yield well-separated distinct country pairs from one entity pool."""
    pairs = [(pool[i], pool[j]) for i in range(len(pool)) for j in range(i + 1, len(pool))]
    rng.shuffle(pairs)
    used = set()
    for (a, capa), (b, capb) in pairs:
        key = tuple(sorted((a.casefold(), b.casefold())))
        if key in used:
            continue
        if not rb4._phrases_well_separated((a, b, capa, capb)):
            continue
        used.add(key)
        yield a, b, capa, capb


def build_known(split: str, pool: Sequence[Tuple[str, str]],
                rng: random.Random) -> List[audit.IndependentRoleRecord]:
    """Build paraphrased known records: each fact across every family in the split."""
    families = SPLIT_PLAN[split]["families"]
    facts_used = SPLIT_PLAN[split]["facts_used"]
    stream = pair_stream(pool, rng)
    records: List[audit.IndependentRoleRecord] = []
    facts_done = 0
    idx = 0
    for a, b, capa, capb in stream:
        produced_for_fact = 0
        for family, template in families:
            rid = f"{split}-{family}-{idx:04d}"
            rec = rb4._make_record(rid, split, family, template(a, b, capa, capb),
                                   a, b, capa, capb, ambiguous=False)
            if rb4._known_is_nontrivial(rec):
                # Randomize the values-tuple order (expected stays the true
                # binding) so the identity/swapped label is ~50/50 and the model
                # cannot win by always predicting one assignment. Invariant to
                # the position baselines, so the corpus stays non-trivial.
                if rng.random() < 0.5:
                    rec = dataclasses.replace(rec, values=(rec.values[1], rec.values[0]))
                records.append(rec)
                produced_for_fact += 1
                idx += 1
        if produced_for_fact == len(families):
            facts_done += 1
        if facts_done >= facts_used:
            break
    if facts_done < facts_used:
        raise RuntimeError(f"{split}: only {facts_done} fully-paraphrased facts "
                           f"(< {facts_used})")
    return records


def build_ambiguous(split: str, pool: Sequence[Tuple[str, str]],
                    rng: random.Random) -> List[audit.IndependentRoleRecord]:
    """Build ambiguous records for one split from its own entity pool."""
    template = AMBIGUOUS_FAMILIES[split]
    family = f"ambiguous_{split}"
    stream = pair_stream(pool, rng)
    target = SPLIT_PLAN[split]["ambiguous"]
    records: List[audit.IndependentRoleRecord] = []
    for k, (a, b, capa, capb) in enumerate(stream):
        if len(records) >= target:
            break
        rid = f"{split}-{family}-{k:04d}"
        rec = rb4._make_record(rid, split, family, template(a, b, capa, capb),
                               a, b, capa, capb, ambiguous=True)
        if audit.inventory_present(rec) and audit.provenance_is_auditable(rec.provenance):
            records.append(rec)
    if len(records) < target:
        raise RuntimeError(f"{split}: only {len(records)} ambiguous (< {target})")
    return records


def write_corpus(records: Sequence[audit.IndependentRoleRecord], out_path: Path) -> str:
    """Write the corpus as deterministic LF JSONL and return its SHA-256."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=True)
                     for r in records) + "\n"
    encoded = body.encode("utf-8")
    out_path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the role-binding evolution corpus")
    parser.add_argument("--pinned", default=str(
        rb4.PINNED_DIR / ("country_capitals_samayo_country_json_pinned_"
                          "807c7db9907bffd7f1e469d166224ad330a52b773ab44f905a9435b5e0dbf026.json")))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    print(SEP, flush=True)
    print("[INFO] Role-binding evolution corpus (family + entity holdout)", flush=True)
    pinned = Path(args.pinned)
    if not pinned.exists():
        print(f"[ERROR] Pinned source missing: {pinned}", flush=True)
        return 2
    facts = rb4.load_clean_facts(pinned)
    pools = split_fact_pools(facts)
    print(f"[INFO] Clean facts {len(facts)}; pools train={len(pools['train'])} "
          f"validation={len(pools['validation'])} evaluation={len(pools['evaluation'])} "
          f"(disjoint entities)", flush=True)

    rng = random.Random(SEED)
    records: List[audit.IndependentRoleRecord] = []
    for split in ("train", "validation", "evaluation"):
        known = build_known(split, pools[split], rng)
        ambiguous = build_ambiguous(split, pools[split], rng)
        records.extend(known)
        records.extend(ambiguous)
        fams = sorted({r.construction_family for r in known})
        print(f"[INFO] {split}: known={len(known)} ambiguous={len(ambiguous)} "
              f"families={fams}", flush=True)

    out_path = Path(args.out)
    sha = write_corpus(records, out_path)
    print(f"✓ Wrote {len(records)} records -> {out_path}", flush=True)
    print(f"✓ Corpus SHA-256: {sha}", flush=True)
    print(SEP, flush=True)
    print("CORPUS_SHA256 " + sha, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
