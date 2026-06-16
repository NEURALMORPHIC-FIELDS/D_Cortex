# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex role-binding STRUCTURAL corpus validator (vnext3). Verifies the corpus
# is non-trivial PER CELL (lexical/position baseline ~0% on every relation x
# construction cell, not just in aggregate), that entity pools are disjoint across
# splits (no entity leakage) and no exact source-text record crosses splits, and
# that the held-out STRUCTURAL constructions (long_gap, embedded) are genuinely
# distinct in structure from training (token filler-gap distance and clause
# embedding depth) -- the evidence behind the G_STRUCTURAL gate. Data-only; the
# substrate/model are never loaded or trained here.

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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

import tiktoken
import independent_role_corpus_audit as audit
from dcortex.semantic_role_conditioned import phrase_token_positions
from build_role_struct_corpus import ctype_of, structural_class_of, axis_of

SEP = "=" * 70
ENC = tiktoken.get_encoding("gpt2")
TOK = lambda t: ENC.encode_ordinary(t)  # noqa: E731

LEXICAL_CEILING = 0.05   # per-cell lexical/position baseline must stay at/under this


def to_record(r: Dict) -> audit.IndependentRoleRecord:
    return audit.IndependentRoleRecord(
        record_id=r["record_id"], split=r["split"],
        construction_family=r["construction_family"], source_text=r["source_text"],
        attribute=r["attribute"], entities=tuple(r["entities"]),
        values=tuple(r["values"]), expected=tuple(tuple(x) for x in r["expected"]),
        ambiguous=r["ambiguous"], provenance=r["provenance"])


def cell_lexical_baseline(records: Sequence[Dict]) -> float:
    """Best of the three position/lexical baselines on a set of known records."""
    objs = [to_record(r) for r in records if not r["ambiguous"]]
    if not objs:
        return 0.0
    best = 0.0
    for name in ("ordered_first_occurrence", "minimum_distance", "lexical_cartesian"):
        fn = audit.BASELINES[name]
        best = max(best, sum(int(fn(o) == o.expected) for o in objs) / len(objs))
    return best


def min_token_gap(text: str, entity: str, value: str) -> int:
    """Minimum token distance between any occurrence of entity and of value."""
    ids = TOK(text)
    ep = phrase_token_positions(ids, entity, TOK)
    vp = phrase_token_positions(ids, value, TOK)
    if not ep or not vp:
        return -1
    return min(abs(e - v) for e in ep for v in vp)


def structure_signature(r: Dict) -> Tuple[int, int]:
    """(max bound-pair filler-gap in tokens, clause-embedding depth)."""
    a, b = r["entities"]
    va, vb = r["values"]
    # recover the true binding from expected facts
    bind = {e: v for e, _attr, v in r["expected"]}
    text = r["source_text"]
    gaps = []
    for ent in (a, b):
        val = bind.get(ent)
        if val is None:
            continue
        g = min_token_gap(text, ent, val)
        if g >= 0:
            gaps.append(g)
    max_gap = max(gaps) if gaps else -1
    low = text.lower()
    depth = sum(low.count(m) for m in ("which", " that ", "(", "whose"))
    return max_gap, depth


def agg(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"median": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {"median": round(statistics.median(values), 2), "min": round(min(values), 2),
            "max": round(max(values), 2), "n": len(values)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate the structural role-binding corpus")
    ap.add_argument("--corpus", default=str(REPO_ROOT / "data" / "role_struct" / "role_struct_corpus.jsonl"))
    ap.add_argument("--calib", default=str(REPO_ROOT / "data" / "role_struct" / "role_struct_calibration.jsonl"))
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "role_struct" / "validation.json"))
    args = ap.parse_args()

    recs = [json.loads(l) for l in Path(args.corpus).read_text(encoding="utf-8").splitlines() if l.strip()]
    calib = [json.loads(l) for l in Path(args.calib).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(SEP, flush=True)
    print(f"[INFO] Validating {len(recs)} main + {len(calib)} calibration records", flush=True)

    # 1) per-cell lexical/position baseline (every relation x construction in eval)
    eval_known = [r for r in recs if r["split"] == "evaluation" and not r["ambiguous"]]
    cells: Dict[str, List[Dict]] = {}
    for r in eval_known:
        cells.setdefault(f"{ctype_of(r['construction_family'])} | {r['attribute']}", []).append(r)
    cell_baselines = {k: cell_lexical_baseline(v) for k, v in cells.items()}
    worst_cell = max(cell_baselines.items(), key=lambda kv: kv[1])
    per_cell_ok = all(b <= LEXICAL_CEILING for b in cell_baselines.values())
    print(f"[{'PASS' if per_cell_ok else 'FAIL'}] per-cell lexical baseline: worst {worst_cell[0]} "
          f"= {worst_cell[1]:.1%} (ceiling {LEXICAL_CEILING:.0%}); {len(cell_baselines)} cells", flush=True)

    # 2) entity disjointness across splits + no fact shared train<->eval
    by_split_entities: Dict[str, set] = {}
    by_split_facts: Dict[str, set] = {}
    all_split = recs + calib
    for r in all_split:
        if r["ambiguous"]:
            continue
        by_split_entities.setdefault(r["split"], set()).update(e.casefold() for e in r["entities"])
        bind = {(e.casefold(), v.casefold()) for e, _a, v in r["expected"]}
        by_split_facts.setdefault(r["split"], set()).update(bind)
    splits = sorted(by_split_entities)
    ent_overlaps = {}
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1:]:
            inter = by_split_entities[s1] & by_split_entities[s2]
            if inter:
                ent_overlaps[f"{s1}^{s2}"] = len(inter)
    entity_disjoint = not ent_overlaps
    fact_overlap = len(by_split_facts.get("train", set()) & by_split_facts.get("evaluation", set()))
    print(f"[{'PASS' if entity_disjoint else 'FAIL'}] entity pools disjoint across splits; "
          f"overlaps={ent_overlaps or 'none'}", flush=True)
    print(f"[{'PASS' if fact_overlap == 0 else 'FAIL'}] train<->eval shared (entity,value) facts = {fact_overlap}",
          flush=True)

    # 3) no exact source_text duplicate across splits
    text_split: Dict[str, str] = {}
    cross_dups = 0
    for r in all_split:
        t = r["source_text"]
        if t in text_split and text_split[t] != r["split"]:
            cross_dups += 1
        text_split[t] = r["split"]
    print(f"[{'PASS' if cross_dups == 0 else 'FAIL'}] exact source-text duplicates across splits = {cross_dups}",
          flush=True)

    # 4) structural distinctness: per-construction gap + depth signatures
    by_ct: Dict[str, Dict[str, List[int]]] = {}
    for r in recs:
        if r["ambiguous"]:
            continue
        ct = ctype_of(r["construction_family"])
        gap, depth = structure_signature(r)
        if gap < 0:
            continue
        d = by_ct.setdefault(ct, {"gap": [], "depth": [], "sclass": structural_class_of(r["construction_family"])})
        d["gap"].append(gap)
        d["depth"].append(depth)
    sig = {ct: {"gap": agg(d["gap"]), "depth": agg(d["depth"]), "sclass": d["sclass"]}
           for ct, d in by_ct.items()}
    local_cts = [ct for ct, d in sig.items() if d["sclass"] == "local"]
    local_gap_max = max((sig[ct]["gap"]["max"] for ct in local_cts), default=0)
    local_depth_max = max((sig[ct]["depth"]["max"] for ct in local_cts), default=0)
    # long_gap is distinct if its MIN gap exceeds the local MAX gap;
    # embedded is distinct if its MIN depth exceeds the local MAX depth.
    long_gap_min = min((sig[ct]["gap"]["min"] for ct, d in sig.items() if d["sclass"] == "long_gap"), default=0)
    embedded_depth_min = min((sig[ct]["depth"]["min"] for ct, d in sig.items() if d["sclass"] == "embedded"), default=0)
    long_gap_distinct = long_gap_min > local_gap_max
    embedded_distinct = embedded_depth_min > local_depth_max
    structural_distinct = long_gap_distinct and embedded_distinct
    print(SEP, flush=True)
    print("[INFO] structure signatures (token filler-gap | clause depth) per construction:", flush=True)
    for ct in sorted(sig):
        s = sig[ct]
        print(f"   {ct:14s} [{s['sclass']:8s}] gap med={s['gap']['median']:.0f} "
              f"[{s['gap']['min']:.0f}-{s['gap']['max']:.0f}]  depth med={s['depth']['median']:.0f} "
              f"[{s['depth']['min']:.0f}-{s['depth']['max']:.0f}]", flush=True)
    print(f"[{'PASS' if long_gap_distinct else 'FAIL'}] long_gap distinct: min gap {long_gap_min:.0f} "
          f"> local max gap {local_gap_max:.0f}", flush=True)
    print(f"[{'PASS' if embedded_distinct else 'FAIL'}] embedded distinct: min depth {embedded_depth_min:.0f} "
          f"> local max depth {local_depth_max:.0f}", flush=True)

    all_ok = (per_cell_ok and entity_disjoint and fact_overlap == 0 and cross_dups == 0
              and structural_distinct)
    out = {
        "corpus": args.corpus, "n_main": len(recs), "n_calibration": len(calib),
        "per_cell_lexical": {"ok": per_cell_ok, "worst_cell": worst_cell[0],
                             "worst_value": round(worst_cell[1], 4), "cells": cell_baselines},
        "entity_disjoint": {"ok": entity_disjoint, "overlaps": ent_overlaps},
        "fact_overlap_train_eval": fact_overlap, "cross_split_text_duplicates": cross_dups,
        "structure_signatures": sig,
        "structural_distinctness": {"ok": structural_distinct,
                                    "long_gap_min": long_gap_min, "local_gap_max": local_gap_max,
                                    "embedded_depth_min": embedded_depth_min,
                                    "local_depth_max": local_depth_max},
        "all_ok": all_ok,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(SEP, flush=True)
    print(f"[INFO] Overall corpus validity: {'ALL CHECKS PASS' if all_ok else 'CHECK FAILURE (reported)'}",
          flush=True)
    print("STRUCT_VALIDATION_JSON " + json.dumps({"all_ok": all_ok, "per_cell_ok": per_cell_ok,
          "entity_disjoint": entity_disjoint, "structural_distinct": structural_distinct,
          "worst_cell_baseline": round(worst_cell[1], 4)}), flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
