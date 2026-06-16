# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Neural binder grounding path. Trains a ContentAddressedRoleBinder on the TARGET
# model's hidden states (the D_Cortex substrate) and benchmarks it head-to-head
# against a deterministic lookup baseline on the SAME 2-entity binding items. This
# measures whether the neural binder adds value over conventional lookup. Honest
# note: deterministic lookup over a known fact table is exact, so on committed facts
# it is the ceiling; the binder's role is grounding from representation, not beating
# an exact table. The gap is reported precisely either way.

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
for extra in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import torch

STRUCT_CORPUS = REPO_ROOT / "data" / "role_struct" / "role_struct_corpus.jsonl"


@dataclass
class BinderBenchmark:
    available: bool
    reason: str = ""
    n_items: int = 0
    binder_exact: float = 0.0
    binder_wrong: float = 0.0
    baseline_exact: float = 0.0          # PROPER (entity, attribute)-keyed lookup = the fair baseline
    baseline_wrong: float = 0.0
    baseline_entity_only_exact: float = 0.0   # naive entity-only lookup (cannot disambiguate relations)
    gap_pp: float = 0.0                  # binder_exact - baseline_exact (positive = binder wins)
    binder_beats_baseline: bool = False  # judged against the PROPER baseline


def _deterministic_lookup_eval(records: List[Dict], eval_idx: List[int]) -> Tuple[float, float, float]:
    """Two deterministic lookup baselines on the same binding items.

    PROPER baseline keys the fact table by (entity, attribute) -- the fair conventional
    lookup, exact on a committed table. NAIVE baseline keys by entity only and so
    conflates relations (the same country has a different value per relation); it is
    reported only to show why entity-only lookup is not the honest comparison."""
    keyed: Dict[Tuple[str, str], str] = {}
    entity_only: Dict[str, str] = {}
    for r in records:
        if r["ambiguous"]:
            continue
        for e, a, v in r["expected"]:
            keyed[((e or "").lower(), (a or "").lower())] = v
            entity_only[(e or "").lower()] = v
    n = keyed_ok = entity_ok = 0
    for gi in eval_idx:
        r = records[gi]
        if r["ambiguous"]:
            continue
        n += 1
        keyed_match = entity_match = True
        for e, a, v in r["expected"]:
            if keyed.get(((e or "").lower(), (a or "").lower())) != v:
                keyed_match = False
            if entity_only.get((e or "").lower()) != v:
                entity_match = False
        keyed_ok += int(keyed_match)
        entity_ok += int(entity_match)
    if n == 0:
        return 0.0, 0.0, 0.0
    return keyed_ok / n, 1 - keyed_ok / n, entity_ok / n


def run_binder_vs_baseline(seed: int = 2024, device: str = "cuda") -> BinderBenchmark:
    """Train one binder on the substrate hidden states and compare to lookup."""
    import json
    if not STRUCT_CORPUS.exists():
        return BinderBenchmark(False, reason=f"binding corpus missing: {STRUCT_CORPUS}")
    try:
        from dcortex_professional.runtime import SubstrateLM
        from train_role_evolution import extract_features, train_one_seed, label_of, evaluate
    except Exception as exc:  # noqa: BLE001
        return BinderBenchmark(False, reason=f"binder imports failed: {type(exc).__name__}: {exc}")

    lm = SubstrateLM(device=device)
    if not lm.available:
        # hidden states inaccessible -> binder path blocked, reported with reason
        return BinderBenchmark(False, reason=f"hidden states inaccessible: {lm.reason}")

    dev = lm.device
    backend = lm.hidden_states("warmup")
    records = [json.loads(l) for l in STRUCT_CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    reps, ok = extract_features(records, backend)
    used = [records[i] for i in ok]
    labels = torch.tensor([label_of(r) for r in used], dtype=torch.long)
    known = torch.tensor([not r["ambiguous"] for r in used], dtype=torch.bool)
    splits = {s: [i for i, r in enumerate(used) if r["split"] == s]
              for s in ("train", "validation", "evaluation")}
    reps = reps.to(dev)
    head = train_one_seed(reps, labels, known, torch.tensor(splits["train"]),
                          torch.tensor(splits["validation"]), reps.shape[-1], dev, seed)
    ev = evaluate(head, reps, labels, used, splits["evaluation"], dev)
    binder_exact = ev["aggregate"]["exact"]
    binder_wrong = ev["aggregate"]["wrong"]
    base_exact, base_wrong, base_entity_only = _deterministic_lookup_eval(used, splits["evaluation"])
    gap = round((binder_exact - base_exact) * 100, 2)
    return BinderBenchmark(available=True, n_items=len(splits["evaluation"]),
                           binder_exact=binder_exact, binder_wrong=binder_wrong,
                           baseline_exact=base_exact, baseline_wrong=base_wrong,
                           baseline_entity_only_exact=base_entity_only, gap_pp=gap,
                           binder_beats_baseline=gap >= 2.0)
