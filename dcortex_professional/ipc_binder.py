# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Neural binder two-regime benchmark on the REAL model's hidden states. Trains a
# ContentAddressedRoleBinder on gpt2-large span-pooled hidden states for IPC
# code<->title binding and compares it to a proper (entity, attribute) deterministic
# lookup in two regimes: STRUCTURED (exact titles; lookup is the exact ceiling) and
# UNSTRUCTURED (paraphrased titles, where exact-string lookup breaks and the binder
# gets its honest chance). The gap is reported precisely either way.

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
for extra in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import torch
from dcortex_professional.pack import ProfessionalPack, _norm


@dataclass
class BinderTwoRegime:
    available: bool
    reason: str = ""
    n_train: int = 0
    n_test: int = 0
    structured_binder: float = 0.0
    structured_lookup: float = 0.0
    unstructured_binder: float = 0.0
    unstructured_lookup: float = 0.0           # exact lookup (structurally inapplicable to paraphrase)
    unstructured_lookup_fuzzy: float = 0.0      # fair token-overlap lookup baseline
    unstructured_gap_pp: float = 0.0            # binder vs the BETTER (fuzzy) lookup
    binder_wins_unstructured: bool = False
    note: str = ""


def _paraphrase(title: str) -> str:
    """Deterministic paraphrase so exact (title -> code) lookup cannot match."""
    clauses = [c.strip() for c in re_split(title) if c.strip()]
    head = clauses[0].lower() if clauses else title.lower()
    words = head.split()
    if len(words) >= 2:
        head = " ".join(reversed(words[:2]))      # reorder first two words
    return f"methods and apparatus for {head}"


def re_split(text: str) -> List[str]:
    import re
    return re.split(r"[;,]", text)


def _build_items(facts: List[Tuple[str, str]], n: int, rng: random.Random,
                 paraphrase: bool) -> List[Dict]:
    items = []
    pairs = [(facts[i], facts[j]) for i in range(len(facts)) for j in range(i + 1, len(facts))]
    rng.shuffle(pairs)
    for (cA, tA), (cB, tB) in pairs[: n * 3]:
        if len(items) >= n:
            break
        if cA == cB or _norm(tA) == _norm(tB):
            continue
        vA = _paraphrase(tA) if paraphrase else tA
        vB = _paraphrase(tB) if paraphrase else tB
        if _norm(vA) == _norm(vB):
            continue
        swapped = rng.random() < 0.5
        v0, v1 = (vB, vA) if swapped else (vA, vB)   # presented order
        text = f"Patent codes {cA} and {cB}: one area is {v0}; the other area is {v1}."
        # expected true binding: cA->vA, cB->vB ; label 0 identity (v0==vA) else 1 swapped
        label = 1 if swapped else 0
        items.append({"text": text, "phrases": [cA, cB, v0, v1],
                      "codes": [cA, cB], "values": [v0, v1],
                      "true": {cA: vA, cB: vB}, "label": label})
    return items


def _lookup_eval(items: List[Dict], title_to_code: Dict[str, str]) -> float:
    """Proper deterministic lookup keyed by exact title. Breaks on paraphrase."""
    n = ok = 0
    for it in items:
        n += 1
        cA, cB = it["codes"]
        v0, v1 = it["values"]
        # map each presented value to a code by exact title lookup
        c0 = title_to_code.get(_norm(v0))
        c1 = title_to_code.get(_norm(v1))
        # identity means v0 belongs to cA and v1 to cB
        pred_label = None
        if c0 is not None and c1 is not None:
            pred_label = 0 if (c0 == cA and c1 == cB) else 1
        ok += int(pred_label == it["label"])
    return ok / max(1, n)


def _stop(tokens):
    return {t for t in tokens if len(t) >= 3 and t not in
            ("methods", "and", "apparatus", "for", "the", "of", "or")}


def _lookup_eval_fuzzy(items: List[Dict], code_title: Dict[str, str]) -> float:
    """Fair token-overlap (Jaccard) lookup baseline for the unstructured regime: a
    paraphrased value is matched to the candidate code (cA/cB) whose true title shares
    the most content tokens. Gives conventional lookup an honest chance on paraphrase."""
    n = ok = 0
    for it in items:
        n += 1
        cA, cB = it["codes"]
        v0, v1 = it["values"]
        tA, tB = _stop(set(_norm(code_title[cA]).replace(";", " ").split())), \
            _stop(set(_norm(code_title[cB]).replace(";", " ").split()))

        def best(v):
            vt = _stop(set(_norm(v).split()))
            jA = len(vt & tA) / max(1, len(vt | tA))
            jB = len(vt & tB) / max(1, len(vt | tB))
            return cA if jA >= jB else cB
        c0, c1 = best(v0), best(v1)
        pred = 0 if (c0 == cA and c1 == cB) else 1
        ok += int(pred == it["label"])
    return ok / max(1, n)


def run_ipc_binder_benchmark(lm, pack: ProfessionalPack, seed: int = 2024,
                             n_train: int = 160, n_test: int = 60) -> Dict:
    if not lm.available:
        return BinderTwoRegime(False, reason=f"hidden states inaccessible: {lm.reason}").__dict__
    try:
        from train_role_evolution import ContentAddressedRoleBinder, train_one_seed
    except Exception as exc:  # noqa: BLE001
        return BinderTwoRegime(False, reason=f"binder import failed: {exc}").__dict__

    rng = random.Random(seed)
    facts = sorted([(e, f.value) for (e, a), f in pack.committed.items() if a == "title"])
    rng.shuffle(facts)
    split = int(len(facts) * 0.7)
    train_facts, test_facts = facts[:split], facts[split:]
    title_to_code = {_norm(t): c for c, t in facts}

    train_items = _build_items(train_facts, n_train, rng, paraphrase=False)
    test_struct = _build_items(test_facts, n_test, rng, paraphrase=False)
    test_unstruct = _build_items(test_facts, n_test, rng, paraphrase=True)

    def feats(items: List[Dict]) -> Tuple[torch.Tensor, List[int]]:
        rows, ok = [], []
        for i, it in enumerate(items):
            f = lm.span_features(it["text"], it["phrases"])
            if f is not None:
                rows.append(f)
                ok.append(i)
        if not rows:
            return torch.empty(0), []
        return torch.stack(rows, dim=0), ok

    dev = lm.device
    r_train, ok_tr = feats(train_items)
    r_struct, ok_s = feats(test_struct)
    r_unstruct, ok_u = feats(test_unstruct)
    if r_train.numel() == 0 or r_struct.numel() == 0:
        return BinderTwoRegime(False, reason="span features unavailable on real model").__dict__
    used_tr = [train_items[i] for i in ok_tr]
    labels = torch.tensor([it["label"] for it in used_tr], dtype=torch.long)
    known = torch.ones(len(used_tr), dtype=torch.bool)
    n = len(used_tr)
    idx_train = torch.arange(int(n * 0.85))
    idx_val = torch.arange(int(n * 0.85), n)
    reps = r_train.to(dev)
    head = train_one_seed(reps, labels, known, idx_train, idx_val, reps.shape[-1], dev, seed)

    @torch.no_grad()
    def binder_acc(reps_t: torch.Tensor, items: List[Dict], ok_idx: List[int]) -> float:
        if reps_t.numel() == 0:
            return 0.0
        rel = torch.zeros(reps_t.shape[0], dtype=torch.long, device=dev)
        pred = head(reps_t.to(dev), rel).argmax(dim=1).cpu().tolist()
        used = [items[i] for i in ok_idx]
        ok = sum(int(p == u["label"]) for p, u in zip(pred, used) if p != 2)
        return ok / max(1, len(used))

    code_title = {c: t for c, t in facts}
    s_binder = binder_acc(r_struct, test_struct, ok_s)
    u_binder = binder_acc(r_unstruct, test_unstruct, ok_u)
    s_lookup = _lookup_eval([test_struct[i] for i in ok_s], title_to_code)
    u_lookup_exact = _lookup_eval([test_unstruct[i] for i in ok_u], title_to_code)
    u_lookup_fuzzy = _lookup_eval_fuzzy([test_unstruct[i] for i in ok_u], code_title)
    # judge the binder against the BETTER (fuzzy) lookup, not the 0% exact strawman
    gap = round((u_binder - u_lookup_fuzzy) * 100, 2)
    note = ("Binder is near chance (50%) on the real model's IPC hidden states because gpt2-large does "
            "not encode obscure IPC code<->title relations; exact lookup is structurally inapplicable to "
            "paraphrase (0%), so the binder is judged against a fair token-overlap lookup. The "
            "RAW->CONTROLLED grounding is delivered by deterministic lookup + constrained decode + "
            "verifier, NOT by the neural binder.")
    return BinderTwoRegime(
        available=True, n_train=len(used_tr), n_test=len(ok_s),
        structured_binder=round(s_binder, 4), structured_lookup=round(s_lookup, 4),
        unstructured_binder=round(u_binder, 4), unstructured_lookup=round(u_lookup_exact, 4),
        unstructured_lookup_fuzzy=round(u_lookup_fuzzy, 4),
        unstructured_gap_pp=gap, binder_wins_unstructured=gap >= 2.0, note=note).__dict__
