# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Region B: the binder-favorable regime. Built from REAL, pinned country -> capital
# facts (samayo, Qwen-known), probe-filtered to keep only facts the capable model
# answers correctly RAW (proof its hidden states encode them). Binding items present
# each capital as a model-generated CLUE that names neither the city nor its country,
# so BOTH exact (entity,attribute) lookup AND fuzzy token-overlap lookup fail, while a
# fresh ContentAddressedRoleBinder over the model's layer -1 hidden states gets a fair
# chance. A pre-declared precondition (fuzzy lookup < 60%) guards the regime validity.

import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
for extra in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import torch

CAPITAL_SRC = REPO_ROOT / "data" / "role_struct" / "source" / "capital_samayo_41d4084b_448e7c9be3b58ee5.json"
FUZZY_PRECONDITION = 0.60        # fuzzy lookup must be BELOW this for the regime to be valid
BINDER_MARGIN = 0.02


@dataclass
class RegionBResult:
    available: bool
    reason: str = ""
    probed: int = 0
    kept: int = 0
    regime_valid: bool = False
    n_test: int = 0
    binder_exact: float = 0.0
    lookup_exact: float = 0.0
    lookup_fuzzy: float = 0.0
    binder_beats_both: bool = False
    margin_binder_vs_fuzzy_pp: float = 0.0
    note: str = ""


def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def _tokens(s: str):
    return {t for t in re.findall(r"[a-z]+", _norm(s)) if len(t) >= 3}


def load_capital_facts() -> List[Tuple[str, str]]:
    data = json.loads(CAPITAL_SRC.read_text(encoding="utf-8"))
    out = []
    for e in data:
        c = str(e.get("country", "")).strip()
        cap = str(e.get("city", "")).strip()
        if c and cap and c.lower() != cap.lower():
            out.append((c, cap))
    return out


def probe_keep(lm, facts: List[Tuple[str, str]], limit: int) -> List[Tuple[str, str]]:
    """Keep only facts the model answers correctly RAW (its representations encode them)."""
    kept = []
    for country, capital in facts[:limit]:
        ans = lm.generate_unconstrained(f"What is the capital city of {country}? Answer with only the city name.", 12)
        if _norm(capital) in _norm(ans):
            kept.append((country, capital))
    return kept


def make_clue(lm, country: str, capital: str) -> Optional[str]:
    """Model-generated clue for the capital that names neither city nor country."""
    prompt = (f"Write one short factual clue (max 16 words) about the city {capital} that does NOT "
              f"contain the word '{capital}' or the word '{country}'. Output only the clue.")
    clue = lm.generate_unconstrained(prompt, 40)
    clue = clue.replace("\n", " ").strip().strip('"')
    low = _norm(clue)
    if not clue or _norm(capital) in low or _norm(country) in low or len(clue) < 12:
        return None
    return clue


def _lookup(items: List[Dict], cap_to_country: Dict[str, str], fuzzy: bool,
            country_cap: Dict[str, str]) -> float:
    n = ok = 0
    for it in items:
        n += 1
        cA, cB = it["countries"]
        v0, v1 = it["values"]      # clues

        def resolve(v):
            if not fuzzy:
                return cap_to_country.get(_norm(v))     # exact: a clue never equals a capital -> None
            vt = _tokens(v)
            jA = len(vt & _tokens(country_cap[cA])) / max(1, len(vt | _tokens(country_cap[cA])))
            jB = len(vt & _tokens(country_cap[cB])) / max(1, len(vt | _tokens(country_cap[cB])))
            return cA if jA >= jB else cB
        c0, c1 = resolve(v0), resolve(v1)
        pred = None
        if c0 is not None and c1 is not None:
            pred = 0 if (c0 == cA and c1 == cB) else 1
        ok += int(pred == it["label"])
    return ok / max(1, n)


def run_region_b(lm, seed: int = 7, probe_limit: int = 120, n_train: int = 120,
                 n_test: int = 50) -> Dict:
    if not lm.available:
        return RegionBResult(False, reason=f"hidden states inaccessible: {lm.reason}").__dict__
    if not CAPITAL_SRC.exists():
        return RegionBResult(False, reason=f"capital source missing: {CAPITAL_SRC}").__dict__
    try:
        from train_role_evolution import ContentAddressedRoleBinder, train_one_seed  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return RegionBResult(False, reason=f"binder import failed: {exc}").__dict__

    rng = random.Random(seed)
    facts = load_capital_facts()
    rng.shuffle(facts)
    kept = probe_keep(lm, facts, probe_limit)
    if len(kept) < 30:
        return RegionBResult(False, probed=min(len(facts), probe_limit), kept=len(kept),
                             reason="too few model-known facts to test the binder fairly").__dict__
    country_cap = {c: cap for c, cap in kept}
    cap_to_country = {_norm(cap): c for c, cap in kept}
    clues = {}
    for country, capital in kept:
        clue = make_clue(lm, country, capital)
        if clue:
            clues[country] = clue
    usable = [(c, country_cap[c]) for c in clues]
    if len(usable) < 30:
        return RegionBResult(False, probed=min(len(facts), probe_limit), kept=len(kept),
                             reason="too few usable clues generated").__dict__

    def build_items(pool: List[Tuple[str, str]], n: int) -> List[Dict]:
        items = []
        pairs = [(pool[i], pool[j]) for i in range(len(pool)) for j in range(i + 1, len(pool))]
        rng.shuffle(pairs)
        for (cA, _a), (cB, _b) in pairs[: n * 3]:
            if len(items) >= n or cA == cB:
                continue
            swapped = rng.random() < 0.5
            vA, vB = clues[cA], clues[cB]
            v0, v1 = (vB, vA) if swapped else (vA, vB)
            text = f"Country {cA} and country {cB}. Clue one: {v0}. Clue two: {v1}."
            items.append({"text": text, "phrases": [cA, cB, v0, v1], "countries": [cA, cB],
                          "values": [v0, v1], "label": 1 if swapped else 0})
        return items

    split = int(len(usable) * 0.7)
    train_items = build_items(usable[:split], n_train)
    test_items = build_items(usable[split:], n_test)

    def feats(items):
        rows, ok = [], []
        for i, it in enumerate(items):
            f = lm.span_features(it["text"], it["phrases"])
            if f is not None:
                rows.append(f)
                ok.append(i)
        return (torch.stack(rows, 0) if rows else torch.empty(0)), ok

    r_tr, ok_tr = feats(train_items)
    r_te, ok_te = feats(test_items)
    if r_tr.numel() == 0 or r_te.numel() == 0:
        return RegionBResult(False, probed=len(kept), kept=len(usable),
                             reason="span features unavailable").__dict__
    dev = lm.model.device
    used_tr = [train_items[i] for i in ok_tr]
    labels = torch.tensor([it["label"] for it in used_tr], dtype=torch.long)
    known = torch.ones(len(used_tr), dtype=torch.bool)
    nn = len(used_tr)
    used_te = [test_items[i] for i in ok_te]

    @torch.no_grad()
    def acc(head):
        rel = torch.zeros(r_te.shape[0], dtype=torch.long, device=dev)
        pred = head(r_te.to(dev), rel).argmax(1).cpu().tolist()
        return sum(int(p == u["label"]) for p, u in zip(pred, used_te) if p != 2) / max(1, len(used_te))

    # multi-seed the binder (features are fixed; only training init varies)
    import statistics
    accs = []
    for s in range(5):
        head = train_one_seed(r_tr.to(dev), labels, known, torch.arange(int(nn * 0.85)),
                              torch.arange(int(nn * 0.85), nn), r_tr.shape[-1], dev, seed + s)
        accs.append(acc(head))
    b = statistics.median(accs)
    b_min, b_max = min(accs), max(accs)
    b_std = statistics.pstdev(accs) if len(accs) > 1 else 0.0
    l_exact = _lookup(used_te, cap_to_country, False, country_cap)
    l_fuzzy = _lookup(used_te, cap_to_country, True, country_cap)
    regime_valid = l_fuzzy < FUZZY_PRECONDITION
    margin = round((b - l_fuzzy) * 100, 2)
    note = ("Region B is binder-favorable: facts the model demonstrably knows (probe-kept), values "
            "presented as clues so exact and fuzzy lookup both fail. " +
            ("Regime VALID (fuzzy < 60%). " if regime_valid else
             "Regime INVALID (fuzzy >= 60%); no binder claim made. ") +
            ("Binder beats both lookups." if (regime_valid and b >= l_fuzzy + BINDER_MARGIN and b >= l_exact + BINDER_MARGIN)
             else "Binder does not beat both lookups (honest)."))
    note = note + f" Binder over 5 seeds: median {b:.0%} [{b_min:.0%}/{b_max:.0%}] std {b_std:.3f}."
    return RegionBResult(
        available=True, probed=len(kept), kept=len(usable), regime_valid=regime_valid,
        n_test=len(used_te), binder_exact=round(b, 4), lookup_exact=round(l_exact, 4),
        lookup_fuzzy=round(l_fuzzy, 4),
        binder_beats_both=bool(regime_valid and b >= l_fuzzy + BINDER_MARGIN and b >= l_exact + BINDER_MARGIN),
        margin_binder_vs_fuzzy_pp=margin, note=note).__dict__
