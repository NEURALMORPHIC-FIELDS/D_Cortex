# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Region B binder validation: decides whether the binder win is REAL or an artifact.
# Splits the persisted Region B facts disjointly by fact (a country->capital pair is
# entirely in train OR held-out test), trains a fresh ContentAddressedRoleBinder on
# Qwen2.5-7B layer -1 hidden states across many seeds, and compares FOUR baselines on
# the HELD-OUT test only: exact (entity,attribute) lookup, fuzzy token-overlap lookup,
# MiniLM embedding cosine retrieval, and the neural binder. A shuffled-hidden-state
# ablation (features permuted across examples, labels kept) tests whether the signal is
# causally in Qwen's hidden states or is head memorization. Verdict is CONFIRMED only
# if the ablation collapses AND the binder beats the embedding baseline on held-out.

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
for extra in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import torch
from dcortex_professional.qwen_runtime import QwenBaseModel
from dcortex_professional.region_b import _norm, _tokens
from train_role_evolution import train_one_seed

SEP = "=" * 70
DATA = REPO_ROOT / "data" / "professional_capable" / "region_b_data.json"
RUN_DIR = REPO_ROOT / "runs" / "region_b_validation"
SEEDS = 12
CHANCE = 0.5
ABLATION_TOL = 0.10          # ablated binder must be <= chance + this
EMBED_MARGIN = 0.05          # binder must beat embedding by >= this on held-out
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_items(pool: List[Dict], n: int, rng: random.Random) -> List[Dict]:
    items = []
    pairs = [(pool[i], pool[j]) for i in range(len(pool)) for j in range(i + 1, len(pool))]
    rng.shuffle(pairs)
    for fa, fb in pairs[: n * 3]:
        if len(items) >= n or fa["country"] == fb["country"]:
            continue
        swapped = rng.random() < 0.5
        vA, vB = fa["clue"], fb["clue"]
        v0, v1 = (vB, vA) if swapped else (vA, vB)
        text = f"Country {fa['country']} and country {fb['country']}. Clue one: {v0}. Clue two: {v1}."
        items.append({"text": text, "phrases": [fa["country"], fb["country"], v0, v1],
                      "countries": [fa["country"], fb["country"]], "values": [v0, v1],
                      "label": 1 if swapped else 0,
                      "caps": [fa["capital"], fb["capital"]]})
    return items


def feats(lm, items):
    rows, ok = [], []
    for i, it in enumerate(items):
        f = lm.span_features(it["text"], it["phrases"])
        if f is not None:
            rows.append(f)
            ok.append(i)
    return (torch.stack(rows, 0) if rows else torch.empty(0)), ok


def exact_lookup(items, cap_to_country) -> float:
    n = ok = 0
    for it in items:
        n += 1
        cA, cB = it["countries"]
        c0 = cap_to_country.get(_norm(it["values"][0]))   # clue never equals a capital -> None
        c1 = cap_to_country.get(_norm(it["values"][1]))
        pred = (0 if (c0 == cA and c1 == cB) else 1) if (c0 and c1) else None
        ok += int(pred == it["label"])
    return ok / max(1, n)


def fuzzy_lookup(items, country_cap) -> float:
    n = ok = 0
    for it in items:
        n += 1
        cA, cB = it["countries"]
        tA, tB = _tokens(country_cap[cA]), _tokens(country_cap[cB])

        def best(v):
            vt = _tokens(v)
            jA = len(vt & tA) / max(1, len(vt | tA))
            jB = len(vt & tB) / max(1, len(vt | tB))
            return cA if jA >= jB else cB
        c0, c1 = best(it["values"][0]), best(it["values"][1])
        ok += int((0 if (c0 == cA and c1 == cB) else 1) == it["label"])
    return ok / max(1, n)


def embed_lookup(items, embedder) -> float:
    """MiniLM cosine: assign each clue to the more similar candidate country."""
    import torch.nn.functional as F
    countries = sorted({c for it in items for c in it["countries"]})
    clues = sorted({v for it in items for v in it["values"]})
    cvec = {c: e for c, e in zip(countries, embedder.encode(countries, convert_to_tensor=True,
                                                            normalize_embeddings=True))}
    vvec = {v: e for v, e in zip(clues, embedder.encode(clues, convert_to_tensor=True,
                                                        normalize_embeddings=True))}
    n = ok = 0
    for it in items:
        n += 1
        cA, cB = it["countries"]

        def best(v):
            sA = float(F.cosine_similarity(vvec[v].unsqueeze(0), cvec[cA].unsqueeze(0)))
            sB = float(F.cosine_similarity(vvec[v].unsqueeze(0), cvec[cB].unsqueeze(0)))
            return cA if sA >= sB else cB
        c0, c1 = best(it["values"][0]), best(it["values"][1])
        ok += int((0 if (c0 == cA and c1 == cB) else 1) == it["label"])
    return ok / max(1, n)


def agg(xs) -> Dict[str, float]:
    return {"median": round(statistics.median(xs), 4), "min": round(min(xs), 4),
            "max": round(max(xs), 4), "std": round(statistics.pstdev(xs) if len(xs) > 1 else 0.0, 4),
            "n": len(xs)}


def binder_eval(r_tr, lab_tr, r_te, items_te, dev, seed, permute=False) -> float:
    reps_tr, reps_te = r_tr.clone(), r_te.clone()
    if permute:
        g = torch.Generator().manual_seed(seed)
        reps_tr = reps_tr[torch.randperm(reps_tr.shape[0], generator=g)]
        g2 = torch.Generator().manual_seed(seed + 99991)
        reps_te = reps_te[torch.randperm(reps_te.shape[0], generator=g2)]
    known = torch.ones(reps_tr.shape[0], dtype=torch.bool)
    nn = reps_tr.shape[0]
    head = train_one_seed(reps_tr.to(dev), lab_tr, known, torch.arange(int(nn * 0.85)),
                          torch.arange(int(nn * 0.85), nn), reps_tr.shape[-1], dev, seed)
    with torch.no_grad():
        rel = torch.zeros(reps_te.shape[0], dtype=torch.long, device=dev)
        pred = head(reps_te.to(dev), rel).argmax(1).cpu().tolist()
    return sum(int(p == it["label"]) for p, it in zip(pred, items_te) if p != 2) / max(1, len(items_te))


def main() -> int:
    ap = argparse.ArgumentParser(description="Region B binder validation")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()
    (RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
    print(SEP, flush=True)
    if not DATA.exists():
        print(f"[BLOCKED] Region B data not persisted: {DATA}", flush=True)
        return 2
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    facts = payload["facts"]
    print(f"[INFO] Region B data: {len(facts)} model-known facts+clues (probe_kept {payload.get('probe_kept')})",
          flush=True)

    lm = QwenBaseModel()
    if not lm.available:
        print(f"[BLOCKED] Qwen: {lm.reason}", flush=True)
        return 2
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBED_MODEL, device="cuda")
    print(f"[INFO] model {lm.model_name} (dim {lm.hidden_dim}); embedder {EMBED_MODEL}", flush=True)

    # disjoint split BY FACT
    rng = random.Random(args.seed)
    pool = facts[:]
    rng.shuffle(pool)
    split = int(len(pool) * 0.7)
    train_facts, test_facts = pool[:split], pool[split:]
    assert not ({f["country"] for f in train_facts} & {f["country"] for f in test_facts}), "fact leakage"
    country_cap = {f["country"]: f["capital"] for f in facts}
    cap_to_country = {_norm(f["capital"]): f["country"] for f in facts}
    train_items = build_items(train_facts, 140, rng)
    test_items = build_items(test_facts, 60, rng)
    print(f"[INFO] disjoint split: {len(train_facts)} train facts / {len(test_facts)} test facts; "
          f"{len(train_items)} train items / {len(test_items)} HELD-OUT test items", flush=True)

    r_tr, ok_tr = feats(lm, train_items)
    r_te, ok_te = feats(lm, test_items)
    used_tr = [train_items[i] for i in ok_tr]
    used_te = [test_items[i] for i in ok_te]
    lab_tr = torch.tensor([it["label"] for it in used_tr], dtype=torch.long)
    dev = lm.model.device

    # baselines on HELD-OUT
    b_exact = exact_lookup(used_te, cap_to_country)
    b_fuzzy = fuzzy_lookup(used_te, country_cap)
    b_embed = embed_lookup(used_te, embedder)

    # binder + ablation across seeds (held-out only)
    real = [binder_eval(r_tr, lab_tr, r_te, used_te, dev, 1000 + s, permute=False) for s in range(args.seeds)]
    abl = [binder_eval(r_tr, lab_tr, r_te, used_te, dev, 3000 + s, permute=True) for s in range(args.seeds)]
    # in-sample (logged, never the result)
    insample = binder_eval(r_tr, lab_tr, r_tr, used_tr, dev, 1000, permute=False)

    real_s, abl_s = agg(real), agg(abl)
    print(SEP, flush=True)
    print(f"[INFO] HELD-OUT baselines: exact {b_exact:.1%} | fuzzy {b_fuzzy:.1%} | embed(MiniLM) {b_embed:.1%}",
          flush=True)
    print(f"[INFO] binder held-out ({args.seeds} seeds): median {real_s['median']:.1%} "
          f"[{real_s['min']:.1%}/{real_s['max']:.1%}] std {real_s['std']:.3f}", flush=True)
    print(f"[INFO] ablation (shuffled hidden states): median {abl_s['median']:.1%} "
          f"[{abl_s['min']:.1%}/{abl_s['max']:.1%}] (chance {CHANCE:.0%}; tol +{ABLATION_TOL:.0%})", flush=True)
    print(f"[INFO] in-sample (logged only, NOT the result): {insample:.1%}", flush=True)

    # gates
    g_heldout = True
    g_ablation = abl_s["median"] <= CHANCE + ABLATION_TOL
    g_vs_embed = real_s["median"] >= b_embed + EMBED_MARGIN
    g_seeds = args.seeds >= 10
    if not g_ablation:
        verdict = "REGION_B_BINDER_INVALID"
        reason = f"ablation did not collapse (median {abl_s['median']:.1%} > chance+{ABLATION_TOL:.0%}); head memorization"
    elif g_vs_embed:
        verdict = "REGION_B_BINDER_CONFIRMED"
        reason = f"ablation collapsed and binder beats MiniLM embedding by {real_s['median']-b_embed:+.1%} on held-out"
    else:
        verdict = "REGION_B_BINDER_NEGATIVE"
        reason = f"embedding baseline beats/ties binder on held-out (binder {real_s['median']:.1%} vs embed {b_embed:.1%})"

    out = {
        "verdict": verdict, "reason": reason,
        "model": lm.model_name, "embed_model": EMBED_MODEL,
        "n_train_facts": len(train_facts), "n_test_facts": len(test_facts),
        "n_heldout_items": len(used_te), "seeds": args.seeds,
        "heldout": {"exact_lookup": round(b_exact, 4), "fuzzy_lookup": round(b_fuzzy, 4),
                    "embedding_cosine_minilm": round(b_embed, 4),
                    "binder": real_s, "binder_ablated_shuffled": abl_s},
        "in_sample_binder_logged_only": round(insample, 4),
        "gates": {"G_HELDOUT": bool(g_heldout), "G_ABLATION": bool(g_ablation),
                  "G_VS_EMBED": bool(g_vs_embed), "G_VS_FUZZY_context": round(real_s["median"] - b_fuzzy, 4),
                  "G_SEEDS": bool(g_seeds)},
        "frozen": {"CHANCE": CHANCE, "ABLATION_TOL": ABLATION_TOL, "EMBED_MARGIN": EMBED_MARGIN, "SEEDS": SEEDS},
        "claim_scope": ("A binder win means content-addressed resolution of an obscured entity from Qwen's "
                        "hidden states, beating embedding retrieval; NOT unlocking latent model knowledge. "
                        "MEASURED, Qwen2.5-7B 4-bit, single env. No external citation, no merge to main."),
    }
    (RUN_DIR / "results" / "verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(SEP, flush=True)
    print(f"[INFO] G_ABLATION {'PASS' if g_ablation else 'FAIL'} | G_VS_EMBED {'PASS' if g_vs_embed else 'FAIL'} "
          f"| G_SEEDS {'PASS' if g_seeds else 'FAIL'}", flush=True)
    print(f"[INFO] VERDICT: {verdict} -- {reason}", flush=True)
    print("REGION_B_VALIDATION_JSON " + json.dumps({"verdict": verdict, "binder": real_s["median"],
          "embed": b_embed, "fuzzy": b_fuzzy, "ablation": abl_s["median"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
