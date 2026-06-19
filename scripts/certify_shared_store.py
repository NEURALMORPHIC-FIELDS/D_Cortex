# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Verify the shared store + canonical-direct path (CPU; the canonical path needs no model). Gates:
#   G_CANONICAL_ROUNDTRIP : write_canonical(e,a,v) -> read back v exactly, status=committed,
#                           provenance=user_canonical, token exact.
#   G_ARBITER_ON_TOKENS    : run the L1-L5 honest-mechanics arbiter with CANONICAL token identity
#                           (exact) -> wrong_commit = 0 vs the symbolic oracle (Stage U holds EXACTLY
#                           on canonical tokens - no internalization asterisk; the ~8% floor is the
#                           EXTRACTION path only, deferred to Stage I).
#   G_INSPECTABLE          : dump the store; edit one object; re-read reflects the edit (the user
#                           sees and corrects what the model knows).
# SCOPE: SUBSTRATE track - exact, honest, auditable STORAGE + ADMINISTRATION. NOT reasoning (Stage 5).

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

from integration.organ_client import OrganClient, FOUND_COMMITTED
from stage_u.shared_store import SharedMemoryStore
from stage_u.l_regime import build_regime, ALL_VALUES
from stage_u.neural_arbiter import NeuralCommitArbiter, cosine_same_value

SEP = "=" * 70
RUN_DIR = REPO_ROOT / "runs" / "shared_store"
SUBSTANTIVE = {"RETROGRADE", "PROMOTE", "PRUNE"}


def onehot(token: int, k: int) -> torch.Tensor:
    v = torch.zeros(k)
    v[token] = 1.0
    return v


def symbolic_oracle(seq, organ_cls):
    o = organ_cls()
    if seq.seed is not None:
        e, a, v = seq.seed
        o.begin_episode(); o.write_fact(e, a, v); o.end_episode()
    for episode in seq.episodes:
        o.begin_episode()
        for (e, a, v) in episode:
            o.write_fact(e, a, v)
        o.end_episode()
    committed = {}
    for t in seq.targets:
        r = o.query(*t)
        committed[t] = r.value if r.status == FOUND_COMMITTED else None
    return committed


def main() -> int:
    (RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
    print(SEP, flush=True)
    print("[INFO] Shared store + canonical-direct path verification (CPU)", flush=True)

    # canonical token registry over all regime values (exact, model-free)
    store = SharedMemoryStore()
    for v in ALL_VALUES:
        store.canonical_token(v)
    K = len(ALL_VALUES)
    tok_of = {v: store.canonical_token(v) for v in ALL_VALUES}
    val_of = {t: v for v, t in tok_of.items()}

    # ---- G_CANONICAL_ROUNDTRIP ----
    s2 = SharedMemoryStore()
    o = s2.write_canonical("bear", "color", "red", rule={"constraint": "color in palette"})
    r = s2.get("bear", "color")
    g_roundtrip = (r is not None and r.value == "red" and r.status == "committed"
                   and r.provenance == "user_canonical" and r.memory_token == s2.canonical_token("red"))
    print(f"  [{'PASS' if g_roundtrip else 'FAIL'}] G_CANONICAL_ROUNDTRIP: read {r.value!r} status={r.status} "
          f"prov={r.provenance} token={r.memory_token}", flush=True)

    # ---- G_ARBITER_ON_TOKENS (canonical, exact) on L1-L5 ----
    ents = OrganClient().known_entities
    seqs = build_regime(ents, n_per_level=20)
    same = cosine_same_value(0.5)        # one-hot tokens -> exact equality
    wrong = total = 0
    for seq in seqs:
        oracle = symbolic_oracle(seq, OrganClient)
        arb = NeuralCommitArbiter(same)
        if seq.seed is not None:
            e, a, v = seq.seed
            arb.seed_committed(e, a, onehot(tok_of[v], K), 0)
        for ep, obs in enumerate(seq.episodes, start=1):
            for (e, a, v) in obs:
                arb.observe(e, a, onehot(tok_of[v], K))
            arb.end_episode(ep)
        for t in seq.targets:
            vec = arb.read(*t)
            got = val_of[int(torch.argmax(vec).item())] if vec is not None else None
            total += 1
            if got != oracle[t]:
                wrong += 1
    g_arbiter = (wrong == 0)
    print(f"  [{'PASS' if g_arbiter else 'FAIL'}] G_ARBITER_ON_TOKENS (canonical): wrong_commit {wrong}/{total} "
          f"(0 expected - canonical tokens are exact)", flush=True)

    # ---- G_INSPECTABLE: dump, edit, re-read ----
    s2.write_canonical("dog", "size", "big")
    before = s2.dump()
    s2.edit("bear", "color", "blue")
    after = s2.get("bear", "color")
    g_inspect = (after.value == "blue" and after.memory_token == s2.canonical_token("blue") and "blue" in s2.dump())
    print(f"  [{'PASS' if g_inspect else 'FAIL'}] G_INSPECTABLE: edited bear.color red->blue; re-read {after.value!r} "
          f"token={after.memory_token}", flush=True)
    print("  store dump after edit:", flush=True)
    for ln in s2.dump().splitlines():
        print("    " + ln, flush=True)

    verdict = "SHARED_STORE_VERIFIED" if (g_roundtrip and g_arbiter and g_inspect) else "SHARED_STORE_PARTIAL"
    out = {"verdict": verdict,
           "G_CANONICAL_ROUNDTRIP": bool(g_roundtrip),
           "G_ARBITER_ON_TOKENS_canonical": {"wrong_commit": wrong, "total": total, "pass": bool(g_arbiter)},
           "G_INSPECTABLE": bool(g_inspect),
           "scope": ("SUBSTRATE track: exact, honest, auditable STORAGE + ADMINISTRATION. Canonical writes are "
                     "EXACT (token from the known value, no internalization drift) -> wrong_commit=0 holds without "
                     "asterisk. The ~8% internalization floor is the EXTRACTION path (Stage I) only. This does NOT "
                     "add reasoning - the model does not reason over memory yet (Stage C refuted 2-hop); that is "
                     "Stage 5, the separate frontier. v1 = honest scalable store + retrieval + no out-of-memory "
                     "hallucination; NOT 'the model reasons over your domain'.")}
    (RUN_DIR / "results" / "verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(SEP, flush=True)
    print(f"[INFO] VERDICT: {verdict}", flush=True)
    print("SHARED_STORE_JSON " + json.dumps({"verdict": verdict, "roundtrip": g_roundtrip,
          "arbiter_wrong_commit": wrong, "inspectable": g_inspect}), flush=True)
    return 0 if (g_roundtrip and g_arbiter and g_inspect) else 1


if __name__ == "__main__":
    sys.exit(main())
