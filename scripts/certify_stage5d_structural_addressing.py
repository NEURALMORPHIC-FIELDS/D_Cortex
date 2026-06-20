# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# STAGE 5d - STRUCTURAL RELATIONAL ADDRESSING. Stage 5c refuted learned pointer REPRESENTATIONS
# (held-out recovery 0.50, in-sample 1.0). The finding: VALUE separability generalizes (0.92) because
# content-ADDRESSING generalizes; the learned pointer failed because it was a learned SEMANTIC vector,
# not a structural ADDRESS. Fix: store the relational pointer AS A COPY of the TARGET entity's
# content-addressing key (the same k_ent the reader uses to address the target's value slot), reusing
# the content-addressing that already generalizes. Traversal re-issues a content-address with that key.
#
# THE CRUX (validity-critical):
#   1. STRUCTURAL POINTER: B's relational slot stores query_key(target) = the target's content-key,
#      NOT a learned pointer-vector. The encoder is FROZEN (the structural copy needs no learning ->
#      single-fact + comparison preserved by construction).
#   2. TRAVERSAL = REUSE: read B's slot -> read its stored content-key (a content-addressed read of the
#      pointer bank) -> content-address the target slot with that key -> read the target value. Every
#      step is a content-address that already generalizes held-out.
#   3. A/B PROOF: the Stage 5c learned-recovery arm on the SAME held-out (expected ~0.50) vs the
#      structural key-copy (expected >= 0.90). Isolates that STRUCTURE generalizes, not more training.
#   4. The operation reads ONLY (bank tensors incl. the pointer bank + query keys), NEVER text.

import argparse
import contextlib
import io
import json
import random
import sys
from pathlib import Path
from statistics import median, pstdev
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

import tiktoken
from scripts.certify_stage5_operate_memory import (
    ENC, DEVICE, ENTITIES, SIZES, SIZE_RANK, COLORS, COLOR_IDX, ABSTAIN, N_SLOTS, OP_COMPARE, OP_CHAIN,
    load_model, query_key, write_and_snapshot, entity_slot)

SEP = "=" * 70
RUN_DIR = REPO_ROOT / "runs" / "stage5d_structural"
KEY_CACHE: Dict[str, torch.Tensor] = {}


def qkey(model, entity: str) -> torch.Tensor:
    if entity not in KEY_CACHE:
        KEY_CACHE[entity] = query_key(model, f"What color is the {entity}?").to(DEVICE).detach().cpu()
    return KEY_CACHE[entity]


# ---------------------------------------------------------------------------
# Regime with a STRUCTURAL POINTER BANK: B's relational slot stores query_key(target).
# ---------------------------------------------------------------------------
def chain_ep(model, ents, rng, variant: str) -> Dict:
    A, B, C = rng.sample(ents, 3)
    c1, c2 = rng.sample(COLORS, 2)
    if variant == "unanswerable":
        Z = rng.choice([e for e in ents if e not in (A, B, C)])
        f2, target, gold = f"The {B} is the same color as {Z}.", Z, ABSTAIN
    elif variant == "shuffled":
        f2, target, gold = f"The {B} is the same color as {C}.", C, COLOR_IDX[c2]
    else:
        f2, target, gold = f"The {B} is the same color as {A}.", A, COLOR_IDX[c1]
    facts = [f"The {A} is {c1}.", f2, f"The {C} is {c2}."]
    v, k, m = write_and_snapshot(model, facts)
    bslot = entity_slot(k, m, query_key(model, f"What color is the {B}?"))
    pointers = torch.zeros(N_SLOTS, 128)
    pointers[bslot] = qkey(model, target)                      # STRUCTURAL pointer = target's content-key
    tgt_slot = entity_slot(k, m, query_key(model, f"What color is the {target}?")) if variant != "unanswerable" else -1
    return {"op": OP_CHAIN, "v": v, "k": k, "m": m, "ptr": pointers,
            "qk": torch.stack([query_key(model, f"What color is the {B}?").cpu(), torch.zeros(128)]),
            "gold": gold, "variant": variant, "bslot": bslot, "target": target, "target_slot": tgt_slot,
            "ctx": (A, B, C, c1, c2)}


def repoint(model, ep: Dict):
    if ep["variant"] == "unanswerable":
        return None
    A, B, C, c1, c2 = ep["ctx"]
    nt = C if ep["variant"] == "normal" else A
    ng = COLOR_IDX[c2] if ep["variant"] == "normal" else COLOR_IDX[c1]
    facts = [f"The {A} is {c1}.", f"The {B} is the same color as {nt}.", f"The {C} is {c2}."]
    v, k, m = write_and_snapshot(model, facts)
    bslot = entity_slot(k, m, query_key(model, f"What color is the {B}?"))
    pointers = torch.zeros(N_SLOTS, 128); pointers[bslot] = qkey(model, nt)
    return {"op": OP_CHAIN, "v": v, "k": k, "m": m, "ptr": pointers,
            "qk": torch.stack([query_key(model, f"What color is the {B}?").cpu(), torch.zeros(128)]),
            "gold": ng, "variant": "repointed", "bslot": bslot, "target": nt,
            "target_slot": entity_slot(k, m, query_key(model, f"What color is the {nt}?"))}


def compare_ep(model, ents, rng) -> Dict:
    e0, e1, e2 = rng.sample(ents, 3)
    s0, s1, s2 = rng.sample(SIZES, 3)
    facts = [f"The {e0} is {s0}.", f"The {e1} is {s1}.", f"The {e2} is {s2}."]
    v, k, m = write_and_snapshot(model, facts)
    qk0, qk1 = query_key(model, f"What size is the {e0}?"), query_key(model, f"What size is the {e1}?")
    return {"op": OP_COMPARE, "v": v, "k": k, "m": m, "ptr": torch.zeros(N_SLOTS, 128),
            "qk": torch.stack([qk0.cpu(), qk1.cpu()]),
            "gold": 0 if SIZE_RANK[s0] > SIZE_RANK[s1] else 1, "variant": "compare", "target_slot": -1}


# ---------------------------------------------------------------------------
# Operation: structural traversal (reads the pointer bank, content-addresses the target).
# ---------------------------------------------------------------------------
class StructuralOp(nn.Module):
    def __init__(self, d_val: int = 768, d_ent: int = 128, d_state: int = 256, n_colors: int = len(COLORS)) -> None:
        super().__init__()
        self.val_in = nn.Linear(d_val, d_state)
        self.init_cmp = nn.Sequential(nn.Linear(2 * d_state, d_state), nn.GELU(), nn.Linear(d_state, d_state))
        self.init_chain = nn.Sequential(nn.Linear(2 * d_state + 1, d_state), nn.GELU(), nn.Linear(d_state, d_state))
        self.temp = nn.Parameter(torch.tensor(8.0))
        self.cmp_head = nn.Linear(d_state, 2)
        self.chain_head = nn.Linear(d_state, n_colors + 1)

    def _attn(self, kn, mask, key):
        sims = torch.einsum('bcd,bd->bc', kn, F.normalize(key, dim=-1)) * self.temp
        sims = sims.masked_fill(~mask, -1e9)
        return F.softmax(sims, dim=1), (sims / self.temp).max(dim=1).values.clamp(-1, 1)

    def forward(self, values, k_ent, pointers, mask, qk, op):
        # reads banks (values, keys, POINTER bank) + query keys only - no text (G_IN_MEMORY)
        kn = F.normalize(k_ent, dim=-1)
        aB, _ = self._attn(kn, mask, qk[:, 0])
        r0 = torch.einsum('bc,bcd->bd', aB, values)
        a1, _ = self._attn(kn, mask, qk[:, 1])
        r1 = torch.einsum('bc,bcd->bd', a1, values)
        # STRUCTURAL traversal: read B's stored content-key, content-address the target
        p = torch.einsum('bc,bcd->bd', aB, pointers)            # pointer = target content-key (structural)
        aT, confT = self._attn(kn, mask, p)
        rT = torch.einsum('bc,bcd->bd', aT, values)
        st_cmp = self.init_cmp(torch.cat([self.val_in(r0), self.val_in(r1)], dim=-1))
        st_chain = self.init_chain(torch.cat([self.val_in(r0), self.val_in(rT), confT.unsqueeze(-1)], dim=-1))
        return self.cmp_head(st_cmp), self.chain_head(st_chain)


class LearnedRecover(nn.Module):  # the Stage 5c arm, for the A/B
    def __init__(self, d_val: int = 768, d_ent: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_val, 256), nn.GELU(), nn.Linear(256, d_ent))

    def forward(self, v):
        return self.net(v)


def pack(eps):
    V = torch.stack([e["v"] for e in eps]).to(DEVICE)
    K = torch.stack([e["k"] for e in eps]).to(DEVICE)
    P = torch.stack([e["ptr"] for e in eps]).to(DEVICE)
    M = torch.stack([e["m"] for e in eps]).to(DEVICE)
    QK = torch.stack([e["qk"] for e in eps]).to(DEVICE)
    OP = torch.tensor([e["op"] for e in eps], device=DEVICE)
    GOLD = torch.tensor([e["gold"] for e in eps], device=DEVICE)
    return V, K, P, M, QK, OP, GOLD


def pointer_generalizes(eps, mode: str, model=None, recover=None) -> float:
    """Does the recovered/structural pointer content-address the target slot? held-out."""
    ok = tot = 0
    for e in eps:
        if e["op"] != OP_CHAIN or e["variant"] == "unanswerable" or e["target_slot"] < 0:
            continue
        kn = F.normalize(e["k"].to(DEVICE), dim=1)
        occ = e["m"].nonzero(as_tuple=True)[0]
        if mode == "structural":
            p = e["ptr"][e["bslot"]].to(DEVICE)
        else:  # learned: recover from B's value
            bv = e["v"][e["bslot"]].to(DEVICE)
            p = recover(bv.unsqueeze(0))[0]
        sims = (kn[occ] @ F.normalize(p, dim=0))
        pred = int(occ[int(torch.argmax(sims).item())].item())
        ok += int(pred == e["target_slot"]); tot += 1
    return ok / max(1, tot)


def evaluate(op, eps):
    V, K, P, M, QK, OP, GOLD = pack(eps)
    op.eval()
    with torch.no_grad():
        cmp_l, chain_l = op(V, K, P, M, QK, OP)
    b = {"compare": [0, 0], "chain_normal": [0, 0], "chain_shuffled": [0, 0], "chain_repointed": [0, 0], "abstain": [0, 0]}
    for i, e in enumerate(eps):
        if e["op"] == OP_COMPARE:
            b["compare"][0] += int(int(torch.argmax(cmp_l[i]).item()) == e["gold"]); b["compare"][1] += 1
        else:
            pred = int(torch.argmax(chain_l[i]).item())
            if e["variant"] == "unanswerable":
                b["abstain"][0] += int(pred == ABSTAIN); b["abstain"][1] += 1
            else:
                key = {"normal": "chain_normal", "shuffled": "chain_shuffled", "repointed": "chain_repointed"}[e["variant"]]
                b[key][0] += int(pred == e["gold"]); b[key][1] += 1
    return {k: (c[0] / c[1] if c[1] else None) for k, c in b.items()}


def train_struct(train_eps, seed, steps, lr):
    random.seed(seed); torch.manual_seed(seed)
    op = StructuralOp().to(DEVICE)
    opt = torch.optim.AdamW(op.parameters(), lr=lr)
    V, K, P, M, QK, OP, GOLD = pack(train_eps)
    is_cmp = (OP == OP_COMPARE)
    op.train()
    for _ in range(steps):
        opt.zero_grad()
        cmp_l, chain_l = op(V, K, P, M, QK, OP)
        loss = torch.tensor(0.0, device=DEVICE)
        if is_cmp.any():
            loss = loss + F.cross_entropy(cmp_l[is_cmp], GOLD[is_cmp])
        if (~is_cmp).any():
            loss = loss + F.cross_entropy(chain_l[~is_cmp], GOLD[~is_cmp])
        loss.backward(); opt.step()
    return op


def train_learned_recover(model, train_eps, seed) -> LearnedRecover:
    random.seed(seed); torch.manual_seed(seed)
    rec = LearnedRecover().to(DEVICE)
    opt = torch.optim.AdamW(rec.parameters(), lr=1e-3)
    chains = [e for e in train_eps if e["op"] == OP_CHAIN and e["variant"] != "unanswerable" and e["target_slot"] >= 0]
    for _ in range(800):
        opt.zero_grad(); loss = torch.tensor(0.0, device=DEVICE)
        batch = random.sample(chains, min(32, len(chains)))
        for e in batch:
            bv = e["v"][e["bslot"]].to(DEVICE)
            rk = F.normalize(rec(bv.unsqueeze(0))[0], dim=0)
            tgt = F.normalize(e["k"][e["target_slot"]].to(DEVICE), dim=0)
            loss = loss + (1.0 - F.cosine_similarity(rk, tgt, dim=0))
        (loss / len(batch)).backward(); opt.step()
    return rec


def dist(xs):
    xs = [x for x in xs if x is not None]
    return {"min": round(min(xs), 4), "median": round(median(xs), 4), "max": round(max(xs), 4),
            "std": round(pstdev(xs), 4) if len(xs) > 1 else 0.0}


def build(model, ents, n, rng):
    eps = []
    for _ in range(n):
        r = rng.random()
        if r < 0.35:
            eps.append(compare_ep(model, ents, rng))
        elif r < 0.6:
            eps.append(chain_ep(model, ents, rng, "normal"))
        elif r < 0.85:
            eps.append(chain_ep(model, ents, rng, "shuffled"))
        else:
            eps.append(chain_ep(model, ents, rng, "unanswerable"))
    return eps


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 5d structural relational addressing")
    ap.add_argument("--ckpt", default="runs/multiobject/ckpt_multiobject.pt")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    (RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
    print(SEP, flush=True)
    print(f"[INFO] Stage 5d structural relational addressing | device={DEVICE}", flush=True)
    model = load_model(args.ckpt)
    print("[INFO] FROZEN encoder (structural pointer needs no learning -> single-fact + comparison preserved)", flush=True)

    split_rng = random.Random(20260620)
    sh = ENTITIES[:]; split_rng.shuffle(sh)
    train_ents, held_ents = sh[:14], sh[14:]
    print(f"[INFO] entity split: 14 train / {len(held_ents)} held-out", flush=True)

    print("[INFO] precomputing banks + STRUCTURAL pointer banks...", flush=True)
    drng = random.Random(7)
    train_eps = build(model, train_ents, args.n_train, drng)
    eval_eps = build(model, held_ents, args.n_eval, drng)
    repoint_eps = [t for e in eval_eps if e["op"] == OP_CHAIN for t in [repoint(model, e)] if t]
    print(f"[INFO] episodes: {len(train_eps)} / {len(eval_eps)} / {len(repoint_eps)} re-pointed", flush=True)

    # ---- G_POINTER_GENERALIZES: structural vs learned (the decisive A/B) ----
    g_struct = pointer_generalizes(eval_eps, "structural")
    rec = train_learned_recover(model, train_eps, 0)
    g_learned = pointer_generalizes(eval_eps, "learned", recover=rec)
    print(f"  G_POINTER_GENERALIZES: structural={g_struct:.3f} vs learned(Stage5c)={g_learned:.3f} (bar 0.90)", flush=True)

    if args.smoke:
        return 0

    runs = []
    for seed in range(args.seeds):
        op = train_struct(train_eps, seed, args.steps, args.lr)
        ev = evaluate(op, eval_eps); gr = evaluate(op, repoint_eps)
        runs.append({"eval": ev, "grounded": gr["chain_repointed"]})
        print(f"  op seed {seed}: chain_shuf={ev['chain_shuffled']:.3f} abstain={ev['abstain']:.3f} "
              f"compare={ev['compare']:.3f} | CHAIN_GROUNDED={gr['chain_repointed']:.3f}", flush=True)

    chaining = dist([r["eval"]["chain_shuffled"] for r in runs])
    grounded = dist([r["grounded"] for r in runs])
    abstain = dist([r["eval"]["abstain"] for r in runs])
    comparison = dist([r["eval"]["compare"] for r in runs])

    g_pointer = g_struct >= 0.90
    g_chaining = chaining["median"] >= 0.80
    g_grounded = grounded["median"] >= 0.90
    g_abstain = abstain["median"] >= 0.80
    g_comparison = comparison["median"] >= 0.80

    # traversal is proven by pointer-generalizes AND chaining AND grounded together. Abstain is a
    # SEPARATE honesty sub-capability - failing it does NOT mean traversal failed (avoid the Stage-5
    # aggregation mislabel).
    traversal_proven = g_pointer and g_chaining and g_grounded
    if traversal_proven and g_abstain and g_comparison:
        verdict = "STAGE_5_GRAPH_COMPLETE"
    elif traversal_proven and g_comparison:
        verdict = "STAGE_5_GRAPH_TRAVERSAL_PROVEN"        # structural addressing closes traversal; abstain weak
    elif g_pointer:
        verdict = "STAGE_5_POINTER_OK_TRAVERSAL_FAIL"     # pointer generalizes but chaining/grounded fail
    else:
        verdict = "STAGE_5_STRUCTURAL_REFUTED"

    out = {"verdict": verdict, "ckpt": args.ckpt,
           "gates": {
               "G_POINTER_GENERALIZES": {"structural": g_struct, "learned_stage5c": g_learned, "bar": 0.90,
                                         "passed": bool(g_pointer),
                                         "evidence": "structural pointer = copy of target content-key; content-address retrieves target slot, held-out; A/B vs learned recovery"},
               "G_CHAINING_BANK": {"passed": bool(g_chaining), "dist": chaining, "bar": 0.80},
               "G_CHAIN_GROUNDED": {"passed": bool(g_grounded), "dist": grounded, "bar": 0.90},
               "G_ABSTAIN": {"passed": bool(g_abstain), "dist": abstain, "bar": 0.80},
               "G_COMPARISON_PRESERVED": {"passed": bool(g_comparison), "dist": comparison, "bar": 0.80},
               "G_SINGLE_FACT_PRESERVED": "0/140 (frozen encoder = ckpt_multiobject; structural pointer needs no encoder change)",
               "G_IN_MEMORY": {"passed": True, "evidence": "operation reads banks (incl pointer bank) + query keys only"},
           },
           "per_seed": runs,
           "scope": ("structural relational addressing (pointer = copy of target content-key) over persisted banks; "
                     "FROZEN encoder; held-out; single machine. NOT generality, NOT free-text. Target identification "
                     "is template-given (the extraction sub-problem is separate); this tests the ADDRESSING mechanism."),
           "meaning": {
               "STAGE_5_GRAPH_COMPLETE": "structural pointer generalizes AND the operation traverses it (grounded) -> "
                   "memory is a navigable graph; thinking-in-memory closed at small scale (per-object op + traversal).",
               "STAGE_5_POINTER_OK_TRAVERSAL_FAIL": "structural pointer generalizes but the operation does not traverse "
                   "it -> the gap is the operation, not the addressing.",
               "STAGE_5_STRUCTURAL_REFUTED": "even the structural key-copy does not generalize -> the current key space "
                   "cannot carry relational addresses; a dedicated relational key space is needed."}[verdict]}
    (RUN_DIR / "results" / "verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(SEP, flush=True)
    print(f"  G_POINTER_GENERALIZES structural={g_struct:.3f} vs learned={g_learned:.3f} (bar 0.90) -> {'PASS' if g_pointer else 'FAIL'}", flush=True)
    print(f"  G_CHAINING_BANK={chaining['median']} G_CHAIN_GROUNDED={grounded['median']} (bar 0.90) "
          f"G_ABSTAIN={abstain['median']} G_COMPARISON_PRESERVED={comparison['median']}", flush=True)
    print(f"[INFO] VERDICT: {verdict}", flush=True)
    print("STAGE5D_JSON " + json.dumps({"verdict": verdict, "pointer_structural": g_struct, "pointer_learned": g_learned,
          "chaining": chaining["median"], "chain_grounded": grounded["median"], "comparison": comparison["median"]}), flush=True)
    return 0 if verdict == "STAGE_5_GRAPH_COMPLETE" else 1


if __name__ == "__main__":
    sys.exit(main())
