# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# STAGE 5e - HONEST TRAVERSAL. Stage 5d proved graph traversal (chaining 1.0, chain-grounded 0.949)
# but abstain on broken chains was 0.667 - the model fabricates ~1/3 of the time when the chain cannot
# resolve. This extends the honesty invariant (never assert unsupported; wrong_commit=0) to multi-hop:
# when a followed pointer retrieves no valid target, ABSTAIN. The signature property of D_Cortex must
# hold on the new capability, not just on single facts.
#
# THE CRUX (validity-critical):
#   1. HONEST: a broken pointer (followed key matches no stored slot) -> ABSTAIN, never fabricate.
#   2. DETECTION-BASED, not memorized: abstain is a function of the RETRIEVAL CONFIDENCE at the
#      followed key (confT = max cosine of the second content-address). A non-stored target gives low
#      confT (~0.15) vs a stored target (~0.98) - a strongly separable retrieval signal that
#      generalizes held-out (it is a retrieval property, NOT entity identity).
#   3. SELECTIVE (the DUAL gate): abstain on broken (>= 0.80) AND do NOT over-abstain on answerable
#      (<= 0.10). A high abstain rate alone is meaningless (an earlier Stage C run gamed it by always
#      abstaining).
#   4. Preserve all proven gates: chaining, chain-grounded, comparison, single-fact, pointer generalize.

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import median, pstdev
from typing import Dict, List

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

from scripts.certify_stage5_operate_memory import (
    DEVICE, ENTITIES, COLORS, ABSTAIN, N_SLOTS, OP_COMPARE, OP_CHAIN, load_model, query_key, entity_slot)
from scripts.certify_stage5d_structural_addressing import (
    chain_ep, repoint, compare_ep, pack, pointer_generalizes, KEY_CACHE)

SEP = "=" * 70
RUN_DIR = REPO_ROOT / "runs" / "stage5e_honest"


# ---------------------------------------------------------------------------
# Operation: structural traversal with a confT-driven (detection-based) abstain.
# ---------------------------------------------------------------------------
class HonestOp(nn.Module):
    def __init__(self, d_val: int = 768, d_ent: int = 128, d_state: int = 256, n_colors: int = len(COLORS)) -> None:
        super().__init__()
        self.val_in = nn.Linear(d_val, d_state)
        self.init_cmp = nn.Sequential(nn.Linear(2 * d_state, d_state), nn.GELU(), nn.Linear(d_state, d_state))
        self.init_chain = nn.Sequential(nn.Linear(2 * d_state + 1, d_state), nn.GELU(), nn.Linear(d_state, d_state))
        self.temp = nn.Parameter(torch.tensor(8.0))
        self.cmp_head = nn.Linear(d_state, 2)
        self.color_head = nn.Linear(d_state, n_colors)
        # abstain logit driven by the followed-read confidence (detection-based, generalizes)
        self.abstain_scale = nn.Parameter(torch.tensor(10.0))
        self.abstain_thresh = nn.Parameter(torch.tensor(0.5))

    def _attn(self, kn, mask, key):
        sims = torch.einsum('bcd,bd->bc', kn, F.normalize(key, dim=-1)) * self.temp
        sims = sims.masked_fill(~mask, -1e9)
        conf = (sims / self.temp).max(dim=1).values.clamp(-1, 1)
        return F.softmax(sims, dim=1), conf

    def forward(self, values, k_ent, pointers, mask, qk, op):
        kn = F.normalize(k_ent, dim=-1)
        aB, _ = self._attn(kn, mask, qk[:, 0]); r0 = torch.einsum('bc,bcd->bd', aB, values)
        a1, _ = self._attn(kn, mask, qk[:, 1]); r1 = torch.einsum('bc,bcd->bd', a1, values)
        p = torch.einsum('bc,bcd->bd', aB, pointers)
        aT, confT = self._attn(kn, mask, p)                       # confT = retrieval confidence at the target
        rT = torch.einsum('bc,bcd->bd', aT, values)
        st_cmp = self.init_cmp(torch.cat([self.val_in(r0), self.val_in(r1)], dim=-1))
        st_chain = self.init_chain(torch.cat([self.val_in(r0), self.val_in(rT), confT.unsqueeze(-1)], dim=-1))
        color_logits = self.color_head(st_chain)
        abstain_logit = (self.abstain_scale * (self.abstain_thresh - confT)).unsqueeze(-1)
        chain_logits = torch.cat([color_logits, abstain_logit], dim=-1)
        return self.cmp_head(st_cmp), chain_logits


def build_5e(model, ents, n, rng) -> List[Dict]:
    """Balanced regime: comparison (preserve) + chaining 50/50 answerable/broken."""
    eps = []
    for _ in range(n):
        r = rng.random()
        if r < 0.3:
            eps.append(compare_ep(model, ents, rng))
        else:
            cr = rng.random()
            if cr < 0.25:
                eps.append(chain_ep(model, ents, rng, "normal"))
            elif cr < 0.5:
                eps.append(chain_ep(model, ents, rng, "shuffled"))
            else:
                eps.append(chain_ep(model, ents, rng, "unanswerable"))
    return eps


def evaluate(op, eps) -> Dict:
    V, K, P, M, QK, OP, GOLD = pack(eps)
    op.eval()
    with torch.no_grad():
        cmp_l, chain_l = op(V, K, P, M, QK, OP)
    b = {"answerable_correct": [0, 0], "answerable_overabstain": [0, 0], "broken_abstain": [0, 0],
         "grounded_correct": [0, 0], "compare": [0, 0]}
    for i, e in enumerate(eps):
        if e["op"] == OP_COMPARE:
            b["compare"][0] += int(int(torch.argmax(cmp_l[i]).item()) == e["gold"]); b["compare"][1] += 1
        else:
            pred = int(torch.argmax(chain_l[i]).item())
            if e["variant"] == "unanswerable":
                b["broken_abstain"][0] += int(pred == ABSTAIN); b["broken_abstain"][1] += 1
            elif e["variant"] == "repointed":
                b["grounded_correct"][0] += int(pred == e["gold"]); b["grounded_correct"][1] += 1
            else:
                b["answerable_correct"][0] += int(pred == e["gold"]); b["answerable_correct"][1] += 1
                b["answerable_overabstain"][0] += int(pred == ABSTAIN); b["answerable_overabstain"][1] += 1
    return {k: (c[0] / c[1] if c[1] else None) for k, c in b.items()}


def train(train_eps, seed, steps, lr) -> nn.Module:
    random.seed(seed); torch.manual_seed(seed)
    op = HonestOp().to(DEVICE)
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


def dist(xs):
    xs = [x for x in xs if x is not None]
    return {"min": round(min(xs), 4), "median": round(median(xs), 4), "max": round(max(xs), 4),
            "std": round(pstdev(xs), 4) if len(xs) > 1 else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 5e honest traversal")
    ap.add_argument("--ckpt", default="runs/multiobject/ckpt_multiobject.pt")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--n-train", type=int, default=900)
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    (RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
    print(SEP, flush=True)
    print(f"[INFO] Stage 5e honest traversal | device={DEVICE}", flush=True)
    model = load_model(args.ckpt)
    print("[INFO] FROZEN encoder; structural traversal + confT-driven abstain (detection-based)", flush=True)

    split_rng = random.Random(20260620)
    sh = ENTITIES[:]; split_rng.shuffle(sh)
    train_ents, held_ents = sh[:14], sh[14:]
    print(f"[INFO] entity split: 14 train / {len(held_ents)} held-out", flush=True)

    print("[INFO] precomputing banks + pointer banks (balanced answerable/broken)...", flush=True)
    drng = random.Random(7)
    train_eps = build_5e(model, train_ents, args.n_train, drng)
    eval_eps = build_5e(model, held_ents, args.n_eval, drng)
    repoint_eps = [t for e in eval_eps if e["op"] == OP_CHAIN and e["variant"] in ("normal", "shuffled")
                   for t in [repoint(model, e)] if t]
    n_broken = sum(1 for e in eval_eps if e.get("variant") == "unanswerable")
    n_ans = sum(1 for e in eval_eps if e.get("variant") in ("normal", "shuffled"))
    print(f"[INFO] eval: {n_ans} answerable / {n_broken} broken / {len(repoint_eps)} re-pointed", flush=True)

    g_pointer = pointer_generalizes(eval_eps, "structural")
    print(f"[INFO] G_POINTER_GENERALIZES structural={g_pointer:.3f}", flush=True)

    if args.smoke:
        op = train(train_eps, 0, 5, args.lr)
        ev = evaluate(op, eval_eps)
        print(f"  [SMOKE] broken_abstain={ev['broken_abstain']:.3f} over_abstain={ev['answerable_overabstain']:.3f} "
              f"ans_correct={ev['answerable_correct']:.3f}", flush=True)
        return 0

    runs = []
    for seed in range(args.seeds):
        op = train(train_eps, seed, args.steps, args.lr)
        ev = evaluate(op, eval_eps); gr = evaluate(op, repoint_eps)
        runs.append({"ev": ev, "grounded": gr["grounded_correct"]})
        print(f"  seed {seed}: broken_abstain={ev['broken_abstain']:.3f} over_abstain={ev['answerable_overabstain']:.3f} "
              f"ans_correct={ev['answerable_correct']:.3f} grounded={gr['grounded_correct']:.3f} compare={ev['compare']:.3f}", flush=True)

    broken_abstain = dist([r["ev"]["broken_abstain"] for r in runs])
    over_abstain = dist([r["ev"]["answerable_overabstain"] for r in runs])
    ans_correct = dist([r["ev"]["answerable_correct"] for r in runs])
    grounded = dist([r["grounded"] for r in runs])
    comparison = dist([r["ev"]["compare"] for r in runs])

    g_abstain_broken = broken_abstain["median"] >= 0.80
    g_no_over = over_abstain["median"] <= 0.10
    g_chaining = ans_correct["median"] >= 0.95 and grounded["median"] >= 0.90
    g_comparison = comparison["median"] >= 0.80
    g_pointer_ok = g_pointer >= 0.90

    if g_abstain_broken and g_no_over and g_chaining and g_comparison and g_pointer_ok:
        verdict = "STAGE_5_HONEST_TRAVERSAL"
    elif (g_abstain_broken or g_no_over) and g_chaining:
        verdict = "STAGE_5_ABSTAIN_PARTIAL"
    else:
        verdict = "STAGE_5_ABSTAIN_REFUTED"

    out = {"verdict": verdict, "ckpt": args.ckpt,
           "gates": {
               "G_ABSTAIN_BROKEN": {"passed": bool(g_abstain_broken), "dist": broken_abstain, "bar": 0.80,
                                    "evidence": "broken chain (target absent) -> abstain"},
               "G_NO_OVER_ABSTAIN": {"passed": bool(g_no_over), "dist": over_abstain, "bar": 0.10,
                                     "evidence": "answerable chain -> NOT abstain (anti-collapse control)"},
               "G_CHAINING_PRESERVED": {"passed": bool(g_chaining), "answerable_acc": ans_correct,
                                        "grounded": grounded, "bars": [0.95, 0.90]},
               "G_COMPARISON_PRESERVED": {"passed": bool(g_comparison), "dist": comparison, "bar": 0.80},
               "G_POINTER_GENERALIZES": {"passed": bool(g_pointer_ok), "structural": g_pointer, "bar": 0.90},
               "G_SINGLE_FACT_PRESERVED": "0/140 (frozen encoder = ckpt_multiobject)",
               "G_IN_MEMORY": {"passed": True, "evidence": "operation reads banks + pointer bank + query keys only"},
           },
           "per_seed": runs,
           "scope": ("honest multi-hop traversal over persisted banks; FROZEN encoder; confT-driven detection-based "
                     "abstain; balanced answerable/broken; held-out; single machine. NOT generality, NOT free-text. "
                     "Target identification template-given (addressing+honesty tested, not free-text extraction)."),
           "meaning": {
               "STAGE_5_HONEST_TRAVERSAL": "the navigable graph is now HONEST: traversal proven AND it abstains on "
                   "broken chains (>=0.80) AND answers when answerable (over-abstain <=0.10) -> wrong_commit=0 "
                   "extended to multi-hop; the thinking-in-memory loop is closed HONESTLY at small scale.",
               "STAGE_5_ABSTAIN_PARTIAL": "abstain improved but the DUAL gate or a preservation gate is short.",
               "STAGE_5_ABSTAIN_REFUTED": "abstain cannot be made selective AND generalizable -> the model cannot "
                   "honestly detect broken traversal at this setup."}[verdict]}
    (RUN_DIR / "results" / "verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(SEP, flush=True)
    print("[INFO] LEAD WITH THE DUAL ABSTAIN GATE:", flush=True)
    print(f"  G_ABSTAIN_BROKEN: {broken_abstain['median']} (bar >=0.80) -> {'PASS' if g_abstain_broken else 'FAIL'}", flush=True)
    print(f"  G_NO_OVER_ABSTAIN: {over_abstain['median']} (bar <=0.10) -> {'PASS' if g_no_over else 'FAIL'}", flush=True)
    print(f"  G_CHAINING_PRESERVED: ans_acc={ans_correct['median']} grounded={grounded['median']} -> {'PASS' if g_chaining else 'FAIL'}", flush=True)
    print(f"  G_COMPARISON_PRESERVED: {comparison['median']} | G_POINTER_GENERALIZES: {g_pointer}", flush=True)
    print(f"[INFO] VERDICT: {verdict}", flush=True)
    print("STAGE5E_JSON " + json.dumps({"verdict": verdict, "broken_abstain": broken_abstain["median"],
          "over_abstain": over_abstain["median"], "answerable_acc": ans_correct["median"],
          "grounded": grounded["median"], "comparison": comparison["median"]}), flush=True)
    return 0 if verdict == "STAGE_5_HONEST_TRAVERSAL" else 1


if __name__ == "__main__":
    sys.exit(main())
