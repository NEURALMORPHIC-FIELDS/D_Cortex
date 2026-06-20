# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# STAGE 5 (operate-over-memory), first piece: COMPARISON. Stage C showed the sealed decoder reads
# memory by a SUM of slots (blends objects) and does not learn comparison even when trained (C2 at
# chance). The vision's layer 5 is therefore an EXPLICIT OPERATION MODULE over the separable object
# representations - exactly what Stage C's diagnosis called for ("BUILD the cognitive-operation
# layer"), NOT a decoder modification (dcortex/ is sealed).
#
# This cert measures the FIRST question (measure-before-build, like the step-1 readout ladder): is
# COMPARISON reachable by an explicit operation module reading two SEPARABLE object representations,
# GENERALIZING to held-out entities? Comparison = an ORDERING operation (tiny<small<big<huge);
# separability gives distinction (red!=blue), not order - so the module must LEARN the size order and
# transfer it across entities. A/B: ckpt_multiattr (non-separable) vs ckpt_multiobject (separable)
# isolates whether separability is the enabler.
#
# Diagnostics:
#   ORDINALITY : a head reading ONE object rep -> its size rank (0..3), held-out. Tests value identity.
#   COMPARISON : a module reading (v_A, v_B) of two co-occurring objects -> "A bigger?", held-out.
#                Symmetric pairs (both orders) to remove positional bias. This is the operation.

import argparse
import contextlib
import io
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

import tiktoken
from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model
from scripts.certify_stage_i_extraction import (
    ENC, DEVICE, ENTITIES, SIZES, ATTR_FACT, find_entity_pos, enc_hidden,
)

SEP = "=" * 70
RUN_DIR = REPO_ROOT / "runs" / "stage5_comparison"
SIZE_RANK = {s: i for i, s in enumerate(SIZES)}            # tiny<small<big<huge -> 0..3


def build_pairs(model, ents: List[str], n: int, rng: random.Random) -> List[Dict]:
    """Multi-entity SIZE texts; cache the two entities' separable reps + their size ranks."""
    out = []
    for _ in range(n):
        if len(ents) < 2:
            break
        e1, e2 = rng.sample(ents, 2)
        s1, s2 = rng.sample(SIZES, 2)
        text = ATTR_FACT["size"].format(e=e1, v=s1) + " " + ATTR_FACT["size"].format(e=e2, v=s2)
        ids = ENC.encode_ordinary(text)
        p1, p2 = find_entity_pos(ids, e1), find_entity_pos(ids, e2)
        if p1 is None or p2 is None or p1 == p2:
            continue
        h = enc_hidden(model, ids)
        out.append({"vA": h[p1], "vB": h[p2], "rA": SIZE_RANK[s1], "rB": SIZE_RANK[s2]})
    return out


class OrdinalHead(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 4))

    def forward(self, v):
        return self.net(v)


class CompareHead(nn.Module):
    """Operation module: (v_A, v_B) -> P(A bigger). Symmetric usage removes positional bias."""
    def __init__(self, d: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(),
                                 nn.Linear(128, 1))

    def forward(self, va, vb):
        return self.net(torch.cat([va, vb], dim=-1)).squeeze(-1)


def run(model, train_pairs, eval_pairs, seed: int, steps: int, lr: float) -> Dict:
    random.seed(seed); torch.manual_seed(seed)
    d = train_pairs[0]["vA"].shape[0]
    ordh = OrdinalHead(d).to(DEVICE)
    cmph = CompareHead(d).to(DEVICE)
    opt = torch.optim.AdamW(list(ordh.parameters()) + list(cmph.parameters()), lr=lr, weight_decay=0.0)

    VA = torch.stack([p["vA"] for p in train_pairs]).to(DEVICE)
    VB = torch.stack([p["vB"] for p in train_pairs]).to(DEVICE)
    rA = torch.tensor([p["rA"] for p in train_pairs], device=DEVICE)
    rB = torch.tensor([p["rB"] for p in train_pairs], device=DEVICE)
    a_big = (rA > rB).float()                              # excludes equal ranks (rng.sample distinct sizes)

    for _ in range(steps):
        opt.zero_grad()
        # ordinality on both slots
        loss_ord = F.cross_entropy(ordh(VA), rA) + F.cross_entropy(ordh(VB), rB)
        # comparison, symmetric (A,B)->a_big and (B,A)->1-a_big
        logit_ab = cmph(VA, VB)
        logit_ba = cmph(VB, VA)
        loss_cmp = F.binary_cross_entropy_with_logits(logit_ab, a_big) + \
            F.binary_cross_entropy_with_logits(logit_ba, 1.0 - a_big)
        (loss_ord + loss_cmp).backward()
        opt.step()

    ordh.eval(); cmph.eval()
    with torch.no_grad():
        eVA = torch.stack([p["vA"] for p in eval_pairs]).to(DEVICE)
        eVB = torch.stack([p["vB"] for p in eval_pairs]).to(DEVICE)
        erA = torch.tensor([p["rA"] for p in eval_pairs], device=DEVICE)
        erB = torch.tensor([p["rB"] for p in eval_pairs], device=DEVICE)
        ea_big = (erA > erB).float()
        ord_acc = ((torch.argmax(ordh(eVA), 1) == erA).float().mean().item() +
                   (torch.argmax(ordh(eVB), 1) == erB).float().mean().item()) / 2
        # symmetric comparison accuracy
        pred_ab = (torch.sigmoid(cmph(eVA, eVB)) > 0.5).float()
        pred_ba = (torch.sigmoid(cmph(eVB, eVA)) > 0.5).float()
        cmp_acc = ((pred_ab == ea_big).float().mean().item() +
                   (pred_ba == (1.0 - ea_big)).float().mean().item()) / 2
    return {"ordinality_acc": ord_acc, "comparison_acc": cmp_acc}


def dist(xs):
    return {"min": round(min(xs), 4), "median": round(median(xs), 4), "max": round(max(xs), 4),
            "std": round(pstdev(xs), 4) if len(xs) > 1 else 0.0}


def load(ckpt):
    with contextlib.redirect_stdout(io.StringIO()):
        m = DCortexV2Model(DCortexConfig()).to(DEVICE).eval()
    ck = torch.load(ckpt, map_location=DEVICE)
    m.load_state_dict(ck["model"])
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 5 comparison operation probe (A/B)")
    ap.add_argument("--separable", default="runs/multiobject/ckpt_multiobject.pt")
    ap.add_argument("--baseline", default="runs/stage_u/results/ckpt_multiattr.pt")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-eval", type=int, default=240)
    args = ap.parse_args()
    (RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
    print(SEP, flush=True)
    print(f"[INFO] Stage 5 comparison operation probe | device={DEVICE}", flush=True)

    split_rng = random.Random(20260620)
    shuffled = ENTITIES[:]; split_rng.shuffle(shuffled)
    train_ents, held_ents = shuffled[:20], shuffled[20:]
    print(f"[INFO] entity split: 20 train / 10 held-out", flush=True)

    out = {"gate_comparison_heldout": 0.80, "arms": {}}
    for arm, ckpt in (("baseline_ckpt_multiattr", args.baseline), ("separable_ckpt_multiobject", args.separable)):
        model = load(ckpt)
        drng = random.Random(7)
        train_pairs = build_pairs(model, train_ents, args.n_train, drng)
        eval_pairs = build_pairs(model, held_ents, args.n_eval, drng)
        runs = [run(model, train_pairs, eval_pairs, s, args.steps, args.lr) for s in range(args.seeds)]
        agg = {k: dist([r[k] for r in runs]) for k in ("ordinality_acc", "comparison_acc")}
        out["arms"][arm] = {"agg": agg, "per_seed": runs, "ckpt": ckpt,
                            "train_pairs": len(train_pairs), "eval_pairs": len(eval_pairs)}
        print(f"  [{arm:28s}] held-out ordinality={agg['ordinality_acc']['median']:.4f} "
              f"comparison={agg['comparison_acc']['median']:.4f} (chance 0.5)", flush=True)

    base_c = out["arms"]["baseline_ckpt_multiattr"]["agg"]["comparison_acc"]["median"]
    sep_c = out["arms"]["separable_ckpt_multiobject"]["agg"]["comparison_acc"]["median"]
    sep_pass = sep_c >= 0.80
    enabler = (sep_c - base_c) >= 0.10

    if sep_pass and enabler:
        verdict = "COMPARISON_REACHABLE_SEPARABILITY_ENABLES"
    elif sep_pass:
        verdict = "COMPARISON_REACHABLE"
    elif sep_c >= 0.65:
        verdict = "COMPARISON_PARTIAL"
    else:
        verdict = "COMPARISON_NOT_REACHABLE_NEED_ORDINALITY"
    out["verdict"] = verdict
    out["meaning"] = {
        "COMPARISON_REACHABLE_SEPARABILITY_ENABLES": "an explicit operation module over the SEPARABLE object "
            "reps solves comparison held-out, and separability is the enabler (>=0.10 over baseline) -> layer 5 "
            "operation is buildable on the substrate; Stage C's C2 failure was the missing operation module, not "
            "the representation. Next: wire operation for robust chaining + multi-object memory read.",
        "COMPARISON_REACHABLE": "operation module solves comparison held-out on the separable base (separability "
            "not clearly the differentiator vs baseline).",
        "COMPARISON_PARTIAL": "operation module lifts comparison above chance but not to 0.80 -> ordinal structure "
            "is weak; an explicit ordinality objective on the value space is needed.",
        "COMPARISON_NOT_REACHABLE_NEED_ORDINALITY": "even an explicit module cannot order the sizes held-out -> the "
            "value space lacks ordinal structure; inject ordinality (ranking objective) before the operation.",
    }[verdict]
    out["scope"] = ("Stage 5 first piece; FROZEN bases; explicit operation module (the layer Stage C said must be "
                    "BUILT) over the separable per-entity object reps; held-out entities; A/B separable vs baseline. "
                    "Comparison reads two co-occurring object reps. NOT a decoder modification (dcortex/ sealed).")
    (RUN_DIR / "results" / "verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(SEP, flush=True)
    print(f"  comparison held-out: baseline={base_c:.4f}  separable={sep_c:.4f}  gate=0.80", flush=True)
    print(f"[INFO] VERDICT: {verdict}", flush=True)
    print("STAGE5_JSON " + json.dumps({"verdict": verdict, "baseline_comparison": base_c,
          "separable_comparison": sep_c}), flush=True)
    return 0 if sep_pass else 1


if __name__ == "__main__":
    sys.exit(main())
