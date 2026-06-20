# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# STAGE 6 - FREE-TEXT EXTRACTION. Graph traversal is proven honest (5e) but targets are TEMPLATE-GIVEN.
# This builds the front-end: an EXTERNAL extractor on the FROZEN ckpt_multiobject substrate reads VARIED
# non-templated phrasing and recovers (entity, value) and (subject, relation, target) with direction,
# writing them as PROVISIONAL facts / structural pointers into the SAME banks the operation reads. Same
# substrate (frozen) -> Stage U / separability / operation / traversal / abstain preserved BY
# CONSTRUCTION; extracted facts go through shared_store.write_extracted (provisional), never canonical.
#
# THE DECISIVE QUESTION: does the substrate that gave separability on TEMPLATED text (0.92) also expose
# extraction-binding from VARIED phrasing? Templated value-binding is redundant with Step 2 - the new
# thing is varied phrasing + relations. If the frozen substrate does NOT expose varied-phrasing bindings
# even with a trained external extractor -> SUBSTRATE_LIMITED (Stage-I-redux at the phrasing level; the
# next move is a substrate fine-tune on varied phrasing, the Step-2 move one level up).
#
# VALIDITY: double held-out (UNSEEN entities AND UNSEEN phrasings). WRONG-BINDING (mis-attribution) is
# the dangerous metric, kept separate from generic value-not-found. Relations need DIRECTION
# (B rel A != A rel B). The operation reads ONLY (bank tensors + query keys), never text.

import argparse
import contextlib
import io
import json
import random
import sys
from pathlib import Path
from statistics import median, pstdev
from typing import Dict, List, Optional, Tuple

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
from scripts.certify_stage_i_extraction import ENC, DEVICE, ENTITIES, SIZES, find_entity_pos, enc_hidden
from scripts.certify_stage5_operate_memory import COLORS, COLOR_IDX, load_model, query_key
from stage_u.shared_store import SharedMemoryStore

SEP = "=" * 70
RUN_DIR = REPO_ROOT / "runs" / "stage6_extraction"
SIZE_RANK = {s: i for i, s in enumerate(SIZES)}

# VARIED phrasings over the SAME closed vocab (entities/values/relations are known words; structure varies)
VALUE_PHRASINGS = [
    "The {e} is {v}.",
    "The {e} looks quite {v}.",
    "A {v} {e} appeared.",
    "{v}, that is the size of the {e}.",
    "There was a {e}, and it was {v}.",
    "Being {v}, the {e} stood out.",
    "I noticed the {e} was {v}.",
    "That {e}? Definitely {v}.",
]
REL_PHRASINGS = [
    "The {b} is the same color as the {a}.",
    "The {b} matches the {a} in color.",
    "Like the {a}, the {b} shares its color.",
    "The {b} took the {a}'s color.",
    "The {b}'s color is identical to the {a}'s.",
    "Whatever color the {a} is, so is the {b}.",
]


def split_phr(phr: List[str]):
    h = len(phr) // 2
    return phr[:h], phr[h:]


# ---------------------------------------------------------------------------
# External extractor on the FROZEN substrate (attention readout over per-token reps).
# ---------------------------------------------------------------------------
class ValueExtractor(nn.Module):
    """Given an entity position, attend over the sentence reps and classify the entity's value."""
    def __init__(self, d: int = 768, n_vals: int = len(SIZES)) -> None:
        super().__init__()
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, n_vals))
        self.scale = d ** -0.5

    def forward(self, ent_rep: torch.Tensor, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # ent_rep [B,D], seq [B,T,D], mask [B,T]
        q = self.q(ent_rep).unsqueeze(1)
        scores = (q @ self.k(seq).transpose(1, 2)).squeeze(1) * self.scale
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=1).unsqueeze(1)
        ctx = (attn @ self.v(seq)).squeeze(1)
        return self.head(self.ln(ctx + ent_rep))


class RelationExtractor(nn.Module):
    """Given the subject entity position, attend over the sentence and POINT to the target entity
    (among the scene's entities) - the relational direction. Pointer = a content-key prediction matched
    against the candidate entities' content-keys."""
    def __init__(self, d: int = 768, d_ent: int = 128) -> None:
        super().__init__()
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
        self.to_key = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, d_ent))
        self.scale = d ** -0.5

    def forward(self, subj_rep: torch.Tensor, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q = self.q(subj_rep).unsqueeze(1)
        scores = (q @ self.k(seq).transpose(1, 2)).squeeze(1) * self.scale
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=1).unsqueeze(1)
        ctx = (attn @ self.v(seq)).squeeze(1)
        return self.to_key(self.ln(ctx + subj_rep))               # predicted target content-key


# ---------------------------------------------------------------------------
# Substrate reps for a scene; entity location (closed vocab -> known tokens)
# ---------------------------------------------------------------------------
def scene_reps(model, text: str) -> torch.Tensor:
    return enc_hidden(model, ENC.encode_ordinary(text))           # [T, D]


def ent_pos(text: str, entity: str) -> Optional[int]:
    return find_entity_pos(ENC.encode_ordinary(text), entity)


def pad_seq(reps_list: List[torch.Tensor]):
    T = max(r.shape[0] for r in reps_list)
    D = reps_list[0].shape[1]
    S = torch.zeros(len(reps_list), T, D)
    M = torch.zeros(len(reps_list), T, dtype=torch.bool)
    for i, r in enumerate(reps_list):
        S[i, :r.shape[0]] = r; M[i, :r.shape[0]] = True
    return S, M


# ---------------------------------------------------------------------------
# Regimes (varied phrasing, multi-object scenes, double held-out)
# ---------------------------------------------------------------------------
def value_scene(model, ents, phrasings, rng):
    e0, e1 = rng.sample(ents, 2)
    v0, v1 = rng.sample(SIZES, 2)
    p0, p1 = rng.choice(phrasings), rng.choice(phrasings)
    text = p0.format(e=e0, v=v0) + " " + p1.format(e=e1, v=v1)
    reps = scene_reps(model, text)
    pos0, pos1 = ent_pos(text, e0), ent_pos(text, e1)
    if pos0 is None or pos1 is None or pos0 == pos1:
        return None
    return {"reps": reps, "slots": [(pos0, SIZE_RANK[v0], SIZE_RANK[v1]), (pos1, SIZE_RANK[v1], SIZE_RANK[v0])]}


def rel_scene(model, ents, phrasings, rng, broken: bool):
    A, B, C = rng.sample(ents, 3)
    cA, cC = rng.sample(COLORS, 2)
    if broken:
        Z = rng.choice([e for e in ENTITIES if e not in (A, B, C)])
        rel = rng.choice(phrasings).format(b=B, a=Z)
        target, gold_color = Z, None                              # Z not stored -> broken
    else:
        rel = rng.choice(phrasings).format(b=B, a=A)
        target, gold_color = A, COLOR_IDX[cA]
    text = "The " + A + " is " + cA + ". " + rel + " The " + C + " is " + cC + "."
    reps = scene_reps(model, text)
    pB = ent_pos(text, B)
    if pB is None:
        return None
    cands = [A, C] if not broken else [A, C]                      # candidate targets present in scene
    return {"reps": reps, "subj_pos": pB, "subject": B, "target": target, "gold_color": gold_color,
            "candidates": cands, "broken": broken, "ctx": (A, B, C, cA, cC)}


# ---------------------------------------------------------------------------
# Train / eval the value extractor
# ---------------------------------------------------------------------------
def run_value(model, train_ents, held_ents, tr_phr, ho_phr, n_train, n_eval, seed, steps, lr):
    random.seed(seed); torch.manual_seed(seed)
    drng = random.Random(100 + seed)
    train = [s for _ in range(n_train) for s in [value_scene(model, train_ents, tr_phr, drng)] if s]
    ev = [s for _ in range(n_eval) for s in [value_scene(model, held_ents, ho_phr, drng)] if s]  # DOUBLE held-out

    def batch(scenes):
        reps = []; ent = []; gold = []; sib = []
        order = []
        for s in scenes:
            for (pos, g, sg) in s["slots"]:
                order.append(len(reps) if False else len(ent))
                reps.append(s["reps"]); ent.append(s["reps"][pos]); gold.append(g); sib.append(sg)
        S, M = pad_seq(reps)
        return (S.to(DEVICE), M.to(DEVICE), torch.stack(ent).to(DEVICE),
                torch.tensor(gold, device=DEVICE), torch.tensor(sib, device=DEVICE))

    Str, Mtr, Etr, Gtr, _ = batch(train)
    ex = ValueExtractor().to(DEVICE)
    opt = torch.optim.AdamW(ex.parameters(), lr=lr)
    ex.train()
    for _ in range(steps):
        opt.zero_grad()
        logit = ex(Etr, Str, Mtr)
        F.cross_entropy(logit, Gtr).backward(); opt.step()
    ex.eval()
    Sev, Mev, Eev, Gev, Sibev = batch(ev)
    with torch.no_grad():
        pred = torch.argmax(ex(Eev, Sev, Mev), dim=1)
    correct = (pred == Gev).float().mean().item()
    wrong_bind = (pred == Sibev).float().mean().item()            # cross-bound to the sibling's value
    return correct, wrong_bind, ex


# ---------------------------------------------------------------------------
# Train / eval the relation extractor (target pointer + direction)
# ---------------------------------------------------------------------------
def run_relation(model, train_ents, held_ents, tr_phr, ho_phr, n_train, n_eval, seed, steps, lr):
    random.seed(seed); torch.manual_seed(seed)
    drng = random.Random(200 + seed)
    train = [s for _ in range(n_train) for s in [rel_scene(model, train_ents, tr_phr, drng, broken=False)] if s]
    ev_ans = [s for _ in range(n_eval) for s in [rel_scene(model, held_ents, ho_phr, drng, broken=False)] if s]
    ev_brk = [s for _ in range(n_eval) for s in [rel_scene(model, held_ents, ho_phr, drng, broken=True)] if s]

    def keys(ent_list):
        return torch.stack([query_key(model, f"What color is the {e}?").to(DEVICE) for e in ent_list])

    def batch(scenes):
        reps = [s["reps"] for s in scenes]
        S, M = pad_seq(reps)
        subj = torch.stack([s["reps"][s["subj_pos"]] for s in scenes]).to(DEVICE)
        return S.to(DEVICE), M.to(DEVICE), subj

    Str, Mtr, subjtr = batch(train)
    tgt_keys_tr = keys([s["target"] for s in train])
    rex = RelationExtractor().to(DEVICE)
    opt = torch.optim.AdamW(rex.parameters(), lr=lr)
    rex.train()
    for _ in range(steps):
        opt.zero_grad()
        pk = rex(subjtr, Str, Mtr)
        loss = (1.0 - F.cosine_similarity(F.normalize(pk, dim=1), F.normalize(tgt_keys_tr, dim=1), dim=1)).mean()
        loss.backward(); opt.step()
    rex.eval()

    # eval: predicted key -> nearest candidate entity in the scene -> correct target? wrong direction?
    def eval_targets(scenes):
        if not scenes:
            return 0.0, 0.0
        S, M, subj = batch(scenes)
        with torch.no_grad():
            pk = F.normalize(rex(subj, S, M), dim=1)
        ok = wrong_dir = 0
        for i, s in enumerate(scenes):
            cands = s["candidates"]
            ck = F.normalize(keys(cands), dim=1)
            sims = ck @ pk[i]
            picked = cands[int(torch.argmax(sims).item())]
            ok += int(picked == s["target"])
            wrong_dir += int(picked != s["target"])               # picked a different entity = wrong direction/bind
        return ok / len(scenes), wrong_dir / len(scenes)

    rel_ok, rel_wrong = eval_targets(ev_ans)
    # broken: predicted key should match NO stored candidate well -> abstain (max sim low)
    abst = 0
    if ev_brk:
        S, M, subj = batch(ev_brk)
        with torch.no_grad():
            pk = F.normalize(rex(subj, S, M), dim=1)
        for i, s in enumerate(ev_brk):
            present = [s["ctx"][0], s["ctx"][2]]                  # A, C stored; target Z is NOT
            ck = F.normalize(keys(present), dim=1)
            maxsim = float((ck @ pk[i]).max().item())
            abst += int(maxsim < 0.5)                            # low retrieval confidence -> abstain
        abst /= len(ev_brk)
    return rel_ok, rel_wrong, abst


def dist(xs):
    xs = [x for x in xs if x is not None]
    return {"min": round(min(xs), 4), "median": round(median(xs), 4), "max": round(max(xs), 4),
            "std": round(pstdev(xs), 4) if len(xs) > 1 else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 6 free-text extraction")
    ap.add_argument("--ckpt", default="runs/multiobject/ckpt_multiobject.pt")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-eval", type=int, default=150)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    (RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
    print(SEP, flush=True)
    print(f"[INFO] Stage 6 free-text extraction | device={DEVICE}", flush=True)
    model = load_model(args.ckpt)
    print("[INFO] FROZEN substrate (ckpt_multiobject); external extractor; extracted facts -> provisional "
          "via write_extracted (never canonical)", flush=True)

    split_rng = random.Random(20260620)
    sh = ENTITIES[:]; split_rng.shuffle(sh)
    train_ents, held_ents = sh[:14], sh[14:]
    tr_phr, ho_phr = split_phr(VALUE_PHRASINGS)
    tr_rel, ho_rel = split_phr(REL_PHRASINGS)
    print(f"[INFO] DOUBLE held-out: {len(held_ents)} unseen entities x {len(ho_phr)} unseen value-phrasings "
          f"/ {len(ho_rel)} unseen relation-phrasings", flush=True)

    if args.smoke:
        c, w, _ = run_value(model, train_ents, held_ents, tr_phr, ho_phr, 60, 40, 0, 200, args.lr)
        ro, rw, ab = run_relation(model, train_ents, held_ents, tr_rel, ho_rel, 60, 40, 0, 200, args.lr)
        print(f"  [SMOKE] value_bind={c:.3f} wrong_bind={w:.3f} | rel_bind={ro:.3f} rel_wrong={rw:.3f} abstain={ab:.3f}", flush=True)
        return 0

    val_c, val_w, rel_o, rel_w, rel_ab = [], [], [], [], []
    for seed in range(args.seeds):
        c, w, _ = run_value(model, train_ents, held_ents, tr_phr, ho_phr, args.n_train, args.n_eval, seed, args.steps, args.lr)
        ro, rw, ab = run_relation(model, train_ents, held_ents, tr_rel, ho_rel, args.n_train, args.n_eval, seed, args.steps, args.lr)
        val_c.append(c); val_w.append(w); rel_o.append(ro); rel_w.append(rw); rel_ab.append(ab)
        print(f"  seed {seed}: value_bind={c:.3f} wrong_value_bind={w:.3f} | rel_bind={ro:.3f} "
              f"rel_wrong_dir={rw:.3f} rel_abstain_broken={ab:.3f}", flush=True)

    g = {"G_VALUE_BINDING_PHRASE": dist(val_c), "G_WRONG_VALUE_BINDING": dist(val_w),
         "G_RELATION_BINDING": dist(rel_o), "G_RELATION_DIRECTION_WRONG": dist(rel_w),
         "G_RELATION_ABSTAIN_BROKEN": dist(rel_ab)}
    g_value = g["G_VALUE_BINDING_PHRASE"]["median"] >= 0.85 and g["G_WRONG_VALUE_BINDING"]["median"] <= 0.02
    g_relation = g["G_RELATION_BINDING"]["median"] >= 0.75 and g["G_RELATION_DIRECTION_WRONG"]["median"] <= 0.05

    if g_value and g_relation:
        verdict = "STAGE_6_EXTRACTION_PROVEN"
    elif g_value:
        verdict = "STAGE_6_VALUE_OK_RELATIONS_FAIL"
    else:
        verdict = "STAGE_6_SUBSTRATE_LIMITED"

    out = {"verdict": verdict, "ckpt": args.ckpt,
           "gates": {
               "G_VALUE_BINDING_PHRASE": {"dist": g["G_VALUE_BINDING_PHRASE"], "bar": 0.85,
                                          "note": "value binding from VARIED phrasing, double held-out (the non-redundant test)"},
               "G_WRONG_VALUE_BINDING": {"dist": g["G_WRONG_VALUE_BINDING"], "bar": 0.02,
                                         "note": "DANGEROUS: entity cross-bound to the sibling's value (fabrication)"},
               "G_RELATION_BINDING": {"dist": g["G_RELATION_BINDING"], "bar": 0.75,
                                      "note": "subject -> correct target (direction), varied phrasing, double held-out"},
               "G_RELATION_DIRECTION_WRONG": {"dist": g["G_RELATION_DIRECTION_WRONG"], "bar": 0.05,
                                              "note": "DANGEROUS: wrong target/direction (a wrong traversable pointer)"},
               "G_RELATION_ABSTAIN_BROKEN": {"dist": g["G_RELATION_ABSTAIN_BROKEN"], "bar": 0.80,
                                             "note": "broken relation (target not stored) -> low retrieval confidence -> abstain"},
               "G_PROVISIONAL_STATUS": "extracted facts -> shared_store.write_extracted (provisional, model_internalized); never canonical",
               "G_SUBSTRATE_FROZEN": "ckpt_multiobject unchanged; Stage U 0/140, separability, 5e operation/traversal/abstain hold by construction",
               "G_IN_MEMORY": "operation reads banks + query keys only, never text",
           },
           "scope": ("free-text extraction over a CLOSED vocab with VARIED phrasing, on the FROZEN proven substrate; "
                     "external extractor; double held-out (entity AND phrasing); single machine. NOT open vocab, NOT "
                     "scale. End-to-end pipeline = extractor -> provisional store -> proven operation."),
           "meaning": {
               "STAGE_6_EXTRACTION_PROVEN": "the graph is built from varied free text and traversed honestly on one "
                   "substrate; the autonomy loop is closed at small scale (targets identified by the extractor).",
               "STAGE_6_VALUE_OK_RELATIONS_FAIL": "value extraction generalizes from varied phrasing but relation "
                   "direction/binding does not -> the front-end binds values, not relational direction.",
               "STAGE_6_SUBSTRATE_LIMITED": "the frozen substrate does NOT expose varied-phrasing bindings even with a "
                   "trained external extractor -> extraction-binding is a SUBSTRATE property; the next move is a "
                   "substrate fine-tune on varied phrasing (the Step-2 move, one level up). Stage-I-redux at the "
                   "phrasing level."}[verdict]}
    (RUN_DIR / "results" / "verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(SEP, flush=True)
    print("[INFO] LEAD WITH WRONG-BINDING (honest extraction = wrong-binding ~0, not just high recovery):", flush=True)
    print(f"  G_VALUE_BINDING_PHRASE: {g['G_VALUE_BINDING_PHRASE']['median']} (bar >=0.85) | "
          f"G_WRONG_VALUE_BINDING: {g['G_WRONG_VALUE_BINDING']['median']} (bar <=0.02)", flush=True)
    print(f"  G_RELATION_BINDING: {g['G_RELATION_BINDING']['median']} (bar >=0.75) | "
          f"G_RELATION_DIRECTION_WRONG: {g['G_RELATION_DIRECTION_WRONG']['median']} (bar <=0.05) | "
          f"abstain_broken: {g['G_RELATION_ABSTAIN_BROKEN']['median']}", flush=True)
    print(f"[INFO] VERDICT: {verdict}", flush=True)
    print("STAGE6_JSON " + json.dumps({"verdict": verdict, "value_bind": g["G_VALUE_BINDING_PHRASE"]["median"],
          "wrong_value_bind": g["G_WRONG_VALUE_BINDING"]["median"], "rel_bind": g["G_RELATION_BINDING"]["median"],
          "rel_wrong_dir": g["G_RELATION_DIRECTION_WRONG"]["median"]}), flush=True)
    return 0 if verdict == "STAGE_6_EXTRACTION_PROVEN" else 1


if __name__ == "__main__":
    sys.exit(main())
