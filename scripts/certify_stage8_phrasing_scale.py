# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# STAGE 8 - PHRASING-DIVERSITY SCALE (the entity-count analog from Step 2). Stage 7 refuted phrasing-
# invariance with a NAMED confound: only ~4 training phrasings (vs Step 2's 14+ entities). The mandatory
# arc re-verification PROVED the gentle substrate fine-tune is SAFE to iterate (foundation intact). So
# scale phrasing diversity to >=14 distinct phrasings and re-run the proven-safe pipeline.
#
# THE DECISIVE QUESTION: does enough phrasing diversity make the substrate abstract phrasing-INVARIANCE
# (generalize to STRUCTURALLY-DISTINCT held-out phrasings), as 14+ entities made it abstract entity-
# invariance? Calibration: entity-invariance generalizes over a CONTENT slot the substrate is built to
# hold; phrasing-invariance is generalization over SURFACE FORM (paraphrase-robustness) - a different,
# harder abstraction pretrained models get free from diverse text but a from-scratch closed-vocab
# substrate may lack the capacity for. Two clean outcomes: extraction closes, OR it is coupled to
# pretraining -> the next move merges the extraction and scale frontiers (port the proven arc to a
# pretrained base).
#
# VALIDITY-CRITICAL: the held-out phrasings are a STRUCTURALLY DISTINCT FAMILY (value-first / inverted /
# embedded), NOT near-duplicates of the training family. A pass on near-duplicates proves in-family
# generalization, NOT invariance, and would be a false positive on the wrong test. Double held-out:
# unseen entities AND unseen, structurally-distinct phrasings.

import argparse
import contextlib
import io
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median, pstdev
from typing import Dict, List, Optional

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

from scripts.certify_stage_i_extraction import ENC, DEVICE, ENTITIES, SIZES, find_entity_pos, enc_hidden
from scripts.certify_stage5_operate_memory import COLORS, COLOR_IDX, load_model, query_key
from scripts.certify_stage6_extraction import ValueExtractor, RelationExtractor, pad_seq

SEP = "=" * 70
RUN_DIR = REPO_ROOT / "runs" / "stage8_phrasing"
OUT_CKPT = REPO_ROOT / "runs" / "multiobject" / "ckpt_multiobject_phrase8.pt"
SIZE_RANK = {s: i for i, s in enumerate(SIZES)}
VAL_VOCAB = SIZES + COLORS
VAL_IDX = {v: i for i, v in enumerate(VAL_VOCAB)}
TRAINABLE = ("encoder.emb_norm", "encoder.blocks", "encoder.final_norm")
KEY_CACHE: Dict[str, torch.Tensor] = {}

# FAMILY A (training): entity-then-value, adjacent declarative (>=14)
VALUE_A = [
    "The {e} is {v}.", "The {e} looks {v}.", "The {e} appears {v}.", "The {e} seems {v}.",
    "The {e} is quite {v}.", "The {e} is rather {v}.", "The {e} is very {v}.", "That {e} is {v}.",
    "This {e} is {v}.", "The {e} was {v}.", "The {e} stayed {v}.", "The {e} remained {v}.",
    "Our {e} is {v}.", "The {e} here is {v}.", "The {e} is clearly {v}.", "The {e} is definitely {v}.",
]
# FAMILY B (held-out): value-first / inverted / distant / embedded - STRUCTURALLY DISTINCT
VALUE_B = [
    "{v} is what the {e} is.", "Quite {v}, that {e}.", "It was {v}, the {e}.", "A {v} thing, the {e}.",
    "What the {e} is, is {v}.", "So {v}, this {e}.", "As for the {e}, {v} describes it.",
    "Among them, the {e} stands out as {v}.", "Of all, the {e} turned out {v}.", "Truly {v} was the {e}.",
]
REL_A = [
    "The {b} is the same color as the {a}.", "The {b} matches the {a} in color.",
    "The {b} shares the {a}'s color.", "The {b} has the same color as the {a}.",
    "The {b} took the {a}'s color.", "The {b} is colored like the {a}.",
    "The {b} looks the same color as the {a}.", "In color, the {b} equals the {a}.",
    "The {b}'s color copies the {a}'s.", "The {b} mirrors the {a}'s color.",
    "The {b} is the {a}'s color.", "The {b} carries the {a}'s color.",
]
REL_B = [
    "Whatever color the {a} is, so is the {b}.", "The {a}'s color is also the {b}'s.",
    "Like the {a}, the {b} is colored.", "The color of the {a} is shared by the {b}.",
    "As the {a} is colored, so is the {b}.", "It is the {a} whose color the {b} takes.",
    "The {a} sets the color; the {b} follows.", "From the {a}, the {b} draws its color.",
]


def hpool_tok(model, ids):
    enc = model.encoder
    B, T = ids.shape
    pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
    h = enc.token_emb(ids) + enc.pos_emb(pos)
    h = enc.emb_norm(h); h = enc.emb_drop(h)
    for blk in enc.blocks:
        h = blk(h)
    h = enc.final_norm(h)
    return h, h.mean(dim=1)


def bank_value(model, hp):
    w = model.encoder.writer
    return w.value_head(w.norm(hp))


def key_of(model, e):
    if e not in KEY_CACHE:
        KEY_CACHE[e] = query_key(model, f"What color is the {e}?").to(DEVICE).detach()
    return KEY_CACHE[e]


def pad(idlist):
    T = max(len(x) for x in idlist)
    eot = ENC.encode_ordinary(" .")[0]
    return torch.tensor([x + [eot] * (T - len(x)) for x in idlist], device=DEVICE)


# ---------------------------------------------------------------------------
# Fine-tune the substrate on FAMILY A (scaled diversity) + templated + distillation
# ---------------------------------------------------------------------------
def finetune(frozen, train_ents, steps, lr, distill_w, seed):
    random.seed(seed); torch.manual_seed(seed)
    model = load_model("runs/multiobject/ckpt_multiobject.pt")
    for n, p in model.named_parameters():
        p.requires_grad_(any(n.startswith(t) for t in TRAINABLE))
    enc_params = [p for n, p in model.named_parameters() if p.requires_grad]
    vhead = nn.Linear(768, len(VAL_VOCAB)).to(DEVICE)
    rhead = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 128)).to(DEVICE)
    opt = torch.optim.AdamW(enc_params + list(vhead.parameters()) + list(rhead.parameters()), lr=lr)
    rng = random.Random(50 + seed)
    model.train()
    for step in range(steps):
        opt.zero_grad()
        loss = torch.tensor(0.0, device=DEVICE)
        # value at entity position (Family A + templated anchor)
        vt, vp, vg = [], [], []
        for _ in range(48):
            e = rng.choice(train_ents); fam = rng.choice(["size", "color"])
            v = rng.choice(SIZES if fam == "size" else COLORS)
            ph = "The {e} is {v}." if rng.random() < 0.3 else rng.choice(VALUE_A)
            ids = ENC.encode_ordinary(ph.format(e=e, v=v)); p = find_entity_pos(ids, e)
            if p is not None:
                vt.append(ids); vp.append(p); vg.append(VAL_IDX[v])
        for L, idxs in _bylen(vt).items():
            ids = torch.tensor([vt[i] for i in idxs], device=DEVICE)
            reps, _ = hpool_tok(model, ids)
            er = torch.stack([reps[j, vp[idxs[j]]] for j in range(len(idxs))])
            loss = loss + F.cross_entropy(vhead(er), torch.tensor([vg[i] for i in idxs], device=DEVICE)) * (len(idxs) / max(1, len(vt)))
        # relation subject->target key (Family A)
        rt = []
        for _ in range(32):
            A, B = rng.sample(train_ents, 2)
            ids = ENC.encode_ordinary(rng.choice(REL_A).format(b=B, a=A)); p = find_entity_pos(ids, B)
            if p is not None:
                rt.append((ids, p, A))
        for L, idxs in _bylen([x[0] for x in rt]).items():
            ids = torch.tensor([rt[i][0] for i in idxs], device=DEVICE)
            reps, _ = hpool_tok(model, ids)
            subj = torch.stack([reps[j, rt[idxs[j]][1]] for j in range(len(idxs))])
            pk = F.normalize(rhead(subj), dim=1)
            tk = torch.stack([F.normalize(key_of(model, rt[i][2]), dim=0) for i in idxs])
            loss = loss + (1.0 - F.cosine_similarity(pk, tk, dim=1)).mean() * (len(idxs) / max(1, len(rt)))
        # distillation: templated bank-write stays close to frozen
        dt = [ENC.encode_ordinary("The {e} is {v}.".format(e=rng.choice(train_ents), v=rng.choice(VAL_VOCAB))) for _ in range(32)]
        ids = pad(dt)
        _, hpn = hpool_tok(model, ids)
        with torch.no_grad():
            _, hpo = hpool_tok(frozen, ids)
            bvo = bank_value(frozen, hpo)
        loss = loss + distill_w * F.mse_loss(bank_value(model, hpn), bvo)
        loss.backward(); opt.step()
    model.eval()
    return model


def _bylen(idlist):
    b = defaultdict(list)
    for i, t in enumerate(idlist):
        b[len(t)].append(i)
    return b


# ---------------------------------------------------------------------------
# Extraction re-test: train extractor on FAMILY A reps of the fine-tuned substrate, eval on FAMILY B
# ---------------------------------------------------------------------------
def value_scene(model, ents, phrasings, rng):
    e0, e1 = rng.sample(ents, 2); v0, v1 = rng.sample(SIZES, 2)
    text = rng.choice(phrasings).format(e=e0, v=v0) + " " + rng.choice(phrasings).format(e=e1, v=v1)
    ids = ENC.encode_ordinary(text); p0, p1 = find_entity_pos(ids, e0), find_entity_pos(ids, e1)
    if p0 is None or p1 is None or p0 == p1:
        return None
    reps = enc_hidden(model, ids)
    return {"reps": reps, "slots": [(p0, SIZE_RANK[v0], SIZE_RANK[v1]), (p1, SIZE_RANK[v1], SIZE_RANK[v0])]}


def extraction_retest(model, train_ents, held_ents, seed, steps, lr):
    random.seed(seed); torch.manual_seed(seed)
    drng = random.Random(300 + seed)
    train = [s for _ in range(400) for s in [value_scene(model, train_ents, VALUE_A, drng)] if s]
    ev = [s for _ in range(150) for s in [value_scene(model, held_ents, VALUE_B, drng)] if s]  # DOUBLE held-out + Family B

    def batch(scenes):
        reps, ent, gold, sib = [], [], [], []
        for s in scenes:
            for (pos, g, sg) in s["slots"]:
                reps.append(s["reps"]); ent.append(s["reps"][pos]); gold.append(g); sib.append(sg)
        S, M = pad_seq(reps)
        return S.to(DEVICE), M.to(DEVICE), torch.stack(ent).to(DEVICE), torch.tensor(gold, device=DEVICE), torch.tensor(sib, device=DEVICE)

    Str, Mtr, Etr, Gtr, _ = batch(train)
    ex = ValueExtractor().to(DEVICE)
    opt = torch.optim.AdamW(ex.parameters(), lr=lr); ex.train()
    for _ in range(steps):
        opt.zero_grad(); F.cross_entropy(ex(Etr, Str, Mtr), Gtr).backward(); opt.step()
    ex.eval()
    Sev, Mev, Eev, Gev, Sib = batch(ev)
    with torch.no_grad():
        pred = torch.argmax(ex(Eev, Sev, Mev), dim=1)
    return (pred == Gev).float().mean().item(), (pred == Sib).float().mean().item()


def dist(xs):
    return {"min": round(min(xs), 4), "median": round(median(xs), 4), "max": round(max(xs), 4),
            "std": round(pstdev(xs), 4) if len(xs) > 1 else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 8 phrasing-diversity scale")
    ap.add_argument("--ft-seeds", type=int, default=3)
    ap.add_argument("--ft-steps", type=int, default=1800)
    ap.add_argument("--ft-lr", type=float, default=1e-4)
    ap.add_argument("--distill-w", type=float, default=3.0)
    ap.add_argument("--ex-seeds", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    (RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
    print(SEP, flush=True)
    print(f"[INFO] Stage 8 phrasing-diversity scale | device={DEVICE}", flush=True)
    print(f"[INFO] training family: {len(VALUE_A)} value + {len(REL_A)} relation phrasings | "
          f"HELD-OUT family (structurally distinct): {len(VALUE_B)} value + {len(REL_B)} relation", flush=True)
    frozen = load_model("runs/multiobject/ckpt_multiobject.pt")

    split_rng = random.Random(20260620)
    sh = ENTITIES[:]; split_rng.shuffle(sh)
    train_ents, held_ents = sh[:14], sh[14:]

    if args.smoke:
        m = finetune(frozen, train_ents, 30, args.ft_lr, args.distill_w, 0)
        c, w = extraction_retest(m, train_ents, held_ents, 0, 200, 1e-3)
        print(f"  [SMOKE] FamilyB value_bind={c:.3f} wrong_bind={w:.3f}", flush=True)
        return 0

    vbind, wbind = [], []
    last_model = None
    for s in range(args.ft_seeds):
        print(f"[INFO] fine-tune seed {s} on Family A ({args.ft_steps} steps)...", flush=True)
        m = finetune(frozen, train_ents, args.ft_steps, args.ft_lr, args.distill_w, s)
        last_model = m
        for es in range(args.ex_seeds):
            c, w = extraction_retest(m, train_ents, held_ents, es, 1500, 1e-3)
            vbind.append(c); wbind.append(w)
        print(f"  seed {s}: FamilyB value_bind median so far, last (c={c:.3f} w={w:.3f})", flush=True)

    vb = dist(vbind); wb = dist(wbind)
    # save last fine-tuned substrate for arc re-verification
    torch.save({"model": last_model.state_dict(), "meta": {"base": "ckpt_multiobject", "stage": 8}}, OUT_CKPT)

    g_value = vb["median"] >= 0.85 and wb["median"] <= 0.02
    if g_value:
        verdict = "STAGE_8_PHRASING_ROBUST"          # arc re-verification + end-to-end confirmed separately
    elif vb["median"] >= 0.70:
        verdict = "STAGE_8_PHRASING_PARTIAL"
    else:
        verdict = "STAGE_8_PHRASING_REFUTED_AT_SCALE"

    out = {"verdict": verdict,
           "training_family_phrasings": {"value": len(VALUE_A), "relation": len(REL_A)},
           "heldout_family_phrasings_structurally_distinct": {"value": len(VALUE_B), "relation": len(REL_B)},
           "gates": {
               "G_VALUE_BINDING_PHRASE_FAMILY_B": {"dist": vb, "bar": 0.85,
                   "note": "value binding on STRUCTURALLY-DISTINCT held-out family + held-out entities (paraphrase-invariance, not in-family)"},
               "G_WRONG_VALUE_BINDING": {"dist": wb, "bar": 0.02, "note": "DANGEROUS: cross-bound to sibling (fabrication)"},
           },
           "config": {"ft_seeds": args.ft_seeds, "ft_steps": args.ft_steps, "ex_seeds": args.ex_seeds},
           "new_substrate": str(OUT_CKPT),
           "scope": ("phrasing-diversity scale: 16 value + 12 relation TRAINING phrasings, held-out on a STRUCTURALLY "
                     "DISTINCT family (value-first/inverted/embedded). Fine-tune toward canonical bank targets + "
                     "distillation; arc re-verification run separately on the new checkpoint. Closed vocab, small."),
           "meaning": {
               "STAGE_8_PHRASING_ROBUST": "enough phrasing diversity made the substrate abstract paraphrase-invariance "
                   "(generalizes to structurally-distinct held-out phrasings) -> extraction front-end works; run the arc "
                   "re-verification + end-to-end to close the autonomy loop on a phrasing-robust substrate.",
               "STAGE_8_PHRASING_PARTIAL": "diversity lifted held-out-family binding above chance but not to the bar.",
               "STAGE_8_PHRASING_REFUTED_AT_SCALE": "even >=14 diverse phrasings do NOT make extraction generalize to "
                   "structurally-distinct held-out phrasings -> paraphrase-robustness is a PRETRAINING / language-"
                   "understanding property, not a fine-tuning-diversity property at this from-scratch closed-vocab "
                   "capacity. The next move MERGES the extraction and scale frontiers: port the proven arc (Stage 5->5e) "
                   "to a PRETRAINED base where paraphrase-robustness exists for free. This negative does not stall the "
                   "program; it focuses it."}[verdict]}
    (RUN_DIR / "results" / "verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(SEP, flush=True)
    print(f"  G_VALUE_BINDING_PHRASE (Family B, structurally distinct held-out): {vb['median']} (bar >=0.85)", flush=True)
    print(f"  G_WRONG_VALUE_BINDING: {wb['median']} (bar <=0.02)", flush=True)
    print(f"[INFO] VERDICT: {verdict}", flush=True)
    print(f"[INFO] saved new substrate -> {OUT_CKPT} (for arc re-verification)", flush=True)
    print("STAGE8_JSON " + json.dumps({"verdict": verdict, "value_bind_familyB": vb["median"],
          "wrong_bind": wb["median"]}), flush=True)
    return 0 if verdict == "STAGE_8_PHRASING_ROBUST" else 1


if __name__ == "__main__":
    sys.exit(main())
