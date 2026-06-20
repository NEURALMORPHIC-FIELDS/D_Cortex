# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# STAGE 7 part 1 - SUBSTRATE FINE-TUNE FOR PHRASING-ROBUST EXTRACTION (Step 2, one level up).
# Stage 6 gave SUBSTRATE_LIMITED: the substrate trained only on templates does not expose varied-phrasing
# bindings. Step 2 made bindings generalize over ENTITIES (phrasing fixed); this makes them generalize
# over PHRASINGS. Fine-tune encoder.blocks so the entity-position representation exposes the value/target
# under VARIED phrasing, while a DISTILLATION loss keeps the bank-write on TEMPLATED facts close to the
# original (so the proven downstream still reads the same banks). Save a NEW checkpoint; never overwrite
# ckpt_multiobject. The full arc re-verification is run separately on the new checkpoint.
#
# THE CONSISTENCY OBJECTIVE: any phrasing of a fact -> the same canonical binding the templated fact has.
# Train on phrasing-set-1 / entity-set-1; the re-test cert evaluates on UNSEEN phrasings AND entities.

import argparse
import contextlib
import io
import random
import sys
from collections import defaultdict
from pathlib import Path
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

from scripts.certify_stage_i_extraction import ENC, DEVICE, ENTITIES, SIZES, find_entity_pos
from scripts.certify_stage5_operate_memory import COLORS, COLOR_IDX, load_model, query_key
from scripts.certify_stage6_extraction import VALUE_PHRASINGS, REL_PHRASINGS, split_phr

SEP = "=" * 70
OUT = REPO_ROOT / "runs" / "multiobject" / "ckpt_multiobject_phrase.pt"
TRAINABLE = ("encoder.emb_norm", "encoder.blocks", "encoder.final_norm")
VAL_VOCAB = SIZES + COLORS                                  # the substrate must expose both value families
VAL_IDX = {v: i for i, v in enumerate(VAL_VOCAB)}
TPL = {"size": "The {e} is {v}.", "color": "The {e} is {v}."}
KEY_CACHE: Dict[str, torch.Tensor] = {}


def enc_hpool_and_tok(model, ids: torch.Tensor):
    """Return per-token reps [B,T,D] and pooled h_pool [B,D] (with grad)."""
    enc = model.encoder
    B, T = ids.shape
    pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
    h = enc.token_emb(ids) + enc.pos_emb(pos)
    h = enc.emb_norm(h)
    h = enc.emb_drop(h)
    for blk in enc.blocks:
        h = blk(h)
    h = enc.final_norm(h)
    return h, h.mean(dim=1)


def bank_value(model, h_pool):
    w = model.encoder.writer
    return w.value_head(w.norm(h_pool))


def key_of(model, e):
    if e not in KEY_CACHE:
        KEY_CACHE[e] = query_key(model, f"What color is the {e}?").to(DEVICE).detach()
    return KEY_CACHE[e]


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 7: fine-tune substrate for phrasing-robust extraction")
    ap.add_argument("--ckpt", default="runs/multiobject/ckpt_multiobject.pt")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--distill-w", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    print(SEP, flush=True)
    print(f"[INFO] Stage 7 substrate fine-tune | device={DEVICE} | base={args.ckpt}", flush=True)
    frozen = load_model(args.ckpt)                             # distillation target (original substrate)
    model = load_model(args.ckpt)
    for n, p in model.named_parameters():
        p.requires_grad_(any(n.startswith(t) for t in TRAINABLE))
    enc_params = [p for n, p in model.named_parameters() if p.requires_grad]
    value_head = nn.Linear(768, len(VAL_VOCAB)).to(DEVICE)
    rel_head = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 128)).to(DEVICE)
    print(f"[INFO] trainable encoder params: {len(enc_params)} tensors + value/relation heads. "
          f"Distillation anchors templated bank-writes.", flush=True)

    opt = torch.optim.AdamW(enc_params + list(value_head.parameters()) + list(rel_head.parameters()), lr=args.lr)

    split_rng = random.Random(20260620)
    sh = ENTITIES[:]; split_rng.shuffle(sh)
    train_ents = sh[:14]
    tr_phr, _ = split_phr(VALUE_PHRASINGS)
    tr_rel, _ = split_phr(REL_PHRASINGS)
    rng = random.Random(7)

    def value_batch():
        texts, ents, gold = [], [], []
        for _ in range(48):
            e = rng.choice(train_ents)
            fam = rng.choice(["size", "color"])
            v = rng.choice(SIZES if fam == "size" else COLORS)
            # mix templated (anchor) and varied phrasing
            if rng.random() < 0.4:
                text = TPL[fam].format(e=e, v=v)
            else:
                text = rng.choice(tr_phr).format(e=e, v=v)
            ids = ENC.encode_ordinary(text)
            p = find_entity_pos(ids, e)
            if p is None:
                continue
            texts.append(ids); ents.append(p); gold.append(VAL_IDX[v])
        return texts, ents, gold

    def rel_batch():
        out = []
        for _ in range(32):
            A, B = rng.sample(train_ents, 2)
            text = rng.choice(tr_rel).format(b=B, a=A)
            ids = ENC.encode_ordinary(text)
            p = find_entity_pos(ids, B)
            if p is None:
                continue
            out.append((ids, p, A))
        return out

    def distill_batch():
        texts, ents = [], []
        for _ in range(32):
            e = rng.choice(train_ents); v = rng.choice(VAL_VOCAB)
            text = "The {e} is {v}.".format(e=e, v=v)        # templated -> anchor existing banks
            texts.append(ENC.encode_ordinary(text))
        return texts

    def pad(idlist):
        T = max(len(x) for x in idlist)
        eot = ENC.encode_ordinary(" .")[0]
        return torch.tensor([x + [eot] * (T - len(x)) for x in idlist], device=DEVICE)

    model.train()
    for step in range(args.steps):
        opt.zero_grad()
        # value-at-entity-position objective (phrasing-robust binding)
        vt, vp, vg = value_batch()
        # group by length for batched forward (encoder has no pad mask)
        bylen = defaultdict(list)
        for i, t in enumerate(vt):
            bylen[len(t)].append(i)
        loss = torch.tensor(0.0, device=DEVICE)
        for L, idxs in bylen.items():
            ids = torch.tensor([vt[i] for i in idxs], device=DEVICE)
            reps, _ = enc_hpool_and_tok(model, ids)
            ent_rep = torch.stack([reps[j, vp[idxs[j]]] for j in range(len(idxs))])
            g = torch.tensor([vg[i] for i in idxs], device=DEVICE)
            loss = loss + F.cross_entropy(value_head(ent_rep), g) * (len(idxs) / max(1, len(vt)))
        # relation subject->target-key objective
        rb = rel_batch()
        rbylen = defaultdict(list)
        for i, (t, _, _) in enumerate(rb):
            rbylen[len(t)].append(i)
        for L, idxs in rbylen.items():
            ids = torch.tensor([rb[i][0] for i in idxs], device=DEVICE)
            reps, _ = enc_hpool_and_tok(model, ids)
            subj = torch.stack([reps[j, rb[idxs[j]][1]] for j in range(len(idxs))])
            pk = F.normalize(rel_head(subj), dim=1)
            tk = torch.stack([F.normalize(key_of(model, rb[i][2]), dim=0) for i in idxs])
            loss = loss + (1.0 - F.cosine_similarity(pk, tk, dim=1)).mean() * (len(idxs) / max(1, len(rb)))
        # distillation: templated bank-write stays close to frozen
        dt = distill_batch()
        ids = pad(dt)
        _, hp_new = enc_hpool_and_tok(model, ids)
        with torch.no_grad():
            _, hp_old = enc_hpool_and_tok(frozen, ids)
        bv_new = bank_value(model, hp_new)
        with torch.no_grad():
            bv_old = bank_value(frozen, hp_old)
        loss = loss + args.distill_w * F.mse_loss(bv_new, bv_old)
        loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
            print(f"  step {step+1}/{args.steps} | loss {float(loss):.3f}", flush=True)

    model.eval()
    # preservation sanity: templated bank-value cosine to frozen (should stay high)
    with torch.no_grad():
        ids = pad(distill_batch())
        _, hp = enc_hpool_and_tok(model, ids)
        _, hpo = enc_hpool_and_tok(frozen, ids)
        pres = F.cosine_similarity(bank_value(model, hp), bank_value(frozen, hpo), dim=1).mean().item()
    torch.save({"model": model.state_dict(),
                "meta": {"base": args.ckpt, "trained": "encoder phrasing-robust + distill",
                         "templated_bank_preservation_cosine": pres}}, OUT)
    print(SEP, flush=True)
    print(f"[INFO] templated bank-value preservation (cosine to frozen): {pres:.4f}", flush=True)
    print(f"[INFO] saved phrasing-robust substrate -> {OUT} (ckpt_multiobject NOT overwritten)", flush=True)
    print("SUBSTRATE_PHRASE_SAVED " + str(OUT), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
