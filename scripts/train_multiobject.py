# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# STEP 2 of the multi-object road: TRAIN the base to MAINTAIN separable co-occurring objects. Step 1
# measured that the frozen rep lacks a generalizable separable-object structure (no modest readout
# recovers held-out binding). The base was trained single-fact-per-encode, never multi-fact. Here we
# fine-tune the encoder's CONTEXTUAL path (encoder.emb_norm + the 4 encoder blocks + encoder.final_norm
# = 52 params) plus LINEAR per-entity heads, on a mix of single- and multi-entity text, so the
# representation exposes each co-occurring fact as a separable, GENERALIZABLE binding at its entity
# position. Frozen: the shared embeddings, the writer, the decoder, the memory banks (so the memory
# write/read machinery is untouched - Stage U honesty is re-certified afterwards, not assumed).
#
# Success signal: after fine-tuning, a LINEAR head at the entity position GENERALIZES to HELD-OUT
# entities (held-out value-accuracy rises far above the frozen baseline ~0.52). That proves the
# representation became linearly separable for co-occurring objects - the root the whole vision needs.
#
# NON-DESTRUCTIVE: loads ckpt_multiattr, saves a COPY to runs/multiobject/ckpt_multiobject.pt. The
# proven checkpoint is never overwritten. Re-certification (Stage U / Stage I / Stage C / readout
# ladder) on the new copy is step 3.

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

import tiktoken
from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model
from scripts.certify_stage_i_extraction import (
    ENC, DEVICE, ENTITIES, ATTRS, ATTR_VALUES, ATTR_FACT, ATTR_IDX, ALL_VALUES, find_entity_pos,
)

SEP = "=" * 70
OUT_DIR = REPO_ROOT / "runs" / "multiobject"
VALUE_IDX = {v: i for i, v in enumerate(ALL_VALUES)}
TRAINABLE_PREFIXES = ("encoder.emb_norm", "encoder.blocks", "encoder.final_norm")


def enc_forward_grad(model: DCortexV2Model, ids: torch.Tensor) -> torch.Tensor:
    """Batched encoder contextual path WITH grad (no padding: caller buckets by length)."""
    enc = model.encoder
    B, T = ids.shape
    pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
    h = enc.token_emb(ids) + enc.pos_emb(pos)
    h = enc.emb_norm(h)
    h = enc.emb_drop(h)
    for block in enc.blocks:
        h = block(h)
    h = enc.final_norm(h)
    return h                                                        # [B, T, D]


def make_examples(ents: List[str], n: int, rng: random.Random, multi_prob: float = 0.7) -> List[Dict]:
    out = []
    dropped = 0
    for _ in range(n):
        attr = rng.choice(ATTRS)
        vals = ATTR_VALUES[attr]
        multi = rng.random() < multi_prob and len(ents) >= 2 and len(vals) >= 2
        if multi:
            e1, e2 = rng.sample(ents, 2)
            v1, v2 = rng.sample(vals, 2)
            text = ATTR_FACT[attr].format(e=e1, v=v1) + " " + ATTR_FACT[attr].format(e=e2, v=v2)
            pairs = [(e1, v1, v2), (e2, v2, v1)]
        else:
            e1 = rng.choice(ents); v1 = rng.choice(vals)
            text = ATTR_FACT[attr].format(e=e1, v=v1)
            pairs = [(e1, v1, None)]
        ids = ENC.encode_ordinary(text)
        slots = []
        ok = True
        for (e, v, sib) in pairs:
            p = find_entity_pos(ids, e)
            if p is None or p >= len(ids):
                ok = False; break
            slots.append({"pos": p, "value": v, "sibling": sib, "attr": attr})
        if not ok or len({s["pos"] for s in slots}) != len(slots):
            dropped += 1; continue
        out.append({"ids": ids, "slots": slots})
    if dropped:
        print(f"  [WARN] dropped {dropped} examples (position not locatable)", flush=True)
    return out


def buckets_by_len(examples: List[Dict]) -> Dict[int, List[Dict]]:
    b = defaultdict(list)
    for ex in examples:
        b[len(ex["ids"])].append(ex)
    return b


def gather_slots(model, batch: List[Dict]):
    """Run encoder on a same-length batch, return (vecs[M,D], y_val[M], y_attr[M], meta)."""
    ids = torch.tensor([ex["ids"] for ex in batch], device=DEVICE)
    h = enc_forward_grad(model, ids)                                # [B,T,D]
    vecs, yv, ya, meta = [], [], [], []
    for bi, ex in enumerate(batch):
        for s in ex["slots"]:
            vecs.append(h[bi, s["pos"]])
            yv.append(VALUE_IDX[s["value"]]); ya.append(ATTR_IDX[s["attr"]])
            meta.append(s)
    return torch.stack(vecs), torch.tensor(yv, device=DEVICE), torch.tensor(ya, device=DEVICE), meta


@torch.no_grad()
def evaluate(model, value_head, attribute_head, examples: List[Dict]) -> Dict:
    model.eval(); value_head.eval(); attribute_head.eval()
    bk = buckets_by_len(examples)
    val_ok = attr_ok = cross = multi_n = tot = 0
    for T, batch in bk.items():
        vecs, yv, ya, meta = gather_slots(model, batch)
        vp = torch.argmax(value_head(vecs), dim=1).tolist()
        ap = torch.argmax(attribute_head(vecs), dim=1).tolist()
        for k, s in enumerate(meta):
            tot += 1
            pred = ALL_VALUES[int(vp[k])]
            val_ok += int(pred == s["value"]); attr_ok += int(ATTRS[int(ap[k])] == s["attr"])
            if s["sibling"] is not None:
                multi_n += 1; cross += int(pred == s["sibling"])
    return {"value_acc": val_ok / tot, "attr_acc": attr_ok / tot,
            "cross_binding": (cross / multi_n) if multi_n else 0.0, "n": tot}


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 2: train base to maintain separable co-occurring objects")
    ap.add_argument("--ckpt", default="runs/stage_u/results/ckpt_multiattr.pt")
    ap.add_argument("--out", default="runs/multiobject/ckpt_multiobject.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-train", type=int, default=2500)
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--lr-base", type=float, default=2e-4)
    ap.add_argument("--lr-head", type=float, default=2e-3)
    ap.add_argument("--accum", type=int, default=4)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(SEP, flush=True)
    print(f"[INFO] STEP 2 train_multiobject | device={DEVICE} | base={args.ckpt}", flush=True)

    with contextlib.redirect_stdout(io.StringIO()):
        model = DCortexV2Model(DCortexConfig()).to(DEVICE)
    ck = torch.load(args.ckpt, map_location=DEVICE)
    model.load_state_dict(ck["model"])

    for n, p in model.named_parameters():
        p.requires_grad_(any(n.startswith(pref) for pref in TRAINABLE_PREFIXES))
    base_params = [p for n, p in model.named_parameters() if p.requires_grad]
    value_head = nn.Linear(768, len(ALL_VALUES)).to(DEVICE)
    attribute_head = nn.Linear(768, 3).to(DEVICE)
    n_train_params = sum(p.numel() for p in base_params)
    print(f"[INFO] trainable encoder params: {len(base_params)} tensors ({n_train_params} weights) "
          f"+ 2 linear heads. Frozen: shared embeddings, writer, decoder, banks.", flush=True)

    opt = torch.optim.AdamW([
        {"params": base_params, "lr": args.lr_base},
        {"params": list(value_head.parameters()) + list(attribute_head.parameters()), "lr": args.lr_head},
    ], weight_decay=0.0)

    split_rng = random.Random(20260620)
    shuffled = ENTITIES[:]; split_rng.shuffle(shuffled)
    train_ents, held_ents = shuffled[:20], shuffled[20:]
    print(f"[INFO] entity split: 20 train / 10 held-out (held: {held_ents})", flush=True)

    data_rng = random.Random(11)
    train_ex = make_examples(train_ents, args.n_train, data_rng)
    eval_ex = make_examples(held_ents, args.n_eval, data_rng)
    print(f"[INFO] examples: {len(train_ex)} train / {len(eval_ex)} held-out", flush=True)

    base0 = evaluate(model, value_head, attribute_head, eval_ex)
    print(f"[INFO] held-out BEFORE training (random heads): value_acc={base0['value_acc']:.4f} "
          f"cross_binding={base0['cross_binding']:.4f}", flush=True)

    use_amp = (DEVICE == "cuda")
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train(); value_head.train(); attribute_head.train()
        bk = buckets_by_len(train_ex)
        batches = []
        for T, exs in bk.items():
            data_rng.shuffle(exs)
            for i in range(0, len(exs), 32):
                batches.append(exs[i:i + 32])
        data_rng.shuffle(batches)
        opt.zero_grad()
        run_loss = 0.0
        for bi, batch in enumerate(batches):
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
            with ctx:
                vecs, yv, ya, _ = gather_slots(model, batch)
                loss = F.cross_entropy(value_head(vecs.float()), yv) + F.cross_entropy(attribute_head(vecs.float()), ya)
            (loss / args.accum).backward()
            run_loss += float(loss.item())
            if (bi + 1) % args.accum == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        ev = evaluate(model, value_head, attribute_head, eval_ex)
        tr = evaluate(model, value_head, attribute_head, train_ex)
        best_acc = max(best_acc, ev["value_acc"])
        print(f"  epoch {epoch:2d} | loss={run_loss/max(1,len(batches)):.3f} | "
              f"HELD-OUT value_acc={ev['value_acc']:.4f} cross_bind={ev['cross_binding']:.4f} "
              f"attr_acc={ev['attr_acc']:.4f} | train value_acc={tr['value_acc']:.4f}", flush=True)

    final = evaluate(model, value_head, attribute_head, eval_ex)
    torch.save({"model": model.state_dict(),
                "meta": {"base": args.ckpt, "trained": "encoder.emb_norm+blocks+final_norm",
                         "held_out_value_acc": final["value_acc"], "held_out_cross_binding": final["cross_binding"],
                         "epochs": args.epochs}},
               args.out)
    print(SEP, flush=True)
    print(f"[INFO] FINAL held-out: value_acc={final['value_acc']:.4f} (baseline ~0.52) "
          f"cross_binding={final['cross_binding']:.4f} (frozen ~0.19) attr_acc={final['attr_acc']:.4f}", flush=True)
    print(f"[INFO] saved fine-tuned COPY -> {args.out} (ckpt_multiattr NOT overwritten)", flush=True)
    sep = "SEPARABILITY_LEARNED" if (final["value_acc"] >= 0.90 and final["cross_binding"] <= 0.05) else \
          ("SEPARABILITY_PARTIAL" if final["value_acc"] >= 0.70 else "SEPARABILITY_NOT_LEARNED")
    print(f"TRAIN_MULTIOBJECT_JSON {{\"result\": \"{sep}\", \"held_out_value_acc\": {final['value_acc']:.4f}, "
          f"\"held_out_cross_binding\": {final['cross_binding']:.4f}}}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
