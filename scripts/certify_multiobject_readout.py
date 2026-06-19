# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# STEP 1 of the multi-object road: MEASURE how far the FROZEN base is from holding co-occurring facts
# as separable objects. Stage I showed a LINEAR probe at the entity position cannot recover the
# per-entity value (wrong_binding 0.21, cross-bind 0.19). This does NOT prove the info is absent.
# Here we climb a READOUT LADDER on the SAME frozen rep, same multi-entity regime, same codebook:
#   LINEAR  : entity-position vector -> heads (the Stage I baseline, reproduced in-run).
#   MLP     : one hidden layer (GELU) at the entity position - is the binding NONLINEARLY present AT
#             the entity token?
#   ATTENTION: a learned query from the entity-position vector attends over the FULL sequence hidden
#             states - is the binding present SOMEWHERE in the sequence, attention-recoverable given
#             the entity (e.g. living at the value-token position)?
# DIRECTION read: if a modest nonlinear/attention readout recovers binding (wrong_binding -> ~0), the
# info is PRESENT-but-not-linear (CLOSE: a readout/training change unlocks it). If even attention
# fails, the frozen rep LACKS separable objects (FAR: the base must be retrained - step 2 mandatory).
# This is a measurement of the ONE direction, not an option. Frozen base; heads only; held-out 10/30.

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
from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model
from stage_u.memory_tokenizer import MemoryTokenizer
# reuse the validated Stage I helpers (frozen-rep access, codebook fit, vocab, positions)
from scripts.certify_stage_i_extraction import (
    ENC, DEVICE, ENTITIES, ATTRS, ATTR_VALUES, ATTR_FACT, ATTR_IDX, ALL_VALUES, VALUE_ATTR,
    enc_hidden, w_value_contextual, find_entity_pos, fit_codebook,
)

SEP = "=" * 70
RUN_DIR = REPO_ROOT / "runs" / "multiobject_readout"


# ---------------------------------------------------------------------------
# Dataset: cache the FULL sequence hidden states (needed for the attention readout), padded + masked.
# ---------------------------------------------------------------------------
def build_examples(model: DCortexV2Model, ents: List[str], n: int, rng: random.Random):
    examples = []
    dropped = 0
    for _ in range(n):
        attr = rng.choice(ATTRS)
        vals = ATTR_VALUES[attr]
        if len(ents) < 2 or len(vals) < 2:
            continue
        e1, e2 = rng.sample(ents, 2)
        v1, v2 = rng.sample(vals, 2)
        text = ATTR_FACT[attr].format(e=e1, v=v1) + " " + ATTR_FACT[attr].format(e=e2, v=v2)
        ids = ENC.encode_ordinary(text)
        p1, p2 = find_entity_pos(ids, e1), find_entity_pos(ids, e2)
        if p1 is None or p2 is None or p1 == p2:
            dropped += 1
            continue
        h = enc_hidden(model, ids)                                  # [T, D]
        examples.append({"h": h, "T": h.shape[0],
                         "slots": [{"pos": p1, "attr": attr, "value": v1, "sibling": v2},
                                   {"pos": p2, "attr": attr, "value": v2, "sibling": v1}]})
    if dropped:
        print(f"  [WARN] dropped {dropped} examples (position not locatable)", flush=True)
    return examples


def pack(examples, tk: MemoryTokenizer, max_t: int):
    """Return padded H_seq [E,maxT,D], mask [E,maxT], and per-slot index tensors."""
    E = len(examples)
    D = examples[0]["h"].shape[1]
    H = torch.zeros(E, max_t, D)
    mask = torch.zeros(E, max_t, dtype=torch.bool)
    s_ex, s_pos, s_yval, s_yattr, s_sib = [], [], [], [], []
    meta = []
    for i, ex in enumerate(examples):
        T = min(ex["T"], max_t)
        H[i, :T] = ex["h"][:T]
        mask[i, :T] = True
        for sl in ex["slots"]:
            if sl["pos"] >= max_t:
                continue
            s_ex.append(i); s_pos.append(sl["pos"])
            s_yval.append(tk.value_token[sl["value"]]); s_yattr.append(ATTR_IDX[sl["attr"]])
            s_sib.append(tk.value_token[sl["sibling"]])
            meta.append({"value": sl["value"], "sibling": sl["sibling"], "attr": sl["attr"]})
    return (H.to(DEVICE), mask.to(DEVICE),
            torch.tensor(s_ex, device=DEVICE), torch.tensor(s_pos, device=DEVICE),
            torch.tensor(s_yval, device=DEVICE), torch.tensor(s_yattr, device=DEVICE),
            torch.tensor(s_sib, device=DEVICE), meta)


# ---------------------------------------------------------------------------
# Three readouts. Uniform interface: forward(qvec[M,D], seq[M,maxT,D], mask[M,maxT]) -> (vp[M,D], ap[M,3])
# qvec = entity-position hidden state. LINEAR/MLP ignore seq/mask. ATTENTION attends over the sequence.
# ---------------------------------------------------------------------------
class LinearReadout(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.value = nn.Linear(d, d)
        self.attr = nn.Linear(d, 3)

    def forward(self, q, seq, mask):
        return self.value(q), self.attr(q)


class MLPReadout(nn.Module):
    def __init__(self, d: int, hid: int = 512) -> None:
        super().__init__()
        self.value = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Linear(hid, d))
        self.attr = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 3))

    def forward(self, q, seq, mask):
        return self.value(q), self.attr(q)


class AttnReadout(nn.Module):
    """Multi-head cross-attention (pre-norm): query from the entity position, keys/values = full
    sequence. The natural binding readout - 'given the entity, fetch its value from the text'.
    Made strong enough to FIT in-sample so a held-out failure indicts the rep, not the readout."""
    def __init__(self, d: int, n_heads: int = 4) -> None:
        super().__init__()
        self.h = n_heads
        self.hd = d // n_heads
        self.scale = self.hd ** -0.5
        self.lnq = nn.LayerNorm(d)
        self.lnk = nn.LayerNorm(d)
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.lnc = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.value = nn.Linear(d, d)
        self.attr = nn.Linear(d, 3)

    def forward(self, q, seq, mask):
        M, T, D = seq.shape
        Q = self.wq(self.lnq(q)).view(M, self.h, 1, self.hd)
        sn = self.lnk(seq)
        K = self.wk(sn).view(M, T, self.h, self.hd).transpose(1, 2)   # [M,h,T,hd]
        V = self.wv(sn).view(M, T, self.h, self.hd).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)).squeeze(2) * self.scale     # [M,h,T]
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(scores, dim=-1).unsqueeze(2)                  # [M,h,1,T]
        ctx = (attn @ V).squeeze(2).reshape(M, D)                      # [M,D]
        ctx = self.out(ctx)
        ctx = ctx + self.mlp(self.lnc(ctx))                           # residual MLP
        return self.value(ctx), self.attr(ctx)


READOUTS = {"linear": LinearReadout, "mlp": MLPReadout, "attention": AttnReadout}
# per-readout optimization (attention needs lower lr + more steps to fit; others are fast)
READOUT_OPT = {"linear": {"lr_mult": 1.0, "step_mult": 1.0},
               "mlp": {"lr_mult": 1.0, "step_mult": 1.0},
               "attention": {"lr_mult": 0.5, "step_mult": 2.0}}


def run_readout(name: str, train_pack, eval_pack, tk: MemoryTokenizer, seed: int,
                steps: int, lr: float, temp: float) -> Dict:
    random.seed(seed); torch.manual_seed(seed)
    codebook = tk.codebook.to(DEVICE)
    Htr, mtr, ex_tr, pos_tr, yv_tr, ya_tr, sib_tr, _ = train_pack
    head = READOUTS[name](Htr.shape[2]).to(DEVICE)
    cfg = READOUT_OPT[name]
    lr = lr * cfg["lr_mult"]
    steps = int(steps * cfg["step_mult"])
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)

    def batch(pack):
        H, msk, ex, pos, yv, ya, sib, meta = pack
        q = H[ex, pos]                                              # [M,D]
        seq = H[ex]; m = msk[ex]                                    # [M,maxT,D],[M,maxT]
        return q, seq, m, yv, ya, sib, meta

    qtr, seqtr, mtr2, yvtr, yatr, sibtr, _ = batch(train_pack)
    head.train()
    for _ in range(steps):
        opt.zero_grad()
        vp, ap = head(qtr, seqtr, mtr2)
        logits_v = F.normalize(vp, dim=1) @ codebook.t() * temp
        loss = F.cross_entropy(logits_v, yvtr) + F.cross_entropy(ap, yatr)
        loss.backward(); opt.step()

    def evaluate(pack) -> Dict:
        q, seq, m, yv, ya, sib, meta = batch(pack)
        head.eval()
        with torch.no_grad():
            vp, ap = head(q, seq, m)
            toks = torch.argmax(F.normalize(vp, dim=1) @ codebook.t(), dim=1).tolist()
            apred = torch.argmax(ap, dim=1).tolist()
        cb = vd = ae = ve = wb = wc = 0
        n = len(meta)
        for k, mt in enumerate(meta):
            value_pred = tk.decode(int(toks[k]))
            attr_pred = ATTRS[int(apred[k])]
            c = (value_pred == mt["sibling"]); d_ = (value_pred != mt["value"] and value_pred != mt["sibling"])
            a = (attr_pred != mt["attr"]); v = (value_pred != mt["value"])
            cb += c; vd += d_; ae += a; ve += v; wb += (c or a); wc += (v or a)
        return {"cross_binding": cb / n, "value_drift": vd / n, "attribute_error": ae / n,
                "value_error": ve / n, "wrong_binding": wb / n, "wrong_commit_total": wc / n}

    held = evaluate(eval_pack)
    tr = evaluate(train_pack)
    held["train_wrong_binding"] = tr["wrong_binding"]; held["train_value_error"] = tr["value_error"]
    held["final_train_loss"] = float(loss.item())
    return held


def dist(xs: List[float]) -> Dict[str, float]:
    return {"min": round(min(xs), 4), "median": round(median(xs), 4), "max": round(max(xs), 4),
            "std": round(pstdev(xs), 4) if len(xs) > 1 else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-object readout ladder (step 1)")
    ap.add_argument("--ckpt", default="runs/stage_u/results/ckpt_multiattr.pt")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--temp", type=float, default=10.0)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-eval", type=int, default=200)
    args = ap.parse_args()
    (RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
    print(SEP, flush=True)
    print(f"[INFO] Multi-object readout ladder (step 1) | device={DEVICE} | ckpt={args.ckpt}", flush=True)

    with contextlib.redirect_stdout(io.StringIO()):
        model = DCortexV2Model(DCortexConfig()).to(DEVICE).eval()
    ck = torch.load(args.ckpt, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    print("[INFO] base model loaded and FROZEN", flush=True)

    split_rng = random.Random(20260620)
    shuffled = ENTITIES[:]; split_rng.shuffle(shuffled)
    train_ents, held_ents = shuffled[:20], shuffled[20:]
    print(f"[INFO] entity split: {len(train_ents)} train / {len(held_ents)} held-out", flush=True)

    tk = fit_codebook(model, train_ents)
    print(f"[INFO] frozen codebook: {len(tk.token_value)} values", flush=True)

    data_rng = random.Random(7)
    tr_ex = build_examples(model, train_ents, args.n_train, data_rng)
    ev_ex = build_examples(model, held_ents, args.n_eval, data_rng)
    max_t = max(max(e["T"] for e in tr_ex), max(e["T"] for e in ev_ex))
    train_pack = pack(tr_ex, tk, max_t)
    eval_pack = pack(ev_ex, tk, max_t)
    print(f"[INFO] examples: {len(tr_ex)} train / {len(ev_ex)} eval | max_t={max_t} | "
          f"slots: {len(train_pack[7])} train / {len(eval_pack[7])} eval", flush=True)

    ladder = {}
    for name in ("linear", "mlp", "attention"):
        runs = [run_readout(name, train_pack, eval_pack, tk, seed, args.steps, args.lr, args.temp)
                for seed in range(args.seeds)]
        agg = {k: dist([r[k] for r in runs]) for k in
               ("wrong_binding", "cross_binding", "attribute_error", "value_error",
                "train_wrong_binding", "train_value_error")}
        ladder[name] = {"agg": agg, "per_seed": runs}
        print(f"  [{name:9s}] wrong_binding median={agg['wrong_binding']['median']:.4f} "
              f"(cross_bind={agg['cross_binding']['median']:.4f} attr_err={agg['attribute_error']['median']:.4f}) "
              f"value_error={agg['value_error']['median']:.4f} | "
              f"in-sample wb={agg['train_wrong_binding']['median']:.4f}", flush=True)

    lin = ladder["linear"]["agg"]["wrong_binding"]["median"]
    mlp = ladder["mlp"]["agg"]["wrong_binding"]["median"]
    attn = ladder["attention"]["agg"]["wrong_binding"]["median"]
    best_nl = min(mlp, attn)

    # DIRECTION verdict (pre-declared)
    if best_nl <= 0.05:
        direction = "BINDING_PRESENT_NONLINEAR"          # CLOSE: info is there, readout/training unlocks
    elif best_nl <= 0.15 and best_nl < lin - 0.05:
        direction = "BINDING_PARTIAL_NONLINEAR"          # present but weak
    else:
        direction = "BINDING_ABSENT_IN_FROZEN_REP"       # FAR: base retrain (step 2) mandatory

    print(SEP, flush=True)
    print("[INFO] READOUT LADDER (frozen rep, wrong_binding median, held-out):", flush=True)
    print(f"  linear={lin:.4f}  mlp={mlp:.4f}  attention={attn:.4f}  best_nonlinear={best_nl:.4f}", flush=True)
    print(f"[INFO] DIRECTION: {direction}", flush=True)
    meanings = {
        "BINDING_PRESENT_NONLINEAR": "co-occurring binding IS in the frozen rep, just not linear -> CLOSE; a "
                                     "nonlinear/attention readout (or light training) unlocks Stage I + Stage C.",
        "BINDING_PARTIAL_NONLINEAR": "partially recoverable -> the rep half-separates objects; base training will "
                                     "likely finish it.",
        "BINDING_ABSENT_IN_FROZEN_REP": "even attention over the sequence cannot recover per-entity binding -> the "
                                        "frozen rep does NOT hold separable objects; step 2 (retrain the base to "
                                        "maintain co-occurring objects) is mandatory, not optional.",
    }
    print(f"  meaning: {meanings[direction]}", flush=True)

    out = {"direction": direction, "ckpt": args.ckpt, "device": DEVICE,
           "config": {"seeds": args.seeds, "steps": args.steps, "lr": args.lr, "temp": args.temp,
                      "max_t": max_t, "train_slots": len(train_pack[7]), "eval_slots": len(eval_pack[7])},
           "ladder_wrong_binding_median": {"linear": lin, "mlp": mlp, "attention": attn, "best_nonlinear": best_nl},
           "ladder_full": {k: v["agg"] for k, v in ladder.items()},
           "per_seed": {k: v["per_seed"] for k, v in ladder.items()},
           "meaning": meanings[direction],
           "scope": ("step 1 of the multi-object road; FROZEN base; readout-only; modest machinery (one MLP hidden "
                     "layer / single-head attention) - a PASS = recoverable with modest readout, not deep "
                     "recomputation. Measures the ONE direction (multi-object separability), not an option."),
           "next": ("if PRESENT/PARTIAL -> step 2 trains the base to maintain separable co-occurring objects "
                    "(scoped by this gap), then re-test Stage I + Stage C. if ABSENT -> step 2 is mandatory and "
                    "heavier (the rep must be rebuilt).")}
    (RUN_DIR / "results" / "verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(SEP, flush=True)
    print("MULTIOBJECT_READOUT_JSON " + json.dumps({"direction": direction, "linear": lin, "mlp": mlp,
          "attention": attn, "best_nonlinear": best_nl}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
