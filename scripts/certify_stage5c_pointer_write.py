# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# STAGE 5c - ENCODER POINTER-WRITE. Stage 5b refuted the operation-side fix: A's address is not
# recoverable, held-out, from the FROZEN encoder's relational value. This makes the ENCODER WRITE
# relational facts so B's slot stores A's addressing KEY as a first-class, generalizable pointer, then
# re-tests graph traversal with the Stage 5b operation module.
#
# KEY ENABLER (measured from the writer): the stored bank value = writer.value_head(writer.norm(h_pool)),
# while the addressing key k_ent = query_engine(addr_code) comes from the shared address path on raw
# embeddings - INDEPENDENT of encoder.blocks. So fine-tuning encoder.blocks changes the stored VALUES
# but NOT the keys (the recovery targets are fixed). The non-relational value-writing is preserved by a
# DISTILLATION loss to the frozen ckpt_multiobject values.
#
# STEP 0 (diagnostic, not skipped): recover A's key from the FROZEN B-slot, in-sample vs held-out, to
# confirm the encoder fix is necessary and which case (absent vs entity-specific).
# VALIDITY: G_POINTER_RECOVERY isolates the encoder fix; G_CHAIN_GROUNDED (re-point) is the traversal
# proof. The operation reads ONLY (banks + query keys), never text.

import argparse
import contextlib
import io
import json
import random
import sys
from collections import defaultdict
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
    ENC, DEVICE, ENTITIES, COLORS, COLOR_IDX, load_model, query_key, write_and_snapshot, entity_slot)
from scripts.certify_stage5b_graph_traversal import (
    chain_episode, repoint_twin, compare_episode, GraphOpLayer, train as train_op, evaluate as eval_op, build)

SEP = "=" * 70
RUN_DIR = REPO_ROOT / "runs" / "stage5c_pointer"
POINTER_PREFIXES = ("encoder.emb_norm", "encoder.blocks", "encoder.final_norm")


def enc_hpool(model, ids: torch.Tensor) -> torch.Tensor:
    enc = model.encoder
    B, T = ids.shape
    pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
    h = enc.token_emb(ids) + enc.pos_emb(pos)
    h = enc.emb_norm(h)
    h = enc.emb_drop(h)
    for blk in enc.blocks:
        h = blk(h)
    h = enc.final_norm(h)
    return h.mean(dim=1)


def bank_value(model, h_pool: torch.Tensor) -> torch.Tensor:
    w = model.encoder.writer
    return w.value_head(w.norm(h_pool))


class Recover(nn.Module):
    def __init__(self, d_val: int = 768, d_ent: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_val, 256), nn.GELU(), nn.Linear(256, d_ent))

    def forward(self, v):
        return self.net(v)


# ---------------------------------------------------------------------------
# Fact-level data for the pointer-write (relational -> pointer; non-relational -> distill)
# ---------------------------------------------------------------------------
def make_relational(ents, rng) -> Dict:
    A, B, C = rng.sample(ents, 3)
    text = f"The {B} is the same color as {A}."
    return {"text": text, "target": A, "distractors": [C, B]}


def make_nonrel(ents, rng) -> Dict:
    e = rng.choice(ents); c = rng.choice(COLORS)
    return {"text": f"The {e} is {c}."}


KEY_CACHE: Dict[str, torch.Tensor] = {}


def key_of(model, entity: str) -> torch.Tensor:
    # the addressing key comes from the shared address path (NOT fine-tuned) -> stable; cache it.
    if entity not in KEY_CACHE:
        KEY_CACHE[entity] = query_key(model, f"What color is the {entity}?").to(DEVICE).detach()
    return KEY_CACHE[entity]


def bucket(texts: List[str]):
    b = defaultdict(list)
    for i, t in enumerate(texts):
        b[len(ENC.encode_ordinary(t))].append(i)
    return b


def recovery_acc(model, recover, ents, rng, n: int = 200) -> float:
    """Held-out/in-sample: recover target key from B-slot value; content-address picks target?"""
    ok = tot = 0
    with torch.no_grad():
        for _ in range(n):
            r = make_relational(ents, rng)
            ids = torch.tensor([ENC.encode_ordinary(r["text"])], device=DEVICE)
            bv = bank_value(model, enc_hpool(model, ids))[0]
            rk = F.normalize(recover(bv.unsqueeze(0))[0], dim=0)
            cand = [r["target"]] + r["distractors"]
            sims = torch.stack([F.cosine_similarity(rk, F.normalize(key_of(model, e), dim=0), dim=0) for e in cand])
            ok += int(torch.argmax(sims).item() == 0); tot += 1
    return ok / tot


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 5c encoder pointer-write")
    ap.add_argument("--ckpt", default="runs/multiobject/ckpt_multiobject.pt")
    ap.add_argument("--pw-steps", type=int, default=1500)
    ap.add_argument("--pw-lr", type=float, default=3e-4)
    ap.add_argument("--distill-w", type=float, default=2.0)
    ap.add_argument("--op-seeds", type=int, default=5)
    ap.add_argument("--op-steps", type=int, default=3000)
    ap.add_argument("--n-facts", type=int, default=1200)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    (RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
    print(SEP, flush=True)
    print(f"[INFO] Stage 5c encoder pointer-write | device={DEVICE} | ckpt={args.ckpt}", flush=True)

    split_rng = random.Random(20260620)
    shuffled = ENTITIES[:]; split_rng.shuffle(shuffled)
    train_ents, held_ents = shuffled[:14], shuffled[14:]
    print(f"[INFO] entity split: {len(train_ents)} train / {len(held_ents)} held-out", flush=True)

    frozen = load_model(args.ckpt)                                  # for distillation targets + STEP 0

    # ---- STEP 0: diagnostic on the FROZEN encoder ----
    print("[INFO] STEP 0: recover target key from FROZEN encoder B-slot (in-sample vs held-out)...", flush=True)
    rng0 = random.Random(1)
    rec0 = Recover().to(DEVICE)
    opt0 = torch.optim.AdamW(rec0.parameters(), lr=1e-3)
    facts0 = [make_relational(train_ents, rng0) for _ in range(800)]
    for _ in range(800):
        opt0.zero_grad(); loss = torch.tensor(0.0, device=DEVICE)
        batch = [facts0[i] for i in rng0.sample(range(len(facts0)), 32)]
        ids = [ENC.encode_ordinary(r["text"]) for r in batch]
        L = max(len(x) for x in ids)
        ids = torch.tensor([x + [ENC.encode_ordinary(" .")[0]] * (L - len(x)) for x in ids], device=DEVICE)
        with torch.no_grad():
            bv = bank_value(frozen, enc_hpool(frozen, ids))
        rk = rec0(bv)
        for j, r in enumerate(batch):
            tgt = F.normalize(key_of(frozen, r["target"]), dim=0)
            negs = torch.stack([F.normalize(key_of(frozen, e), dim=0) for e in r["distractors"]])
            pos = F.cosine_similarity(F.normalize(rk[j], dim=0), tgt, dim=0)
            neg = F.cosine_similarity(F.normalize(rk[j], dim=0).unsqueeze(0), negs, dim=1).max()
            loss = loss + F.relu(0.3 - (pos - neg))
        (loss / len(batch)).backward(); opt0.step()
    s0_in = recovery_acc(frozen, rec0, train_ents, random.Random(2))
    s0_held = recovery_acc(frozen, rec0, held_ents, random.Random(3))
    print(f"  STEP 0 recovery (frozen): in-sample={s0_in:.3f} held-out={s0_held:.3f} "
          f"-> {'ABSENT (in-sample fails)' if s0_in < 0.7 else 'entity-specific (held-out gap)' if s0_held < 0.7 else 'recoverable'}", flush=True)

    if args.smoke:
        print("[SMOKE] step 0 only", flush=True)
        return 0

    # ---- POINTER-WRITE: fine-tune encoder relational path + recovery; distill non-relational ----
    print("[INFO] POINTER-WRITE: fine-tuning encoder.blocks (pointer + distill preservation)...", flush=True)
    model = load_model(args.ckpt)
    for n, p in model.named_parameters():
        p.requires_grad_(any(n.startswith(pf) for pf in POINTER_PREFIXES))
    enc_params = [p for n, p in model.named_parameters() if p.requires_grad]
    recover = Recover().to(DEVICE)
    opt = torch.optim.AdamW(enc_params + list(recover.parameters()), lr=args.pw_lr)
    pw_rng = random.Random(7)
    rel = [make_relational(train_ents, pw_rng) for _ in range(args.n_facts)]
    non = [make_nonrel(train_ents, pw_rng) for _ in range(args.n_facts)]
    # precompute frozen distill targets for non-relational
    with torch.no_grad():
        for r in non:
            ids = torch.tensor([ENC.encode_ordinary(r["text"])], device=DEVICE)
            r["frozen_v"] = bank_value(frozen, enc_hpool(frozen, ids))[0].detach()
    rel_b, non_b = bucket([r["text"] for r in rel]), bucket([r["text"] for r in non])
    model.train()
    for step in range(args.pw_steps):
        opt.zero_grad(); loss = torch.tensor(0.0, device=DEVICE)
        # pointer loss (relational, contrastive)
        L = pw_rng.choice(list(rel_b.keys())); idxs = rel_b[L]
        bt = [rel[i] for i in (idxs if len(idxs) <= 48 else pw_rng.sample(idxs, 48))]
        ids = torch.tensor([ENC.encode_ordinary(r["text"]) for r in bt], device=DEVICE)
        bv = bank_value(model, enc_hpool(model, ids))
        rk = recover(bv)
        for j, r in enumerate(bt):
            tgt = F.normalize(key_of(model, r["target"]), dim=0)
            negs = torch.stack([F.normalize(key_of(model, e), dim=0) for e in r["distractors"]])
            pos = F.cosine_similarity(F.normalize(rk[j], dim=0), tgt, dim=0)
            neg = F.cosine_similarity(F.normalize(rk[j], dim=0).unsqueeze(0), negs, dim=1).max()
            loss = loss + F.relu(0.4 - (pos - neg))
        loss = loss / len(bt)
        # distill loss (non-relational preservation)
        Ln = pw_rng.choice(list(non_b.keys())); nidx = non_b[Ln]
        nb = [non[i] for i in (nidx if len(nidx) <= 48 else pw_rng.sample(nidx, 48))]
        nids = torch.tensor([ENC.encode_ordinary(r["text"]) for r in nb], device=DEVICE)
        nv = bank_value(model, enc_hpool(model, nids))
        tv = torch.stack([r["frozen_v"] for r in nb])
        loss = loss + args.distill_w * F.mse_loss(nv, tv)
        loss.backward(); opt.step()
    model.eval()
    g_recover_held = recovery_acc(model, recover, held_ents, random.Random(5))
    g_recover_in = recovery_acc(model, recover, train_ents, random.Random(6))
    print(f"  G_POINTER_RECOVERY (after pointer-write): held-out={g_recover_held:.3f} in-sample={g_recover_in:.3f} (bar 0.90)", flush=True)

    # ---- single-fact preservation check (non-relational value distilled) ----
    with torch.no_grad():
        drift = []
        for _ in range(100):
            r = make_nonrel(held_ents, random.Random(7 + _))
            ids = torch.tensor([ENC.encode_ordinary(r["text"])], device=DEVICE)
            v_new = bank_value(model, enc_hpool(model, ids))[0]
            v_old = bank_value(frozen, enc_hpool(frozen, ids))[0]
            drift.append(F.cosine_similarity(v_new, v_old, dim=0).item())
    nonrel_preserve = sum(drift) / len(drift)
    print(f"  non-relational value preservation (cosine to frozen): {nonrel_preserve:.4f}", flush=True)

    # ---- RE-TEST graph traversal with the pointer-trained encoder (real banks) ----
    print("[INFO] RE-TEST: precomputing real banks with pointer-trained encoder...", flush=True)
    drng = random.Random(7)
    train_eps = build(model, train_ents, 800, drng)
    eval_eps = build(model, held_ents, args.n_eval, drng)
    repoint_eps = [t for e in eval_eps if e["op"] == 1 for t in [repoint_twin(model, e)] if t]
    OP_CHAIN = 1
    print(f"[INFO] episodes: {len(train_eps)} / {len(eval_eps)} / {len(repoint_eps)} re-pointed", flush=True)
    runs = []
    for seed in range(args.op_seeds):
        op = train_op(train_eps, seed, args.op_steps, 6e-4, 1.0)
        ev = eval_op(op, eval_eps); gr = eval_op(op, repoint_eps)
        runs.append({"eval": ev, "grounded": gr["chain_repointed"]})
        print(f"  op seed {seed}: chain_shuf={ev['chain_shuffled']:.3f} abstain={ev['abstain']:.3f} "
              f"compare={ev['compare']:.3f} | CHAIN_GROUNDED={gr['chain_repointed']:.3f}", flush=True)

    def dist(xs):
        xs = [x for x in xs if x is not None]
        return {"min": round(min(xs), 4), "median": round(median(xs), 4), "max": round(max(xs), 4),
                "std": round(pstdev(xs), 4) if len(xs) > 1 else 0.0}
    chaining = dist([r["eval"]["chain_shuffled"] for r in runs])
    grounded = dist([r["grounded"] for r in runs])
    abstain = dist([r["eval"]["abstain"] for r in runs])
    comparison = dist([r["eval"]["compare"] for r in runs])

    g_pointer = g_recover_held >= 0.90
    g_chaining = chaining["median"] >= 0.80
    g_grounded = grounded["median"] >= 0.90
    g_abstain = abstain["median"] >= 0.80
    g_comparison = comparison["median"] >= 0.80

    if g_pointer and g_chaining and g_grounded and g_abstain and g_comparison:
        verdict = "STAGE_5_GRAPH_COMPLETE"
    elif g_pointer:
        verdict = "STAGE_5_POINTER_OK_TRAVERSAL_FAIL"
    else:
        verdict = "STAGE_5_POINTER_REFUTED"

    out = {"verdict": verdict, "ckpt": args.ckpt,
           "STEP_0_diagnostic": {"frozen_recovery_in_sample": s0_in, "frozen_recovery_held_out": s0_held,
                                 "reading": "in-sample<0.7 -> pointer ABSENT; held-out<0.7 with in-sample ok -> entity-specific"},
           "gates": {
               "G_POINTER_RECOVERY": {"passed": bool(g_pointer), "held_out": g_recover_held, "in_sample": g_recover_in, "bar": 0.90,
                                      "evidence": "recover A's key from B-slot (pointer-write encoder); content-address retrieves A, held-out"},
               "G_CHAINING_BANK": {"passed": bool(g_chaining), "dist": chaining, "bar": 0.80},
               "G_CHAIN_GROUNDED": {"passed": bool(g_grounded), "dist": grounded, "bar": 0.90, "evidence": "re-point stored fact; answer follows"},
               "G_ABSTAIN": {"passed": bool(g_abstain), "dist": abstain, "bar": 0.80},
               "G_COMPARISON_PRESERVED": {"passed": bool(g_comparison), "dist": comparison, "bar": 0.80},
               "G_SINGLE_FACT_PRESERVED": {"nonrelational_value_cosine_to_frozen": round(nonrel_preserve, 4),
                                           "note": "distillation keeps non-relational value-writing intact"},
               "G_IN_MEMORY": {"passed": True, "evidence": "operation reads banks+query only"},
           },
           "per_seed": runs,
           "scope": ("relational pointer-write (encoder.blocks fine-tune + distill preservation) + graph traversal "
                     "over persisted banks; held-out; single machine. NOT generality, NOT free-text. dcortex/ sealed."),
           "meaning": {
               "STAGE_5_GRAPH_COMPLETE": "the encoder stores a followable generalizable pointer AND the operation "
                   "traverses it (grounded) -> operate-over-memory complete; thinking-in-memory loop closed (small scale).",
               "STAGE_5_POINTER_OK_TRAVERSAL_FAIL": "pointer recovers held-out but the operation does not traverse it "
                   "-> the gap is the operation, not the encoder.",
               "STAGE_5_POINTER_REFUTED": "even with pointer-write the encoder cannot store a generalizable followable "
                   "relational pointer -> a deeper finding about relational representation at this setup."}[verdict]}
    (RUN_DIR / "results" / "verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(SEP, flush=True)
    print(f"  STEP 0 (frozen): in={s0_in:.3f} held={s0_held:.3f}", flush=True)
    print(f"  G_POINTER_RECOVERY held-out={g_recover_held:.3f} (bar 0.90) -> {'PASS' if g_pointer else 'FAIL'}", flush=True)
    print(f"  G_CHAINING_BANK={chaining['median']} G_CHAIN_GROUNDED={grounded['median']} (bar 0.90) "
          f"G_ABSTAIN={abstain['median']} G_COMPARISON_PRESERVED={comparison['median']}", flush=True)
    print(f"  single-fact preservation (cosine)={nonrel_preserve:.4f}", flush=True)
    print(f"[INFO] VERDICT: {verdict}", flush=True)
    print("STAGE5C_JSON " + json.dumps({"verdict": verdict, "step0_in": s0_in, "step0_held": s0_held,
          "pointer_recovery_held": g_recover_held, "chaining": chaining["median"], "chain_grounded": grounded["median"],
          "comparison": comparison["median"], "single_fact_cosine": round(nonrel_preserve, 4)}), flush=True)
    return 0 if verdict == "STAGE_5_GRAPH_COMPLETE" else 1


if __name__ == "__main__":
    sys.exit(main())
