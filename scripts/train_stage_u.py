# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Stage U - Step 2 training: train DCortexV2Model on the structural F/L regime with its NATIVE
# losses (L_emit answer-from-memory + L_sel key-query alignment + L_sep_neg key separation -
# replicated from colab/step2_training_v11.py), and MEASURE the value-separability margin
# against the Step 1 bar at intervals. The order is the owner's: TRAIN (natural objective) ->
# MEASURE -> constrain only if the margin does not cross 0. This run adds NO separability loss
# on values; it asks whether the natural objective alone lifts the margin from the init
# baseline (~ -0.03 at lexical_alpha 0.9) above 0. Structural episodes only (LM episodes from
# the reference are omitted to keep the harness self-contained; noted as a simplification).

import argparse
import contextlib
import io
import json
import random
import sys
from pathlib import Path

import torch
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
from stage_u.margin_probe import collect_values, margin
from stage_u.observe_geometry import observe

SEP = "=" * 70
RUN_DIR = REPO_ROOT / "runs" / "stage_u"
ENC = tiktoken.get_encoding("gpt2")
EOT = ENC.eot_token
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENTITIES = ["cat", "dog", "bird", "fish", "rabbit", "horse", "bear", "fox", "lion", "tiger",
            "monkey", "penguin", "owl", "wolf", "deer", "dragon", "knight", "wizard", "princess",
            "fairy", "goblin", "witch", "pirate", "giant", "ghost", "robot", "queen", "king", "dwarf", "elf"]
COLORS = ["red", "blue", "green", "yellow", "black", "white", "brown", "pink", "orange", "purple",
          "golden", "silver", "crimson", "gray", "violet"]
ANIMALS, FANTASY = ENTITIES[:15], ENTITIES[15:]

# TrainConfig. NOTE: training is batch=1 (the model's memory banks are per-episode stateful,
# not batched), so the GPU sits ~20% utilized regardless of device - this is a throughput limit
# of the architecture, not a CPU fallback. We shrink per-step cost (shorter seq, fewer facts,
# smaller grad_accum) to reach an observable trained state fast; true GPU utilization would need
# a batched-memory model change (flagged for later).
SEQ_LEN, GRAD_ACCUM, LR, MIN_LR, WD = 32, 16, 6e-4, 6e-5, 0.1
WARMUP, MIN_FACTS, MAX_FACTS = 150, 3, 3
W_EMIT, W_SEL, W_SEP, LEX_ALPHA = 1.0, 1.0, 0.5, 0.9
SIMPLE, UPDATE = 0.5, 0.25


def _pad(ids, n):
    return ids[:n] if len(ids) > n else ids + [EOT] * (n - len(ids))


def _fact(e, c, idx):
    return {"text": f"The {e} is {c}.", "entity": e, "value": c, "idx": idx,
            "ans": ENC.encode_ordinary(f" {c}")[0]}


def gen_episode(rng):
    r = rng.random()
    n = rng.randint(MIN_FACTS, MAX_FACTS)
    if r < SIMPLE:                                            # simple
        ents, cols = rng.sample(ENTITIES, n), rng.sample(COLORS, n)
        facts = [_fact(e, c, i) for i, (e, c) in enumerate(zip(ents, cols))]
        t = rng.randint(0, n - 1)
        return {"facts": facts, "prompt": f"What color is the {ents[t]}? The {ents[t]} is",
                "target": t, "ans": facts[t]["ans"], "type": "simple"}
    if r < SIMPLE + UPDATE:                                   # update
        ents, cols = rng.sample(ENTITIES, n), rng.sample(COLORS, n + 1)
        facts = [_fact(e, c, i) for i, (e, c) in enumerate(zip(ents, cols[:n]))]
        ut = rng.randint(0, n - 1)
        nc = cols[n]
        facts.append({"text": f"The {ents[ut]} is now {nc}.", "entity": ents[ut], "value": nc,
                      "idx": n, "ans": ENC.encode_ordinary(f" {nc}")[0]})
        return {"facts": facts, "prompt": f"What color is the {ents[ut]} now? The {ents[ut]} is",
                "target": n, "ans": facts[-1]["ans"], "type": "update"}
    cluster = rng.choice([ANIMALS, FANTASY])                  # distractor
    n = min(n, len(cluster))
    ents, cols = rng.sample(cluster, n), rng.sample(COLORS, n)
    facts = [_fact(e, c, i) for i, (e, c) in enumerate(zip(ents, cols))]
    t = rng.randint(0, n - 1)
    return {"facts": facts, "prompt": f"What color is the {ents[t]}? The {ents[t]} is",
            "target": t, "ans": facts[t]["ans"], "type": "distractor"}


def structural_loss(model, ep, alpha=LEX_ALPHA):
    with contextlib.redirect_stdout(io.StringIO()):
        model.reset_memory()
    model.begin_episode()
    keys = []
    for f in ep["facts"]:
        xf = torch.tensor([_pad(ENC.encode_ordinary(f["text"]) + [EOT], SEQ_LEN)], device=DEVICE)
        ans = torch.tensor([f["ans"]], device=DEVICE)
        aux = model.encode(xf, answer_token_id=ans, lexical_alpha=alpha, force_bank="working")
        keys.append(aux["w_k_ent"][0])
    xp = torch.tensor([_pad(ENC.encode_ordinary(ep["prompt"]), SEQ_LEN)], device=DEVICE)
    logits, retrieved = model.decode(xp, return_retrieved=True)
    aux_logits = model.aux_answer_head(retrieved)
    tgt = torch.tensor([ep["ans"]], device=DEVICE)
    l_emit = F.cross_entropy(aux_logits, tgt)
    # L_sel: query-key alignment to the target fact
    pos = torch.arange(xp.shape[1], device=DEVICE).unsqueeze(0)
    q_emb = model.shared_token_emb(xp) + model.shared_pos_emb(pos)
    q_addr = model.shared_address_encoder(q_emb)
    q_k_ent, _, _ = model.shared_query_engine(q_addr)
    Kn = F.normalize(torch.stack(keys, 0), dim=-1)
    sim = (F.normalize(q_k_ent, dim=-1) @ Kn.t()).squeeze(0)
    l_sel = -F.log_softmax(sim * 5.0, dim=-1)[ep["target"]]
    # L_sep_neg: push distinct fact keys apart
    l_sep = torch.tensor(0.0, device=DEVICE)
    if len(keys) >= 2:
        s = Kn @ Kn.t()
        mask = torch.eye(len(keys), device=DEVICE, dtype=torch.bool)
        l_sep = F.relu(s[~mask] - 0.5).pow(2).mean()
    total = W_EMIT * l_emit + W_SEL * l_sel + W_SEP * l_sep
    top1 = int(aux_logits[0].argmax().item() == ep["ans"])
    return total, {"emit": l_emit.item(), "sel": l_sel.item(), "sep": float(l_sep), "top1": top1}


def measure(model, alphas=(0.9, 0.0)):
    """Full geometry (linear probe = the internalization signal) at each requested alpha."""
    model.eval()
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for alpha in alphas:
            out[f"alpha{alpha}"] = observe(model, ENC, alpha, device=DEVICE, seed=1234)
    model.train()
    return out


def main() -> int:
    import math
    import statistics
    ap = argparse.ArgumentParser(description="Stage U training (Step 2 baseline / Step B internalization)")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--measure-every", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--lexical-alpha", type=float, default=LEX_ALPHA,
                    help="TRAINING lexical_alpha (0.0 = copy-proof STRICT regime; Step B)")
    ap.add_argument("--anneal-alpha", type=float, nargs=2, default=None, metavar=("FROM", "TO"),
                    help="linearly anneal training lexical_alpha FROM->TO over the first half (Step B ANNEAL)")
    ap.add_argument("--measure-alpha", type=float, default=0.0,
                    help="alpha at which the linear probe (the internalization signal) is measured")
    args = ap.parse_args()
    (RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    ma = round(args.measure_alpha, 3)
    meas_alphas = [ma] if ma == 0.9 else [ma, 0.9]          # measure-alpha + 0.9 reference

    def alpha_at(step):
        if args.anneal_alpha is None:
            return args.lexical_alpha
        a0, a1 = args.anneal_alpha
        return a0 + (a1 - a0) * min(1.0, step / max(1, args.steps // 2))

    def probe(geom, alpha):
        return geom[f"alpha{alpha}"]["linear_probe_accuracy"]

    train_desc = f"anneal{args.anneal_alpha}" if args.anneal_alpha else f"{args.lexical_alpha}"
    print(SEP, flush=True)
    print(f"[INFO] Stage U training on {DEVICE}; steps={args.steps} grad_accum={GRAD_ACCUM} "
          f"train_alpha={train_desc} measure_alpha={ma}", flush=True)
    if DEVICE != "cuda":
        print("[WARN] no CUDA - this will be slow", flush=True)

    with contextlib.redirect_stdout(io.StringIO()):
        model = DCortexV2Model(DCortexConfig()).to(DEVICE)
    model.train()
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 else nodecay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD},
                             {"params": nodecay, "weight_decay": 0.0}], lr=LR, betas=(0.9, 0.95))

    def lr_at(step):
        if step < WARMUP:
            return LR * (step + 1) / WARMUP
        prog = (step - WARMUP) / max(1, args.steps - WARMUP)
        return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * min(1.0, prog)))

    traj = []
    m0 = measure(model, meas_alphas)
    traj.append({"step": 0, "geometry": m0})
    print(f"  step 0 (init): probe@{ma}={probe(m0, ma):.3f} probe@0.9={probe(m0, 0.9):.3f}", flush=True)

    run_top1, run_emit = [], []
    for step in range(args.steps):
        a = alpha_at(step)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        acc_top1 = acc_emit = 0.0
        for _ in range(GRAD_ACCUM):
            ep = gen_episode(rng)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16) if DEVICE == "cuda" else contextlib.nullcontext():
                loss, m = structural_loss(model, ep, alpha=a)
            (loss / GRAD_ACCUM).backward()
            acc_top1 += m["top1"] / GRAD_ACCUM
            acc_emit += m["emit"] / GRAD_ACCUM
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run_top1.append(acc_top1); run_emit.append(acc_emit)
        if (step + 1) % 50 == 0:
            print(f"  step {step+1:4d} | lr {lr_at(step):.2e} | a {a:.2f} | "
                  f"top1 {statistics.mean(run_top1[-50:]):.3f} | L_emit {statistics.mean(run_emit[-50:]):.3f}", flush=True)
        if (step + 1) % args.measure_every == 0:
            mm = measure(model, meas_alphas)
            t1 = round(statistics.mean(run_top1[-args.measure_every:]), 3)
            traj.append({"step": step + 1, "geometry": mm, "top1": t1})
            g = mm[f"alpha{ma}"]
            print(f"  >>> step {step+1}: probe@{ma}={g['linear_probe_accuracy']:.3f} "
                  f"decode@{ma}={g['decode_head_value_accuracy']:.3f} top1~{t1:.3f}", flush=True)
            (RUN_DIR / "results" / "train_trajectory.json").write_text(
                json.dumps({"trajectory": traj, "steps_done": step + 1}, indent=2), encoding="utf-8")

    # final geometry + INTERNALIZATION VERDICT on the measure-alpha linear probe (the Step B gate)
    print(SEP, flush=True)
    final = traj[-1]["geometry"]
    (RUN_DIR / "results" / "organic_geometry.json").write_text(
        json.dumps({"organic": final, "top1_final": traj[-1].get("top1")}, indent=2), encoding="utf-8")
    try:
        torch.save({"model": model.state_dict(), "steps": args.steps}, RUN_DIR / "results" / "ckpt_stage_u.pt")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] checkpoint save failed: {exc}", flush=True)

    # Gate on DECODE (the model's OWN answer-head reading the value), NOT the linear probe:
    # the probe is high-from-init (value-identity is linearly present in w_value even untrained),
    # so a probe-RISE gate mislabels; decode rising 0->high is the internalization signal.
    # ACHIEVED is only confirmed by the adversarial control (stage_u/control_internalization.py).
    def dec(geom, alpha):
        return geom[f"alpha{alpha}"].get("decode_head_value_accuracy", 0.0)
    p0, pF = probe(m0, ma), probe(final, ma)
    d0, dF = dec(m0, ma), dec(final, ma)
    t1F = traj[-1].get("top1") or 0.0
    rose = dF > d0 + 0.10
    if dF >= 0.80 and rose:
        verdict = "INTERNALIZATION_ACHIEVED_PENDING_CONTROL"
    elif dF < 0.50:
        verdict = "INTERNALIZATION_REFUTED"
    else:
        verdict = "INTERNALIZATION_PARTIAL"
    summary = {"verdict": verdict, "measure_alpha": ma, "train_alpha": train_desc,
               "decode_init": round(d0, 4), "decode_final": round(dF, 4), "decode_rose": bool(rose),
               "probe_init": round(p0, 4), "probe_final": round(pF, 4),
               "top1_final": t1F, "steps": args.steps, "trajectory": traj,
               "honest_scope": ("single model, small synthetic regime, single machine; the linear PROBE "
                                "(not top1) is the internalization gate; NOT a generality claim.")}
    (RUN_DIR / "results" / "train_trajectory.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO] VERDICT: {verdict} | decode@{ma} {d0:.3f} -> {dF:.3f} (rose={rose}) | "
          f"probe@{ma} {pF:.3f} | top1 {t1F:.3f} (control required to confirm ACHIEVED)", flush=True)
    print("STAGE_U_TRAIN_JSON " + json.dumps({"verdict": verdict, "decode_init": round(d0, 4),
          "decode_final": round(dF, 4), "probe_final": round(pF, 4), "top1": t1F,
          "measure_alpha": ma, "train_alpha": train_desc}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
