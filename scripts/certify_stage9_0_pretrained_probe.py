# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# STAGE 9.0 - FROZEN PRETRAINED BINDING READOUT PROBE. Stage 8 measured that the small from-scratch
# closed-vocab substrate does NOT abstract phrasing-invariance (value-binding on the structurally-
# distinct Family-B held-out = 0.337, near chance), and concluded paraphrase-robustness is a PRETRAINING
# property. This probe tests that premise DIRECTLY and CHEAPLY: does a FROZEN pretrained base already
# expose entity-value binding (and the relational pointer = traversal precondition) from the SAME
# structurally-distinct phrasings, where the small substrate failed?
#
# NECESSARY-not-sufficient: 9.0 proves the binding is THERE in the pretrained hidden states. Whether it
# is USABLE-as-honest-memory is Stage 9.1 (re-stabilize the proven D_Cortex DESIGN on these hidden
# states, with the anti-cheat controls). This file does NOT build memory; it only reads hidden states.
#
# CROSS-MODEL: Qwen2.5-7B-Instruct AND Mistral-7B-Instruct-v0.3 (4-bit NF4, frozen). A binding-exposure
# result on two independent pretrained bases is far more credible than on one.
#
# VALIDITY: SAME regime as Stage 8 (train Family A = entity-then-value; eval Family B = value-first /
# inverted / embedded, STRUCTURALLY DISTINCT). DOUBLE held-out (unseen entities AND unseen phrasings).
# Wrong-binding (cross-bound to the sibling) is the dangerous metric. Ladder linear -> MLP (linear = the
# binding is linearly there; MLP = nonlinearly there). The probe is trained; the base is FROZEN.

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

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from scripts.certify_stage_i_extraction import ENTITIES, SIZES
from scripts.certify_stage5_operate_memory import COLORS, COLOR_IDX
from scripts.certify_stage8_phrasing_scale import VALUE_A, VALUE_B, REL_A, REL_B

SEP = "=" * 70
RUN_DIR = REPO_ROOT / "runs" / "stage9_0_probe"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIZE_RANK = {s: i for i, s in enumerate(SIZES)}
LAYER_FRAC = 0.66                                          # declared up front: a mid-late hidden layer
MODELS = ["Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]


def load_4bit(model_id: str):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_id)
    with contextlib.redirect_stdout(io.StringIO()):
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb,
                                                     device_map={"": 0}, output_hidden_states=True)
    model.eval()
    return tok, model


@torch.no_grad()
def hidden_layer(model, tok, text: str, layer: int) -> Tuple[torch.Tensor, "list"]:
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc)
    h = out.hidden_states[layer][0].float().cpu()          # [T, D]
    return h, offsets


def entity_token_pos(text: str, entity: str, offsets) -> Optional[int]:
    # last token whose char span overlaps the entity word occurrence
    ci = text.find(entity)
    if ci < 0:
        return None
    cj = ci + len(entity)
    last = None
    for ti, (a, b) in enumerate(offsets):
        if a == b:
            continue
        if a < cj and b > ci:                              # token overlaps [ci, cj)
            last = ti
    return last


# ---------------------------------------------------------------------------
# Probe ladder (linear + MLP). Base FROZEN; only the probe trains.
# ---------------------------------------------------------------------------
class LinearProbe(nn.Module):
    def __init__(self, d: int, n: int) -> None:
        super().__init__()
        self.net = nn.Linear(d, n)

    def forward(self, x):
        return self.net(x)


class MLPProbe(nn.Module):
    def __init__(self, d: int, n: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, n))

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Build cached reps for value scenes (Family A train / Family B eval) and relation scenes
# ---------------------------------------------------------------------------
def value_reps(model, tok, layer, ents, phrasings, n, rng):
    out = []
    for _ in range(n):
        if len(ents) < 2:
            break
        e0, e1 = rng.sample(ents, 2)
        v0, v1 = rng.sample(SIZES, 2)
        text = rng.choice(phrasings).format(e=e0, v=v0) + " " + rng.choice(phrasings).format(e=e1, v=v1)
        h, off = hidden_layer(model, tok, text, layer)
        p0, p1 = entity_token_pos(text, e0, off), entity_token_pos(text, e1, off)
        if p0 is None or p1 is None or p0 == p1:
            continue
        out.append((h[p0], SIZE_RANK[v0], SIZE_RANK[v1]))
        out.append((h[p1], SIZE_RANK[v1], SIZE_RANK[v0]))
    return out


def relation_reps(model, tok, layer, ents, phrasings, n, rng):
    out = []
    for _ in range(n):
        if len(ents) < 3:
            break
        A, B, C = rng.sample(ents, 3)
        cA, cC = rng.sample(COLORS, 2)
        text = "The " + A + " is " + cA + ". " + rng.choice(phrasings).format(b=B, a=A) + " The " + C + " is " + cC + "."
        h, off = hidden_layer(model, tok, text, layer)
        pB = entity_token_pos(text, B, off)
        pA = entity_token_pos(text, A, off)
        pC = entity_token_pos(text, C, off)
        if None in (pA, pB, pC) or len({pA, pB, pC}) < 3:
            continue
        out.append({"subj": h[pB], "target": h[pA], "distractor": h[pC]})  # target = A (the pointed-to)
    return out


def probe_value(train, evl, ProbeCls, seed, steps=1500, lr=2e-3):
    random.seed(seed); torch.manual_seed(seed)
    d = train[0][0].shape[0]
    Xtr = torch.stack([r for r, _, _ in train]).to(DEVICE)
    ytr = torch.tensor([g for _, g, _ in train], device=DEVICE)
    probe = ProbeCls(d, len(SIZES)).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    probe.train()
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(probe(Xtr), ytr).backward(); opt.step()
    probe.eval()
    Xev = torch.stack([r for r, _, _ in evl]).to(DEVICE)
    yev = torch.tensor([g for _, g, _ in evl], device=DEVICE)
    sib = torch.tensor([s for _, _, s in evl], device=DEVICE)
    with torch.no_grad():
        pred = torch.argmax(probe(Xev), dim=1)
    return (pred == yev).float().mean().item(), (pred == sib).float().mean().item()


def probe_relation(train, evl, seed, steps=1500, lr=2e-3):
    random.seed(seed); torch.manual_seed(seed)
    d = train[0]["subj"].shape[0]
    proj = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, d)).to(DEVICE)
    opt = torch.optim.AdamW(proj.parameters(), lr=lr)
    Subj = torch.stack([r["subj"] for r in train]).to(DEVICE)
    Tgt = torch.stack([r["target"] for r in train]).to(DEVICE)
    proj.train()
    for _ in range(steps):
        opt.zero_grad()
        pk = F.normalize(proj(Subj), dim=1)
        loss = (1.0 - F.cosine_similarity(pk, F.normalize(Tgt, dim=1), dim=1)).mean()
        loss.backward(); opt.step()
    proj.eval()
    ok = wrong = 0
    with torch.no_grad():
        for r in evl:
            pk = F.normalize(proj(r["subj"].unsqueeze(0).to(DEVICE)), dim=1)[0]
            cands = torch.stack([r["target"], r["distractor"]]).to(DEVICE)
            sims = F.normalize(cands, dim=1) @ pk
            pick = int(torch.argmax(sims).item())           # 0 = target (correct), 1 = distractor
            ok += int(pick == 0); wrong += int(pick == 1)
    n = len(evl)
    return ok / n, wrong / n


def dist(xs):
    return {"min": round(min(xs), 4), "median": round(median(xs), 4), "max": round(max(xs), 4),
            "std": round(pstdev(xs), 4) if len(xs) > 1 else 0.0}


def probe_model(model_id, train_ents, held_ents, n_train, n_eval, seeds):
    print(f"[INFO] loading {model_id} (4-bit NF4, frozen)...", flush=True)
    tok, model = load_4bit(model_id)
    n_layers = model.config.num_hidden_layers
    layer = int(LAYER_FRAC * n_layers)
    print(f"[INFO] {model_id}: {n_layers} layers, probing hidden layer {layer} (frac {LAYER_FRAC})", flush=True)
    drng = random.Random(7)
    print("[INFO] caching frozen hidden-state reps (this is the slow part)...", flush=True)
    v_train = value_reps(model, tok, layer, train_ents, VALUE_A, n_train, drng)
    v_eval = value_reps(model, tok, layer, held_ents, VALUE_B, n_eval, drng)     # Family B, double held-out
    r_train = relation_reps(model, tok, layer, train_ents, REL_A, n_train, drng)
    r_eval = relation_reps(model, tok, layer, held_ents, REL_B, n_eval, drng)
    del model
    torch.cuda.empty_cache()
    print(f"[INFO] reps: value {len(v_train)}/{len(v_eval)} | relation {len(r_train)}/{len(r_eval)}", flush=True)

    lin_c, lin_w, mlp_c, mlp_w, rel_o, rel_w = [], [], [], [], [], []
    for s in range(seeds):
        c, w = probe_value(v_train, v_eval, LinearProbe, s); lin_c.append(c); lin_w.append(w)
        c, w = probe_value(v_train, v_eval, MLPProbe, s); mlp_c.append(c); mlp_w.append(w)
        ro, rw = probe_relation(r_train, r_eval, s); rel_o.append(ro); rel_w.append(rw)
    res = {"layer": layer, "n_layers": n_layers,
           "value_linear": dist(lin_c), "value_linear_wrong": dist(lin_w),
           "value_mlp": dist(mlp_c), "value_mlp_wrong": dist(mlp_w),
           "relation_bind": dist(rel_o), "relation_wrong_dir": dist(rel_w)}
    best_v = max(res["value_linear"]["median"], res["value_mlp"]["median"])
    res["value_best"] = best_v
    print(f"  [{model_id}] value_linear={res['value_linear']['median']} mlp={res['value_mlp']['median']} "
          f"(best {best_v}) wrong={min(res['value_linear_wrong']['median'], res['value_mlp_wrong']['median'])} | "
          f"relation={res['relation_bind']['median']} wrong_dir={res['relation_wrong_dir']['median']}", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 9.0 frozen pretrained binding probe")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=250)
    ap.add_argument("--n-eval", type=int, default=120)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    (RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
    print(SEP, flush=True)
    print(f"[INFO] Stage 9.0 frozen pretrained binding probe | device={DEVICE}", flush=True)
    print(f"[INFO] Family A (train, entity-then-value) {len(VALUE_A)} | Family B (eval, structurally distinct) "
          f"{len(VALUE_B)} | the SAME test where the small substrate got value 0.337", flush=True)

    split_rng = random.Random(20260620)
    sh = ENTITIES[:]; split_rng.shuffle(sh)
    train_ents, held_ents = sh[:14], sh[14:]
    if args.smoke:
        r = probe_model(args.models[0], train_ents, held_ents, 20, 20, 1)
        print(f"  [SMOKE] {r['value_best']}", flush=True)
        return 0

    per_model = {}
    for mid in args.models:
        per_model[mid] = probe_model(mid, train_ents, held_ents, args.n_train, args.n_eval, args.seeds)

    # cross-model verdict (calibrated thresholds from the design)
    BAR_VALUE, BAR_WRONG = 0.70, 0.10
    all_value = [m["value_best"] for m in per_model.values()]
    all_wrong = [min(m["value_linear_wrong"]["median"], m["value_mlp_wrong"]["median"]) for m in per_model.values()]
    both_pass = all(v >= BAR_VALUE for v in all_value) and all(w <= BAR_WRONG for w in all_wrong)
    any_pass = any(v >= BAR_VALUE for v in all_value)
    if both_pass:
        verdict = "PRETRAINING_EXPOSES_BINDING"
    elif any_pass:
        verdict = "PRETRAINING_EXPOSES_BINDING_PARTIAL"
    else:
        verdict = "PRETRAINING_BINDING_REFUTED"

    out = {"verdict": verdict, "models": args.models, "layer_frac": LAYER_FRAC,
           "bars": {"value_binding": BAR_VALUE, "wrong_binding": BAR_WRONG},
           "small_substrate_baseline_familyB": 0.337,
           "per_model": per_model,
           "scope": ("frozen pretrained binding probe; SAME structurally-distinct Family-B test as Stage 8; double "
                     "held-out (entities AND phrasings); cross-model (Qwen + Mistral). NECESSARY-not-sufficient: "
                     "proves binding is in the hidden states, not that it is usable-as-honest-memory (that is 9.1)."),
           "meaning": {
               "PRETRAINING_EXPOSES_BINDING": "both pretrained bases expose value-binding from structurally-distinct "
                   "phrasings (>=0.70, wrong<=0.10) where the small substrate failed (0.337) -> paraphrase-invariant "
                   "binding IS a pretraining property; the premise for Stage 9.1 (re-stabilize the D_Cortex design on "
                   "pretrained hidden states) holds. Confirm addressing (relation probe) before 9.1.",
               "PRETRAINING_EXPOSES_BINDING_PARTIAL": "one base exposes it, the other does not, or wrong-binding is "
                   "above bar -> model-dependent; investigate before committing to 9.1.",
               "PRETRAINING_BINDING_REFUTED": "neither frozen pretrained base exposes the binding from structurally-"
                   "distinct phrasings either -> binding is not a simple readout even on a pretrained base; the port "
                   "needs more than a frozen readout (e.g. a trained adapter, or the problem is deeper)."}[verdict]}
    (RUN_DIR / "results" / "verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(SEP, flush=True)
    print("[INFO] vs small substrate Family-B value-binding = 0.337 (Stage 8):", flush=True)
    for mid, m in per_model.items():
        print(f"  {mid}: value_best={m['value_best']} wrong={min(m['value_linear_wrong']['median'], m['value_mlp_wrong']['median'])} "
              f"relation={m['relation_bind']['median']} (layer {m['layer']}/{m['n_layers']})", flush=True)
    print(f"[INFO] VERDICT: {verdict}", flush=True)
    print("STAGE9_0_JSON " + json.dumps({"verdict": verdict, "value_best_per_model": dict(zip(args.models, all_value)),
          "wrong_per_model": dict(zip(args.models, all_wrong))}), flush=True)
    return 0 if verdict == "PRETRAINING_EXPOSES_BINDING" else 1


if __name__ == "__main__":
    sys.exit(main())
