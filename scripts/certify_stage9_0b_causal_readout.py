# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# STAGE 9.0b - CAUSAL-VALID PRETRAINED BINDING READOUT PROBE (corrects a measurement artifact in 9.0).
#
# WHY 9.0 IS SUPERSEDED FOR THE VALUE PROBE: 9.0 read the entity-VALUE binding at the entity TOKEN
# position on a bare scene, and TRAINED on Family A ("entity-then-value", e.g. "the bear is big"). On a
# CAUSAL decoder the hidden state at the "bear" token has NOT yet seen "big" - it cannot, physically,
# carry the value. The probe trained on labels uncorrelated with its input -> learned noise -> exactly
# chance (Qwen 0.2042, Mistral 0.2458; chance 0.25). That near-chance "PRETRAINING_BINDING_REFUTED" was a
# PROBE ARTIFACT, not a property of the model. (9.0's RELATION result IS valid: there the pointed-to
# entity appears BEFORE the subject, so the subject rep is causally downstream of it.)
#
# THE FIX: read the value at a CAUSALLY VALID position - append an entity-naming continuation query
# ("... The {e} is") and read the LAST-token rep, which has seen the whole scene. This is the standard
# causal readout: a trained probe (= the Stage 9.1 adapter) recovers the in-context binding from frozen
# hidden states. Entity-specific (the query names the entity), attribute-implicit (value scenes carry
# only size), causal-valid.
#
# CONTROLS that keep a POSITIVE honest (the danger now flips to leakage):
#   - wrong-binding: does the probe surface the SIBLING entity's value? (cross-bind, must stay low)
#   - COUNTERFACTUAL-FOLLOW: rebuild each eval scene with the two entities' values SWAPPED (same entities,
#     same phrasings) - the readout must FOLLOW the swap. This is the real anti-prior control AND the
#     primary Stage 9.1 anti-cheat: a fixed entity world-prior ("bears are big") would NOT flip, so high
#     follow proves the readout reads the SCENE assignment. (Values are already randomly assigned per
#     scene, so value-binding is prior-immune by construction; the swap makes it explicit and paired.)
#   - NATIVE-READOUT (train-free baseline): the frozen model's OWN next-token argmax over the SIZES tokens
#     at the readout position. If native ~ probed value-binding, 9.0b largely measures the model's native
#     in-context answering -> 9.1's burden is honesty/auditability/abstain ADDED over native capability,
#     not capability itself. Reported, NOT gated - it CONTEXTUALIZES the headline.
#   - ENTITY-POS (the 9.0 buggy mode): reproduced on the eval set - expected ~chance, which DEMONSTRATES
#     the original artifact was the causal position, not the model.
#   - layer selection on a SHUFFLED TRAIN-internal val split (never on eval); full per-layer curve reported.
#
# SCOPE (do not over-read): a PASS scopes the READOUT precondition ONLY. It does NOT test simultaneous
# separability of >=2 facts held as operable objects, operate-over-memory / compare / chain, abstain/confT,
# or counterfactual-OVERWRITE of a stored fact - exactly where the small substrate passed decodability yet
# failed operability (root cause: multi-object separability). The readout QUERY ("The {e} is") is a
# canonical frame; only the SCENE phrasing (Family B) is held-out at the readout position. The relation
# probe shows pointer-RECOVERABILITY only (reported, NOT gated), not dereference-to-value / traversal.
#
# CROSS-MODEL (Qwen2.5-7B-Instruct + Mistral-7B-Instruct-v0.3, 4-bit NF4, frozen). DOUBLE held-out
# (unseen entities AND structurally-distinct Family-B SCENE phrasings - the same held-out scene family as
# Stage 8; the readout query itself is a canonical frame, see SCOPE). NECESSARY-not-sufficient: proves the
# binding is causally readable from frozen hidden states; whether it is usable-as-honest-memory is Stage 9.1.

import argparse
import contextlib
import io
import json
import random
import re
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
from scripts.certify_stage5_operate_memory import COLORS
from scripts.certify_stage8_phrasing_scale import VALUE_A, VALUE_B, REL_A, REL_B

SEP = "=" * 70
RUN_DIR = REPO_ROOT / "runs" / "stage9_0b_causal"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIZE_RANK = {s: i for i, s in enumerate(SIZES)}
N_SIZE = len(SIZES)
CHANCE = 1.0 / N_SIZE
PRIMARY_FRAC = 0.66
LAYER_FRACS = [0.40, 0.50, 0.66, 0.80, 0.90]               # declared up front; selected on train-CV
MODELS = ["Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]
CONT = "The {e} is"                                         # causal readout query (attribute-implicit)


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
def last_rep_layers(model, tok, text: str, layers: List[int]) -> Dict[int, torch.Tensor]:
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc)
    return {L: out.hidden_states[L][0, -1].float().cpu() for L in layers}


@torch.no_grad()
def full_hidden_layers(model, tok, text: str, layers: List[int]):
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc)
    return {L: out.hidden_states[L][0].float().cpu() for L in layers}, offsets


@torch.no_grad()
def forward_last(model, tok, text: str, layers: List[int]):
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc)
    reps = {L: out.hidden_states[L][0, -1].float().cpu() for L in layers}
    logits = out.logits[0, -1].float().cpu()
    return reps, logits


def size_token_ids(tok) -> List[int]:
    # first sub-token id of each SIZES word as a continuation (with leading space)
    return [tok(" " + s, add_special_tokens=False).input_ids[0] for s in SIZES]


def entity_token_pos(text: str, entity: str, offsets) -> Optional[int]:
    m = re.search(r"\b" + re.escape(entity) + r"\b", text)
    if m:
        ci, cj = m.start(), m.end()
    else:
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
# Probe ladder. Base FROZEN; only the probe trains.
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
# Rep caching
#   value_train: Family A scenes, CAUSAL query readout only, all candidate layers.
#   value_eval : Family B scenes (held-out entities + structurally-distinct phrasing), three readouts:
#                causal-query, no-scene (leakage control), entity-pos (reproduce the 9.0 artifact).
# Each entry: (rep, own_value_idx, sibling_value_idx)
# ---------------------------------------------------------------------------
def value_train_reps(model, tok, layers, ents, n, rng):
    out = {L: [] for L in layers}
    for _ in range(n):
        if len(ents) < 2:
            break
        e0, e1 = rng.sample(ents, 2)
        v0, v1 = rng.sample(SIZES, 2)
        scene = rng.choice(VALUE_A).format(e=e0, v=v0) + " " + rng.choice(VALUE_A).format(e=e1, v=v1)
        for e, v, sv in ((e0, v0, v1), (e1, v1, v0)):
            H = last_rep_layers(model, tok, scene + " " + CONT.format(e=e), layers)
            for L in layers:
                out[L].append((H[L], SIZE_RANK[v], SIZE_RANK[sv]))
    return out


def value_eval_reps(model, tok, layers, ents, n, rng, size_ids, phrasings):
    # phrasings = VALUE_B -> held-out family (headline); VALUE_A -> in-family reference + CLEAN entity-pos
    # repro of the 9.0 artifact (value-after-entity, so the entity-token rep is causally upstream of the
    # value -> entity_pos should fall to ~chance, where Family-B value-first does not).
    caus = {L: [] for L in layers}                          # causal query readout (headline)
    cf = {L: [] for L in layers}                            # counterfactual: values SWAPPED, must follow
    epos = {L: [] for L in layers}                          # 9.0 buggy mode (reproduce the artifact)
    native = []                                             # (pred_idx, true_idx) - model's own argmax
    for _ in range(n):
        if len(ents) < 2:
            break
        e0, e1 = rng.sample(ents, 2)
        v0, v1 = rng.sample(SIZES, 2)
        phr0, phr1 = rng.choice(phrasings), rng.choice(phrasings)
        scene = phr0.format(e=e0, v=v0) + " " + phr1.format(e=e1, v=v1)
        scene_cf = phr0.format(e=e0, v=v1) + " " + phr1.format(e=e1, v=v0)     # same entities, values swapped
        # entity-pos control (9.0 buggy mode): rep at entity token of the bare scene
        Hf, off = full_hidden_layers(model, tok, scene, layers)
        p0, p1 = entity_token_pos(scene, e0, off), entity_token_pos(scene, e1, off)
        epos_ok = (p0 is not None and p1 is not None and p0 != p1)
        for e, v, sv, p in ((e0, v0, v1, p0), (e1, v1, v0, p1)):
            Hc, lg = forward_last(model, tok, scene + " " + CONT.format(e=e), layers)   # causal query
            for L in layers:
                caus[L].append((Hc[L], SIZE_RANK[v], SIZE_RANK[sv]))
                if epos_ok:
                    epos[L].append((Hf[L][p], SIZE_RANK[v], SIZE_RANK[sv]))
            native.append((int(torch.argmax(torch.tensor([lg[i] for i in size_ids])).item()), SIZE_RANK[v]))
        for e, v, sv in ((e0, v1, v0), (e1, v0, v1)):       # swapped scene: e0 now has v1, e1 has v0
            Hc2, _ = forward_last(model, tok, scene_cf + " " + CONT.format(e=e), layers)
            for L in layers:
                cf[L].append((Hc2[L], SIZE_RANK[v], SIZE_RANK[sv]))
    return caus, cf, epos, native


def relation_reps(model, tok, layers, ents, phrasings, n, rng):
    out = {L: [] for L in layers}
    for _ in range(n):
        if len(ents) < 3:
            break
        A, B, C = rng.sample(ents, 3)
        cA, cC = rng.sample(COLORS, 2)
        text = "The " + A + " is " + cA + ". " + rng.choice(phrasings).format(b=B, a=A) + " The " + C + " is " + cC + "."
        Hf, off = full_hidden_layers(model, tok, text, layers)
        pA, pB, pC = entity_token_pos(text, A, off), entity_token_pos(text, B, off), entity_token_pos(text, C, off)
        if None in (pA, pB, pC) or len({pA, pB, pC}) < 3:
            continue
        for L in layers:
            out[L].append({"subj": Hf[L][pB], "target": Hf[L][pA], "distractor": Hf[L][pC]})
    return out


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------
def _train_value_probe(train, ProbeCls, seed, steps=1500, lr=2e-3):
    random.seed(seed); torch.manual_seed(seed)
    d = train[0][0].shape[0]
    Xtr = torch.stack([r for r, _, _ in train]).to(DEVICE)
    ytr = torch.tensor([g for _, g, _ in train], device=DEVICE)
    probe = ProbeCls(d, N_SIZE).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    probe.train()
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(probe(Xtr), ytr).backward(); opt.step()
    probe.eval()
    return probe


def _eval_value_probe(probe, evl) -> Tuple[float, float]:
    Xev = torch.stack([r for r, _, _ in evl]).to(DEVICE)
    yev = torch.tensor([g for _, g, _ in evl], device=DEVICE)
    sib = torch.tensor([s for _, _, s in evl], device=DEVICE)
    with torch.no_grad():
        pred = torch.argmax(probe(Xev), dim=1)
    return (pred == yev).float().mean().item(), (pred == sib).float().mean().item()


def _native_acc(native):
    return round(sum(int(pr == tr) for pr, tr in native) / len(native), 4) if native else None


def probe_value_selected(train_by_L, families, layers, seeds):
    # families: ordered dict name -> {"caus":..., "cf":..., "epos":..., "native":[...]}.
    # Select (layer, probe class) ONCE on a SHUFFLED TRAIN-internal val split (never on eval), train the
    # seed probes ONCE, then evaluate the SAME probes on every family (apples-to-apples comparison).
    best = None
    for L in layers:
        tr = list(train_by_L[L])
        random.Random(1234).shuffle(tr)                    # decorrelate val tail from generation order
        cut = max(1, int(0.8 * len(tr)))
        fit, val = tr[:cut], tr[cut:]
        for ProbeCls in (LinearProbe, MLPProbe):
            p = _train_value_probe(fit, ProbeCls, 0)
            vacc, _ = _eval_value_probe(p, val if val else fit)
            if best is None or vacc > best[0]:
                best = (vacc, L, ProbeCls)
    _, selL, selCls = best
    seed_probes = [_train_value_probe(train_by_L[selL], selCls, s) for s in range(seeds)]
    per_family = {}
    for name, fam in families.items():
        cacc, cwrong, cfacc, eacc = [], [], [], []
        for p in seed_probes:
            c, w = _eval_value_probe(p, fam["caus"][selL]); cacc.append(c); cwrong.append(w)
            cf_a, _ = _eval_value_probe(p, fam["cf"][selL]); cfacc.append(cf_a)
            if fam["epos"][selL]:
                e, _ = _eval_value_probe(p, fam["epos"][selL]); eacc.append(e)
        per_family[name] = {"value_binding": dist(cacc), "wrong_binding": dist(cwrong),
                            "counterfactual_follow": dist(cfacc), "native_readout": _native_acc(fam["native"]),
                            "entity_pos_artifact": dist(eacc) if eacc else None}
    # full per-layer eval curve (per family, seed 0) for transparency - no eval-peeking in selection
    curves = {}
    for name, fam in families.items():
        curve = {}
        for L in layers:
            row = {}
            for nm, Cls in (("linear", LinearProbe), ("mlp", MLPProbe)):
                p = _train_value_probe(train_by_L[L], Cls, 0)
                c, _ = _eval_value_probe(p, fam["caus"][L]); row[nm] = round(c, 4)
            curve[L] = row
        curves[name] = curve
    return {"selected_layer": selL, "selected_probe": selCls.__name__,
            "per_family": per_family, "per_layer_curves": curves}


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
            ok += int(int(torch.argmax(sims).item()) == 0); wrong += int(int(torch.argmax(sims).item()) == 1)
    n = len(evl)
    return ok / n, wrong / n


def dist(xs):
    if not xs:
        return None
    return {"min": round(min(xs), 4), "median": round(median(xs), 4), "max": round(max(xs), 4),
            "std": round(pstdev(xs), 4) if len(xs) > 1 else 0.0}


def probe_model(model_id, train_ents, held_ents, n_train, n_eval, seeds):
    print(f"[INFO] loading {model_id} (4-bit NF4, frozen)...", flush=True)
    tok, model = load_4bit(model_id)
    n_layers = model.config.num_hidden_layers
    layers = sorted({min(n_layers, max(1, int(f * n_layers))) for f in LAYER_FRACS})
    primary = min(n_layers, max(1, int(PRIMARY_FRAC * n_layers)))
    print(f"[INFO] {model_id}: {n_layers} layers, candidate layers {layers} (primary {primary})", flush=True)
    size_ids = size_token_ids(tok)
    drng = random.Random(7)
    print("[INFO] caching frozen hidden-state reps (causal query readout; this is the slow part)...", flush=True)
    v_train = value_train_reps(model, tok, layers, train_ents, n_train, drng)
    # Family B (held-out, headline) FIRST so its RNG draws (and the relation draws) are byte-identical to
    # the prior 9.0b run; the Family A reference is appended AFTER relation, perturbing nothing upstream.
    vB_caus, vB_cf, vB_epos, vB_native = value_eval_reps(model, tok, layers, held_ents, n_eval, drng, size_ids, VALUE_B)
    r_train = relation_reps(model, tok, layers, train_ents, REL_A, n_train, drng)
    r_eval = relation_reps(model, tok, layers, held_ents, REL_B, n_eval, drng)
    vA_caus, vA_cf, vA_epos, vA_native = value_eval_reps(model, tok, layers, held_ents, n_eval, drng, size_ids, VALUE_A)
    del model
    torch.cuda.empty_cache()
    nL0 = layers[0]
    print(f"[INFO] reps: value train {len(v_train[nL0])} | evalB causal {len(vB_caus[nL0])} epos {len(vB_epos[nL0])} "
          f"| evalA causal {len(vA_caus[nL0])} epos {len(vA_epos[nL0])} | relation {len(r_train[nL0])}/{len(r_eval[nL0])}",
          flush=True)

    families = {"family_B": {"caus": vB_caus, "cf": vB_cf, "epos": vB_epos, "native": vB_native},
                "family_A": {"caus": vA_caus, "cf": vA_cf, "epos": vA_epos, "native": vA_native}}
    val = probe_value_selected(v_train, families, layers, seeds)
    rel_o, rel_w = [], []
    rL = primary if primary in r_train else layers[len(layers) // 2]
    for s in range(seeds):
        ro, rw = probe_relation(r_train[rL], r_eval[rL], s); rel_o.append(ro); rel_w.append(rw)
    fb, fa = val["per_family"]["family_B"], val["per_family"]["family_A"]
    res = {"n_layers": n_layers, "candidate_layers": layers, "primary_layer": primary, "relation_layer": rL,
           "selected_layer": val["selected_layer"], "selected_probe": val["selected_probe"],
           "value_binding": fb["value_binding"], "wrong_binding": fb["wrong_binding"],
           "counterfactual_follow": fb["counterfactual_follow"], "native_readout": fb["native_readout"],
           "entity_pos_artifact": fb["entity_pos_artifact"],
           "family_A_reference": fa, "per_layer_curves": val["per_layer_curves"],
           "relation_bind": dist(rel_o), "relation_wrong_dir": dist(rel_w)}
    epa_b = res["entity_pos_artifact"]["median"] if res["entity_pos_artifact"] else None
    epa_a = fa["entity_pos_artifact"]["median"] if fa["entity_pos_artifact"] else None
    print(f"  [{model_id}] (Family B held-out) value_binding={res['value_binding']['median']} (layer "
          f"{res['selected_layer']} {res['selected_probe']}) wrong={res['wrong_binding']['median']} "
          f"cf_follow={res['counterfactual_follow']['median']} native={res['native_readout']} "
          f"entity_pos_B={epa_b} | relation={res['relation_bind']['median']} wrong_dir={res['relation_wrong_dir']['median']}",
          flush=True)
    print(f"  [{model_id}] (Family A reference) value_binding={fa['value_binding']['median']} "
          f"wrong={fa['wrong_binding']['median']} cf_follow={fa['counterfactual_follow']['median']} "
          f"native={fa['native_readout']} entity_pos_A={epa_a} (chance {round(CHANCE,3)}; CLEAN 9.0-artifact repro, "
          f"value-after-entity -> expect ~chance)", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 9.0b causal-valid pretrained binding probe")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    (RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
    print(SEP, flush=True)
    print(f"[INFO] Stage 9.0b causal-valid pretrained binding probe | device={DEVICE} | chance={round(CHANCE,3)}", flush=True)
    print(f"[INFO] readout: append '{CONT}' and read the LAST token (causally sees the whole scene). "
          f"Train Family A, eval Family B (structurally distinct) + held-out entities.", flush=True)

    split_rng = random.Random(20260620)
    sh = ENTITIES[:]; split_rng.shuffle(sh)
    train_ents, held_ents = sh[:14], sh[14:]
    if args.smoke:
        r = probe_model(args.models[0], train_ents, held_ents, 24, 16, 1)
        print(f"  [SMOKE] value={r['value_binding']['median']} epos_artifact="
              f"{r['entity_pos_artifact']['median'] if r['entity_pos_artifact'] else None}", flush=True)
        return 0

    per_model = {}
    for mid in args.models:
        per_model[mid] = probe_model(mid, train_ents, held_ents, args.n_train, args.n_eval, args.seeds)

    # corrected verdict: causal readout must expose value-binding that FOLLOWS the scene (counterfactual swap),
    # with low cross-bind. (no-scene was dropped - vacuous by construction; the swap is the real anti-prior test.)
    BAR_VALUE, BAR_WRONG, BAR_CF = 0.70, 0.15, 0.60
    def passes(m):
        return (m["value_binding"]["median"] >= BAR_VALUE
                and m["wrong_binding"]["median"] <= BAR_WRONG
                and m["counterfactual_follow"]["median"] >= BAR_CF)
    both = all(passes(m) for m in per_model.values())
    any_ = any(passes(m) for m in per_model.values())
    verdict = ("PRETRAINING_BINDING_CAUSALLY_READABLE" if both else
               "PRETRAINING_BINDING_PARTIAL" if any_ else "PRETRAINING_BINDING_NOT_READABLE")

    out = {"verdict": verdict, "models": args.models, "chance": round(CHANCE, 4),
           "bars": {"value_binding": BAR_VALUE, "wrong_binding_max": BAR_WRONG, "counterfactual_follow_min": BAR_CF},
           "supersedes": "certify_stage9_0_pretrained_probe.py value verdict (PRETRAINING_BINDING_REFUTED was a "
                         "causal-position artifact: the 9.0 value probe read the entity token on a bare Family-A scene, "
                         "upstream of the value on a causal decoder; entity_pos_artifact below reproduces it ~chance).",
           "small_substrate_baseline_familyB": 0.337,
           "scope_caveat": ("PASS scopes the READOUT precondition ONLY: it does NOT test simultaneous separability of "
                            ">=2 facts as operable objects, operate-over-memory / compare / chain, abstain/confT, or "
                            "counterfactual-OVERWRITE of a stored fact (Stage 9.1). The readout query is a canonical "
                            "frame; only the SCENE phrasing (Family B) is held-out. native_readout (the model's own "
                            "argmax) contextualizes the headline: if native ~ value_binding, 9.1's burden is honesty/"
                            "auditability/abstain ADDED over native capability. relation_* is reported, NOT gated "
                            "(pointer-recoverability only, not dereference-to-value/traversal)."),
           "per_model": per_model,
           "meaning": {
               "PRETRAINING_BINDING_CAUSALLY_READABLE": "both frozen bases expose entity-value binding from "
                   "structurally-distinct Family-B phrasings at a causal readout (>=0.70, cross-bind<=0.15, and the "
                   "counterfactual value-swap is FOLLOWED >=0.60 -> reads the scene, not an entity prior), where the "
                   "small substrate got 0.337 -> the binding IS there in pretrained hidden states and recoverable by a "
                   "trained adapter. Stage 9.1 readout-premise holds. Addressing precondition = relation probe (reported).",
               "PRETRAINING_BINDING_PARTIAL": "one base or one control fails -> model-dependent or a residual confound; "
                   "investigate before 9.1.",
               "PRETRAINING_BINDING_NOT_READABLE": "even a causal readout does not recover the binding from structurally-"
                   "distinct phrasings -> the binding is not a simple frozen readout; 9.1 needs more than an adapter."}[verdict]}
    (RUN_DIR / "results" / "verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(SEP, flush=True)
    print("[INFO] vs small substrate Family-B value-binding = 0.337 (Stage 8); chance =", round(CHANCE, 3), flush=True)

    def _epa(d):
        return d["median"] if d else None
    for mid, m in per_model.items():
        fa = m["family_A_reference"]
        print(f"  {mid} (Family B held-out): value={m['value_binding']['median']} wrong={m['wrong_binding']['median']} "
              f"cf_follow={m['counterfactual_follow']['median']} native={m['native_readout']} "
              f"entity_pos_B={_epa(m['entity_pos_artifact'])} relation={m['relation_bind']['median']} "
              f"(layer {m['selected_layer']}/{m['n_layers']})", flush=True)
        print(f"  {mid} (Family A ref):       value={fa['value_binding']['median']} wrong={fa['wrong_binding']['median']} "
              f"cf_follow={fa['counterfactual_follow']['median']} native={fa['native_readout']} "
              f"entity_pos_A={_epa(fa['entity_pos_artifact'])} <- CLEAN 9.0 repro (expect ~{round(CHANCE,2)})", flush=True)
    print(f"[INFO] VERDICT: {verdict}", flush=True)
    print("STAGE9_0B_JSON " + json.dumps({"verdict": verdict,
          "value_per_model": {k: v["value_binding"]["median"] for k, v in per_model.items()},
          "wrong_per_model": {k: v["wrong_binding"]["median"] for k, v in per_model.items()},
          "counterfactual_follow_per_model": {k: v["counterfactual_follow"]["median"] for k, v in per_model.items()},
          "native_readout_per_model": {k: v["native_readout"] for k, v in per_model.items()},
          "entity_pos_B_per_model": {k: _epa(v["entity_pos_artifact"]) for k, v in per_model.items()},
          "entity_pos_A_per_model": {k: _epa(v["family_A_reference"]["entity_pos_artifact"]) for k, v in per_model.items()},
          "value_A_per_model": {k: v["family_A_reference"]["value_binding"]["median"] for k, v in per_model.items()}}),
          flush=True)
    return 0 if verdict == "PRETRAINING_BINDING_CAUSALLY_READABLE" else 1


if __name__ == "__main__":
    sys.exit(main())
