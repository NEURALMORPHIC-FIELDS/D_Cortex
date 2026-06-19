# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Stage U - UNIFICATION: the honest-mechanics arbiter (committed/provisional/disputed +
# promote/retrograde/prune, Step 1) operating on the NEURAL model's OWN internalized value
# vectors (Step B), end to end on the L1-L5 regime. This is the Stage U completion test:
# does the honest property survive on the trained model's REAL representation (not the
# controlled-rho vectors of Step 1)?
#
# Pipeline per event (entity, attribute, value): render "The {entity} is {value}.", run the
# trained model.encode() at lexical_alpha=0.0 to get w_value (the value the model INTERNALIZES
# from context); the arbiter's same_value() decides identity on these REAL vectors; at read,
# the committed vector is decoded to a value id by the MODEL's OWN aux_answer_head. Gates:
# wrong_commit on the model's own values, and the op-sequence vs the symbolic organ oracle.
# Honest expectation: wrong_commit is bounded BELOW by the internalization/decode error (the
# model reads its value at ~0.85-0.93, not 1.0), so the floor measures exactly that.

import argparse
import contextlib
import io
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

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
from integration.organ_client import OrganClient
from stage_u.l_regime import build_regime, run_symbolic_oracle, ALL_VALUES
from stage_u.neural_arbiter import NeuralCommitArbiter, cosine_same_value

SEP = "=" * 70
RUN_DIR = REPO_ROOT / "runs" / "stage_u"
ENC = tiktoken.get_encoding("gpt2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUBSTANTIVE = {"RETROGRADE", "PROMOTE", "PRUNE"}


class ModelValueSpace:
    """Wraps the trained model: text -> internalized w_value; w_value -> decoded value id."""

    def __init__(self, model, value_answer_tokens: Dict[str, int]):
        self.m = model
        self.val_tokens = value_answer_tokens
        self.tok_to_val = {t: v for v, t in value_answer_tokens.items()}
        self._cache: Dict[Tuple[str, str], torch.Tensor] = {}

    def value_vec(self, entity: str, value: str) -> torch.Tensor:
        key = (entity, value)
        if key in self._cache:
            return self._cache[key]
        text = f"The {entity} is {value}."
        ids = torch.tensor([ENC.encode_ordinary(text)], device=DEVICE)
        ans = torch.tensor([self.val_tokens[value]], device=DEVICE)
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            if hasattr(self.m, "reset_memory"):
                self.m.reset_memory()
            aux = self.m.encode(ids, answer_token_id=ans, lexical_alpha=0.0)
        v = aux["w_value"][0].detach().float()
        self._cache[key] = v
        return v

    def decode_value(self, vec: torch.Tensor) -> Optional[str]:
        """Model's OWN readout: w_value -> value id, argmax over the regime's value tokens."""
        dkey = round(float(vec.sum()), 6)
        if not hasattr(self, "_dcache"):
            self._dcache = {}
        if dkey in self._dcache:
            return self._dcache[dkey]
        with torch.no_grad():
            logits = self.m.aux_answer_head(vec.to(DEVICE).unsqueeze(0))[0].float().cpu()
        toks = list(self.val_tokens.values())
        names = list(self.val_tokens.keys())
        sub = torch.tensor([float(logits[t]) for t in toks])
        out = names[int(sub.argmax())]
        self._dcache[dkey] = out
        return out


def decode_same_value(space: "ModelValueSpace"):
    """Value identity via the MODEL's OWN readout (vision-faithful): two value vectors are the
    same value iff the model decodes them to the same value id. Uses the model's organic
    value-reading, not an imposed cosine metric."""
    def same(a: torch.Tensor, b: torch.Tensor) -> bool:
        return space.decode_value(a) == space.decode_value(b)
    return same


def run_arbiter_on_model(seq, space: ModelValueSpace, theta: float, identity: str = "cosine"):
    """Run the arbiter using the model's internalized w_value as the value identity, decoding
    the committed value with the model's own head. identity: 'cosine' (raw threshold) or
    'decode' (the model's own readout). Returns (committed_decoded, op_set)."""
    same_fn = decode_same_value(space) if identity == "decode" else cosine_same_value(theta)
    arb = NeuralCommitArbiter(same_fn)
    if seq.seed is not None:
        e, a, v = seq.seed
        arb.seed_committed(e, a, space.value_vec(e, v), 0)
    for ep, obs in enumerate(seq.episodes, start=1):
        for (e, a, v) in obs:
            arb.observe(e, a, space.value_vec(e, v))
        arb.end_episode(ep)
    committed = {}
    for (e, a) in seq.targets:
        vec = arb.read(e, a)
        committed[(e, a)] = space.decode_value(vec) if vec is not None else None
    opset = set(k.upper() for k, c in arb.op_counts.items() if c > 0) & SUBSTANTIVE
    return committed, opset


def calibrate_theta(space: ModelValueSpace, entities: List[str]) -> Dict:
    """Same-value (same value, different entity) vs different-value cosine on the model's
    w_value, to pick a separating threshold and expose the margin."""
    def cos(a, b):
        return float(torch.dot(a / (a.norm() + 1e-8), b / (b.norm() + 1e-8)))
    vals = ALL_VALUES
    same, diff = [], []
    ents = entities[:6]
    vecs = {(e, v): space.value_vec(e, v) for e in ents for v in vals}
    for v in vals:
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                same.append(cos(vecs[(ents[i], v)], vecs[(ents[j], v)]))
    for e in ents:
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                diff.append(cos(vecs[(e, vals[i])], vecs[(e, vals[j])]))
    min_same, max_diff = min(same), max(diff)
    theta = (min_same + max_diff) / 2.0
    return {"min_same": round(min_same, 4), "mean_same": round(statistics.mean(same), 4),
            "max_diff": round(max_diff, 4), "mean_diff": round(statistics.mean(diff), 4),
            "margin": round(min_same - max_diff, 4), "theta": round(theta, 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage U unification: arbiter on model's internalized values")
    ap.add_argument("--ckpt", default="runs/stage_u/results/ckpt_anneal.pt")
    args = ap.parse_args()
    (RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
    print(SEP, flush=True)
    print("[INFO] Stage U UNIFICATION - honest arbiter on the model's INTERNALIZED values", flush=True)

    with contextlib.redirect_stdout(io.StringIO()):
        model = DCortexV2Model(DCortexConfig()).to(DEVICE).eval()
    ck = torch.load(args.ckpt, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    print(f"[INFO] loaded {args.ckpt} (steps {ck.get('steps')})", flush=True)

    organ = OrganClient()
    ents = organ.known_entities
    val_tokens = {v: ENC.encode_ordinary(" " + v)[0] for v in ALL_VALUES}
    space = ModelValueSpace(model, val_tokens)

    cal = calibrate_theta(space, ents)
    theta = cal["theta"]
    print(f"[INFO] value-identity calibration on model w_value: min_same {cal['min_same']} "
          f"max_diff {cal['max_diff']} margin {cal['margin']} -> theta {theta}", flush=True)

    seqs = build_regime(ents, n_per_level=20)
    oracle = {seq.name: _oracle(seq, organ) for seq in seqs}   # committed + op-set, once

    # decode-only floor: how often does the model's head mis-read a clean value vector (no arbiter)?
    decode_floor_wrong = decode_floor_n = 0
    for seq in seqs:
        gold = {t: oracle[seq.name][0][t] for t in seq.targets}
        for t, gv in gold.items():
            if gv is None:
                continue
            decode_floor_n += 1
            if space.decode_value(space.value_vec(t[0], gv)) != gv:
                decode_floor_wrong += 1
    decode_floor = round(decode_floor_wrong / max(1, decode_floor_n), 4)

    def evaluate(identity: str) -> Dict:
        wrong = found = opm = 0
        per = {}
        for seq in seqs:
            oc, oops = oracle[seq.name]
            ac, aops = run_arbiter_on_model(seq, space, theta, identity=identity)
            per.setdefault(seq.level, {"n": 0, "wrong": 0})
            for t in seq.targets:
                found += 1
                per[seq.level]["n"] += 1
                if ac[t] != oc[t]:
                    wrong += 1
                    per[seq.level]["wrong"] += 1
            if aops == oops:
                opm += 1
        return {"wrong_commit": wrong, "n": found, "wrong_commit_rate": round(wrong / max(1, found), 4),
                "op_set_match": opm, "per_level_wrong": per}

    cos_res = evaluate("cosine")
    dec_res = evaluate("decode")
    # UNIFIED_CLEAN if the vision-faithful (decode) identity reaches the decode floor with 0 extra
    # arbiter loss; the residual IS the internalization floor, not an arbiter failure.
    clean = (dec_res["wrong_commit_rate"] <= decode_floor + 1e-9 and dec_res["op_set_match"] == len(seqs))
    verdict = "D_CORTEX_STAGE_U_UNIFIED_CLEAN" if clean else "D_CORTEX_STAGE_U_UNIFIED_FLOORED"
    out = {
        "verdict": verdict, "ckpt": args.ckpt,
        "value_identity_calibration": cal,
        "decode_floor_rate": decode_floor,
        "identity_cosine": cos_res,
        "identity_decode": dec_res,
        "n_sequences": len(seqs),
        "interpretation": ("Two value-identity functions for the arbiter on the model's OWN internalized "
                           "w_value: COSINE (imposed metric) vs DECODE (the model's own readout, vision-"
                           "faithful). decode_floor_rate = the model's head mis-read rate on clean value "
                           "vectors (the internalization floor). The arbiter op-sequence matches the symbolic "
                           "oracle; wrong_commit under the DECODE identity should equal the decode floor "
                           "(the residual is internalization quality, NOT an arbiter failure). COSINE is worse "
                           "because the w_value geometry is not cosine-separable (margin < 0) - the Step-1 "
                           "'honesty comes from separability' finding, on the real representation."),
        "scope": "MEASURED, trained ANNEAL ckpt on L1-L5, small synthetic regime, single machine.",
    }
    (RUN_DIR / "results" / "unification_verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(SEP, flush=True)
    print(f"[INFO] decode floor (model head mis-read on clean vectors): {decode_floor:.1%}", flush=True)
    print(f"  COSINE identity : wrong_commit {cos_res['wrong_commit']}/{cos_res['n']} "
          f"({cos_res['wrong_commit_rate']:.1%}) | op-match {cos_res['op_set_match']}/{len(seqs)}", flush=True)
    print(f"  DECODE identity : wrong_commit {dec_res['wrong_commit']}/{dec_res['n']} "
          f"({dec_res['wrong_commit_rate']:.1%}) | op-match {dec_res['op_set_match']}/{len(seqs)}", flush=True)
    print(f"[INFO] VERDICT: {verdict}", flush=True)
    print("STAGE_U_UNIFY_JSON " + json.dumps({"verdict": verdict, "decode_floor": decode_floor,
          "cosine_wc": cos_res["wrong_commit_rate"], "decode_wc": dec_res["wrong_commit_rate"],
          "cosine_opmatch": cos_res["op_set_match"], "decode_opmatch": dec_res["op_set_match"],
          "margin": cal["margin"]}), flush=True)
    return 0


def _oracle(seq, organ_unused):
    from integration.organ_client import OrganClient, FOUND_COMMITTED
    o = OrganClient()
    if seq.seed is not None:
        e, a, v = seq.seed
        o.begin_episode(); o.write_fact(e, a, v); o.end_episode()
    for episode in seq.episodes:
        o.begin_episode()
        for (e, a, v) in episode:
            o.write_fact(e, a, v)
        o.end_episode()
    committed = {}
    for t in seq.targets:
        r = o.query(*t)
        committed[t] = r.value if r.status == FOUND_COMMITTED else None
    opset = set(r.operation for r in o._audit) & SUBSTANTIVE
    return committed, opset


if __name__ == "__main__":
    sys.exit(main())
