# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# STAGE 9.1-B - CONFUSABLE-ENTITY ADDRESSING STRESS TEST (the actual multi-object separability frontier).
#
# 9.1-A stabilized as a faithful content-addressable KV store / RAG-equivalent: value recovery is a FROZEN
# base property (does not beat a frozen lookup) and addressing@10=1.0 was TRIVIAL - 1-of-10 routing among
# MAXIMALLY-DISTINCT (orthogonal) invented entity strings. The one genuinely-trained, load-bearing component
# is the ADDRESSING head; it was never stressed on the thing the program exists to prove: holding several
# co-occurring CONFUSABLE facts as separable objects (root-cause-multiobject-separability).
#
# THIS TEST: entities that are MINIMAL-PAIR CONFUSABLE ("node 41" .. "node 90", differing only by a number),
# whose frozen reps are CLOSE (measured and reported). The decisive comparison is TRAINED addressing vs a
# FROZEN-ROUTING baseline (route the value-free query by raw entity-rep cosine, no trained head): if the
# entities are confusable, frozen routing FAILS; the honest claim is that the TRAINED address head SEPARATES
# them and GENERALIZES to HELD-OUT confusable entities, by a real MARGIN over frozen routing. Scale n=2/10/50.
# Cross-model (Qwen + Mistral), reported separately. The value path is unchanged (RAG-style, not the point).

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean, pstdev
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

from scripts.certify_stage9_1a_adapter import (load_4bit, reps_for_text, CortexBank, Adapters, ATTRS,
                                               LAYER_FRAC, DEVICE, D_ENT)

SEP = "=" * 70
RUN_DIR = REPO_ROOT / "runs" / "stage9_1b_confusable"
MODELS = ["Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]
N_SLOTS = [2, 10, 50]
# ENTANGLED entities: ORDERED PAIRS of shared symbols ("alpha beta" vs "beta alpha") - IDENTICAL token bag,
# differing ONLY by ORDER/binding. The routing rep is read at a DOWNSTREAM token that has seen the whole
# ordered entity (no single token distinguishes the entities), so separating them requires reading the BINDING,
# not a clean distinguishing feature. This is the actual multi-object separability / binding frontier.
SYMBOLS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
           "iota", "kappa", "lambda", "sigma", "omega", "tau"]
# KEY (write) and QUERY use DIFFERENT phrasings; both read at the LAST token (integrates the ordered entity).
KEY_T = "node {e} has a recorded checksum"
QUERY_T = "the checksum on file for node {e} reads"


def gen_entangled(n: int, rng) -> List[str]:
    pairs = [f"{a} {b}" for a in SYMBOLS for b in SYMBOLS if a != b]   # ordered pairs, shared token bag
    rng.shuffle(pairs)
    return pairs[:n]


def build_facts(entities, rng) -> List[Dict]:
    return [{"entity": e, "attribute": "checksum"} for e in entities]


@torch.no_grad()
def cache(model, tok, layer, facts):
    c = {}
    for f in facts:
        e = f["entity"]
        kr = reps_for_text(model, tok, KEY_T.format(e=e), layer, {})      # last token = ordered-entity key
        qr = reps_for_text(model, tok, QUERY_T.format(e=e), layer, {})    # last token, DIFFERENT phrasing
        c[e] = {"ent_w": kr["_last"], "ent_q": qr["_last"]}
    return c


@torch.no_grad()
def measure_confusability(cache, facts) -> Dict[str, float]:
    # mean / max off-diagonal cosine of the value-free QUERY entity reps (the routing keys)
    R = F.normalize(torch.stack([cache[f["entity"]]["ent_q"] for f in facts]), dim=1)
    S = R @ R.T
    n = S.shape[0]
    off = S[~torch.eye(n, dtype=torch.bool)]
    return {"mean_cos": round(off.mean().item(), 4), "max_cos": round(off.max().item(), 4),
            "p90_cos": round(off.quantile(0.9).item(), 4)}


def scene_groups(facts, n_slots, rng):
    fs = facts[:]; rng.shuffle(fs)
    return [fs[i:i + n_slots] for i in range(0, len(fs) - n_slots + 1, n_slots)]


def train_address(ad, cache, train_facts, seed, steps=500, lr=2e-3, n_slots=8):
    torch.manual_seed(seed); rng = random.Random(seed)
    opt = torch.optim.AdamW(ad.parameters(), lr=lr)
    ad.train()
    kt = ad.k_typ(torch.tensor(ATTRS["checksum"], device=DEVICE))
    for _ in range(steps):
        groups = scene_groups(train_facts, min(n_slots, len(train_facts)), rng)
        opt.zero_grad(); loss = torch.zeros((), device=DEVICE); seen = 0
        for grp in groups:
            keys = torch.stack([ad.k_ent(cache[f["entity"]]["ent_w"].to(DEVICE)) for f in grp])  # [n, D_ENT]
            for idx, f in enumerate(grp):
                q = ad.k_ent(cache[f["entity"]]["ent_q"].to(DEVICE))
                s = F.normalize(q, dim=0) @ F.normalize(keys, dim=1).T                  # [n] routing logits
                loss = loss + F.cross_entropy((s / 0.07).unsqueeze(0), torch.tensor([idx], device=DEVICE))
                seen += 1
        (loss / max(1, seen)).backward(); opt.step()
    ad.eval()


@torch.no_grad()
def eval_addressing(ad, cache, facts, n_slots, rng, mode):
    # mode='trained' -> route by trained AddressEncoder keys; 'frozen' -> route by RAW entity reps (baseline)
    groups = scene_groups(facts, n_slots, rng)
    ok, tot = 0, 0
    for grp in groups:
        if mode == "trained":
            keys = F.normalize(torch.stack([ad.k_ent(cache[f["entity"]]["ent_w"].to(DEVICE)) for f in grp]), dim=1)
        else:
            keys = F.normalize(torch.stack([cache[f["entity"]]["ent_w"].to(DEVICE) for f in grp]), dim=1)
        for idx, f in enumerate(grp):
            if mode == "trained":
                q = F.normalize(ad.k_ent(cache[f["entity"]]["ent_q"].to(DEVICE)), dim=0)
            else:
                q = F.normalize(cache[f["entity"]]["ent_q"].to(DEVICE), dim=0)
            pick = int(torch.argmax(keys @ q).item())
            ok += int(pick == idx); tot += 1
    return ok / tot if tot else None


def _dist(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"median": round(sorted(xs)[len(xs) // 2], 4), "min": round(min(xs), 4),
            "max": round(max(xs), 4), "std": round(pstdev(xs), 4) if len(xs) > 1 else 0.0}


def run_model(model_id, seeds, smoke):
    print(f"[INFO] loading {model_id} (4-bit NF4, frozen)...", flush=True)
    tok, model = load_4bit(model_id)
    n_layers = model.config.num_hidden_layers
    layer = max(1, min(n_layers, int(LAYER_FRAC * n_layers)))
    erng = random.Random(20260620)
    n_ent = 40 if smoke else 180
    entities = gen_entangled(n_ent, erng)
    facts = build_facts(entities, erng)
    print(f"[INFO] {model_id}: layer {layer}; {len(facts)} ENTANGLED entities (ordered symbol pairs; order-only "
          f"difference, read at a downstream token)", flush=True)
    c = cache(model, tok, layer, facts)
    conf = measure_confusability(c, facts)
    d_in = c[facts[0]["entity"]]["ent_w"].shape[0]
    del model; torch.cuda.empty_cache()
    print(f"[INFO] frozen entity-rep confusability (off-diag cosine): mean={conf['mean_cos']} "
          f"p90={conf['p90_cos']} max={conf['max_cos']}", flush=True)

    # held-out CONFUSABLE entities: train on 60%, test on unseen confusable entities
    sh = facts[:]; random.Random(7).shuffle(sh)
    cut = int(0.6 * len(sh)); train_facts, test_facts = sh[:cut], sh[cut:]
    n_list = [n for n in ([2, 8] if smoke else N_SLOTS) if n <= len(test_facts)] or [2]

    per_n = {n: {"trained": [], "frozen": []} for n in n_list}
    for s in range(seeds):
        ad = Adapters(d_in, len(ATTRS)).to(DEVICE)
        train_address(ad, c, train_facts, seed=s, steps=200 if smoke else 600)
        for n in n_list:
            per_n[n]["trained"].append(eval_addressing(ad, c, test_facts, n, random.Random(100 + s), "trained"))
            if s == 0:                                          # frozen routing is seed-independent
                per_n[n]["frozen"].append(eval_addressing(ad, c, test_facts, n, random.Random(100 + s), "frozen"))
    res = {"layer": layer, "n_confusable": len(facts), "confusability": conf, "slot_sizes": n_list,
           "n_test": len(test_facts),
           "per_n": {str(n): {"trained": _dist(per_n[n]["trained"]),
                              "frozen": _dist(per_n[n]["frozen"]),
                              "margin": (round(_dist(per_n[n]["trained"])["median"] - _dist(per_n[n]["frozen"])["median"], 4)
                                         if per_n[n]["trained"] and per_n[n]["frozen"] else None)}
                     for n in n_list}}
    for n in n_list:
        pn = res["per_n"][str(n)]
        print(f"  [{model_id}] n={n}: addressing TRAINED={pn['trained']['median']} FROZEN={pn['frozen']['median']} "
              f"MARGIN={pn['margin']}", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 9.1-B confusable-entity addressing stress test")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    (RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
    print(SEP, flush=True)
    print(f"[INFO] Stage 9.1-B confusable-entity addressing | device={DEVICE} | trained addressing vs FROZEN routing",
          flush=True)
    if args.smoke:
        args.models = args.models[:1]; args.seeds = 2

    per_model = {mid: run_model(mid, args.seeds, args.smoke) for mid in args.models}

    # gates (pre-declared): the entities must BE confusable (frozen routing materially below 1.0 at scale), AND
    # the trained head must separate them at scale (>=0.80) AND beat frozen routing by a real margin (>=0.20),
    # on HELD-OUT confusable entities. Cross-model: BOTH. This is the multi-object separability claim, honestly.
    G_CONF, G_ADDR, G_MARGIN = 0.80, 0.80, 0.20

    def gates(m):
        nmx = str(max(int(k) for k in m["per_n"]))
        pn = m["per_n"][nmx]
        return {
            "entities_confusable(frozen@max<0.80)": (pn["frozen"] is not None and pn["frozen"]["median"] < G_CONF),
            f"trained_addressing@{nmx}>=0.80": (pn["trained"] is not None and pn["trained"]["median"] >= G_ADDR),
            f"margin_over_frozen@{nmx}>=0.20": (pn["margin"] is not None and pn["margin"] >= G_MARGIN),
        }
    per_gates = {mid: gates(m) for mid, m in per_model.items()}
    per_pass = {mid: all(g.values()) for mid, g in per_gates.items()}
    both = all(per_pass.values()) and len(per_pass) >= 2
    any_ = any(per_pass.values())
    verdict = ("STAGE_9_1B_CONFUSABLE_SEPARABILITY_PROVEN" if both else
               "STAGE_9_1B_MODEL_DEPENDENT_PARTIAL" if any_ else "STAGE_9_1B_CONFUSABLE_SEPARABILITY_REFUTED")

    out = {"verdict": verdict, "models": args.models,
           "gates": {"confusable_frozen_max": G_CONF, "trained_addressing_min": G_ADDR, "margin_min": G_MARGIN},
           "per_model": per_model, "per_model_gates": per_gates, "per_model_pass": per_pass,
           "meaning": ("PROVEN: on CONFUSABLE entities (frozen routing materially fails at scale), a trained address "
                       "head separates them >=0.80 at the max slot count AND beats frozen routing by >=0.20, on "
                       "HELD-OUT confusable entities, on BOTH bases -> genuine learned multi-object separability "
                       "BEYOND the orthogonal-key (RAG) routing of 9.1-A. REFUTED: the trained head does not separate "
                       "confusable entities (or does not beat frozen routing) -> the separability frontier is NOT "
                       "crossed by a frozen-base + trained-addressing adapter; it needs more (base fine-tune / LoRA).")}
    (RUN_DIR / "results" / "verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(SEP, flush=True)
    for mid, m in per_model.items():
        nmx = str(max(int(k) for k in m["per_n"]))
        pn = m["per_n"][nmx]
        print(f"  {mid}: PASS={per_pass[mid]} | confusability mean={m['confusability']['mean_cos']} | "
              f"addr@{nmx} TRAINED={pn['trained']['median']} FROZEN={pn['frozen']['median']} MARGIN={pn['margin']}",
              flush=True)
        print("      gates: " + ", ".join(f"{k}={'P' if v else 'F'}" for k, v in per_gates[mid].items()), flush=True)
    print(f"[INFO] VERDICT: {verdict}", flush=True)
    print("STAGE9_1B_JSON " + json.dumps({"verdict": verdict, "pass": per_pass,
          "confusability": {k: v["confusability"]["mean_cos"] for k, v in per_model.items()},
          "trained_addr@max": {k: v["per_n"][str(max(int(x) for x in v["per_n"]))]["trained"]["median"]
                               for k, v in per_model.items()},
          "frozen_addr@max": {k: v["per_n"][str(max(int(x) for x in v["per_n"]))]["frozen"]["median"]
                              for k, v in per_model.items()},
          "margin@max": {k: v["per_n"][str(max(int(x) for x in v["per_n"]))]["margin"]
                         for k, v in per_model.items()}}), flush=True)
    return 0 if verdict == "STAGE_9_1B_CONFUSABLE_SEPARABILITY_PROVEN" else 1


if __name__ == "__main__":
    sys.exit(main())
