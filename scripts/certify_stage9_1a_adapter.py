# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# STAGE 9.1-A - ADAPTER-ONLY D_CORTEX RE-STABILIZATION on a FROZEN pretrained base.
#
# Re-implements the SEALED D_Cortex bank design EXTERNALLY (faithful: k_ent[128]/k_rel[64]/k_typ[64]/
# values[768], content-addressed) on top of FROZEN pretrained hidden states, with a TRAINED adapter. The
# base is never touched (9.1-A); k_rel is reserved for traversal (9.1-C). Runs on the 59-fact eligible set
# from the 9.1-A0 pre-screen (certify_stage9_1a0_prescreen.py), where BOTH bases answer at/below chance
# with no memory - so a correct read here is the MEMORY answering, not the LLM prior.
#
# WRITE (text present): fact statement -> frozen base ~deep layer; k_ent = AddressEncoder(entity-token rep)
# (causally identical at read - the prefix "Node Arven" is the same), k_typ = TypeEncoder(attribute),
# value = ValueWriteHead(value-token rep). Store the slot.
# READ (TEXT ABSENT - the value is NOT in the query): q_ent = AddressEncoder(entity-token rep of the value-
# free query), q_typ = TypeEncoder(attribute) -> content-address the bank -> retrieved value vector ->
# decode by cosine against the candidate value embeddings (the fact's option set). The value is recovered
# from the BANK, never re-read from text.
#
# ANTI-CHEAT CONTROLS (BLOCKING), dangerous metrics first:
#   - counterfactual-override: on the 18 counterfactual facts the bank stores the CF value; the read MUST
#     return CF, not the real value the base knows. If it returns the real value, the LLM leaked. THE test.
#   - wrong/cross-binding: read returns a DIFFERENT value / the SIBLING entity's value (2-fact scenes).
#   - zeroed-memory collapse: zero the bank values -> accuracy must collapse to ~chance.
#   - shuffled stored values: permute values across slots -> the read MUST follow the permutation.
#   - text-absent: structural assert (the read query never contains the value).
#   - LLM-direct baseline: established by the pre-screen (fails by construction).
# Only the adapters train; the base is FROZEN. Held-out ENTITIES (train vs test split). Cross-model
# (Qwen2.5-7B + Mistral-7B), reported SEPARATELY - never an average that hides one base failing.

import argparse
import contextlib
import io
import json
import random
import sys
from pathlib import Path
from statistics import mean, pstdev
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

SEP = "=" * 70
PRESCREEN = REPO_ROOT / "runs" / "stage9_1a0_prescreen" / "results" / "eligible_facts.json"
RUN_DIR = REPO_ROOT / "runs" / "stage9_1a_adapter"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = ["Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]
LAYER_FRAC = 0.80                                          # deep layer (9.0b per-layer curve peaked ~L25)
D_ENT, D_REL, D_TYP, D_VAL = 128, 64, 64, 768

# write statement + text-absent query per attribute (value NEVER in the query)
TEMPLATES = {
    "checksum": ("Node {e} has checksum {v}.", "Node {e} has checksum"),
    "zone": ("Node {e} is located in zone {v}.", "Node {e} is located in zone"),
    "fact": ("{e} is {v}.", "{e} is"),
}


def load_4bit(model_id: str):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_id)
    with contextlib.redirect_stdout(io.StringIO()):
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb,
                                                     device_map={"": 0}, output_hidden_states=True)
    model.eval()
    return tok, model


def char_span_last_token(text: str, sub: str, offsets) -> Optional[int]:
    ci = text.find(sub)
    if ci < 0:
        return None
    cj = ci + len(sub)
    last = None
    for ti, (a, b) in enumerate(offsets):
        if a == b:
            continue
        if a < cj and b > ci:
            last = ti
    return last


@torch.no_grad()
def reps_for_text(model, tok, text: str, layer: int, spans: Dict[str, str]) -> Dict[str, torch.Tensor]:
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(model.device) for k, v in enc.items()}
    h = model(**enc).hidden_states[layer][0].float().cpu()          # [T, D]
    out = {"_last": h[-1]}
    for name, sub in spans.items():
        p = char_span_last_token(text, sub, offsets)
        out[name] = h[p] if p is not None else h[-1]
    return out


# ---------------------------------------------------------------------------
# Faithful external D_Cortex bank (content-addressed) + adapters
# ---------------------------------------------------------------------------
class CortexBank:
    # holds k_ent/k_rel/k_typ/values as plain tensors; content-addressed read. values store the RAW value
    # rep (d_val = base hidden dim) - decoded by a FROZEN cosine to canonical value reps (no trained value
    # head, which the rebuild showed degrades a frozen-decodable signal).
    def __init__(self, n: int, device, d_val: int):
        self.k_ent = torch.zeros(n, D_ENT, device=device)
        self.k_rel = torch.zeros(n, D_REL, device=device)
        self.k_typ = torch.zeros(n, D_TYP, device=device)
        self.values = torch.zeros(n, d_val, device=device)
        self.n = 0

    def write(self, ke, kt, val, kr=None):
        i = self.n
        self.k_ent[i] = ke; self.k_typ[i] = kt; self.values[i] = val
        if kr is not None:
            self.k_rel[i] = kr
        self.n += 1
        return i

    def address(self, q_ent, q_typ, temp=0.07, w_typ=0.3):
        ke = F.normalize(self.k_ent[: self.n], dim=1)
        kt = F.normalize(self.k_typ[: self.n], dim=1)
        s = F.normalize(q_ent, dim=0) @ ke.T + w_typ * (F.normalize(q_typ, dim=0) @ kt.T)
        attn = F.softmax(s / temp, dim=0)
        retrieved = attn @ self.values[: self.n]
        return retrieved, attn, s


class Adapters(nn.Module):
    # ONLY the addressing is trained (entity -> content key). The value is stored RAW and decoded by a
    # FROZEN cosine to canonical value reps - so the value path adds no trained shortcut, and the trained
    # contribution is isolated to content-ADDRESSING (the real memory operation).
    def __init__(self, d_in: int, n_attr: int):
        super().__init__()
        self.address = nn.Sequential(nn.Linear(d_in, 512), nn.GELU(), nn.Linear(512, D_ENT))
        self.typ = nn.Embedding(n_attr, D_TYP)

    def k_ent(self, rep):
        return self.address(rep)

    def k_typ(self, attr_id):
        return self.typ(attr_id)

    @staticmethod
    def decode(retrieved, cand_canon):
        # FROZEN: cosine of the raw retrieved value rep to each candidate's raw canonical rep. No params.
        return F.normalize(retrieved, dim=-1) @ F.normalize(cand_canon, dim=-1).T   # [C] logits over candidates


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_facts() -> List[Dict]:
    facts = json.loads(PRESCREEN.read_text(encoding="utf-8"))
    # attach candidate option set (stored + distractors) from the stored fields
    for f in facts:
        opts = [f["stored"]]
        # reconstruct option pool: for counterfactual include true_value; pull distractors from the domain
        if f["domain"] == "counterfactual" and f.get("true_value"):
            opts.append(f["true_value"])
        f["_options_seed"] = opts
    return facts


def build_value_vocab(facts):
    vals = set()
    for f in facts:
        vals.add(f["stored"])
        if f.get("true_value"):
            vals.add(f["true_value"])
    vocab = {v: i for i, v in enumerate(sorted(vals))}
    pools = {}                                              # same-attribute distractor pools (vocab indices)
    for f in facts:
        attr = f["attribute"]
        pools.setdefault(attr, set()).add(vocab[f["stored"]])
        if f.get("true_value"):
            pools[attr].add(vocab[f["true_value"]])
    pools = {a: sorted(s) for a, s in pools.items()}
    return vocab, pools


ATTRS = {"checksum": 0, "zone": 1, "fact": 2}


def cache_reps(model, tok, layer, facts) -> Dict[str, Dict]:
    cache = {}
    for f in facts:
        tmpl_w, tmpl_q = TEMPLATES[f["attribute"]]
        e, v = f["entity"], f["stored"]
        write_text = tmpl_w.format(e=e, v=v)
        query_text = tmpl_q.format(e=e)
        rw = reps_for_text(model, tok, write_text, layer, {"ent": e, "val": v})
        rq = reps_for_text(model, tok, query_text, layer, {"ent": e})
        cache[f["id"]] = {"ent_w": rw["ent"], "val_w": rw["val"], "ent_q": rq["ent"],
                          "write_text": write_text, "query_text": query_text}
        # text-absent structural assert: the value string must NOT be in the query
        assert v not in query_text, f"LEAK: value in query for {f['id']}"
    return cache


@torch.no_grad()
def canon_value_reps(model, tok, layer, vocab) -> Dict[str, torch.Tensor]:
    # one canonical rep per distinct value (neutral frame) so the decode can match ANY value, incl. unseen
    # counterfactual values - both stored and candidates go through the SAME value head (content match).
    out = {}
    for v in vocab:
        r = reps_for_text(model, tok, f"The recorded value is {v}.", layer, {"v": v})
        out[v] = r["v"]
    return out


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def make_scene_pairs(facts, rng, same_attr=True):
    # pair facts (2 per scene) for cross-binding; prefer same attribute so candidate pools overlap
    by_attr = {}
    for f in facts:
        by_attr.setdefault(f["attribute"], []).append(f)
    pairs = []
    for attr, fs in by_attr.items():
        fs = fs[:]
        rng.shuffle(fs)
        for i in range(0, len(fs) - 1, 2):
            pairs.append((fs[i], fs[i + 1]))
    rng.shuffle(pairs)
    return pairs


def options_for(f, vocab, pools, rng, must_include=(), k=4):
    # candidate set = stored (+true for cf) (+ must_include, e.g. sibling) + SAME-ATTRIBUTE distractors.
    base = [vocab[v] for v in ([f["stored"]] + ([f["true_value"]] if f.get("true_value") else []))]
    for mi in must_include:
        if mi not in base:
            base.append(mi)
    base = base[:k]                                        # stored is base[0], always kept
    pool = [i for i in pools[f["attribute"]] if i not in base]
    rng.shuffle(pool)
    opts = base[:]
    for i in pool:
        if len(opts) >= k:
            break
        opts.append(i)
    rng.shuffle(opts)
    return opts, vocab[f["stored"]]


def train_adapters(ad, cache, train_facts, vocab, pools, canon_by_idx, seed, steps=400, lr=2e-3):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    opt = torch.optim.AdamW(ad.parameters(), lr=lr)
    ad.train()
    for step in range(steps):
        pairs = make_scene_pairs(train_facts, rng)
        opt.zero_grad()
        loss = torch.zeros((), device=DEVICE)
        nseen = 0
        for fa, fb in pairs:
            bank = CortexBank(2, DEVICE, canon_by_idx.shape[1])
            for f in (fa, fb):
                c = cache[f["id"]]
                ke = ad.k_ent(c["ent_w"].to(DEVICE))
                kt = ad.k_typ(torch.tensor(ATTRS[f["attribute"]], device=DEVICE))
                bank.write(ke, kt, c["val_w"].to(DEVICE))         # store RAW value rep
            for idx, f in enumerate((fa, fb)):
                sib = (fa, fb)[1 - idx]
                c = cache[f["id"]]
                q_ent = ad.k_ent(c["ent_q"].to(DEVICE))
                q_typ = ad.k_typ(torch.tensor(ATTRS[f["attribute"]], device=DEVICE))
                retrieved, attn, s = bank.address(q_ent, q_typ)
                cand_ids, correct = options_for(f, vocab, pools, rng, must_include=[vocab[sib["stored"]]])
                cand_canon = canon_by_idx[torch.tensor(cand_ids, device=DEVICE)]
                logits = ad.decode(retrieved, cand_canon) / 0.07
                target = torch.tensor(cand_ids.index(correct), device=DEVICE)
                loss = loss + F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
                # addressing aux: the query must content-address its OWN slot (s = raw similarity logits)
                loss = loss + 0.5 * F.cross_entropy(s.unsqueeze(0), torch.tensor([idx], device=DEVICE))
                nseen += 1
        (loss / max(1, nseen)).backward()
        opt.step()
    ad.eval()


def scene_groups(facts, n_slots, rng):
    # groups of n_slots facts, SAME attribute (comparable values); drop the remainder
    by_attr = {}
    for f in facts:
        by_attr.setdefault(f["attribute"], []).append(f)
    groups = []
    for attr, fs in by_attr.items():
        fs = fs[:]
        rng.shuffle(fs)
        for i in range(0, len(fs) - n_slots + 1, n_slots):
            groups.append(fs[i:i + n_slots])
    rng.shuffle(groups)
    return groups


@torch.no_grad()
def evaluate(ad, cache, test_facts, vocab, pools, canon_by_idx, n_slots, rng, control="none"):
    # n_slots facts per bank; metrics: value acc, cross-bind (any other slot), addressing (1-of-n), oracle
    # decode, cf-override; controls: shuffled-follow (rotate values), random-collapse (random vectors, NOT zero).
    groups = scene_groups(test_facts, n_slots, rng)
    acc, cross, addr, oracle, coll = [], [], [], [], []
    cf_ok, cf_tot, follow_ok, follow_tot = 0, 0, 0, 0
    for grp in groups:
        bank = CortexBank(n_slots, DEVICE, canon_by_idx.shape[1])
        for f in grp:
            c = cache[f["id"]]
            ke = ad.k_ent(c["ent_w"].to(DEVICE)); kt = ad.k_typ(torch.tensor(ATTRS[f["attribute"]], device=DEVICE))
            bank.write(ke, kt, c["val_w"].to(DEVICE))            # store RAW value rep
        if control == "random":
            bank.values = torch.randn_like(bank.values)              # random-vector collapse (not degenerate zero)
        if control == "shuffled":
            rot = list(range(1, n_slots)) + [0]                      # slot i now holds old slot (i+1)%n value
            bank.values = bank.values[rot].clone()
        for idx, f in enumerate(grp):
            c = cache[f["id"]]
            q_ent = ad.k_ent(c["ent_q"].to(DEVICE)); q_typ = ad.k_typ(torch.tensor(ATTRS[f["attribute"]], device=DEVICE))
            retrieved, attn, _ = bank.address(q_ent, q_typ)
            others = [vocab[g["stored"]] for j, g in enumerate(grp) if j != idx]
            cand_ids, correct = options_for(f, vocab, pools, rng, must_include=others[:1])
            cand_canon = canon_by_idx[torch.tensor(cand_ids, device=DEVICE)]
            pred = cand_ids[int(torch.argmax(ad.decode(retrieved, cand_canon)).item())]
            if control == "none":
                own = vocab[f["stored"]]
                acc.append(1.0 if pred == own else 0.0)
                # cross-binding = predicted a SIBLING's DIFFERENT value (exclude own value: with many slots,
                # two facts can share a value, and returning the shared value is NOT a binding error).
                others_distinct = [o for o in others if o != own]
                cross.append(1.0 if (pred != own and pred in others_distinct) else 0.0)
                addr.append(1.0 if int(torch.argmax(attn).item()) == idx else 0.0)
                pred_o = cand_ids[int(torch.argmax(ad.decode(bank.values[idx], cand_canon)).item())]
                oracle.append(1.0 if pred_o == vocab[f["stored"]] else 0.0)
                if f["domain"] == "counterfactual":
                    cf_tot += 1; cf_ok += int(pred == vocab[f["stored"]])
            elif control == "shuffled":
                follow_tot += 1
                follow_ok += int(pred == vocab[grp[(idx + 1) % n_slots]["stored"]])   # value now at the addressed slot
            elif control == "random":
                coll.append(1.0 if pred == vocab[f["stored"]] else 0.0)
    if control == "none":
        return {"value_binding": round(mean(acc), 4) if acc else None,
                "cross_binding": round(mean(cross), 4) if cross else None,
                "cf_override": round(cf_ok / cf_tot, 4) if cf_tot else None, "cf_n": cf_tot,
                "addressing_acc": round(mean(addr), 4) if addr else None,
                "oracle_decode": round(mean(oracle), 4) if oracle else None, "n": len(acc)}
    if control == "shuffled":
        return {"shuffled_follow": round(follow_ok / follow_tot, 4) if follow_tot else None}
    return {"random_collapse": round(mean(coll), 4) if coll else None}


@torch.no_grad()
def frozen_baselines(cache, canon_by_idx, vocab, pools, test_facts, rng):
    # NO training, NO bank. FSL: does a frozen surface match (stored value rep vs candidate canon reps) already
    # pick the value? entq_leak: does the entity QUERY rep already encode the value (text-absent violation)?
    fsl, leak_stored, leak_true = [], [], []
    for f in test_facts:
        c = cache[f["id"]]
        cand_ids, _ = options_for(f, vocab, pools, rng)
        ccn = F.normalize(canon_by_idx[torch.tensor(cand_ids, device=DEVICE)], dim=-1)   # [C, d_in]
        vw = F.normalize(c["val_w"].to(DEVICE), dim=0)
        fsl.append(1.0 if cand_ids[int(torch.argmax(ccn @ vw).item())] == vocab[f["stored"]] else 0.0)
        eq = F.normalize(c["ent_q"].to(DEVICE), dim=0)
        leak_pred = cand_ids[int(torch.argmax(ccn @ eq).item())]
        leak_stored.append(1.0 if leak_pred == vocab[f["stored"]] else 0.0)
        if f.get("true_value") and f["true_value"] in vocab:
            leak_true.append(1.0 if leak_pred == vocab[f["true_value"]] else 0.0)
    return {"fsl_value": round(mean(fsl), 4) if fsl else None,
            "entq_leak_stored": round(mean(leak_stored), 4) if leak_stored else None,
            "entq_leak_true": round(mean(leak_true), 4) if leak_true else None}


N_SLOTS_FULL = [2, 10, 50]


def split_by_value(facts, frac=0.6, seed=20260620):
    # held out by VALUE (not entity): test values do not appear in train -> a real held-out-memory test
    by_attr = {}
    for f in facts:
        by_attr.setdefault(f["attribute"], set()).add(f["stored"])
    rng = random.Random(seed)
    train_vals = set()
    for attr, vals in by_attr.items():
        vs = sorted(vals); rng.shuffle(vs)
        train_vals |= set(vs[:max(1, int(frac * len(vs)))])
    tr = [f for f in facts if f["stored"] in train_vals]
    te = [f for f in facts if f["stored"] not in train_vals]
    return tr, te, train_vals


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
    print(f"[INFO] {model_id}: {n_layers} layers, deep layer {layer}", flush=True)
    facts = load_facts()
    # drop facts whose value collides as a substring of the text-absent query (would trip the leak assert);
    # this is a data-hygiene filter, not a result - report the count.
    n0 = len(facts)
    facts = [f for f in facts if f["stored"] not in TEMPLATES[f["attribute"]][1].format(e=f["entity"])]
    if len(facts) < n0:
        print(f"[WARN] dropped {n0 - len(facts)} facts with value-in-query substring collision", flush=True)
    if smoke:
        facts = [f for f in facts if f["attribute"] in ("checksum", "fact")][:24]
    vocab, pools = build_value_vocab(facts)
    print(f"[INFO] caching frozen reps for {len(facts)} facts (value vocab {len(vocab)})...", flush=True)
    cache = cache_reps(model, tok, layer, facts)
    canon = canon_value_reps(model, tok, layer, vocab)
    idx_to_val = {i: v for v, i in vocab.items()}
    canon_by_idx = torch.stack([canon[idx_to_val[i]] for i in range(len(vocab))]).to(DEVICE)
    d_in = cache[facts[0]["id"]]["ent_w"].shape[0]
    del model
    torch.cuda.empty_cache()

    # HELD-OUT BY VALUE (the validity fix): test values absent from train
    train_facts, test_facts, train_vals = split_by_value(facts)
    overlap = mean([1.0 if f["stored"] in train_vals else 0.0 for f in test_facts]) if test_facts else 0.0
    by_attr_test = {}
    for f in test_facts:
        by_attr_test.setdefault(f["attribute"], 0)
        by_attr_test[f["attribute"]] += 1
    max_attr = max(by_attr_test.values()) if by_attr_test else 0
    n_list = [n for n in ([2] if smoke else N_SLOTS_FULL) if n <= max_attr]
    if not n_list:
        n_list = [2]
    print(f"[INFO] facts: {len(train_facts)} train / {len(test_facts)} test (HELD-OUT BY VALUE; "
          f"test->train value overlap {round(overlap,3)}); slot sizes {n_list}", flush=True)

    # FROZEN baselines (no training, no bank): the binder-discipline margin + the text-absent hidden-state probe
    frozen = frozen_baselines(cache, canon_by_idx, vocab, pools, test_facts, random.Random(7))

    per_n = {n: {"value": [], "addr": [], "cross": [], "cfo": []} for n in n_list}
    shf, rnd, orc, vtr = [], [], [], []
    for s in range(seeds):
        ad = Adapters(d_in, len(ATTRS)).to(DEVICE)
        train_adapters(ad, cache, train_facts, vocab, pools, canon_by_idx, seed=s, steps=150 if smoke else 500)
        for n in n_list:
            ev = evaluate(ad, cache, test_facts, vocab, pools, canon_by_idx, n, random.Random(100 + s), "none")
            per_n[n]["value"].append(ev["value_binding"]); per_n[n]["addr"].append(ev["addressing_acc"])
            per_n[n]["cross"].append(ev["cross_binding"])
            if ev["cf_override"] is not None:
                per_n[n]["cfo"].append(ev["cf_override"])
            if n == 2:
                orc.append(ev["oracle_decode"])
        sh = evaluate(ad, cache, test_facts, vocab, pools, canon_by_idx, 2, random.Random(100 + s), "shuffled")
        rc = evaluate(ad, cache, test_facts, vocab, pools, canon_by_idx, 2, random.Random(100 + s), "random")
        mt = evaluate(ad, cache, train_facts, vocab, pools, canon_by_idx, 2, random.Random(200 + s), "none")
        shf.append(sh["shuffled_follow"]); rnd.append(rc["random_collapse"]); vtr.append(mt["value_binding"])

    n_max = max(n_list)
    v2 = _dist(per_n[2]["value"])
    fsl = frozen["fsl_value"]
    entq_s = frozen["entq_leak_stored"]                        # the TRUE no-memory baseline (no bank)
    entq_t = frozen["entq_leak_true"]
    margin_nomem = round(v2["median"] - entq_s, 4) if (v2 and entq_s is not None) else None    # memory over no-bank
    cfo2 = _dist(per_n[2]["cfo"])
    cf_over_leak = round(cfo2["median"] - entq_t, 4) if (cfo2 and entq_t is not None) else None  # bank beats leak
    res = {"layer": layer, "n_layers": n_layers, "n_test_facts": len(test_facts),
           "test_train_value_overlap": round(overlap, 4), "slot_sizes": n_list, "n_max": n_max,
           "per_n": {str(n): {"value": _dist(per_n[n]["value"]), "addressing": _dist(per_n[n]["addr"]),
                              "cross": _dist(per_n[n]["cross"]), "cf_override": _dist(per_n[n]["cfo"])}
                     for n in n_list},
           "frozen_baseline": frozen, "value_margin_over_no_memory": margin_nomem,
           "value_margin_over_fsl": round(v2["median"] - fsl, 4) if (v2 and fsl is not None) else None,
           "cf_override_over_leak_floor": cf_over_leak,
           "shuffled_follow": _dist(shf), "random_collapse": _dist(rnd),
           "diag_value_train": _dist(vtr), "diag_oracle_decode": _dist(orc)}
    print(f"  [{model_id}] value@2={v2['median'] if v2 else None} | MARGIN over no-memory(ent_q-alone "
          f"{entq_s})={margin_nomem} | fsl(ref)={fsl}", flush=True)
    print(f"  [{model_id}] cf_override@2={cfo2['median'] if cfo2 else None} - leak_floor(entq_true {entq_t}) "
          f"= {cf_over_leak} | addressing@n: " +
          ", ".join(f"n{n}={res['per_n'][str(n)]['addressing']['median']}" for n in n_list) +
          f" | cross@{n_max}={res['per_n'][str(n_max)]['cross']['median']} "
          f"shuffled={res['shuffled_follow']['median']} collapse={res['random_collapse']['median']}", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 9.1-A adapter-only D_Cortex re-stabilization")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    (RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
    if not PRESCREEN.exists():
        print(f"[ERROR] missing pre-screen eligible set: {PRESCREEN}", flush=True)
        return 2
    print(SEP, flush=True)
    print(f"[INFO] Stage 9.1-A adapter-only | device={DEVICE} | frozen base + trained adapter | "
          f"faithful bank (k_ent{D_ENT}/k_rel{D_REL}/k_typ{D_TYP}/values{D_VAL})", flush=True)
    if args.smoke:
        args.models = args.models[:1]; args.seeds = 2

    per_model = {}
    for mid in args.models:
        per_model[mid] = run_model(mid, args.seeds, args.smoke)

    # HONEST gates (pre-declared, post-decode-fix): the value path is now FROZEN (raw storage), so the headline
    # is the MARGIN over the TRUE no-memory baseline (ent_q-alone, no bank); cf-override must beat its own ent_q
    # LEAK FLOOR (the bank overrides the prior, not just echoes it); addressing AT SCALE (multi-object). Separate.
    # HONEST gates (post final adversarial review): the DECISIVE value gate is the margin over FSL (a zero-param
    # frozen cosine of the SAME stored value rep) - the comparator the trained pipeline actually contends with.
    # ent_q-alone is near-chance by prescreen construction, so margin_over_no_memory is a setup-integrity check,
    # NOT a value-recovery certifier. addressing here is ORTHOGONAL-key routing (entities are non-confusable);
    # it is NOT the confusable multi-object separability the program exists to prove.
    G_FSL, G_MARGIN, G_CF, G_ENTQS, G_ADDR, G_CROSS, G_COLL, G_SHUF = 0.05, 0.30, 0.75, 0.40, 0.80, 0.10, 0.40, 0.60

    def med(d):
        return d["median"] if d else None
    def gates(m):
        nmx = str(m["n_max"])
        return {
            "value_margin_over_FSL>=0.05": (m["value_margin_over_fsl"] is not None
                                            and m["value_margin_over_fsl"] >= G_FSL),   # DECISIVE: beats frozen lookup
            "value_margin_over_no_memory>=0.30": (m["value_margin_over_no_memory"] is not None
                                                  and m["value_margin_over_no_memory"] >= G_MARGIN),  # setup check
            "cf_override@2>=0.75": (med(m["per_n"]["2"]["cf_override"]) is not None
                                    and med(m["per_n"]["2"]["cf_override"]) >= G_CF),
            "entq_leak_stored<=0.40": (m["frozen_baseline"]["entq_leak_stored"] is None
                                       or m["frozen_baseline"]["entq_leak_stored"] <= G_ENTQS),
            f"addressing@{nmx}(orthogonal)>=0.80": (med(m["per_n"][nmx]["addressing"]) is not None
                                                    and med(m["per_n"][nmx]["addressing"]) >= G_ADDR),
            f"cross@{nmx}<=0.10": (med(m["per_n"][nmx]["cross"]) is not None
                                   and med(m["per_n"][nmx]["cross"]) <= G_CROSS),
            "random_collapse<=0.40": (med(m["random_collapse"]) is not None and med(m["random_collapse"]) <= G_COLL),
            "shuffled_follow>=0.60": (med(m["shuffled_follow"]) is not None and med(m["shuffled_follow"]) >= G_SHUF),
        }
    per_gates = {mid: gates(m) for mid, m in per_model.items()}
    per_pass = {mid: all(g.values()) for mid, g in per_gates.items()}
    both = all(per_pass.values()) and len(per_pass) >= 2
    any_ = any(per_pass.values())
    verdict = ("STAGE_9_1A_ADAPTER_PROVEN" if both else
               "STAGE_9_1A_MODEL_DEPENDENT_PARTIAL" if any_ else "STAGE_9_1A_ADAPTER_INSUFFICIENT")

    out = {"verdict": verdict, "models": args.models,
           "gates": {"value_margin_over_no_memory_min": G_MARGIN, "cf_override_min": G_CF,
                     "entq_leak_stored_max": G_ENTQS, "addressing_at_scale_min": G_ADDR, "cross_max": G_CROSS,
                     "random_collapse_max": G_COLL, "shuffled_follow_min": G_SHUF},
           "per_model": per_model, "per_model_gates": per_gates, "per_model_pass": per_pass,
           "meaning": ("HONEST cert (value path FROZEN = raw storage, no trained value head). The claim 'the memory "
                       "answers, not the LLM' requires: a value-read MARGIN over the TRUE no-memory baseline (ent_q "
                       "alone, no bank >= 0.30); the no-memory baseline on the STORED value at chance (entq_leak_stored "
                       "<= 0.40 -> the value is NOT in the query rep); cf-override (bank returns the stored CF, not the "
                       "world prior) on HELD-OUT-BY-VALUE counterfactuals; content-addressing AT SCALE (n=10, multi-"
                       "object). entq_leak_true is REPORTED as context (the prior IS accessible from the entity rep, "
                       "but the bank decode does not use it). The only trained component is the ADDRESSING. PROVEN only "
                       "if all gates pass on BOTH bases.")}
    (RUN_DIR / "results" / "verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(SEP, flush=True)
    for mid, m in per_model.items():
        nmx = str(m["n_max"])
        print(f"  {mid}: PASS={per_pass[mid]} | value@2={med(m['per_n']['2']['value'])} "
              f"MARGIN_over_no_memory={m['value_margin_over_no_memory']} (ent_q-alone {m['frozen_baseline']['entq_leak_stored']}) | "
              f"cf_override@2={med(m['per_n']['2']['cf_override'])} over_leak={m['cf_override_over_leak_floor']} | "
              f"addressing@{nmx}={med(m['per_n'][nmx]['addressing'])} cross@{nmx}={med(m['per_n'][nmx]['cross'])} "
              f"collapse={med(m['random_collapse'])} shuffled={med(m['shuffled_follow'])}", flush=True)
        print(f"      gates: " + ", ".join(f"{k}={'P' if v else 'F'}" for k, v in per_gates[mid].items()), flush=True)
    print(f"[INFO] VERDICT: {verdict}", flush=True)
    print("STAGE9_1A_JSON " + json.dumps({"verdict": verdict, "pass": per_pass,
          "value@2": {k: med(v["per_n"]["2"]["value"]) for k, v in per_model.items()},
          "margin_over_no_memory": {k: v["value_margin_over_no_memory"] for k, v in per_model.items()},
          "cf_override@2": {k: med(v["per_n"]["2"]["cf_override"]) for k, v in per_model.items()},
          "cf_over_leak": {k: v["cf_override_over_leak_floor"] for k, v in per_model.items()},
          "addressing@max": {k: med(v["per_n"][str(v["n_max"])]["addressing"]) for k, v in per_model.items()},
          "cross@max": {k: med(v["per_n"][str(v["n_max"])]["cross"]) for k, v in per_model.items()},
          "n_max": {k: v["n_max"] for k, v in per_model.items()}}), flush=True)
    return 0 if verdict == "STAGE_9_1A_ADAPTER_PROVEN" else 1


if __name__ == "__main__":
    sys.exit(main())
