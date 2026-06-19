# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Stage C certification: does DCortexV2Model REASON OVER memory (chain facts, compare values) or
# only token-flow? Trains the dual-agent model on the C regime (encode facts -> memory; decode
# query -> answer) and evaluates the decisive gates on HELD-OUT entities:
#   acc(memory)        - 2-hop / comparison answered from memory only.
#   acc(text_context)  - facts also in the query (token-flow control; memory must not need it).
#   shuffled           - the binding is permuted; genuine reasoning FOLLOWS it.
#   unanswerable       - the chain breaks; the honest answer is ABSTAIN.
# Verdict THINKS / PARTIAL / REFUTED per the pre-declared bars in docs/STAGE_C_EXEC.md. Leads with
# the negative: the memory-vs-text gap + the shuffled control are the gates, not raw accuracy.

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
from stage_c import reasoning_regime as R

SEP = "=" * 70
RUN_DIR = REPO_ROOT / "runs" / "stage_c"
ENC = tiktoken.get_encoding("gpt2")
EOT = ENC.eot_token
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 48
ABSTAIN_TOK = ENC.encode_ordinary(" unknown")[0]

TRAIN_ENTS = R.ENTITIES[:14]
TEST_ENTS = R.ENTITIES[14:]


def tok(word):
    return ENC.encode_ordinary(" " + word)[0]


def _pad(ids):
    return ids[:SEQ_LEN] if len(ids) > SEQ_LEN else ids + [EOT] * (SEQ_LEN - len(ids))


def fact_answer_tok(fact):
    e, rel, val = fact
    return tok(val) if rel in ("color", "size") else tok(val)  # color/size -> value; same_color -> target entity


def query_answer_tok(item):
    return ABSTAIN_TOK if item.answer == R.ABSTAIN else tok(item.answer)


def candidates(item):
    if item.family in ("C0", "C1"):     # color answers (C0 direct + C1 relational); C2 = entity answers
        toks = [tok(c) for c in R.COLORS] + [ABSTAIN_TOK]
    else:
        ents = [f[0] for f in item.facts] + [item.query.split("the ")[-1].rstrip("?")]
        ents = list(dict.fromkeys([e for f in item.facts for e in [f[0]]]))
        # include both queried entities from the query text
        q = item.query
        import re
        qents = re.findall(r"the (\w+)", q)
        toks = [tok(e) for e in dict.fromkeys(qents)] + [ABSTAIN_TOK]
    return toks


def run_item(model, item, alpha, train=True):
    with contextlib.redirect_stdout(io.StringIO()):
        model.reset_memory()
    model.begin_episode()
    for f, txt in zip(item.facts, item.fact_texts):
        xf = torch.tensor([_pad(ENC.encode_ordinary(txt) + [EOT])], device=DEVICE)
        a = torch.tensor([fact_answer_tok(f)], device=DEVICE)
        model.encode(xf, answer_token_id=a, lexical_alpha=alpha, force_bank="working")
    xq = torch.tensor([_pad(ENC.encode_ordinary(item.query))], device=DEVICE)
    _, retrieved = model.decode(xq, return_retrieved=True)
    aux = model.aux_answer_head(retrieved)
    gold = query_answer_tok(item)
    if train:
        return F.cross_entropy(aux, torch.tensor([gold], device=DEVICE))
    cand = candidates(item)
    sub = torch.tensor([float(aux[0, t]) for t in cand])
    pred = cand[int(sub.argmax())]
    return int(pred == gold), int(pred == ABSTAIN_TOK)


def evaluate(model, n=80, seed=999):
    model.eval()
    rng = random.Random(seed)
    res = {}
    with torch.no_grad():
        for fam in ("C0", "C1", "C2"):
            for var in R.VARIANTS:
                ok = ab = tot = 0
                for _ in range(n):
                    it = R.build(rng, fam, var, TEST_ENTS)
                    correct, abst = run_item(model, it, 0.0, train=False)
                    ok += correct; ab += abst; tot += 1
                res[f"{fam}_{var}"] = {"acc": round(ok / tot, 3), "abstain_rate": round(ab / tot, 3)}
    model.train()
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage C: reasoning over memory")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--measure-every", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    (RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    import math, statistics
    print(SEP, flush=True)
    print(f"[INFO] Stage C reasoning-over-memory on {DEVICE}; steps={args.steps}", flush=True)

    with contextlib.redirect_stdout(io.StringIO()):
        model = DCortexV2Model(DCortexConfig()).to(DEVICE)
    model.train()
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": nodecay, "weight_decay": 0.0}], lr=6e-4, betas=(0.9, 0.95))
    LR, MIN_LR, WARM, GA = 6e-4, 6e-5, 200, 16

    def lr_at(s):
        if s < WARM:
            return LR * (s + 1) / WARM
        p = (s - WARM) / max(1, args.steps - WARM)
        return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * min(1.0, p)))

    def alpha_at(s):
        return 0.5 * (1 - min(1.0, s / (args.steps // 2)))   # anneal 0.5 -> 0.0 over first half

    traj = []
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        a = alpha_at(step)
        for _ in range(GA):
            fam = rng.choice(["C0", "C1", "C2"])   # include 1-hop sanity; memory-only (no abstain collapse)
            it = R.build(rng, fam, "memory", TRAIN_ENTS)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16) if DEVICE == "cuda" else contextlib.nullcontext():
                loss = run_item(model, it, a, train=True)
            (loss / GA).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 100 == 0:
            print(f"  step {step+1} | lr {lr_at(step):.2e} | a {a:.2f} | loss {float(loss):.3f}", flush=True)
        if (step + 1) % args.measure_every == 0:
            ev = evaluate(model)
            traj.append({"step": step + 1, "eval": ev})
            print(f"  >>> step {step+1} | C1 mem {ev['C1_memory']['acc']} txt {ev['C1_text_context']['acc']} "
                  f"shuf {ev['C1_shuffled']['acc']} unans-abst {ev['C1_unanswerable']['abstain_rate']} | "
                  f"C2 mem {ev['C2_memory']['acc']} shuf {ev['C2_shuffled']['acc']}", flush=True)
            (RUN_DIR / "results" / "trajectory.json").write_text(
                json.dumps({"trajectory": traj}, indent=2), encoding="utf-8")

    ev = evaluate(model, n=160)
    c0m = ev["C0_memory"]["acc"]
    c1m, c2m = ev["C1_memory"]["acc"], ev["C2_memory"]["acc"]
    mem_ge_text = (c1m >= ev["C1_text_context"]["acc"] - 0.10) and (c2m >= ev["C2_text_context"]["acc"] - 0.10)
    shuf_follows = ev["C1_shuffled"]["acc"] >= 0.80 and ev["C2_shuffled"]["acc"] >= 0.80
    abstains = ev["C1_unanswerable"]["abstain_rate"] >= 0.80 and ev["C2_unanswerable"]["abstain_rate"] >= 0.80
    # SHORTCUT detection: high raw accuracy that does NOT follow the shuffled binding is NOT genuine
    # reasoning - it is a shallow read of the available value. The shuffled control is the real gate.
    shortcut = (c1m >= 0.80 and ev["C1_shuffled"]["acc"] < 0.50)
    thinks = (c1m >= 0.80 and c2m >= 0.80 and mem_ge_text and shuf_follows and abstains and not shortcut)
    refuted = (c1m < 0.50 or not mem_ge_text or shortcut)
    verdict = "C_THINKS_IN_MEMORY" if thinks else ("C_REFUTED_SHORTCUT" if shortcut else
              ("C_REFUTED" if refuted else "C_PARTIAL"))
    storage_works = c0m >= 0.80
    thinking_fails = (c1m < 0.50 and c2m < 0.50)
    out = {"verdict": verdict, "final_eval": ev,
           "C0_direct_1hop_storage": c0m, "C1_relational_2hop": c1m, "C2_comparison": c2m,
           "storage_works_thinking_fails": bool(storage_works and thinking_fails),
           "gates": {"c1_memory>=0.80": c1m >= 0.80, "c2_memory>=0.80": c2m >= 0.80,
                     "memory>=text-0.10": mem_ge_text, "shuffled_follows>=0.80": shuf_follows,
                     "abstains>=0.80": abstains},
           "trajectory": traj,
           "scope": "MEASURED, DCortexV2Model on the C regime, held-out entities, single machine, small synthetic. NOT generality.",
           "DIAGNOSIS": ("If C0 (1-hop storage read) works but C1/C2 (multi-step) fail, the architecture HAS "
                         "STORAGE memory but NO MECHANISM to OPERATE over it: decode is single-pass read->emit, "
                         "with no working scratchpad for intermediate reasoning states. 'Thinking in memory' then "
                         "requires BUILDING the cognitive-operation layer (the vision's Stage 5), NOT training the "
                         "existing substrate. It cannot think IN memory because it has WORKING memory only for "
                         "writing facts, not for the reasoning process.")}
    (RUN_DIR / "results" / "verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(SEP, flush=True)
    print(f"[INFO] VERDICT: {verdict}", flush=True)
    print(f"  C0 1hop-storage {c0m} | C1 2hop {c1m} | C2 compare {c2m} | "
          f"storage_works_thinking_fails={storage_works and thinking_fails}", flush=True)
    print("STAGE_C_JSON " + json.dumps({"verdict": verdict, "c0_storage": c0m, "c1_mem": c1m, "c2_mem": c2m,
          "storage_works_thinking_fails": bool(storage_works and thinking_fails)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
