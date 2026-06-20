# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# STAGE 9.1-A0 - LLM-IGNORANCE PRE-SCREEN (the validity FOUNDATION of the 9.1 arc).
#
# WHY THIS COMES FIRST: Stage 9.1 must prove that the D_Cortex MEMORY answers, not the frozen LLM from its
# prior. 9.0b found native-readout ~ probed value: the base often already answers in-context. So before any
# adapter, FILTER the fact set down to facts where BOTH frozen bases (Qwen2.5-7B + Mistral-7B), queried
# DIRECTLY with NO memory and NO answer in the prompt, score near chance. A fact the base can already answer
# is DROPPED. Only the surviving set feeds the 9.1-A arc. Without this, the arc measures how well we mask an
# LLM answering from prior, not D_Cortex.
#
# REGIME (small NOVEL domain, architect-chosen), two sources, both make base-ignorance MEASURABLE not assumed:
#   - INVENTED private micro-domain: fictional nodes with arbitrary attributes (checksum K-###, zone) the base
#     cannot know -> direct accuracy ~chance.
#   - COUNTERFACTUAL overwrites of REAL entities: the stored value contradicts the world fact (capital of
#     France := Lyon). The base picks the TRUE value (a distractor) -> direct accuracy on the STORED value ~0,
#     confidently wrong -> the hardest, cleanest "memory must override a confident prior" test.
#
# METHOD: 4-option multiple choice (chance 0.25), K phrasings/orderings per fact (a one-shot MC is binary; K
# trials give a per-fact rate that separates "knows" ~1.0 from "guesses" ~0.25). Read the letter logits under
# the instruct chat template (forced pick, no refusals). A fact is ELIGIBLE iff for BOTH models the per-fact
# direct accuracy AND the confidence on the correct/stored option are at/below the declared bars. Report Qwen
# and Mistral SEPARATELY - never an average that hides one base failing.

import argparse
import contextlib
import io
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import torch

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
RUN_DIR = REPO_ROOT / "runs" / "stage9_1a0_prescreen"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = ["Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]
N_OPT = 4
CHANCE = 1.0 / N_OPT
K_PHRASINGS = 4
# eligibility bars (declared up front): a fact survives only if BOTH bases are at/below these
ACC_MAX = 0.50          # per-fact direct accuracy over K phrasings (knows -> ~1.0; guesses -> ~0.25)
CONF_MAX = 0.45         # mean softmax prob on the correct/stored option (knows -> high; guesses -> ~0.25)
MIN_ELIGIBLE = 30       # the surviving set must be usable for the arc
LETTERS = ["A", "B", "C", "D"]

# ---------------------------------------------------------------------------
# Candidate facts
# ---------------------------------------------------------------------------
INVENTED_ENTITIES = [
    "Arven", "Brixil", "Corvane", "Dovrenth", "Eskil-9", "Fyorn", "Grathel", "Hovrin", "Iskandra",
    "Jovric", "Kelthos", "Lumira-X", "Morvath", "Nyxel", "Orveth", "Pravia", "Quenlir", "Rovask",
    "Sythrin", "Tovrel", "Ulmreth", "Vandor", "Wexil", "Xanthe-7", "Yrrden", "Zephrel",
]
CHECKSUMS = [f"K-{n}" for n in (734, 191, 482, 905, 256, 613, 877, 340, 528, 769, 102, 455, 681, 293, 814, 567)]
ZONES = ["Velrith", "Caldun", "Threx", "Ombra", "Pyrne", "Solgrid", "Druan", "Maketh"]

# COUNTERFACTUAL: (question_tail, true_value, counterfactual_stored_value, [extra_distractors])
COUNTERFACTUALS = [
    ("the capital city of France", "Paris", "Lyon", ["Marseille", "Nice"]),
    ("the largest planet in the Solar System", "Jupiter", "Saturn", ["Neptune", "Uranus"]),
    ("the chemical symbol for gold", "Au", "Gd", ["Ag", "Go"]),
    ("the author of the play 'Romeo and Juliet'", "Shakespeare", "Marlowe", ["Jonson", "Webster"]),
    ("the number of continents on Earth", "7", "5", ["6", "8"]),
    ("the currency of Japan", "yen", "won", ["yuan", "ringgit"]),
    ("the tallest mountain above sea level on Earth", "Everest", "K2", ["Lhotse", "Makalu"]),
    ("the first president of the United States", "Washington", "Adams", ["Jefferson", "Madison"]),
    ("the planet known as the Red Planet", "Mars", "Venus", ["Mercury", "Neptune"]),
    ("the longest river commonly cited in the world", "Nile", "Amazon", ["Yangtze", "Congo"]),
    ("the language with the most native speakers", "Mandarin", "Spanish", ["English", "Hindi"]),
    ("the smallest prime number", "2", "3", ["1", "5"]),
    ("the freezing point of water in degrees Celsius", "0", "10", ["-5", "5"]),
    ("the country where the Eiffel Tower stands", "France", "Italy", ["Spain", "Belgium"]),
    ("the chemical symbol for sodium", "Na", "So", ["Sd", "So"]),
    ("the closest planet to the Sun", "Mercury", "Venus", ["Earth", "Mars"]),
    ("the number of sides on a hexagon", "6", "8", ["5", "7"]),
    ("the largest ocean on Earth", "Pacific", "Atlantic", ["Indian", "Arctic"]),
    ("the largest country by area", "Russia", "Canada", ["China", "Brazil"]),
    ("the number of days in a week", "7", "6", ["5", "8"]),
    ("the primary gas in Earth's atmosphere", "nitrogen", "oxygen", ["argon", "helium"]),
    ("the square root of eighty-one", "nine", "seven", ["six", "eight"]),
    ("the capital of Japan", "Tokyo", "Kyoto", ["Osaka", "Nagoya"]),
    ("the hardest natural material commonly cited", "diamond", "quartz", ["graphite", "corundum"]),
    ("the largest land animal", "elephant", "rhino", ["giraffe", "hippo"]),
    ("the number of planets in the Solar System", "8", "9", ["7", "10"]),
    ("the fastest land animal", "cheetah", "lion", ["leopard", "gazelle"]),
    ("the currency of the United Kingdom", "pound", "euro", ["dollar", "franc"]),
    ("the boiling point of water in Celsius at sea level", "100", "90", ["80", "110"]),
    ("the chemical symbol for iron", "Fe", "Ir", ["In", "Fr"]),
    ("the planet with the most prominent rings", "Saturn", "Jupiter", ["Uranus", "Neptune"]),
    ("the author of the play 'Hamlet'", "Shakespeare", "Dickens", ["Chaucer", "Milton"]),
    ("the smallest country by area", "Vatican", "Monaco", ["Nauru", "Malta"]),
    ("the chemical symbol for potassium", "K", "Po", ["Pt", "Ka"]),
]


def build_candidates(rng) -> List[Dict]:
    facts = []
    # invented private micro-domain (base cannot know -> chance)
    for e in INVENTED_ENTITIES:
        v = rng.choice(CHECKSUMS)
        pool = [c for c in CHECKSUMS if c != v]
        facts.append({"id": f"inv_chk_{e}", "domain": "invented", "entity": e, "attribute": "checksum",
                      "question": f"What checksum is assigned to node {e}?", "stored": v,
                      "distractor_pool": pool})
        z = rng.choice(ZONES)
        zpool = [c for c in ZONES if c != z]
        facts.append({"id": f"inv_zone_{e}", "domain": "invented", "entity": e, "attribute": "zone",
                      "question": f"In which zone is node {e} located?", "stored": z, "distractor_pool": zpool})
    # counterfactual overwrites of real entities (stored value contradicts the world fact)
    for tail, true_v, cf_v, extra in COUNTERFACTUALS:
        pool = [true_v] + [d for d in extra if d not in (true_v, cf_v)]
        facts.append({"id": f"cf_{tail[:18].strip().replace(' ', '_')}", "domain": "counterfactual",
                      "entity": tail, "attribute": "fact", "question": f"What is {tail}?",
                      "stored": cf_v, "true_value": true_v, "distractor_pool": pool})
    return facts


# ---------------------------------------------------------------------------
# Model querying (instruct chat template, read letter logits -> forced pick)
# ---------------------------------------------------------------------------
def load_4bit(model_id: str):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_id)
    with contextlib.redirect_stdout(io.StringIO()):
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map={"": 0})
    model.eval()
    return tok, model


def letter_token_ids(tok) -> Dict[str, List[int]]:
    out = {}
    for L in LETTERS:
        ids = set()
        for s in (L, " " + L):
            t = tok(s, add_special_tokens=False).input_ids
            if t:
                ids.add(t[0])
        out[L] = list(ids)
    return out


def build_mc(fact: Dict, rng) -> Tuple[str, int, List[str]]:
    # options = stored (correct) + distractors (incl. true_value for counterfactuals), shuffled
    opts = [fact["stored"]]
    pool = list(fact["distractor_pool"])
    rng.shuffle(pool)
    for d in pool:
        if len(opts) >= N_OPT:
            break
        if d not in opts:
            opts.append(d)
    while len(opts) < N_OPT:                                # pad defensively (should not trigger)
        opts.append(f"none-{len(opts)}")
    rng.shuffle(opts)
    correct_idx = opts.index(fact["stored"])
    return fact["question"], correct_idx, opts


@torch.no_grad()
def query_mc(model, tok, lett_ids, question: str, options: List[str]) -> Tuple[int, List[float]]:
    body = question + "\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    body += "\nAnswer with the single letter of the correct option."
    msgs = [{"role": "user", "content": body}]
    try:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:  # noqa: BLE001
        prompt = body + "\nAnswer:"
    enc = tok(prompt, return_tensors="pt").to(model.device)
    logits = model(**enc).logits[0, -1].float()
    scores = torch.tensor([max(logits[i].item() for i in lett_ids[L]) for L in LETTERS])
    probs = torch.softmax(scores, dim=0)
    return int(torch.argmax(scores).item()), probs.tolist()


def screen_model(model_id: str, facts: List[Dict], qrng_seed: int) -> Dict[str, Dict]:
    print(f"[INFO] loading {model_id} (4-bit NF4, frozen)...", flush=True)
    tok, model = load_4bit(model_id)
    lett_ids = letter_token_ids(tok)
    per_fact = {}
    for fi, fact in enumerate(facts):
        rng = random.Random(qrng_seed + fi)
        accs, confs = [], []
        for _ in range(K_PHRASINGS):
            q, correct_idx, opts = build_mc(fact, rng)
            pick, probs = query_mc(model, tok, lett_ids, q, opts)
            accs.append(1.0 if pick == correct_idx else 0.0)
            confs.append(probs[correct_idx])
        per_fact[fact["id"]] = {"acc": round(mean(accs), 4), "conf_correct": round(mean(confs), 4)}
    del model
    torch.cuda.empty_cache()
    return per_fact


def dist_acc(per_fact: Dict, ids: List[str]) -> float:
    xs = [per_fact[i]["acc"] for i in ids if i in per_fact]
    return round(mean(xs), 4) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 9.1-A0 LLM-ignorance pre-screen")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    (RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
    print(SEP, flush=True)
    print(f"[INFO] Stage 9.1-A0 LLM-ignorance pre-screen | device={DEVICE} | chance={CHANCE} | "
          f"bars: acc<={ACC_MAX} conf<={CONF_MAX} (BOTH bases)", flush=True)

    facts = build_candidates(random.Random(20260620))
    if args.smoke:
        facts = facts[:6] + facts[-4:]
        args.models = args.models[:1]
    n_inv = sum(1 for f in facts if f["domain"] == "invented")
    n_cf = sum(1 for f in facts if f["domain"] == "counterfactual")
    print(f"[INFO] candidates: {len(facts)} ({n_inv} invented, {n_cf} counterfactual)", flush=True)

    per_model = {}
    for mid in args.models:
        per_model[mid] = screen_model(mid, facts, qrng_seed=1000)
        acc_all = dist_acc(per_model[mid], [f["id"] for f in facts])
        print(f"  [{mid}] direct accuracy on ALL candidates = {acc_all} (chance {CHANCE})", flush=True)

    # eligibility: BOTH bases at/below bars (per fact)
    def known(mid, fid):
        r = per_model[mid][fid]
        return r["acc"] > ACC_MAX or r["conf_correct"] > CONF_MAX
    eligible, dropped = [], []
    for f in facts:
        fid = f["id"]
        knowers = [mid for mid in args.models if known(mid, fid)]
        rec = {**{k: f[k] for k in ("id", "domain", "entity", "attribute", "question", "stored")},
               "per_model": {mid: per_model[mid][fid] for mid in args.models},
               "known_by": knowers}
        if f.get("true_value"):
            rec["true_value"] = f["true_value"]
        (dropped if knowers else eligible).append(rec)

    elig_ids = [r["id"] for r in eligible]
    elig_acc = {mid: dist_acc(per_model[mid], elig_ids) for mid in args.models}
    elig_inv = sum(1 for r in eligible if r["domain"] == "invented")
    elig_cf = sum(1 for r in eligible if r["domain"] == "counterfactual")
    # verdict: enough survivors AND the filter actually worked (eligible-set accuracy near chance on both)
    enough = len(eligible) >= (3 if args.smoke else MIN_ELIGIBLE)
    filtered_clean = all((elig_acc[mid] is None or elig_acc[mid] <= CHANCE + 0.10) for mid in args.models)
    verdict = "PRESCREEN_OK" if (enough and filtered_clean) else "PRESCREEN_INSUFFICIENT"

    out = {"verdict": verdict, "models": args.models, "chance": CHANCE,
           "bars": {"acc_max": ACC_MAX, "conf_max": CONF_MAX, "min_eligible": MIN_ELIGIBLE},
           "counts": {"candidates": len(facts), "eligible": len(eligible), "dropped": len(dropped),
                      "eligible_invented": elig_inv, "eligible_counterfactual": elig_cf},
           "eligible_set_direct_accuracy_per_model": elig_acc,
           "all_candidates_direct_accuracy_per_model": {mid: dist_acc(per_model[mid], [f["id"] for f in facts])
                                                        for mid in args.models},
           "meaning": ("PRESCREEN_OK: a >=MIN_ELIGIBLE fact set survives where BOTH frozen bases answer at/below "
                       "chance with no memory -> the 9.1 arc on this set measures D_Cortex memory, not LLM prior. "
                       "PRESCREEN_INSUFFICIENT: too few survivors or the bases still answer the survivors -> the set "
                       "is not safe for the arc; widen the invented domain / add counterfactual overwrites.")}
    (RUN_DIR / "results" / "verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (RUN_DIR / "results" / "eligible_facts.json").write_text(json.dumps(eligible, indent=2), encoding="utf-8")
    (RUN_DIR / "results" / "dropped_facts.json").write_text(json.dumps(dropped, indent=2), encoding="utf-8")

    print(SEP, flush=True)
    print(f"[INFO] eligible {len(eligible)}/{len(facts)} (invented {elig_inv}, counterfactual {elig_cf}); "
          f"dropped {len(dropped)}", flush=True)
    for mid in args.models:
        print(f"  [{mid}] direct acc: ALL={out['all_candidates_direct_accuracy_per_model'][mid]} "
              f"ELIGIBLE={elig_acc[mid]} (target near chance {CHANCE})", flush=True)
    if dropped:
        ex = "; ".join(f"{r['id']} [knew: {', '.join(m.split('/')[-1] for m in r['known_by'])}]" for r in dropped[:6])
        print(f"[INFO] sample dropped (base already knew): {ex}", flush=True)
    print(f"[INFO] VERDICT: {verdict}", flush=True)
    print("STAGE9_1A0_JSON " + json.dumps({"verdict": verdict, "eligible": len(eligible),
          "eligible_acc_per_model": elig_acc,
          "all_acc_per_model": out["all_candidates_direct_accuracy_per_model"]}), flush=True)
    return 0 if verdict == "PRESCREEN_OK" else 1


if __name__ == "__main__":
    sys.exit(main())
