# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex role-binding SCALE certification: train the UNCHANGED
# ContentAddressedRoleBinder on the scaled multi-relation corpus across many
# seeds and run a legal-grade, pre-declared, frozen benchmark per held-out
# construction type and per held-out relation, with a second-environment (CPU)
# reproduction. The substrate is read-only; the binder is reused as-is.

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for extra in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import torch
import independent_role_corpus_audit as audit
from train_role_evolution import (ContentAddressedRoleBinder, big_config, label_of,
                                   extract_features, train_one_seed, agg_stats)
from dcortex.model import DCortexV2Model
from dcortex.semantic_role_conditioned import DCortexTokenContextBackend
import tiktoken

SEP = "=" * 70
ENC = tiktoken.get_encoding("gpt2")
WARMSTART = REPO_ROOT.parent / "D_Cortex-main" / "runs" / "warmstart" / "warmstarted_init.pt"
CORPUS = REPO_ROOT / "data" / "role_scale" / "role_scale_corpus.jsonl"

# --- Pre-declared, FROZEN legal-grade gates ---
SEEDS = 20
G_GEN_EXACT_FLOOR = 0.75
G_GEN_BASELINE_UPLIFT = 0.30
G_PERCON_EXACT_FLOOR = 0.65
G_PERCON_WRONG_CEIL = 0.15
G_SAFE_WRONG_CEIL = 0.12
G_CALIB_ABSTAIN_FLOOR = 0.70
G_STABILITY_MIN_FLOOR = 0.65
G_STABILITY_STD_CEIL = 0.07
G_REPRO_TOL = 0.05


def ctype_of(record: Dict) -> str:
    return record["construction_family"].rsplit("_", 1)[0]


def load_substrate(device: torch.device):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        ckpt = torch.load(WARMSTART, map_location=device, weights_only=False)
        model = DCortexV2Model(big_config()).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return DCortexTokenContextBackend(model, lambda t: ENC.encode_ordinary(t), max_seq_len=128)


@torch.no_grad()
def evaluate_grouped(head, reps, labels, records, indices, device) -> Dict[str, Any]:
    rel = torch.zeros(reps.shape[0], dtype=torch.long, device=device)
    idx = torch.tensor(indices, device=device)
    pred = head(reps[idx], rel[idx]).argmax(dim=1).cpu().tolist()
    by_ctype: Dict[str, Dict[str, int]] = {}
    by_relation: Dict[str, Dict[str, int]] = {}
    amb_total = amb_abstain = 0
    for local, gi in enumerate(indices):
        rec = records[gi]
        p = pred[local]
        if rec["ambiguous"]:
            amb_total += 1
            amb_abstain += int(p == 2)
            continue
        correct = int(labels[gi])
        outcome = "abstain" if p == 2 else ("exact" if p == correct else "wrong")
        for bucket, key in ((by_ctype, ctype_of(rec)), (by_relation, rec["attribute"])):
            d = bucket.setdefault(key, {"n": 0, "exact": 0, "wrong": 0, "abstain": 0})
            d["n"] += 1
            d[outcome] += 1

    def rates(buckets):
        return {k: {"n": d["n"], "exact": d["exact"] / d["n"], "wrong": d["wrong"] / d["n"],
                    "abstain": d["abstain"] / d["n"]} for k, d in buckets.items()}

    n_known = sum(d["n"] for d in by_ctype.values())
    agg = {m: sum(d[m] for d in by_ctype.values()) / max(1, n_known)
           for m in ("exact", "wrong", "abstain")}
    return {"by_ctype": rates(by_ctype), "by_relation": rates(by_relation),
            "aggregate": agg, "ambiguous_abstain_rate": amb_abstain / max(1, amb_total)}


def run_environment(device: torch.device, records, seeds: int, label: str) -> Dict[str, Any]:
    print(f"[INFO] [{label}] loading substrate on {device} and extracting features ...", flush=True)
    backend = load_substrate(device)
    reps, ok = extract_features(records, backend)
    used = [records[i] for i in ok]
    labels = torch.tensor([label_of(r) for r in used], dtype=torch.long)
    known = torch.tensor([not r["ambiguous"] for r in used], dtype=torch.bool)
    splits = {s: [i for i, r in enumerate(used) if r["split"] == s]
              for s in ("train", "validation", "evaluation")}
    idx_train = torch.tensor(splits["train"])
    idx_val = torch.tensor(splits["validation"])
    eval_idx = splits["evaluation"]
    reps = reps.to(device)
    per_seed = []
    for s in range(seeds):
        head = train_one_seed(reps, labels, known, idx_train, idx_val, reps.shape[-1], device, 2000 + s)
        per_seed.append(evaluate_grouped(head, reps, labels, used, eval_idx, device))
        a = per_seed[-1]["aggregate"]
        print(f"  [{label}] seed {s:2d} exact={a['exact']:.1%} wrong={a['wrong']:.1%} "
              f"abstain={a['abstain']:.1%}", flush=True)
    return {"used": used, "eval_idx": eval_idx, "per_seed": per_seed}


def main() -> int:
    ap = argparse.ArgumentParser(description="Role-binding scale certification")
    ap.add_argument("--run-dir", default=str(REPO_ROOT / "runs" / "role_scale"))
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--repro-seeds", type=int, default=8)
    args = ap.parse_args()
    results_dir = Path(args.run_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    corpus_sha = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    substrate_sha = hashlib.sha256(WARMSTART.read_bytes()).hexdigest()
    print(SEP, flush=True)
    print(f"[INFO] Scale certification | corpus {len(records)} rec SHA {corpus_sha[:16]} | "
          f"seeds={args.seeds}", flush=True)

    gpu = run_environment(torch.device("cuda"), records, args.seeds, "ENV1-cuda")
    per_seed = gpu["per_seed"]
    used, eval_idx = gpu["used"], gpu["eval_idx"]

    # lexical baseline on held-out eval
    from train_role_evolution import lexical_baseline_eval
    lex = lexical_baseline_eval(used, eval_idx)

    # aggregate across seeds
    ctypes = sorted({c for ev in per_seed for c in ev["by_ctype"]})
    relations = sorted({r for ev in per_seed for r in ev["by_relation"]})

    def stat(getter):
        return agg_stats([getter(ev) for ev in per_seed])
    agg_exact = stat(lambda e: e["aggregate"]["exact"])
    agg_wrong = stat(lambda e: e["aggregate"]["wrong"])
    agg_abstain = stat(lambda e: e["aggregate"]["abstain"])
    amb_abstain = stat(lambda e: e["ambiguous_abstain_rate"])
    per_ctype = {c: {"exact": stat(lambda e, c=c: e["by_ctype"].get(c, {}).get("exact", 0.0)),
                     "wrong": stat(lambda e, c=c: e["by_ctype"].get(c, {}).get("wrong", 1.0))}
                 for c in ctypes}
    per_relation = {r: {"exact": stat(lambda e, r=r: e["by_relation"].get(r, {}).get("exact", 0.0)),
                        "wrong": stat(lambda e, r=r: e["by_relation"].get(r, {}).get("wrong", 1.0))}
                    for r in relations}

    # second environment (CPU) reproduction
    print(SEP, flush=True)
    cpu = run_environment(torch.device("cpu"), records, args.repro_seeds, "ENV2-cpu")
    cpu_exact = agg_stats([e["aggregate"]["exact"] for e in cpu["per_seed"]])
    repro_delta = abs(cpu_exact["median"] - agg_exact["median"])

    # --- gates ---
    g_gen = (agg_exact["median"] >= G_GEN_EXACT_FLOOR) and (agg_exact["median"] >= lex + G_GEN_BASELINE_UPLIFT)
    percon_fail = [c for c in ctypes if per_ctype[c]["exact"]["median"] < G_PERCON_EXACT_FLOOR
                   or per_ctype[c]["wrong"]["median"] > G_PERCON_WRONG_CEIL]
    g_percon = len(percon_fail) == 0
    g_safe = agg_wrong["median"] <= G_SAFE_WRONG_CEIL
    g_calib = amb_abstain["median"] >= G_CALIB_ABSTAIN_FLOOR
    g_stab = (agg_exact["min"] >= G_STABILITY_MIN_FLOOR) and (agg_exact["std"] <= G_STABILITY_STD_CEIL)
    g_repro = repro_delta <= G_REPRO_TOL
    seals = audit.artifact_hash_report(audit.SEALED_ARTIFACTS)
    g_seals = seals["all_ok"]

    verdict = [
        {"criterion_id": "G_GEN", "passed": bool(g_gen),
         "evidence": f"held-out median exact {agg_exact['median']:.1%} (floor {G_GEN_EXACT_FLOOR:.0%}); lexical {lex:.1%}; uplift {agg_exact['median']-lex:+.1%}."},
        {"criterion_id": "G_PER_CONSTRUCTION", "passed": bool(g_percon),
         "evidence": f"every held-out construction exact>={G_PERCON_EXACT_FLOOR:.0%} & wrong<={G_PERCON_WRONG_CEIL:.0%}; failing={percon_fail or 'none'}; " + "; ".join(f"{c}:{per_ctype[c]['exact']['median']:.0%}/{per_ctype[c]['wrong']['median']:.0%}" for c in ctypes)},
        {"criterion_id": "G_SAFE", "passed": bool(g_safe),
         "evidence": f"aggregate median wrong-mapping {agg_wrong['median']:.1%} (ceiling {G_SAFE_WRONG_CEIL:.0%}; RB3 31.4%)."},
        {"criterion_id": "G_CALIB", "passed": bool(g_calib),
         "evidence": f"ambiguous abstain median {amb_abstain['median']:.1%} (floor {G_CALIB_ABSTAIN_FLOOR:.0%}); known abstain median {agg_abstain['median']:.1%}."},
        {"criterion_id": "G_STABILITY", "passed": bool(g_stab),
         "evidence": f"min exact {agg_exact['min']:.1%} (floor {G_STABILITY_MIN_FLOOR:.0%}; RB3 56.9%) & std {agg_exact['std']:.3f} (ceiling {G_STABILITY_STD_CEIL}) over {args.seeds} seeds."},
        {"criterion_id": "G_REPRO", "passed": bool(g_repro),
         "evidence": f"second env (CPU, {args.repro_seeds} seeds) median exact {cpu_exact['median']:.1%} vs ENV1 {agg_exact['median']:.1%}, delta {repro_delta:.1%} (tol {G_REPRO_TOL:.0%}); same machine, CPU vs CUDA backend (NOT distinct hardware)."},
        {"criterion_id": "G_SEALS", "passed": bool(g_seals),
         "evidence": f"sealed sources byte-identical={g_seals}; substrate SHA {substrate_sha[:16]} read-only."},
    ]
    out = {"verdict": verdict, "reference": {
        "corpus_sha256": corpus_sha, "substrate_sha256": substrate_sha, "seeds": args.seeds,
        "repro_seeds": args.repro_seeds, "lexical_baseline_exact": round(lex, 4),
        "aggregate": {"exact": agg_exact, "wrong": agg_wrong, "abstain": agg_abstain,
                      "ambiguous_abstain": amb_abstain},
        "per_construction": per_ctype, "per_relation": per_relation,
        "second_environment": {"device": "cpu", "median_exact": cpu_exact, "delta_vs_env1": round(repro_delta, 4)},
        "gates_predeclared": {"G_GEN_EXACT_FLOOR": G_GEN_EXACT_FLOOR, "G_PERCON_EXACT_FLOOR": G_PERCON_EXACT_FLOOR,
                              "G_PERCON_WRONG_CEIL": G_PERCON_WRONG_CEIL, "G_SAFE_WRONG_CEIL": G_SAFE_WRONG_CEIL,
                              "G_STABILITY_MIN_FLOOR": G_STABILITY_MIN_FLOOR, "G_STABILITY_STD_CEIL": G_STABILITY_STD_CEIL,
                              "G_REPRO_TOL": G_REPRO_TOL},
        "claim_status": ("Legal-grade attempt: >= 20 seeds + CPU/CUDA second backend (same machine). "
                         "True multi-hardware repro still required for full legal-grade. Not PROVEN.")}}
    (results_dir / "verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(SEP, flush=True)
    for c in ctypes:
        print(f"  [{c}] exact={per_ctype[c]['exact']['median']:.1%} wrong={per_ctype[c]['wrong']['median']:.1%}", flush=True)
    for r in relations:
        print(f"  <rel {r}> exact={per_relation[r]['exact']['median']:.1%} wrong={per_relation[r]['wrong']['median']:.1%}", flush=True)
    print(SEP, flush=True)
    for v in verdict:
        print(f"{'✓ PASS' if v['passed'] else '✗ FAIL'}  [{v['criterion_id']}] {v['evidence']}", flush=True)
    print(SEP, flush=True)
    all_pass = all(v["passed"] for v in verdict)
    print(f"[INFO] Overall: {'ALL GATES PASS' if all_pass else 'GATE FAILURE (reported)'}", flush=True)
    print("SCALE_VERDICT_JSON " + json.dumps({"all_pass": all_pass,
          "exact_median": agg_exact["median"], "exact_min": agg_exact["min"],
          "exact_std": agg_exact["std"], "percon_fail": percon_fail}), flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
