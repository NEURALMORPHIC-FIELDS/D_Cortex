# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex role-binding STRUCTURAL certification (vnext3). Trains the UNCHANGED
# ContentAddressedRoleBinder over the FROZEN substrate on the structural multi-
# relation corpus, and certifies it against pre-declared, frozen, legal-grade
# gates that separate cue-predicate generalization from genuine STRUCTURAL
# generalization (long filler-gap distance, clause embedding). Adds a calibration
# pilot that measures the never-before-measured axes (relative per-seed floor,
# compositional) on an entity-disjoint split and freezes their thresholds; a no-
# memory (zeroed-content) causal control; a failure taxonomy; seed-variance
# localization; and a second-environment (CPU) reproduction. The substrate
# (warmstarted_init.pt) is read-only; the binder is reused as-is; Pas 7a seals are
# verified byte-identical before and after.

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
from train_role_evolution import (big_config, label_of, extract_features, train_one_seed,
                                   agg_stats, lexical_baseline_eval)
from build_role_struct_corpus import ctype_of, axis_of, structural_class_of
from dcortex.model import DCortexV2Model
from dcortex.semantic_role_conditioned import DCortexTokenContextBackend
import tiktoken

SEP = "=" * 70
ENC = tiktoken.get_encoding("gpt2")
WARMSTART = REPO_ROOT.parent / "D_Cortex-main" / "runs" / "warmstart" / "warmstarted_init.pt"
CORPUS = REPO_ROOT / "data" / "role_struct" / "role_struct_corpus.jsonl"
CALIB = REPO_ROOT / "data" / "role_struct" / "role_struct_calibration.jsonl"
VALIDATION = REPO_ROOT / "runs" / "role_struct" / "validation.json"

# --- Pre-declared, FROZEN gates (set before the certification run) ---
SEEDS = 20
PILOT_SEEDS = 4
REPRO_SEEDS = 8
G_GEN_EXACT_FLOOR = 0.75
G_GEN_BASELINE_UPLIFT = 0.30
G_PERCON_EXACT_FLOOR = 0.65
G_PERCON_WRONG_CEIL = 0.15
G_SAFE_WRONG_CEIL = 0.12
G_CALIB_ABSTAIN_FLOOR = 0.70
G_STABILITY_MIN_FLOOR = 0.65
G_STABILITY_STD_CEIL = 0.07
G_NO_MEMORY_GAP = 0.20          # binder must beat zeroed-content control by >= 20pp
G_REPRO_TOL = 0.05
# Pilot-calibrated thresholds (measured on the entity-disjoint calibration split,
# then FROZEN before the certification run); filled in by run_pilot().
PILOT_MARGIN = 0.05
G_RELATIVE_NOSEED_FLOOR_MIN = 0.55     # hard anchor per Section 8 (no seed below this)
G_COMPOSITIONAL_FLOOR_ANCHOR = 0.70    # target region per Section 8


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
def evaluate_struct(head, reps, labels, records, indices, device) -> Dict[str, Any]:
    """Per construction / axis / relation outcomes + ambiguous abstain + no-memory."""
    rel = torch.zeros(reps.shape[0], dtype=torch.long, device=device)
    idx = torch.tensor(indices, device=device)
    pred = head(reps[idx], rel[idx]).argmax(dim=1).cpu().tolist()
    # no-memory control: same head, zeroed content (memory path disabled)
    nm_pred = head(torch.zeros_like(reps[idx]), rel[idx]).argmax(dim=1).cpu().tolist()

    def newbucket() -> Dict[str, int]:
        return {"n": 0, "exact": 0, "wrong": 0, "abstain": 0}
    by_ctype: Dict[str, Dict[str, int]] = {}
    by_axis: Dict[str, Dict[str, int]] = {}
    by_relation: Dict[str, Dict[str, int]] = {}
    amb_total = amb_abstain = 0
    nm_known = nm_exact = 0
    for local, gi in enumerate(indices):
        rec = records[gi]
        p = pred[local]
        if rec["ambiguous"]:
            amb_total += 1
            amb_abstain += int(p == 2)
            continue
        correct = int(labels[gi])
        outcome = "abstain" if p == 2 else ("exact" if p == correct else "wrong")
        ax = axis_of(rec["construction_family"], rec["attribute"], rec["split"])
        for bucket, key in ((by_ctype, ctype_of(rec["construction_family"])),
                            (by_axis, ax), (by_relation, rec["attribute"])):
            d = bucket.setdefault(key, newbucket())
            d["n"] += 1
            d[outcome] += 1
        nm_known += 1
        nm_exact += int(nm_pred[local] != 2 and nm_pred[local] == correct)

    def rates(buckets):
        return {k: {"n": d["n"], "exact": d["exact"] / d["n"], "wrong": d["wrong"] / d["n"],
                    "abstain": d["abstain"] / d["n"]} for k, d in buckets.items()}

    n_known = sum(d["n"] for d in by_ctype.values())
    aggregate = {m: sum(d[m] for d in by_ctype.values()) / max(1, n_known)
                 for m in ("exact", "wrong", "abstain")}
    return {"by_ctype": rates(by_ctype), "by_axis": rates(by_axis), "by_relation": rates(by_relation),
            "aggregate": aggregate, "ambiguous_abstain_rate": amb_abstain / max(1, amb_total),
            "false_abstain_rate": aggregate["abstain"], "missed_abstain_rate": 1 - amb_abstain / max(1, amb_total),
            "no_memory_exact": nm_exact / max(1, nm_known)}


def prepare(device: torch.device, records, backend):
    reps, ok = extract_features(records, backend)
    used = [records[i] for i in ok]
    labels = torch.tensor([label_of(r) for r in used], dtype=torch.long)
    known = torch.tensor([not r["ambiguous"] for r in used], dtype=torch.bool)
    splits = {s: [i for i, r in enumerate(used) if r["split"] == s]
              for s in ("train", "validation", "evaluation")}
    return reps.to(device), labels, known, used, splits


def run_seed_evals(reps, labels, known, splits, used, eval_idx, device, seeds, base_seed, label):
    idx_train = torch.tensor(splits["train"])
    idx_val = torch.tensor(splits["validation"])
    per_seed = []
    heads = []
    for s in range(seeds):
        head = train_one_seed(reps, labels, known, idx_train, idx_val, reps.shape[-1], device, base_seed + s)
        heads.append(head)
        ev = evaluate_struct(head, reps, labels, used, eval_idx, device)
        per_seed.append(ev)
        a = ev["aggregate"]
        print(f"  [{label}] seed {s:2d} exact={a['exact']:.1%} wrong={a['wrong']:.1%} "
              f"abstain={a['abstain']:.1%} no-mem={ev['no_memory_exact']:.1%}", flush=True)
    return per_seed, heads


def run_pilot(device, backend) -> Dict[str, Any]:
    """Measure relative per-seed floor + compositional on the entity-disjoint
    calibration split, then derive and FREEZE the two pilot-calibrated thresholds."""
    print(SEP, flush=True)
    print(f"[INFO] [PILOT] calibration on entity-disjoint split ({PILOT_SEEDS} seeds)", flush=True)
    main = [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    calib = [json.loads(l) for l in CALIB.read_text(encoding="utf-8").splitlines() if l.strip()]
    # train/val from the main corpus; the calibration records act as the held-out eval
    pilot_records = [r for r in main if r["split"] in ("train", "validation")] + \
                    [dict(r, split="evaluation") for r in calib]
    reps, labels, known, used, splits = prepare(device, pilot_records, backend)
    eval_idx = splits["evaluation"]
    per_seed, _ = run_seed_evals(reps, labels, known, splits, used, eval_idx, device,
                                 PILOT_SEEDS, 5000, "PILOT")
    rel_exact = [ev["by_ctype"].get("relative", {}).get("exact", 0.0) for ev in per_seed]
    comp_exact = [ev["by_axis"].get("compositional", {}).get("exact", 0.0) for ev in per_seed]
    rel_min = min(rel_exact) if rel_exact else 0.0
    comp_med = statistics.median(comp_exact) if comp_exact else 0.0
    g_relative_floor = round(max(G_RELATIVE_NOSEED_FLOOR_MIN, rel_min - PILOT_MARGIN), 2)
    g_compositional_floor = round(min(G_COMPOSITIONAL_FLOOR_ANCHOR,
                                      max(0.60, comp_med - PILOT_MARGIN)), 2)
    print(f"[INFO] [PILOT] relative per-seed min={rel_min:.1%} -> G_RELATIVE no-seed floor "
          f"{g_relative_floor:.0%}; compositional median={comp_med:.1%} -> G_COMPOSITIONAL floor "
          f"{g_compositional_floor:.0%} (frozen)", flush=True)
    return {"relative_exact_per_seed": rel_exact, "relative_min": rel_min,
            "compositional_exact_per_seed": comp_exact, "compositional_median": comp_med,
            "g_relative_noseed_floor": g_relative_floor, "g_compositional_floor": g_compositional_floor,
            "pilot_seeds": PILOT_SEEDS}


def stat_over(per_seed, getter):
    return agg_stats([getter(ev) for ev in per_seed])


def main() -> int:
    ap = argparse.ArgumentParser(description="Role-binding STRUCTURAL certification")
    ap.add_argument("--run-dir", default=str(REPO_ROOT / "runs" / "role_struct"))
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--repro-seeds", type=int, default=REPRO_SEEDS)
    args = ap.parse_args()
    results_dir = Path(args.run_dir) / "results"
    reports_dir = REPO_ROOT / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    corpus_sha = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    calib_sha = hashlib.sha256(CALIB.read_bytes()).hexdigest()
    substrate_sha = hashlib.sha256(WARMSTART.read_bytes()).hexdigest()
    seals_before = audit.artifact_hash_report(audit.SEALED_ARTIFACTS)["all_ok"]
    validation = json.loads(VALIDATION.read_text(encoding="utf-8")) if VALIDATION.exists() else {}
    structural_distinct = bool(validation.get("structural_distinctness", {}).get("ok", False))
    corpus_valid = bool(validation.get("all_ok", False))
    print(SEP, flush=True)
    print(f"[INFO] STRUCTURAL certification | corpus {len(records)} rec SHA {corpus_sha[:16]} | "
          f"seals_before={seals_before} | structural_distinct={structural_distinct} | seeds={args.seeds}",
          flush=True)

    device = torch.device("cuda")
    backend = load_substrate(device)

    # --- calibration pilot: freeze the two pilot-calibrated thresholds ---
    pilot = run_pilot(device, backend)
    g_rel_floor = pilot["g_relative_noseed_floor"]
    g_comp_floor = pilot["g_compositional_floor"]

    # --- certification training (ENV1 = CUDA, >= 20 seeds) ---
    print(SEP, flush=True)
    print(f"[INFO] [ENV1-cuda] certification training ({args.seeds} seeds)", flush=True)
    reps, labels, known, used, splits = prepare(device, records, backend)
    eval_idx = splits["evaluation"]
    per_seed, _ = run_seed_evals(reps, labels, known, splits, used, eval_idx, device,
                                 args.seeds, 2000, "ENV1-cuda")
    lex = lexical_baseline_eval(used, eval_idx)

    ctypes = sorted({c for ev in per_seed for c in ev["by_ctype"]})
    axes = sorted({a for ev in per_seed for a in ev["by_axis"]})
    relations = sorted({r for ev in per_seed for r in ev["by_relation"]})

    agg_exact = stat_over(per_seed, lambda e: e["aggregate"]["exact"])
    agg_wrong = stat_over(per_seed, lambda e: e["aggregate"]["wrong"])
    agg_abstain = stat_over(per_seed, lambda e: e["aggregate"]["abstain"])
    amb_abstain = stat_over(per_seed, lambda e: e["ambiguous_abstain_rate"])
    no_mem = stat_over(per_seed, lambda e: e["no_memory_exact"])
    per_ctype = {c: {"exact": stat_over(per_seed, lambda e, c=c: e["by_ctype"].get(c, {}).get("exact", 0.0)),
                     "wrong": stat_over(per_seed, lambda e, c=c: e["by_ctype"].get(c, {}).get("wrong", 1.0))}
                 for c in ctypes}
    per_axis = {a: {"exact": stat_over(per_seed, lambda e, a=a: e["by_axis"].get(a, {}).get("exact", 0.0)),
                    "wrong": stat_over(per_seed, lambda e, a=a: e["by_axis"].get(a, {}).get("wrong", 1.0))}
                for a in axes}
    per_relation = {r: {"exact": stat_over(per_seed, lambda e, r=r: e["by_relation"].get(r, {}).get("exact", 0.0)),
                        "wrong": stat_over(per_seed, lambda e, r=r: e["by_relation"].get(r, {}).get("wrong", 1.0))}
                    for r in relations}
    relative_per_seed = [ev["by_ctype"].get("relative", {}).get("exact", 0.0) for ev in per_seed]
    relative_min = min(relative_per_seed) if relative_per_seed else 0.0

    # --- failure taxonomy (Section 6) ---
    taxonomy = {"wrong_binding": 0, "false_abstain": 0, "missed_abstain": 0, "malformed": 0,
                "per_construction": {}, "per_relation": {}}
    # tallied on the lowest-variance representative seed = the median-exact seed
    med_val = statistics.median([ev["aggregate"]["exact"] for ev in per_seed])
    rep_ev = min(per_seed, key=lambda e: abs(e["aggregate"]["exact"] - med_val))
    for c, d in rep_ev["by_ctype"].items():
        taxonomy["wrong_binding"] += d["n"] * d["wrong"]
        taxonomy["false_abstain"] += d["n"] * d["abstain"]
        taxonomy["per_construction"][c] = {"wrong_binding": round(d["wrong"], 4),
                                           "false_abstain": round(d["abstain"], 4), "n": d["n"]}
    for r, d in rep_ev["by_relation"].items():
        taxonomy["per_relation"][r] = {"wrong_binding": round(d["wrong"], 4),
                                       "false_abstain": round(d["abstain"], 4), "n": d["n"]}
    taxonomy["missed_abstain"] = round(rep_ev["missed_abstain_rate"], 4)
    taxonomy["wrong_binding"] = round(taxonomy["wrong_binding"], 2)
    taxonomy["false_abstain"] = round(taxonomy["false_abstain"], 2)
    taxonomy["note"] = ("3-way head (identity/swapped/abstain): wrong-entity vs wrong-value are "
                        "not separable; wrong_binding = identity<->swapped confusion.")

    # --- seed-variance localization (Section 7) ---
    var_by_ctype = {c: per_ctype[c]["exact"]["std"] for c in ctypes}
    worst_var = max(var_by_ctype.items(), key=lambda kv: kv[1]) if var_by_ctype else ("none", 0.0)
    min_seed = min(range(len(per_seed)), key=lambda i: per_seed[i]["aggregate"]["exact"])
    max_seed = max(range(len(per_seed)), key=lambda i: per_seed[i]["aggregate"]["exact"])
    diag = [
        "# Seed-variance localization (vnext3 structural certification)", "",
        f"Aggregate held-out exact across {args.seeds} seeds: median {agg_exact['median']:.1%}, "
        f"min {agg_exact['min']:.1%}, max {agg_exact['max']:.1%}, std {agg_exact['std']:.3f}.", "",
        f"- Lowest seed: #{min_seed} exact {per_seed[min_seed]['aggregate']['exact']:.1%}; "
        f"highest seed: #{max_seed} exact {per_seed[max_seed]['aggregate']['exact']:.1%}.",
        f"- Construction with the highest cross-seed std: **{worst_var[0]}** (std {worst_var[1]:.3f}).",
        "- Per-construction cross-seed std: " + ", ".join(f"{c}={var_by_ctype[c]:.3f}" for c in ctypes) + ".",
        "- No-memory (zeroed-content) control exact: median "
        f"{no_mem['median']:.1%} (binder uplift {agg_exact['median']-no_mem['median']:+.1%}).", "",
    ]
    if agg_exact["std"] <= 0.02:
        diag.append("Conclusion: cross-seed variance is already low (std <= 2pp). The residual "
                    f"spread is localized to the **{worst_var[0]}** construction; no pipeline "
                    "instability (init / abstain-calibration / wrong-mapping spike) dominates.")
    else:
        diag.append(f"Conclusion: variance is non-negligible and is localized primarily to the "
                    f"**{worst_var[0]}** construction (highest cross-seed std). This is the component "
                    "to stabilize next.")
    (reports_dir / "seed_variance_diagnosis.md").write_text("\n".join(diag) + "\n", encoding="utf-8")

    # --- second environment (CPU) reproduction ---
    print(SEP, flush=True)
    print(f"[INFO] [ENV2-cpu] reproduction ({args.repro_seeds} seeds)", flush=True)
    cpu = torch.device("cpu")
    backend_cpu = load_substrate(cpu)
    reps_c, labels_c, known_c, used_c, splits_c = prepare(cpu, records, backend_cpu)
    cpu_per_seed, _ = run_seed_evals(reps_c, labels_c, known_c, splits_c, used_c,
                                     splits_c["evaluation"], cpu, args.repro_seeds, 9000, "ENV2-cpu")
    cpu_exact = stat_over(cpu_per_seed, lambda e: e["aggregate"]["exact"])
    repro_delta = abs(cpu_exact["median"] - agg_exact["median"])

    seals_after = audit.artifact_hash_report(audit.SEALED_ARTIFACTS)["all_ok"]

    # --- gates (pre-declared / pilot-frozen) ---
    held_ctypes = [c for c in ctypes]   # all eval constructions are held out
    percon_fail = [c for c in held_ctypes if per_ctype[c]["exact"]["median"] < G_PERCON_EXACT_FLOOR
                   or per_ctype[c]["wrong"]["median"] > G_PERCON_WRONG_CEIL]
    structural_ctypes = [c for c in ctypes if structural_class_of(c) in ("long_gap", "embedded")]
    structural_fail = [c for c in structural_ctypes if per_ctype[c]["exact"]["median"] < G_PERCON_EXACT_FLOOR
                       or per_ctype[c]["wrong"]["median"] > G_PERCON_WRONG_CEIL]

    g_gen = (agg_exact["median"] >= G_GEN_EXACT_FLOOR) and (agg_exact["median"] >= lex + G_GEN_BASELINE_UPLIFT)
    g_percon = len(percon_fail) == 0
    g_relative = ("relative" in per_ctype
                  and per_ctype["relative"]["exact"]["median"] >= G_PERCON_EXACT_FLOOR
                  and per_ctype["relative"]["wrong"]["median"] <= G_PERCON_WRONG_CEIL
                  and relative_min >= g_rel_floor)
    g_safe = agg_wrong["median"] <= G_SAFE_WRONG_CEIL
    g_calib = amb_abstain["median"] >= G_CALIB_ABSTAIN_FLOOR
    g_stab = (agg_exact["min"] >= G_STABILITY_MIN_FLOOR) and (agg_exact["std"] <= G_STABILITY_STD_CEIL)
    g_comp = ("compositional" in per_axis
              and per_axis["compositional"]["exact"]["median"] >= g_comp_floor
              and per_axis["compositional"]["wrong"]["median"] <= G_PERCON_WRONG_CEIL)
    g_structural = structural_distinct   # Section 1.8 distinctness verification
    g_no_memory = (agg_exact["median"] - no_mem["median"]) >= G_NO_MEMORY_GAP
    g_repro = repro_delta <= G_REPRO_TOL
    g_seals = bool(seals_before and seals_after)

    def gate(cid, passed, evidence):
        return {"criterion_id": cid, "passed": bool(passed), "evidence": evidence}

    verdict = [
        gate("G_GEN", g_gen, f"held-out median exact {agg_exact['median']:.1%} "
             f"[{agg_exact['min']:.1%}/{agg_exact['max']:.1%}] (floor {G_GEN_EXACT_FLOOR:.0%}); "
             f"lexical {lex:.1%}; uplift {agg_exact['median']-lex:+.1%}."),
        gate("G_PER_CONSTRUCTION", g_percon, f"every held-out construction exact>={G_PERCON_EXACT_FLOOR:.0%} "
             f"& wrong<={G_PERCON_WRONG_CEIL:.0%}; failing={percon_fail or 'none'}; " +
             "; ".join(f"{c}:{per_ctype[c]['exact']['median']:.0%}/{per_ctype[c]['wrong']['median']:.0%}" for c in ctypes)),
        gate("G_RELATIVE", g_relative, f"relative exact {per_ctype.get('relative',{}).get('exact',{}).get('median',0):.1%} "
             f"wrong {per_ctype.get('relative',{}).get('wrong',{}).get('median',1):.1%}; per-seed min "
             f"{relative_min:.1%} >= pilot floor {g_rel_floor:.0%}."),
        gate("G_SAFE", g_safe, f"aggregate wrong-mapping median {agg_wrong['median']:.1%} "
             f"(ceiling {G_SAFE_WRONG_CEIL:.0%}; RB3 31.4%)."),
        gate("G_CALIB", g_calib, f"ambiguous abstain median {amb_abstain['median']:.1%} "
             f"(floor {G_CALIB_ABSTAIN_FLOOR:.0%}); known false-abstain median {agg_abstain['median']:.1%}."),
        gate("G_STABILITY", g_stab, f"min exact {agg_exact['min']:.1%} (floor {G_STABILITY_MIN_FLOOR:.0%}; "
             f"RB3 56.9%) & std {agg_exact['std']:.3f} (ceiling {G_STABILITY_STD_CEIL}) over {args.seeds} seeds."),
        gate("G_COMPOSITIONAL", g_comp, f"compositional (seen relation x seen predicate, unseen pairing) exact "
             f"{per_axis.get('compositional',{}).get('exact',{}).get('median',0):.1%} >= pilot floor "
             f"{g_comp_floor:.0%}; wrong {per_axis.get('compositional',{}).get('wrong',{}).get('median',1):.1%}."),
        gate("G_STRUCTURAL", g_structural, f"held-out structural constructions verified distinct from training "
             f"(long_gap/embedded gap+depth disjoint from local): {structural_distinct}. Structural-construction "
             f"performance: " + "; ".join(f"{c}:{per_ctype[c]['exact']['median']:.0%}/{per_ctype[c]['wrong']['median']:.0%}"
                                          for c in structural_ctypes) + f"; structural_fail={structural_fail or 'none'}."),
        gate("G_NO_MEMORY", g_no_memory, f"binder median exact {agg_exact['median']:.1%} vs zeroed-content control "
             f"{no_mem['median']:.1%}, gap {agg_exact['median']-no_mem['median']:+.1%} (need >= {G_NO_MEMORY_GAP:.0%})."),
        gate("G_REPRO", g_repro, f"second env (CPU, {args.repro_seeds} seeds) median exact {cpu_exact['median']:.1%} "
             f"vs ENV1 {agg_exact['median']:.1%}, delta {repro_delta:.1%} (tol {G_REPRO_TOL:.0%}); same machine, "
             f"CPU vs CUDA backend (NOT distinct hardware)."),
        gate("G_SEALS", g_seals, f"sealed artifacts byte-identical before={seals_before} after={seals_after}; "
             f"substrate SHA {substrate_sha[:16]} read-only."),
    ]

    out = {"verdict": verdict, "reference": {
        "corpus_sha256": corpus_sha, "calibration_sha256": calib_sha, "substrate_sha256": substrate_sha,
        "corpus_valid": corpus_valid, "structural_distinct": structural_distinct,
        "seeds": args.seeds, "repro_seeds": args.repro_seeds, "lexical_baseline_exact": round(lex, 4),
        "pilot": pilot,
        "aggregate": {"exact": agg_exact, "wrong": agg_wrong, "abstain": agg_abstain,
                      "ambiguous_abstain": amb_abstain, "no_memory_exact": no_mem},
        "per_construction": per_ctype, "per_axis": per_axis, "per_relation": per_relation,
        "relative_per_seed_min": round(relative_min, 4),
        "failure_taxonomy": taxonomy,
        "second_environment": {"device": "cpu", "median_exact": cpu_exact, "delta_vs_env1": round(repro_delta, 4)},
        "frozen_gates": {"G_GEN_EXACT_FLOOR": G_GEN_EXACT_FLOOR, "G_PERCON_EXACT_FLOOR": G_PERCON_EXACT_FLOOR,
                         "G_PERCON_WRONG_CEIL": G_PERCON_WRONG_CEIL, "G_SAFE_WRONG_CEIL": G_SAFE_WRONG_CEIL,
                         "G_STABILITY_MIN_FLOOR": G_STABILITY_MIN_FLOOR, "G_STABILITY_STD_CEIL": G_STABILITY_STD_CEIL,
                         "G_NO_MEMORY_GAP": G_NO_MEMORY_GAP, "G_REPRO_TOL": G_REPRO_TOL,
                         "G_RELATIVE_NOSEED_FLOOR": g_rel_floor, "G_COMPOSITIONAL_FLOOR": g_comp_floor},
        "claim_status": ("Legal-grade attempt: >= 20 seeds + CPU/CUDA second backend (same machine), pilot-"
                         "calibrated thresholds, no-memory causal control, structural holdout. True multi-"
                         "hardware repro + independent replication still required for PROVEN.")}}
    (results_dir / "verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(SEP, flush=True)
    for c in ctypes:
        print(f"  [{c:11s}] exact={per_ctype[c]['exact']['median']:.1%} wrong={per_ctype[c]['wrong']['median']:.1%} "
              f"std={per_ctype[c]['exact']['std']:.3f}", flush=True)
    for a in axes:
        print(f"  <axis {a:13s}> exact={per_axis[a]['exact']['median']:.1%} wrong={per_axis[a]['wrong']['median']:.1%}", flush=True)
    for r in relations:
        print(f"  <rel {r:13s}> exact={per_relation[r]['exact']['median']:.1%} wrong={per_relation[r]['wrong']['median']:.1%}", flush=True)
    print(SEP, flush=True)
    for v in verdict:
        print(f"{'✓ PASS' if v['passed'] else '✗ FAIL'}  [{v['criterion_id']}] {v['evidence']}", flush=True)
    print(SEP, flush=True)
    all_pass = all(v["passed"] for v in verdict)
    print(f"[INFO] Overall: {'ALL GATES PASS' if all_pass else 'GATE FAILURE (reported)'}", flush=True)
    print("STRUCT_VERDICT_JSON " + json.dumps({"all_pass": all_pass, "exact_median": agg_exact["median"],
          "exact_min": agg_exact["min"], "exact_std": agg_exact["std"], "no_memory": no_mem["median"],
          "percon_fail": percon_fail, "structural_fail": structural_fail,
          "structural_distinct": structural_distinct}), flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
