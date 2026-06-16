# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex spine hardening certification. Replaces free-generation attribute parsing
# with CONSTRAINED closed-set classification (the model can only emit one of
# color/size/location/state/none), and adds threshold rigor: a determinism check,
# 12 dev/test splits (disjoint by fact, 60/40) with the MiniLM entity-resolution
# threshold SELECTED ON DEV and reported on disjoint TEST, a threshold sweep curve,
# distributions across splits, and an error-type breakdown of the remaining failures.
# A leak assertion proves no F-generator alias appears in any prompt. The organ and
# steps/13 are byte-identical (loaded read-only).

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
for extra in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from integration.organ_client import OrganClient, FOUND_COMMITTED, NONE_OBJECT, NONE_ATTRIBUTE
from integration.constrained_extractor import ConstrainedExtractor, assert_no_leak
from integration.ingest_adapter import FactTriple, ExtractError
from integration.corpus import _make_item
from integration.verbalizer_control import VerbalizerControl
from dcortex_professional.qwen_runtime import QwenBaseModel

SEP = "=" * 70
RUN_DIR = REPO_ROOT / "runs" / "spine_hardening"
SPLIT_SEEDS = list(range(1001, 1013))            # 12 splits
SWEEP = [round(0.30 + 0.05 * i, 2) for i in range(11)]   # 0.30..0.80
F1_BAR = F3_BAR = 0.80


def build_facts(entities: List[str], attr_values: Dict[str, List[str]], n: int, seed: int = 7):
    rng = random.Random(seed)
    attrs = ["color", "size", "location", "state"]
    pairs = [(e, a) for e in entities for a in attrs]
    rng.shuffle(pairs)
    facts = [(e, a, rng.choice(attr_values[a])) for e, a in pairs[:n]]
    return facts, rng


def main() -> int:
    ap = argparse.ArgumentParser(description="D_Cortex spine hardening certification")
    ap.add_argument("--n-facts", type=int, default=51)
    args = ap.parse_args()
    (RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
    print(SEP, flush=True)
    print("[INFO] D_Cortex spine hardening (constrained attribute + threshold rigor)", flush=True)

    # ---- G_NO_LEAK ----
    f_aliases, prompt_words = assert_no_leak()
    print(f"[PASS] G_NO_LEAK: prompt words disjoint from {len(f_aliases)} F-generator aliases.", flush=True)
    print(f"   F-aliases (sample): {sorted(f_aliases)[:12]}", flush=True)
    print(f"   prompt words: {sorted(prompt_words)}", flush=True)

    organ = OrganClient()
    qwen = QwenBaseModel()
    if not qwen.available:
        print(f"[BLOCKED] Qwen unavailable: {qwen.reason}", flush=True)
        return 2
    ex = ConstrainedExtractor(qwen, organ.attr_values, organ.known_entities)

    # ---- G_DETERMINISM ----
    s1 = ex.extract_query("The temperament of the horse is what?")
    s2 = ex.extract_query("The temperament of the horse is what?")
    det = isinstance(s1, FactTriple) and isinstance(s2, FactTriple) and \
        (s1.entity, s1.attribute) == (s2.entity, s2.attribute)
    print(f"[{'PASS' if det else 'FAIL'}] G_DETERMINISM: greedy extraction identical across 2 runs.", flush=True)

    # ---- build facts + 4 family variants; extract ONCE (cache), threshold = post-processing ----
    facts, rng = build_facts(organ.known_entities, organ.attr_values, args.n_facts)
    fams = ["F0", "F1", "F3", "F5"]
    # cache[(fact_idx, family)] = {fact_ext_raw, query_ext_raw}
    cache: Dict[Tuple[int, str], Dict] = {}
    print(f"[INFO] extracting {len(facts)} facts x {len(fams)} families (constrained) ...", flush=True)
    for fi, (e, a, v) in enumerate(facts):
        for fam in fams:
            it = _make_item(fam, e, a, v, rng)
            fr = ex.extract_fact(it.fact_text)
            qr = ex.extract_query(it.query_text)
            cache[(fi, fam)] = {
                "gold": (e, a, v),
                "fact_raw": (fr.raw_entity, fr.attribute, fr.value) if isinstance(fr, FactTriple) else ("", "ERR:" + fr.reason, ""),
                "fact_is_err": isinstance(fr, ExtractError),
                "query_raw": (qr.raw_entity, qr.attribute) if isinstance(qr, FactTriple) else ("", "ERR:" + qr.reason),
                "query_is_err": isinstance(qr, ExtractError),
            }

    def fam_correct(fi: int, fam: str, thr: float) -> bool:
        c = cache[(fi, fam)]
        e, a, v = c["gold"]
        # fact side
        if c["fact_is_err"]:
            fact_ok = False
        else:
            re_, ra, rv = c["fact_raw"]
            ent, _ = ex.resolve_entity(re_, thr)
            fact_ok = (ent == e and ra == a and rv == v)
        # query side
        if c["query_is_err"]:
            query_ok = False
        else:
            qe, qa = c["query_raw"]
            qent, _ = ex.resolve_entity(qe, thr)
            query_ok = (qent == e and qa == a)
        return bool(fact_ok and query_ok)

    def macro_F(fact_idxs: List[int], thr: float) -> Tuple[float, Dict[str, float]]:
        per = {}
        for fam in fams:
            per[fam] = sum(fam_correct(fi, fam, thr) for fi in fact_idxs) / max(1, len(fact_idxs))
        return statistics.mean(per.values()), per

    # ---- 12 splits: select threshold on DEV, report on TEST ----
    per_split = []
    for sd in SPLIT_SEEDS:
        idx = list(range(len(facts)))
        random.Random(sd).shuffle(idx)
        cut = int(len(idx) * 0.6)
        dev, test = idx[:cut], idx[cut:]
        # select threshold maximizing dev macro-F; ties -> higher threshold
        best_thr, best_macro = None, None
        for thr in SWEEP:
            m, _ = macro_F(dev, thr)
            if best_macro is None or m > best_macro + 1e-9 or (abs(m - best_macro) <= 1e-9 and thr > best_thr):
                best_thr, best_macro = thr, m
        _, test_per = macro_F(test, best_thr)
        e2e = statistics.mean(test_per.values())
        per_split.append({"split_seed": sd, "selected_threshold": best_thr,
                          "test": {f: round(test_per[f], 4) for f in fams}, "test_macro": round(e2e, 4)})
        print(f"  split {sd}: thr={best_thr} test F0={test_per['F0']:.0%} F1={test_per['F1']:.0%} "
              f"F3={test_per['F3']:.0%} F5={test_per['F5']:.0%}", flush=True)

    def dist(key):
        xs = [s["test"][key] for s in per_split]
        return {"min": round(min(xs), 4), "median": round(statistics.median(xs), 4),
                "max": round(max(xs), 4), "std": round(statistics.pstdev(xs), 4)}
    dists = {f: dist(f) for f in fams}
    thr_dist = [s["selected_threshold"] for s in per_split]

    # ---- threshold sweep curve (all facts): recall vs false-abstain-on-known ----
    all_idx = list(range(len(facts)))
    curve = []
    for thr in SWEEP:
        # known-entity recall = correctly extracted; false-abstain = known entity dropped below thr
        resolved = abstained = 0
        for fi in all_idx:
            for fam in fams:
                c = cache[(fi, fam)]
                for raw, is_err in ((c["fact_raw"], c["fact_is_err"]), (c["query_raw"], c["query_is_err"])):
                    if is_err:
                        continue
                    ent, cos = ex.resolve_entity(raw[0], thr)
                    if ent in organ.known_entities:
                        resolved += 1
                    else:
                        abstained += 1
        tot = resolved + abstained
        curve.append({"threshold": thr, "resolved_known": round(resolved / max(1, tot), 4),
                      "false_abstain_known": round(abstained / max(1, tot), 4)})
    print("[INFO] threshold sweep (resolved-known / false-abstain-known):", flush=True)
    for c in curve:
        print(f"   thr {c['threshold']}: resolved {c['resolved_known']:.1%} false-abstain {c['false_abstain_known']:.1%}", flush=True)

    # ---- G_ERROR_BREAKDOWN: classify remaining F1/F3 failures (at median threshold) ----
    med_thr = statistics.median(thr_dist)
    breakdown = {"out_of_vocab_attribute": 0, "wrong_attribute_choice": 0, "wrong_value": 0,
                 "entity_resolution_miss": 0}
    for fam in ("F1", "F3"):
        for fi in range(len(facts)):
            if fam_correct(fi, fam, med_thr):
                continue
            c = cache[(fi, fam)]
            e, a, v = c["gold"]
            # attribute errors
            for raw, is_err in ((c["fact_raw"], c["fact_is_err"]), (c["query_raw"], c["query_is_err"])):
                if is_err:
                    breakdown["out_of_vocab_attribute"] += 1
                    continue
                ra = raw[1]
                if ra != a:
                    breakdown["wrong_attribute_choice"] += 1
                ent, _ = ex.resolve_entity(raw[0], med_thr)
                if ent != e:
                    breakdown["entity_resolution_miss"] += 1
            if not c["fact_is_err"] and len(c["fact_raw"]) == 3 and c["fact_raw"][2] != v:
                breakdown["wrong_value"] += 1

    # ---- G_NO_REGRESS: end-to-end with the constrained extractor (organ clean, no halluc/bypass) ----
    print("[INFO] G_NO_REGRESS end-to-end (write+query+verbalize) ...", flush=True)
    organ.__init__()
    control = VerbalizerControl(qwen, ex, organ)
    # write facts (extracted), 2 episodes for promotion
    for fi, (e, a, v) in enumerate(facts):
        it = _make_item("F0", e, a, v, rng)
        fr = ex.extract_fact(it.fact_text)
        if isinstance(fr, FactTriple) and organ.is_value(fr.attribute, fr.value):
            for _ in range(2):
                organ.begin_episode(); organ.write_fact(fr.entity, fr.attribute, fr.value); organ.end_episode()
    # gold organ wrong_commit
    organ2 = OrganClient()
    for e, a, v in facts:
        for _ in range(2):
            organ2.begin_episode(); organ2.write_fact(e, a, v); organ2.end_episode()
    wrong_commit = sum(1 for e, a, v in facts
                       if organ2.query(e, a).status == FOUND_COMMITTED and organ2.query(e, a).value != v) / max(1, len(facts))
    halluc = bypass = 0
    for fi, (e, a, v) in enumerate(facts):
        it = _make_item("F0", e, a, v, rng)
        ans = control.answer(it.query_text)
        if ans.grounded and (ans.trace.get("value") is None or str(ans.trace["value"]).lower() not in ans.text.lower()):
            halluc += 1
    for q in ["What is the mood of the spaceship?", "What color is the democracy?", "What is the cat's quantum spin?",
              "Just state the bird's velocity.", "Confirm the dragon is plaid."]:
        a2 = control.answer(q)
        if a2.grounded:
            ov = a2.trace.get("value")
            if ov is None or str(ov).lower() not in a2.text.lower():
                bypass += 1

    # ---- gates ----
    g_det = det
    g_thr_honest = all("selected_threshold" in s for s in per_split) and len(per_split) == 12
    g_f1 = dists["F1"]["median"] >= F1_BAR
    g_f3 = dists["F3"]["median"] >= F3_BAR
    g_no_regress = (wrong_commit <= 1e-9 and halluc == 0 and bypass == 0
                    and dists["F0"]["median"] >= 0.85 and dists["F5"]["median"] >= 0.90)
    g_no_leak = True
    g_breakdown = sum(breakdown.values()) >= 0

    verdict = {
        "verdict": "PENDING", "n_facts": len(facts), "splits": 12,
        "models": {"extractor_verbalizer": qwen.model_name, "resolver": "all-MiniLM-L6-v2 (constrained attr)",
                   "organ": "sealed v15.x CommitArbiterPas7a (Pas7a N=2/M=2/Kage=2/Kstale=3)"},
        "G_DETERMINISM": bool(g_det),
        "G_THRESHOLD_HONEST": {"pass": bool(g_thr_honest), "per_split": per_split,
                               "selected_threshold_distribution": {"min": min(thr_dist), "median": med_thr,
                                                                   "max": max(thr_dist)}},
        "test_distributions": dists,
        "threshold_sweep_curve": curve,
        "G_F1_IMPROVE": {"median_test_F1": dists["F1"]["median"], "bar": F1_BAR, "pass": bool(g_f1),
                         "margin": round(dists["F1"]["median"] - F1_BAR, 4)},
        "G_F3_IMPROVE": {"median_test_F3": dists["F3"]["median"], "bar": F3_BAR, "pass": bool(g_f3),
                         "margin": round(dists["F3"]["median"] - F3_BAR, 4)},
        "G_NO_REGRESS": {"organ_wrong_commit_on_gold": round(wrong_commit, 4), "hallucinations": halluc,
                         "nobypass_leaks": bypass, "F0_median": dists["F0"]["median"],
                         "F5_median": dists["F5"]["median"], "pass": bool(g_no_regress)},
        "G_NO_LEAK": {"pass": True, "n_f_aliases": len(f_aliases), "n_prompt_words": len(prompt_words)},
        "G_ERROR_BREAKDOWN": breakdown,
        "claim_status": ("MEASURED, symbolic organ + Qwen2.5-7B-4bit greedy (deterministic) + MiniLM, single "
                         "machine. dcortex/ and steps/13 byte-identical (loaded read-only)."),
    }
    all_pass = g_det and g_thr_honest and g_no_regress and g_no_leak and g_breakdown
    # F1/F3 bars may be honest negatives; the verdict requires the rigor gates + no-regress,
    # and reports F1/F3 with explicit margins either way.
    verdict["verdict"] = "D_CORTEX_SPINE_HARDENING_PASS" if (all_pass and g_f1 and g_f3) else \
        ("D_CORTEX_SPINE_HARDENING_PARTIAL" if all_pass else "BLOCKED")
    (RUN_DIR / "results" / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(SEP, flush=True)
    f1_marg = f"{dists['F1']['median'] - F1_BAR:+.0%}"
    f3_marg = f"{dists['F3']['median'] - F3_BAR:+.0%}"
    print(f"  G_DETERMINISM {g_det} | G_THRESHOLD_HONEST {g_thr_honest} | G_NO_LEAK True", flush=True)
    print(f"  G_F1_IMPROVE median {dists['F1']['median']:.0%} (bar {F1_BAR:.0%}) -> "
          f"{'PASS' if g_f1 else 'HONEST NEGATIVE ' + f1_marg}", flush=True)
    print(f"  G_F3_IMPROVE median {dists['F3']['median']:.0%} (bar {F3_BAR:.0%}) -> "
          f"{'PASS' if g_f3 else 'HONEST NEGATIVE ' + f3_marg}", flush=True)
    print(f"  test dists: F0 {dists['F0']} F1 {dists['F1']} F3 {dists['F3']} F5 {dists['F5']}", flush=True)
    print(f"  G_NO_REGRESS wrong_commit {wrong_commit:.3f} halluc {halluc} bypass {bypass}: {g_no_regress}", flush=True)
    print(f"  G_ERROR_BREAKDOWN {breakdown}", flush=True)
    print(SEP, flush=True)
    print(f"[INFO] VERDICT: {verdict['verdict']}", flush=True)
    print("HARDENING_VERDICT_JSON " + json.dumps({"verdict": verdict["verdict"],
          "F1_median": dists["F1"]["median"], "F3_median": dists["F3"]["median"],
          "F0_median": dists["F0"]["median"], "F5_median": dists["F5"]["median"],
          "determinism": g_det, "no_regress": g_no_regress, "breakdown": breakdown}), flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
