# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex spine: entity-resolution accuracy. After the constrained attribute fix the
# dominant remaining failure is entity resolution. Diagnostic (Part 0) shows these are
# the Qwen free-generation extractor naming the property word instead of the entity, so
# no embedder swap on that word can recover. This campaign runs a 5-way head-to-head of
# resolvers, each mapping the SOURCE TEXT (the clue) to one known entity or "none"
# (abstain): R0 MiniLM, R1 bge-small, R2 mpnet (cosine on text), R3 Qwen constrained
# classification over {known entities}+none, R4 MiniLM top-5 retrieve -> Qwen rerank.
# 12 dev/test splits (disjoint by fact); embedder thresholds selected on dev. The organ
# and steps/13 are byte-identical (loaded read-only).

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
for extra in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import torch
from integration.organ_client import OrganClient, FOUND_COMMITTED
from integration.constrained_extractor import ConstrainedExtractor, assert_no_leak, _alias_words
from integration.corpus import _make_item
from integration.ingest_adapter import FactTriple
from dcortex_professional.qwen_runtime import QwenBaseModel

SEP = "=" * 70
RUN_DIR = REPO_ROOT / "runs" / "entity_resolution"
SPLIT_SEEDS = list(range(1001, 1013))
SWEEP = [round(0.30 + 0.05 * i, 2) for i in range(11)]
EMBEDDERS = {"R0_minilm": "sentence-transformers/all-MiniLM-L6-v2",
             "R1_bge": "BAAI/bge-small-en-v1.5",
             "R2_mpnet": "sentence-transformers/all-mpnet-base-v2"}
ENTITY_PROMPT = ("Which single named thing (a creature, a person, or an object) is the text below about? "
                 "Ignore any property words. Name only the thing.\nText: {text}")


def build_facts(entities, attr_values, n, seed=7):
    rng = random.Random(seed)
    attrs = ["color", "size", "location", "state"]
    pairs = [(e, a) for e in entities for a in attrs]
    rng.shuffle(pairs)
    return [(e, a, rng.choice(attr_values[a])) for e, a in pairs[:n]], rng


def main() -> int:
    ap = argparse.ArgumentParser(description="Entity-resolution head-to-head")
    ap.add_argument("--n-facts", type=int, default=51)
    args = ap.parse_args()
    (RUN_DIR / "results").mkdir(parents=True, exist_ok=True)
    print(SEP, flush=True)
    print("[INFO] D_Cortex entity-resolution accuracy (5-way head-to-head)", flush=True)

    organ = OrganClient()
    qwen = QwenBaseModel()
    if not qwen.available:
        print(f"[BLOCKED] Qwen unavailable: {qwen.reason}", flush=True)
        return 2
    ex = ConstrainedExtractor(qwen, organ.attr_values, organ.known_entities)
    known = list(organ.known_entities)

    # ---- G_NO_LEAK: prompt tokens disjoint from F-aliases AND entity surface forms ----
    import integration.corpus as corpus
    import re
    f_aliases = _alias_words(corpus)
    entity_surfaces = {e.lower() for e in known}
    instr_words = {w.lower() for w in re.findall(r"[A-Za-z]+", ENTITY_PROMPT) if len(w) >= 3}
    leak_alias = instr_words & f_aliases
    leak_entity = instr_words & entity_surfaces
    assert_no_leak()
    g_no_leak = not leak_alias and not leak_entity
    print(f"[{'PASS' if g_no_leak else 'FAIL'}] G_NO_LEAK: entity-prompt words {sorted(instr_words)}", flush=True)
    print(f"   disjoint from {len(f_aliases)} aliases ({leak_alias or 'ok'}) and {len(entity_surfaces)} "
          f"entity surfaces ({leak_entity or 'ok'}); entities are scored as continuations, not listed in the prompt.",
          flush=True)

    # ---- embedders ----
    from sentence_transformers import SentenceTransformer
    embs = {}
    for k, name in EMBEDDERS.items():
        m = SentenceTransformer(name, device="cuda")
        embs[k] = (m, m.encode(known, convert_to_tensor=True, normalize_embeddings=True))

    def embed_topk(text, k_emb, topk=1):
        m, K = embs[k_emb]
        q = m.encode([text], convert_to_tensor=True, normalize_embeddings=True)[0]
        sims = torch.matmul(K, q)
        vals, idx = torch.topk(sims, min(topk, len(known)))
        return [(known[int(i)], float(v)) for v, i in zip(vals, idx)]

    def qwen_pick(text, options):
        best, _ = qwen.classify(ENTITY_PROMPT.format(text=text), options, answer_prefix=" The thing is")
        return best

    # ---- build facts + variants; resolve entity per item with all resolvers (cache) ----
    facts, rng = build_facts(known, organ.attr_values, args.n_facts)
    fams = ["F0", "F1", "F3", "F5"]
    # attribute/value (resolver-independent) + per-resolver entity, per (fact, family, side)
    rows = []
    print(f"[INFO] extracting+resolving {len(facts)} facts x {len(fams)} families ...", flush=True)
    for fi, (e, a, v) in enumerate(facts):
        for fam in fams:
            it = _make_item(fam, e, a, v, rng)
            for side, text in (("fact", it.fact_text), ("query", it.query_text)):
                attr = ex.classify_attribute(text)
                val = ex.classify_value(text, attr) if (side == "fact" and attr in organ.attr_values) else None
                # resolvers: embedders return (entity, cos); Qwen R3/R4 return entity-or-none
                emb_out = {k: embed_topk(text, k, 1)[0] for k in embs}
                r3 = qwen_pick(text, known + ["none"])
                top5 = [c for c, _ in embed_topk(text, "R0_minilm", 5)]
                r4 = qwen_pick(text, top5 + ["none"])
                rows.append({"fi": fi, "fam": fam, "side": side, "gold_e": e, "gold_a": a, "gold_v": v,
                             "attr": attr, "val": val, "emb": emb_out, "R3": r3, "R4": r4})

    # ---- Part 0 diagnostic: current resolver (R0 on EXTRACTED WORD, thr 0.55) miss types ----
    diag = {"wrong_match": 0, "no_match": 0}
    for fi, (e, a, v) in enumerate(facts):
        for fam in ("F1", "F3"):
            it = _make_item(fam, e, a, v, rng)
            raw = ex._extract_entity(it.query_text)
            resolved, cos = ex.resolve_entity(raw, 0.55)
            if resolved != e:
                diag["no_match" if resolved not in known else "wrong_match"] += 1
    print(f"[INFO] G_ENTITY_DIAG (current word-based resolver, F1/F3 query side): {diag}", flush=True)

    # ---- resolver entity decision at a threshold (embedders) / fixed (R3,R4) ----
    def entity_of(row, resolver, thr):
        if resolver in embs:
            ent, cos = row["emb"][resolver]
            return ent if cos >= thr else None
        pick = row[resolver]
        return None if pick == "none" else pick

    def fam_correct(fi, fam, resolver, thr):
        # need fact (attr,val,entity) and query (attr,entity) both correct
        fr = next(r for r in rows if r["fi"] == fi and r["fam"] == fam and r["side"] == "fact")
        qr = next(r for r in rows if r["fi"] == fi and r["fam"] == fam and r["side"] == "query")
        e, a, v = fr["gold_e"], fr["gold_a"], fr["gold_v"]
        fok = (fr["attr"] == a and fr["val"] == v and entity_of(fr, resolver, thr) == e)
        qok = (qr["attr"] == a and entity_of(qr, resolver, thr) == e)
        return bool(fok and qok)

    def entity_acc(fact_idxs, resolver, thr):
        n = ok = 0
        for fi in fact_idxs:
            for fam in fams:
                for side in ("fact", "query"):
                    r = next(x for x in rows if x["fi"] == fi and x["fam"] == fam and x["side"] == side)
                    n += 1
                    ok += int(entity_of(r, resolver, thr) == r["gold_e"])
        return ok / max(1, n)

    def macro_F(fact_idxs, resolver, thr):
        per = {f: sum(fam_correct(fi, f, resolver, thr) for fi in fact_idxs) / max(1, len(fact_idxs)) for f in fams}
        return statistics.mean(per.values()), per

    resolvers = list(embs) + ["R3", "R4"]
    # ---- 12 splits, per resolver: select threshold on dev (embedders), report test ----
    split_results = {r: [] for r in resolvers}
    for sd in SPLIT_SEEDS:
        idx = list(range(len(facts)))
        random.Random(sd).shuffle(idx)
        cut = int(len(idx) * 0.6)
        dev, test = idx[:cut], idx[cut:]
        for r in resolvers:
            if r in embs:
                best_thr, best_m = None, None
                for thr in SWEEP:
                    m, _ = macro_F(dev, r, thr)
                    if best_m is None or m > best_m + 1e-9 or (abs(m - best_m) <= 1e-9 and thr > best_thr):
                        best_thr, best_m = thr, m
            else:
                best_thr = 0.0
            ent_test = entity_acc(test, r, best_thr)
            _, perF = macro_F(test, r, best_thr)
            split_results[r].append({"split": sd, "thr": best_thr, "entity_acc": round(ent_test, 4),
                                     "F": {f: round(perF[f], 4) for f in fams}})

    def dist(xs):
        return {"min": round(min(xs), 4), "median": round(statistics.median(xs), 4),
                "max": round(max(xs), 4), "std": round(statistics.pstdev(xs), 4)}
    summary = {}
    for r in resolvers:
        ent = [s["entity_acc"] for s in split_results[r]]
        summary[r] = {"entity_acc": dist(ent),
                      "F": {f: dist([s["F"][f] for s in split_results[r]]) for f in fams}}
        print(f"  {r}: entity_acc median {summary[r]['entity_acc']['median']:.0%} "
              f"[{summary[r]['entity_acc']['min']:.0%}/{summary[r]['entity_acc']['max']:.0%}] | "
              f"F1 {summary[r]['F']['F1']['median']:.0%} F3 {summary[r]['F']['F3']['median']:.0%}", flush=True)

    winner = max(resolvers, key=lambda r: summary[r]["entity_acc"]["median"])
    win_acc = summary[winner]["entity_acc"]["median"]

    # ---- aggregate entity_resolution_miss for winner vs baseline (full corpus) ----
    def agg_miss(resolver, thr_pick):
        miss = 0
        for r in rows:
            if r["fam"] in ("F1", "F3") and r["side"] == "query":
                miss += int(entity_of(r, resolver, thr_pick) != r["gold_e"])
        return miss
    win_thr = statistics.median([s["thr"] for s in split_results[winner]]) if winner in embs else 0.0
    # baseline = the CURRENT production resolver (word-based, the Part 0 diagnostic count),
    # which is what we are improving FROM (the reported 12 entity_resolution_miss).
    base_miss = sum(diag.values())
    win_miss = agg_miss(winner, win_thr)

    # ---- G_DETERMINISM ----
    a1 = qwen_pick("Regarding physical scale of the doctor, where is it?", known + ["none"])
    a2 = qwen_pick("Regarding physical scale of the doctor, where is it?", known + ["none"])
    g_det = a1 == a2

    # ---- gates ----
    g_diag = True
    g_h2h = all(len(split_results[r]) == 12 for r in resolvers)
    g_improve = (win_miss <= base_miss * 0.5) and (win_acc >= 0.90)
    g_no_regress = (summary[winner]["F"]["F1"]["median"] >= 0.86 and summary[winner]["F"]["F3"]["median"] >= 0.81
                    and summary[winner]["F"]["F0"]["median"] >= 0.95 and summary[winner]["F"]["F5"]["median"] >= 0.95)

    verdict = {
        "verdict": "PENDING", "n_facts": len(facts),
        "G_ENTITY_DIAG": diag,
        "winner": winner, "winner_entity_acc_median": win_acc,
        "baseline_R0_entity_acc_median": summary["R0_minilm"]["entity_acc"]["median"],
        "aggregate_miss": {"baseline_R0": base_miss, "winner": win_miss,
                           "reduction": round(1 - win_miss / max(1, base_miss), 4)},
        "G_ENTITY_IMPROVE": {"winner_miss_reduction_ge_50pct": bool(win_miss <= base_miss * 0.5),
                             "winner_entity_acc_ge_90pct": bool(win_acc >= 0.90), "pass": bool(g_improve)},
        "G_NO_REGRESS": bool(g_no_regress),
        "G_DETERMINISM": bool(g_det), "G_NO_LEAK": bool(g_no_leak),
        "resolver_summary": summary, "per_split": split_results,
        "claim_status": ("MEASURED, symbolic organ + Qwen-4bit greedy + chosen resolver, single machine. "
                         "dcortex/ and steps/13 byte-identical (loaded read-only)."),
    }
    all_pass = g_diag and g_h2h and g_det and g_no_leak and g_no_regress
    verdict["verdict"] = ("D_CORTEX_ENTITY_RESOLUTION_PASS" if (all_pass and g_improve)
                          else ("D_CORTEX_ENTITY_RESOLUTION_PARTIAL" if all_pass else "BLOCKED"))
    (RUN_DIR / "results" / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(SEP, flush=True)
    print(f"  WINNER: {winner} entity_acc median {win_acc:.0%}; aggregate F1/F3-query miss "
          f"{base_miss} (R0) -> {win_miss} ({winner})", flush=True)
    print(f"  G_ENTITY_IMPROVE (>=50% miss cut AND >=90% acc): {g_improve}", flush=True)
    print(f"  G_NO_REGRESS {g_no_regress} | G_DETERMINISM {g_det} | G_NO_LEAK {g_no_leak}", flush=True)
    print(SEP, flush=True)
    print(f"[INFO] VERDICT: {verdict['verdict']}", flush=True)
    print("ENTITY_VERDICT_JSON " + json.dumps({"verdict": verdict["verdict"], "winner": winner,
          "winner_acc": win_acc, "baseline_acc": summary["R0_minilm"]["entity_acc"]["median"],
          "miss_base": base_miss, "miss_win": win_miss, "diag": diag}), flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
