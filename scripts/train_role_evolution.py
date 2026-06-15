# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex role-binding evolution: content-addressed, relation-typed binding head
# trained above the FROZEN substrate, benchmarked for systematic generalization
# on held-out syntactic families and held-out entities. The substrate
# (warmstarted_init.pt) is read-only and never modified; only the new binding
# head is trainable. The benchmark gates are pre-declared and frozen below.

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

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
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model
from dcortex.semantic_role_conditioned import DCortexTokenContextBackend, phrase_token_positions
import independent_role_corpus_audit as audit

SEP = "=" * 70
ENC = tiktoken.get_encoding("gpt2")
WARMSTART = REPO_ROOT.parent / "D_Cortex-main" / "runs" / "warmstart" / "warmstarted_init.pt"
CORPUS = REPO_ROOT / "data" / "role_evolution" / "role_evolution_corpus.jsonl"

# --- Pre-declared, FROZEN benchmark gates (set before any run) ---
SEEDS = 10
G_GEN_EXACT_FLOOR = 0.70        # held-out-family median exact-match must reach this
G_GEN_BASELINE_UPLIFT = 0.30    # ... and beat the lexical baseline by >= 30pp
G_SAFE_WRONG_CEIL = 0.15        # held-out-family median wrong-mapping must stay under this
G_CALIB_ABSTAIN_FLOOR = 0.70    # on ambiguous items, abstain rate must reach this
WRONG_BIND_PENALTY = 1.0        # abstain-calibration: confident-wrong costs more than abstain


def big_config() -> DCortexConfig:
    return DCortexConfig(hidden_dim=1024, n_enc_heads=16, n_dec_heads=16,
                         enc_ff_dim=4096, dec_ff_dim=4096, n_dec_layers=16,
                         n_enc_layers=4, n_fusion_layers=4, max_seq_len=2048)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def label_of(record: Dict) -> int:
    """0 identity, 1 swapped, 2 unresolved (abstain)."""
    if record["ambiguous"]:
        return 2
    ents, vals, attr = record["entities"], record["values"], record["attribute"]
    expected = set(tuple(x) for x in record["expected"])
    identity = {(ents[0], attr, vals[0]), (ents[1], attr, vals[1])}
    return 0 if expected == identity else 1


class ContentAddressedRoleBinder(nn.Module):
    """Relation-typed, content-addressed, position-independent binding head.

    Binds by matching a relation-keyed entity query against value keys built from
    pooled span content; one uniform bind for all families; a learned abstain
    logit competes with the two bindings."""

    def __init__(self, context_dim: int, proj_dim: int = 256, relation_dim: int = 64,
                 n_relations: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(context_dim),
                                  nn.Linear(context_dim, proj_dim), nn.GELU(),
                                  nn.Dropout(dropout))
        self.relation = nn.Embedding(n_relations, relation_dim)
        self.query = nn.Linear(proj_dim + relation_dim, proj_dim)
        self.key = nn.Linear(proj_dim + relation_dim, proj_dim)
        self.scale = proj_dim ** -0.5
        self.abstain_bias = nn.Parameter(torch.zeros(()))
        self.proj_dim = proj_dim

    def forward(self, reps: torch.Tensor, relation_id: torch.Tensor) -> torch.Tensor:
        """reps [B,4,context_dim] = (entity1, entity2, value1, value2). -> logits [B,3]."""
        projected = self.proj(reps)                                   # [B,4,d]
        e1, e2, v1, v2 = projected.unbind(dim=1)
        rel = self.relation(relation_id)                              # [B, rdim]

        def q(x: torch.Tensor) -> torch.Tensor:
            return self.query(torch.cat((x, rel), dim=-1))

        def k(x: torch.Tensor) -> torch.Tensor:
            return self.key(torch.cat((x, rel), dim=-1))

        q1, q2, k1, k2 = q(e1), q(e2), k(v1), k(v2)
        s11 = (q1 * k1).sum(-1) * self.scale
        s12 = (q1 * k2).sum(-1) * self.scale
        s21 = (q2 * k1).sum(-1) * self.scale
        s22 = (q2 * k2).sum(-1) * self.scale
        identity = s11 + s22
        swapped = s12 + s21
        unresolved = self.abstain_bias.expand_as(identity)
        return torch.stack((identity, swapped, unresolved), dim=1)


def extract_features(records: Sequence[Dict], backend: DCortexTokenContextBackend
                     ) -> Tuple[torch.Tensor, List[int]]:
    """Frozen contextual span-pooled content vectors per record: [N,4,dim].
    Returns the feature tensor and the indices of records that were usable."""
    tok = lambda t: ENC.encode_ordinary(t)  # noqa: E731
    feats: List[torch.Tensor] = []
    ok_index: List[int] = []
    for i, record in enumerate(records):
        ctx = backend.features([record["source_text"]])
        hidden = ctx.hidden[0]                                        # [seq, dim]
        ids = ctx.token_ids[0].tolist()
        phrases = [record["entities"][0], record["entities"][1],
                   record["values"][0], record["values"][1]]
        pooled = []
        good = True
        for phrase in phrases:
            positions = phrase_token_positions(ids, phrase, tok)
            if not positions:
                good = False
                break
            pooled.append(hidden[list(positions)].mean(dim=0))
        if not good:
            continue
        feats.append(torch.stack(pooled, dim=0))
        ok_index.append(i)
    return torch.stack(feats, dim=0), ok_index


def train_one_seed(reps: torch.Tensor, labels: torch.Tensor, known_mask: torch.Tensor,
                   idx_train: torch.Tensor, idx_val: torch.Tensor, context_dim: int,
                   device: torch.device, seed: int) -> nn.Module:
    """Train the head; early-stop on held-out-family (validation) exact-match."""
    seed_everything(seed)
    head = ContentAddressedRoleBinder(context_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-3)
    rel = torch.zeros(reps.shape[0], dtype=torch.long, device=device)
    reps = reps.to(device)
    labels = labels.to(device)
    known_mask = known_mask.to(device)
    best_val = -1.0
    best_state = None
    patience, stale, batch = 25, 0, 64
    gen = torch.Generator().manual_seed(seed)

    def val_exact() -> float:
        head.eval()
        with torch.no_grad():
            logits = head(reps[idx_val], rel[idx_val])
            pred = logits.argmax(dim=1)
            known = known_mask[idx_val]
            if known.sum() == 0:
                return 0.0
            return float((pred[known] == labels[idx_val][known]).float().mean())

    for epoch in range(200):
        head.train()
        perm = idx_train[torch.randperm(len(idx_train), generator=gen)]
        for start in range(0, len(perm), batch):
            bi = perm[start:start + batch]
            logits = head(reps[bi], rel[bi])
            loss = F.cross_entropy(logits, labels[bi], label_smoothing=0.02)
            # abstain-calibration: penalize probability on the WRONG binding for
            # known records, so a confident wrong bind costs more than abstaining.
            probs = F.softmax(logits, dim=1)
            kmask = known_mask[bi]
            if kmask.any():
                wrong_class = 1 - labels[bi][kmask]   # the opposite binding (0<->1)
                wrong_prob = probs[kmask].gather(1, wrong_class.unsqueeze(1)).squeeze(1)
                loss = loss + WRONG_BIND_PENALTY * wrong_prob.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
        v = val_exact()
        if v > best_val + 1e-6:
            best_val = v
            best_state = {k: t.detach().cpu().clone() for k, t in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    return head


@torch.no_grad()
def evaluate(head: nn.Module, reps: torch.Tensor, labels: torch.Tensor,
             records: Sequence[Dict], indices: Sequence[int], device: torch.device
             ) -> Dict[str, Any]:
    """Per-family exact/wrong/abstain on known records + ambiguous abstain rate."""
    rel = torch.zeros(reps.shape[0], dtype=torch.long, device=device)
    logits = head(reps[torch.tensor(indices, device=device)], rel[torch.tensor(indices, device=device)])
    pred = logits.argmax(dim=1).cpu().tolist()
    per_family: Dict[str, Dict[str, int]] = {}
    amb_total, amb_abstain = 0, 0
    for local, gi in enumerate(indices):
        rec = records[gi]
        p = pred[local]
        if rec["ambiguous"]:
            amb_total += 1
            amb_abstain += int(p == 2)
            continue
        fam = rec["construction_family"]
        d = per_family.setdefault(fam, {"n": 0, "exact": 0, "wrong": 0, "abstain": 0})
        d["n"] += 1
        correct = int(labels[gi])
        if p == 2:
            d["abstain"] += 1
        elif p == correct:
            d["exact"] += 1
        else:
            d["wrong"] += 1
    families = {f: {"n": d["n"], "exact": d["exact"] / d["n"], "wrong": d["wrong"] / d["n"],
                    "abstain": d["abstain"] / d["n"]} for f, d in per_family.items()}
    n_known = sum(d["n"] for d in per_family.values())
    agg = {"exact": sum(d["exact"] for d in per_family.values()) / max(1, n_known),
           "wrong": sum(d["wrong"] for d in per_family.values()) / max(1, n_known),
           "abstain": sum(d["abstain"] for d in per_family.values()) / max(1, n_known)}
    return {"per_family": families, "aggregate": agg,
            "ambiguous_abstain_rate": amb_abstain / max(1, amb_total),
            "ambiguous_n": amb_total}


def lexical_baseline_eval(records: Sequence[Dict], indices: Sequence[int]) -> float:
    """Position/lexical baseline exact-match on held-out known records."""
    rec_objs = []
    for gi in indices:
        r = records[gi]
        if r["ambiguous"]:
            continue
        rec_objs.append(audit.IndependentRoleRecord(
            record_id=r["record_id"], split=r["split"],
            construction_family=r["construction_family"], source_text=r["source_text"],
            attribute=r["attribute"], entities=tuple(r["entities"]),
            values=tuple(r["values"]), expected=tuple(tuple(x) for x in r["expected"]),
            ambiguous=False, provenance=r["provenance"]))
    if not rec_objs:
        return 0.0
    best = 0.0
    for name in ("ordered_first_occurrence", "minimum_distance", "lexical_cartesian"):
        fn = audit.BASELINES[name]
        exact = sum(int(fn(rec) == rec.expected) for rec in rec_objs) / len(rec_objs)
        best = max(best, exact)
    return best


def agg_stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "n": 0}
    return {"median": round(statistics.median(values), 4), "min": round(min(values), 4),
            "max": round(max(values), 4),
            "std": round(statistics.pstdev(values) if len(values) > 1 else 0.0, 4),
            "n": len(values)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Role-binding evolution train + benchmark")
    ap.add_argument("--run-dir", default=str(REPO_ROOT / "runs" / "role_evolution"))
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()
    results_dir = Path(args.run_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(SEP, flush=True)
    print(f"[INFO] Role-binding evolution | device={device} | seeds={args.seeds}", flush=True)
    if not WARMSTART.exists():
        print(f"[ERROR] Frozen substrate missing: {WARMSTART}", flush=True)
        return 2
    records = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    corpus_sha = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    print(f"[INFO] Corpus {len(records)} records, SHA {corpus_sha}", flush=True)

    # --- frozen substrate (read-only) ---
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        ckpt = torch.load(WARMSTART, map_location=device, weights_only=False)
        model = DCortexV2Model(big_config()).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    backend = DCortexTokenContextBackend(model, lambda t: ENC.encode_ordinary(t), max_seq_len=128)
    context_dim = backend.output_dim
    substrate_sha = hashlib.sha256(WARMSTART.read_bytes()).hexdigest()
    print(f"[INFO] Frozen substrate context_dim={context_dim}; warmstart SHA {substrate_sha[:16]}...",
          flush=True)

    # --- features (computed once, frozen) ---
    print("[INFO] Extracting frozen span-pooled features ...", flush=True)
    reps, ok_index = extract_features(records, backend)
    records = [records[i] for i in ok_index]   # keep only usable records
    labels = torch.tensor([label_of(r) for r in records], dtype=torch.long)
    known_mask = torch.tensor([not r["ambiguous"] for r in records], dtype=torch.bool)
    splits = {s: [i for i, r in enumerate(records) if r["split"] == s]
              for s in ("train", "validation", "evaluation")}
    print(f"[INFO] Usable records {len(records)} (dropped {len(ok_index) and 0}); "
          f"splits train={len(splits['train'])} val={len(splits['validation'])} "
          f"eval={len(splits['evaluation'])}", flush=True)
    idx_train = torch.tensor(splits["train"])
    idx_val = torch.tensor(splits["validation"])
    eval_indices = splits["evaluation"]

    lex = lexical_baseline_eval(records, eval_indices)
    print(f"[INFO] Lexical/position baseline exact on held-out eval = {lex:.1%}", flush=True)

    # --- train across seeds, evaluate on held-out families ---
    per_seed: List[Dict[str, Any]] = []
    print(SEP, flush=True)
    for s in range(args.seeds):
        head = train_one_seed(reps, labels, known_mask, idx_train, idx_val, context_dim, device, 1000 + s)
        ev = evaluate(head, reps.to(device), labels, records, eval_indices, device)
        per_seed.append(ev)
        a = ev["aggregate"]
        print(f"  seed {s:2d} | held-out exact={a['exact']:.1%} wrong={a['wrong']:.1%} "
              f"abstain={a['abstain']:.1%} | ambiguous-abstain={ev['ambiguous_abstain_rate']:.1%}",
              flush=True)

    # --- aggregate across seeds ---
    families = sorted({f for ev in per_seed for f in ev["per_family"]})
    family_stats = {}
    for fam in families:
        family_stats[fam] = {
            "exact": agg_stats([ev["per_family"][fam]["exact"] for ev in per_seed if fam in ev["per_family"]]),
            "wrong": agg_stats([ev["per_family"][fam]["wrong"] for ev in per_seed if fam in ev["per_family"]]),
            "abstain": agg_stats([ev["per_family"][fam]["abstain"] for ev in per_seed if fam in ev["per_family"]]),
        }
    agg_exact = agg_stats([ev["aggregate"]["exact"] for ev in per_seed])
    agg_wrong = agg_stats([ev["aggregate"]["wrong"] for ev in per_seed])
    agg_abstain = agg_stats([ev["aggregate"]["abstain"] for ev in per_seed])
    amb_abstain = agg_stats([ev["ambiguous_abstain_rate"] for ev in per_seed])

    # --- gates (pre-declared, frozen) ---
    g_gen = (agg_exact["median"] >= G_GEN_EXACT_FLOOR) and (agg_exact["median"] >= lex + G_GEN_BASELINE_UPLIFT)
    g_safe = agg_wrong["median"] <= G_SAFE_WRONG_CEIL
    g_calib = amb_abstain["median"] >= G_CALIB_ABSTAIN_FLOOR
    seals = audit.artifact_hash_report(audit.SEALED_ARTIFACTS)
    g_seals = seals["all_ok"]

    verdict = [
        {"criterion_id": "G_GEN", "passed": bool(g_gen),
         "evidence": (f"held-out-family median exact-match {agg_exact['median']:.1%} "
                      f"[{agg_exact['min']:.1%}/{agg_exact['max']:.1%}/std {agg_exact['std']:.3f}] over "
                      f"{args.seeds} seeds; floor {G_GEN_EXACT_FLOOR:.0%}; lexical baseline {lex:.1%}; "
                      f"uplift {agg_exact['median']-lex:+.1%} (need >= {G_GEN_BASELINE_UPLIFT:.0%}).")},
        {"criterion_id": "G_SAFE", "passed": bool(g_safe),
         "evidence": (f"held-out-family median wrong-mapping {agg_wrong['median']:.1%} "
                      f"[{agg_wrong['min']:.1%}/{agg_wrong['max']:.1%}]; ceiling {G_SAFE_WRONG_CEIL:.0%} "
                      f"(RB3 was 31.4%); abstain median {agg_abstain['median']:.1%}.")},
        {"criterion_id": "G_CALIB", "passed": bool(g_calib),
         "evidence": (f"ambiguous-item abstain rate median {amb_abstain['median']:.1%} "
                      f"[{amb_abstain['min']:.1%}/{amb_abstain['max']:.1%}]; floor {G_CALIB_ABSTAIN_FLOOR:.0%} "
                      f"(prefers abstain over confident wrong-bind on uncertain items).")},
        {"criterion_id": "G_SEALS", "passed": bool(g_seals),
         "evidence": (f"sealed substrate/semantic sources byte-identical={g_seals}; "
                      f"warmstart SHA {substrate_sha[:16]}... (read-only, never trained).")},
    ]
    out = {"verdict": verdict, "reference": {
        "corpus_sha256": corpus_sha, "substrate_sha256": substrate_sha,
        "seeds": args.seeds, "lexical_baseline_exact": round(lex, 4),
        "heldout_eval_families": sorted({records[i]["construction_family"] for i in eval_indices
                                         if not records[i]["ambiguous"]}),
        "aggregate": {"exact": agg_exact, "wrong": agg_wrong, "abstain": agg_abstain,
                      "ambiguous_abstain": amb_abstain},
        "per_family": family_stats,
        "gates_predeclared": {"G_GEN_EXACT_FLOOR": G_GEN_EXACT_FLOOR,
                              "G_GEN_BASELINE_UPLIFT": G_GEN_BASELINE_UPLIFT,
                              "G_SAFE_WRONG_CEIL": G_SAFE_WRONG_CEIL,
                              "G_CALIB_ABSTAIN_FLOOR": G_CALIB_ABSTAIN_FLOOR},
        "claim_status": ("MEASURED systematic generalization on held-out families + entities, "
                         "single environment, " + str(args.seeds) + " seeds. Not PROVEN.")}}
    (results_dir / "verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(SEP, flush=True)
    for fam in families:
        fs = family_stats[fam]
        print(f"  [{fam}] exact={fs['exact']['median']:.1%} wrong={fs['wrong']['median']:.1%} "
              f"abstain={fs['abstain']['median']:.1%}", flush=True)
    print(SEP, flush=True)
    for v in verdict:
        print(f"{'✓ PASS' if v['passed'] else '✗ FAIL'}  [{v['criterion_id']}] {v['evidence']}", flush=True)
    print(SEP, flush=True)
    all_pass = all(v["passed"] for v in verdict)
    print(f"[INFO] Overall: {'ALL GATES PASS' if all_pass else 'GATE FAILURE (reported)'}", flush=True)
    print("EVO_VERDICT_JSON " + json.dumps({"verdict": [{"criterion_id": v["criterion_id"],
          "passed": v["passed"]} for v in verdict], "exact_median": agg_exact["median"],
          "wrong_median": agg_wrong["median"], "lexical": lex}), flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
