# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# Builds the D_Cortex_PatentAnalyst professional pack (committed / provisional /
# disputed / forbidden facts + sources + abstain rules + schemas). Every committed
# fact is grounded in an auditable repository source (file + reference), pinned by
# the source file SHA. The build is deterministic. Data construction only; no model
# is loaded here.

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

SEP = "=" * 70
REPO_ROOT = Path(__file__).resolve().parent.parent
PACK_DIR = REPO_ROOT / "data" / "professional" / "D_Cortex_PatentAnalyst"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "MISSING"


def provenance(source_id: str, ref: str) -> Dict[str, str]:
    return {"source": source_id, "ref": ref, "document": source_id}


# Auditable repository sources used to ground committed facts.
SOURCES = {
    "project_concept": {"type": "repo_file", "ref": ".claude/project_concept.json",
                        "note": "D_Cortex project concept (source of truth for project facts)"},
    "big_config": {"type": "repo_file", "ref": "scripts/train_role_evolution.py",
                   "note": "big_config() = the BIG DCortexConfig used across role-binding runs"},
    "struct_verdict": {"type": "repo_file", "ref": "data/role_struct/verdict.json",
                       "note": "vnext3 structural certification verdict (measured numbers + claim status)"},
}

# committed facts: (entity, attribute, value, source_id, ref)
COMMITTED = [
    ("D_Cortex", "patent_number", "EP25216372.0", "project_concept", "patent"),
    ("D_Cortex", "owner", "Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.",
     "project_concept", "owner"),
    ("D_Cortex", "architecture", "dual-agent memory-native transformer", "project_concept", "vision"),
    ("D_Cortex", "substrate", "GPT-2-medium warm-started decoder", "big_config",
     "warmstarted_init.pt big config"),
    ("DCortexConfig", "hidden_dim", "1024", "big_config", "big_config().hidden_dim"),
    ("DCortexConfig", "decoder_layers", "16", "big_config", "big_config().n_dec_layers"),
    ("DCortexConfig", "encoder_layers", "4", "big_config", "big_config().n_enc_layers"),
    ("ContentAddressedRoleBinder", "held_out_structural_exact_median", "99.3 percent",
     "struct_verdict", "reference.aggregate.exact.median"),
    ("ContentAddressedRoleBinder", "no_memory_control_exact", "51.6 percent",
     "struct_verdict", "reference.aggregate.no_memory_exact.median"),
    ("ContentAddressedRoleBinder", "claim_status", "MEASURED not PROVEN", "struct_verdict",
     "reference.claim_status"),
]

# provisional facts: stated but not yet independently verified
PROVISIONAL = [
    ("D_Cortex", "multi_hardware_reproduction", "pending", "struct_verdict",
     "second environment is CPU vs CUDA on the same machine; distinct hardware not yet run"),
    ("D_Cortex", "independent_replication", "pending", "struct_verdict",
     "PROVEN requires independent replication; not yet performed"),
]

# disputed facts: conflicting candidate values across sources, kept as uncertain
DISPUTED = [
    {"entity": "D_Cortex", "attribute": "parameter_count",
     "candidates": [
         {"value": "344.70M", "source": "big_config", "ref": "computed from BIG config"},
         {"value": "345M", "source": "project_concept", "ref": "rounded figure in notes"}],
     "note": "exact vs rounded parameter count differ across sources; report as uncertain"},
]

# forbidden claim patterns (mechanically blocked); tie to anti-confabulation rules
FORBIDDEN = [
    {"pattern": "is proven", "reason": "role-binding is MEASURED not PROVEN; asserting proof is forbidden",
     "severity": "high"},
    {"pattern": "fully proven", "reason": "role-binding is MEASURED not PROVEN", "severity": "high"},
    {"pattern": "100% generalization", "reason": "the 100% scale figure was a caught test artifact",
     "severity": "high"},
    {"pattern": "trades profitably", "reason": "no verified trading fills; profitability claim forbidden",
     "severity": "high"},
    {"pattern": "production ready", "reason": "system is a research substrate; not production-certified",
     "severity": "medium"},
    {"pattern": "patent granted", "reason": "application number on file; grant status not in pack",
     "severity": "high"},
]

SCHEMAS = {
    "domain": "patent_analysis",
    "entities": ["D_Cortex", "DCortexConfig", "ContentAddressedRoleBinder"],
    "attributes": {
        "patent_number": "string", "owner": "string", "architecture": "string",
        "substrate": "string", "hidden_dim": "integer", "decoder_layers": "integer",
        "encoder_layers": "integer", "held_out_structural_exact_median": "percent",
        "no_memory_control_exact": "percent", "claim_status": "string",
        "parameter_count": "string", "multi_hardware_reproduction": "status",
        "independent_replication": "status",
    },
    "in_domain_keywords": ["patent", "d_cortex", "dcortex", "binder", "config", "architecture",
                           "substrate", "owner", "layers", "hidden", "claim", "parameter",
                           "role-binding", "memory", "transformer"],
}

ABSTAIN_RULES = {
    "rules": [
        {"id": "R1_unknown_fact", "when": "entity in schema but (entity,attribute) not committed",
         "action": "abstain", "message": "Not grounded in D_Cortex_PatentAnalyst committed memory."},
        {"id": "R2_out_of_domain", "when": "no in-domain keyword and entity not in schema",
         "action": "out_of_domain", "message": "Query is outside the patent-analysis domain."},
        {"id": "R3_disputed", "when": "(entity,attribute) in disputed",
         "action": "uncertain", "message": "Conflicting sources; reported as uncertain."},
        {"id": "R4_forbidden", "when": "answer matches a forbidden pattern",
         "action": "block", "message": "Claim is forbidden in this pack."},
        {"id": "R5_provisional", "when": "(entity,attribute) in provisional",
         "action": "provisional", "message": "Provisional; not independently verified."},
    ],
    "default_action": "abstain",
}


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> str:
    body = "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n"
    encoded = body.encode("utf-8")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, obj: Any) -> str:
    body = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2)
    encoded = body.encode("utf-8")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the D_Cortex_PatentAnalyst pack")
    ap.add_argument("--out", default=str(PACK_DIR))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(SEP, flush=True)
    print("[INFO] Building D_Cortex_PatentAnalyst professional pack", flush=True)

    # pin every source file by SHA
    sources = {sid: {**meta, "sha256": sha256_file(REPO_ROOT / meta["ref"])}
               for sid, meta in SOURCES.items()}
    for sid, meta in sources.items():
        if meta["sha256"] == "MISSING":
            print(f"[WARN] source {sid} file missing: {meta['ref']}", flush=True)

    committed_rows = [{"entity": e, "attribute": a, "value": v,
                       "provenance": provenance(sid, ref)} for e, a, v, sid, ref in COMMITTED]
    provisional_rows = [{"entity": e, "attribute": a, "value": v, "status": "provisional",
                         "provenance": provenance(sid, ref)} for e, a, v, sid, ref in PROVISIONAL]

    shas = {
        "committed.jsonl": write_jsonl(out / "committed.jsonl", committed_rows),
        "provisional.jsonl": write_jsonl(out / "provisional.jsonl", provisional_rows),
        "disputed.jsonl": write_jsonl(out / "disputed.jsonl", DISPUTED),
        "forbidden.jsonl": write_jsonl(out / "forbidden.jsonl", FORBIDDEN),
        "sources.json": write_json(out / "sources.json", sources),
        "abstain_rules.json": write_json(out / "abstain_rules.json", ABSTAIN_RULES),
        "schemas.json": write_json(out / "schemas.json", SCHEMAS),
    }
    manifest = {"pack": "D_Cortex_PatentAnalyst", "files": shas,
                "counts": {"committed": len(committed_rows), "provisional": len(provisional_rows),
                           "disputed": len(DISPUTED), "forbidden": len(FORBIDDEN)}}
    write_json(out / "pack_manifest.json", manifest)

    for name, sha in shas.items():
        print(f"  ✓ {name:22s} sha {sha[:16]}", flush=True)
    print(f"[INFO] committed={len(committed_rows)} provisional={len(provisional_rows)} "
          f"disputed={len(DISPUTED)} forbidden={len(FORBIDDEN)}", flush=True)
    print(SEP, flush=True)
    print("PACK_BUILT D_Cortex_PatentAnalyst", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
