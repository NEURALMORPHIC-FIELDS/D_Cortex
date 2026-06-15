# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b semantic adapter acceptance verifier.

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dcortex.semantic_adapter import (
    ConservativeSemanticAdapter,
    DecisionStatus,
    HypothesisMode,
    RequestedDestination,
    SemanticHypothesis,
)

SEP = "=" * 70
SEALED_SOURCE = REPO_ROOT / "steps" / "13_v15_7a_consolidation" / "code.py"
SEALED_SHA256 = "25b4906ecc05a6b51b10902e54332a0ec2b26c4c622aa4e6ee74bd4961369aa3"


def make_fact(
    hypothesis_id: str,
    episode_id: int,
    value_id: str = "blue",
    destination: RequestedDestination = RequestedDestination.PROVISIONAL_ONLY,
    provenance: Tuple[str, ...] = ("evidence:source-1",),
    confidence: float = 0.8,
    uncertainty: float = 0.2,
) -> SemanticHypothesis:
    """Build one verifier fact hypothesis."""
    return SemanticHypothesis(
        hypothesis_id=hypothesis_id,
        episode_id=episode_id,
        mode=HypothesisMode.FACT,
        source_text="The dragon appears blue.",
        producer="semantic-adapter-verifier",
        producer_version="1.0",
        provenance=provenance,
        confidence=confidence,
        uncertainty=uncertainty,
        requested_destination=destination,
        entity_id="dragon",
        attr_type="color",
        value_id=value_id,
    )


def make_query(
    hypothesis_id: str,
    value_id: str | None = None,
    destination: RequestedDestination = RequestedDestination.QUERY_ONLY,
) -> SemanticHypothesis:
    """Build one verifier query hypothesis."""
    return SemanticHypothesis(
        hypothesis_id=hypothesis_id,
        episode_id=1,
        mode=HypothesisMode.QUERY,
        source_text="What hue does the dragon have?",
        producer="semantic-adapter-verifier",
        producer_version="1.0",
        provenance=("evidence:query-1",),
        confidence=0.75,
        uncertainty=0.25,
        requested_destination=destination,
        entity_id="dragon",
        attr_type="color",
        value_id=value_id,
    )


def gate(
    criterion_id: str, passed: bool, evidence: str, distribution: Dict[str, Any]
) -> Dict[str, Any]:
    """Build one verdict gate."""
    return {
        "criterion_id": criterion_id,
        "passed": bool(passed),
        "evidence": evidence,
        "distribution": distribution,
    }


def run() -> int:
    """Run all frozen semantic-adapter gates and write a verdict."""
    results_dir = REPO_ROOT / "runs" / "semantic_adapter" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    gates: List[Dict[str, Any]] = []

    actual_hash = hashlib.sha256(SEALED_SOURCE.read_bytes()).hexdigest()
    gates.append(
        gate(
            "G0_SEALED_UNTOUCHED",
            actual_hash == SEALED_SHA256,
            f"Pas 7a SHA-256 actual={actual_hash}, frozen={SEALED_SHA256}.",
            {"actual": actual_hash, "frozen": SEALED_SHA256},
        )
    )

    direct_adapter = ConservativeSemanticAdapter()
    direct = [
        direct_adapter.submit(
            make_fact(
                f"direct-{index}",
                index,
                destination=RequestedDestination.COMMITTED_DIRECT,
            )
        )
        for index in range(1, 21)
    ]
    direct_rejected = sum(item.status == DecisionStatus.REJECT for item in direct)
    gates.append(
        gate(
            "G1_NO_DIRECT_COMMIT",
            direct_rejected == len(direct),
            f"Rejected {direct_rejected}/{len(direct)} direct-commit requests.",
            {"rejected": direct_rejected, "total": len(direct)},
        )
    )

    provenance_adapter = ConservativeSemanticAdapter()
    missing = [
        provenance_adapter.submit(make_fact(f"missing-{index}", index, provenance=()))
        for index in range(1, 21)
    ]
    missing_rejected = sum(item.status == DecisionStatus.REJECT for item in missing)
    gates.append(
        gate(
            "G2_PROVENANCE_REQUIRED",
            missing_rejected == len(missing),
            f"Rejected {missing_rejected}/{len(missing)} missing-provenance hypotheses.",
            {"rejected": missing_rejected, "total": len(missing)},
        )
    )

    query_adapter = ConservativeSemanticAdapter()
    query_writes = [
        query_adapter.submit(make_query(f"query-write-{index}", value_id="blue"))
        for index in range(1, 21)
    ]
    query_rejected = sum(item.status == DecisionStatus.REJECT for item in query_writes)
    gates.append(
        gate(
            "G3_QUERY_READ_ONLY",
            query_rejected == len(query_writes),
            f"Rejected {query_rejected}/{len(query_writes)} query-write attempts.",
            {"rejected": query_rejected, "total": len(query_writes)},
        )
    )

    anti = ConservativeSemanticAdapter()
    for index in range(20):
        anti.submit(make_fact(f"same-episode-{index}", 7))
    anti_count = anti.confirmation_count("dragon", "color", "blue")
    gates.append(
        gate(
            "G4_ANTI_INFLATION",
            anti_count == 1,
            f"Twenty same-episode submissions produced {anti_count} distinct confirmation.",
            {"submissions": 20, "confirmation_count": anti_count},
        )
    )

    longitudinal = ConservativeSemanticAdapter()
    longitudinal.submit(make_fact("long-1", 11))
    longitudinal.submit(make_fact("long-2", 12))
    longitudinal_count = longitudinal.confirmation_count("dragon", "color", "blue")
    gates.append(
        gate(
            "G5_LONGITUDINAL_CONFIRMATION",
            longitudinal_count == 2,
            f"Two distinct episodes produced {longitudinal_count} confirmations.",
            {"episodes": [11, 12], "confirmation_count": longitudinal_count},
        )
    )

    conflict = ConservativeSemanticAdapter()
    conflict.submit(make_fact("conflict-blue", 21, "blue"))
    conflict.submit(make_fact("conflict-red", 22, "red"))
    values = sorted({item.value_id for item in conflict.candidates_for_slot("dragon", "color")})
    gates.append(
        gate(
            "G6_CONFLICT_PRESERVATION",
            values == ["blue", "red"],
            f"Conflicting values preserved separately: {values}.",
            {"values": values},
        )
    )

    def deterministic_run() -> str:
        adapter = ConservativeSemanticAdapter()
        adapter.submit(make_fact("det-fact", 31))
        adapter.submit(make_query("det-query"))
        adapter.submit(
            make_fact(
                "det-direct",
                32,
                destination=RequestedDestination.COMMITTED_DIRECT,
            )
        )
        return adapter.audit_json()

    audit_a = deterministic_run()
    audit_b = deterministic_run()
    gates.append(
        gate(
            "G7_DETERMINISTIC_AUDIT",
            audit_a == audit_b,
            f"Repeated audit JSON byte-identical; bytes={len(audit_a.encode('utf-8'))}.",
            {"byte_identical": audit_a == audit_b, "bytes": len(audit_a.encode("utf-8"))},
        )
    )

    original = make_fact("roundtrip", 41)
    rebuilt = SemanticHypothesis.from_dict(
        json.loads(json.dumps(original.to_dict(), ensure_ascii=False))
    )
    gates.append(
        gate(
            "G8_ROUNDTRIP",
            rebuilt == original,
            "SemanticHypothesis JSON roundtrip reconstructed exactly.",
            {"exact": rebuilt == original},
        )
    )

    ranges = ConservativeSemanticAdapter()
    invalid = [
        ranges.submit(make_fact("range-high", 51, confidence=1.01)),
        ranges.submit(make_fact("range-low", 52, confidence=-0.01)),
        ranges.submit(make_fact("unc-high", 53, uncertainty=1.01)),
        ranges.submit(make_fact("unc-low", 54, uncertainty=-0.01)),
    ]
    invalid_rejected = sum(item.status == DecisionStatus.REJECT for item in invalid)
    gates.append(
        gate(
            "G9_INVALID_RANGE_REJECTED",
            invalid_rejected == len(invalid),
            f"Rejected {invalid_rejected}/{len(invalid)} invalid-range hypotheses.",
            {"rejected": invalid_rejected, "total": len(invalid)},
        )
    )

    all_pass = all(item["passed"] for item in gates)
    verdict = {
        "verdict": gates,
        "reference": {
            "scope": "Pas 7b explicit adapter contract only; no semantic producer and no Pas 7a modification.",
            "claim_status": "MEASURED in current local session. Contract validation, not semantic-quality proof.",
            "all_pass": all_pass,
        },
    }
    verdict_path = results_dir / "verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(SEP, flush=True)
    print("[INFO] D_Cortex v15.7b semantic adapter verdict", flush=True)
    print(SEP, flush=True)
    for item in gates:
        label = "✓ PASS" if item["passed"] else "✗ FAIL"
        print(f"{label} [{item['criterion_id']}] {item['evidence']}", flush=True)
    print(SEP, flush=True)
    print(f"[INFO] Overall: {'ALL GATES PASS' if all_pass else 'GATE FAILURE'}", flush=True)
    print(f"[INFO] Verdict: {verdict_path}", flush=True)
    return 0 if all_pass else 1


def main() -> int:
    """CLI entry point."""
    return run()


if __name__ == "__main__":
    sys.exit(main())
