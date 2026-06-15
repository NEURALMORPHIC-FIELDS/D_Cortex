# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b conservative semantic producer probe.

import argparse
import ast
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

import torch
import torch.nn as nn
import tiktoken

from dcortex.semantic_adapter import (
    ConservativeSemanticAdapter,
    DecisionStatus,
    HypothesisMode,
    RequestedDestination,
)
from dcortex.semantic_producer import (
    ConservativePrototypeProducer,
    DCortexTokenEmbeddingBackend,
)

SEP = "=" * 70
ENC = tiktoken.get_encoding("gpt2")
SIMILARITY_THRESHOLD = 0.55
MARGIN_THRESHOLD = 0.025

ENTITY_PROTOTYPES = {
    "dragon": ("dragon",),
    "mermaid": ("mermaid",),
    "phoenix": ("phoenix",),
    "robot": ("robot",),
}
ATTRIBUTE_PROTOTYPES = {
    "color": ("What color is the object?",),
    "location": ("Where is the object located?",),
    "size": ("What size is the object?",),
    "state": ("What state is the object in?",),
}

NOVEL_QUERIES: Tuple[Tuple[str, str, str], ...] = (
    ("Which hue belongs to the dragon?", "dragon", "color"),
    ("Describe the phoenix's coloration.", "phoenix", "color"),
    ("What shade characterizes the mermaid?", "mermaid", "color"),
    ("Identify the robot's chromatic appearance.", "robot", "color"),
    ("How would you describe the dragon's hue?", "dragon", "color"),
    ("How large is the dragon?", "dragon", "size"),
    ("What are the phoenix's dimensions?", "phoenix", "size"),
    ("Describe the mermaid's magnitude.", "mermaid", "size"),
    ("Assess the robot's physical scale.", "robot", "size"),
    ("How big is the phoenix?", "phoenix", "size"),
    ("Where can the dragon be found?", "dragon", "location"),
    ("What place contains the phoenix?", "phoenix", "location"),
    ("Identify the mermaid's whereabouts.", "mermaid", "location"),
    ("In which place is the robot situated?", "robot", "location"),
    ("Where does the dragon reside?", "dragon", "location"),
    ("What condition is the dragon in?", "dragon", "state"),
    ("Describe the phoenix's current status.", "phoenix", "state"),
    ("What is the mermaid's present condition?", "mermaid", "state"),
    ("Identify the robot's current status.", "robot", "state"),
    ("How is the phoenix doing right now?", "phoenix", "state"),
)

AMBIGUOUS_QUERIES: Tuple[str, ...] = (
    "Tell me about the dragon.",
    "What do we know about the phoenix?",
    "Describe the mermaid.",
    "Give information about the robot.",
    "What is notable about the dragon?",
    "Report something concerning the phoenix.",
    "What can be said regarding the mermaid?",
    "Summarize the robot.",
    "Provide details about the dragon.",
    "What is known about the phoenix?",
)


class TokenEmbeddingModel(nn.Module):
    """Minimal wrapper exposing a checkpoint's shared token embeddings."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.shared_token_emb = nn.Embedding.from_pretrained(weight, freeze=True)


def load_embedding_model(checkpoint: Path, device: torch.device) -> TokenEmbeddingModel:
    """Load only the shared-token embedding table from a D_Cortex checkpoint."""
    if not checkpoint.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint}")
    payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    state = payload.get("model", payload)
    key = "shared_token_emb.weight"
    if key not in state:
        raise RuntimeError(f"Checkpoint lacks {key}: {checkpoint}")
    weight = state[key].detach().float().to(device)
    return TokenEmbeddingModel(weight).to(device).eval()


def gate(
    criterion_id: str, passed: bool, evidence: str, distribution: Dict[str, Any]
) -> Dict[str, Any]:
    """Build one producer gate."""
    return {
        "criterion_id": criterion_id,
        "passed": bool(passed),
        "evidence": evidence,
        "distribution": distribution,
    }


def build_producer(
    backend: DCortexTokenEmbeddingBackend,
    similarity: float = SIMILARITY_THRESHOLD,
    margin: float = MARGIN_THRESHOLD,
) -> ConservativePrototypeProducer:
    """Build a producer with a fresh mandatory adapter."""
    return ConservativePrototypeProducer(
        backend,
        ConservativeSemanticAdapter(),
        similarity_threshold=similarity,
        margin_threshold=margin,
    )


def run(args: argparse.Namespace) -> int:
    """Run the frozen real-checkpoint semantic producer probe."""
    checkpoint = Path(args.checkpoint)
    results_dir = Path(args.run_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_embedding_model(checkpoint, device)
    backend = DCortexTokenEmbeddingBackend(
        model=model,
        tokenizer=ENC.encode_ordinary,
        max_seq_len=64,
        backend_version=checkpoint.name,
    )
    gates: List[Dict[str, Any]] = []

    producer = build_producer(backend)
    sample = producer.produce(
        "p0",
        1,
        HypothesisMode.QUERY,
        NOVEL_QUERIES[0][0],
        ENTITY_PROTOTYPES,
        ATTRIBUTE_PROTOTYPES,
        provenance=("probe:p0",),
    )
    p0 = (not sample.emitted) or (
        sample.adapter_decision is not None
        and sample.adapter_decision.status == DecisionStatus.ACCEPT_QUERY
    )
    gates.append(
        gate(
            "P0_ADAPTER_REQUIRED",
            p0,
            "Any emitted sample has an adapter ACCEPT_QUERY decision.",
            {"emitted": sample.emitted, "decision": None if sample.adapter_decision is None
             else sample.adapter_decision.status.value},
        )
    )

    emitted_dest = None if sample.hypothesis is None else sample.hypothesis.requested_destination
    p1 = emitted_dest in (None, RequestedDestination.QUERY_ONLY)
    gates.append(
        gate(
            "P1_NO_DIRECT_COMMIT",
            p1,
            f"Producer destination={None if emitted_dest is None else emitted_dest.value}.",
            {"destination": None if emitted_dest is None else emitted_dest.value},
        )
    )

    strict = build_producer(backend, similarity=1.0, margin=0.0)
    strict_result = strict.produce(
        "strict",
        2,
        HypothesisMode.QUERY,
        NOVEL_QUERIES[0][0],
        ENTITY_PROTOTYPES,
        ATTRIBUTE_PROTOTYPES,
        provenance=("probe:strict",),
    )
    gates.append(
        gate(
            "P2_THRESHOLD_ABSTENTION",
            not strict_result.emitted,
            f"At similarity threshold 1.0, emitted={strict_result.emitted}.",
            {"emitted": strict_result.emitted, "reasons": list(strict_result.reason_codes)},
        )
    )

    tied_entities = {"entity_a": ("dragon",), "entity_b": ("dragon",)}
    tied = build_producer(backend, similarity=0.0, margin=0.001)
    tied_result = tied.produce(
        "tied",
        3,
        HypothesisMode.QUERY,
        "Tell me about the dragon's color.",
        tied_entities,
        ATTRIBUTE_PROTOTYPES,
        provenance=("probe:tied",),
    )
    gates.append(
        gate(
            "P3_MARGIN_ABSTENTION",
            not tied_result.emitted,
            f"Equal entity prototypes produced emitted={tied_result.emitted}.",
            {"emitted": tied_result.emitted, "reasons": list(tied_result.reason_codes)},
        )
    )

    def deterministic() -> Dict[str, Any]:
        return build_producer(backend).produce(
            "det",
            4,
            HypothesisMode.QUERY,
            NOVEL_QUERIES[1][0],
            ENTITY_PROTOTYPES,
            ATTRIBUTE_PROTOTYPES,
            provenance=("probe:det",),
        ).to_dict()

    det_a, det_b = deterministic(), deterministic()
    gates.append(
        gate(
            "P4_DETERMINISTIC",
            det_a == det_b,
            f"Repeated result exact={det_a == det_b}.",
            {"exact": det_a == det_b},
        )
    )

    producer_source = (REPO_ROOT / "dcortex" / "semantic_producer.py").read_text(
        encoding="utf-8"
    )
    ast.parse(producer_source)
    forbidden = ["ALIAS_MAP", "SYNONYM_MAP", "HARDCODED_SYNONYM"]
    present = [item for item in forbidden if item in producer_source]
    gates.append(
        gate(
            "P5_NO_SYNONYM_MAP",
            not present,
            f"Forbidden lexical-map identifiers present={present}.",
            {"present": present},
        )
    )

    query_records: List[Dict[str, Any]] = []
    correct = 0
    emitted = 0
    adapter_accepted = 0
    for index, (text, expected_entity, expected_attr) in enumerate(NOVEL_QUERIES):
        result = build_producer(backend).produce(
            f"novel-{index}",
            100 + index,
            HypothesisMode.QUERY,
            text,
            ENTITY_PROTOTYPES,
            ATTRIBUTE_PROTOTYPES,
            provenance=(f"probe:novel:{index}",),
        )
        emitted += int(result.emitted)
        adapter_accepted += int(
            result.adapter_decision is not None
            and result.adapter_decision.status == DecisionStatus.ACCEPT_QUERY
        )
        prediction = None
        if result.hypothesis is not None:
            prediction = {
                "entity": result.hypothesis.entity_id,
                "attribute": result.hypothesis.attr_type,
            }
            correct += int(
                result.hypothesis.entity_id == expected_entity
                and result.hypothesis.attr_type == expected_attr
            )
        query_records.append(
            {
                "text": text,
                "expected_entity": expected_entity,
                "expected_attribute": expected_attr,
                "prediction": prediction,
                "emitted": result.emitted,
                "reasons": list(result.reason_codes),
                "matches": [match.to_dict() for match in result.matches],
            }
        )
    accuracy = correct / len(NOVEL_QUERIES)
    gates.append(
        gate(
            "P6_QUERY_NOVEL_FORM",
            accuracy >= 0.60,
            f"Novel-query entity+attribute accuracy={accuracy:.1%} "
            f"({correct}/{len(NOVEL_QUERIES)}), emitted={emitted}.",
            {
                "correct": correct,
                "total": len(NOVEL_QUERIES),
                "accuracy": accuracy,
                "emitted": emitted,
                "adapter_accepted": adapter_accepted,
            },
        )
    )

    ambiguous_records: List[Dict[str, Any]] = []
    abstained = 0
    for index, text in enumerate(AMBIGUOUS_QUERIES):
        result = build_producer(backend).produce(
            f"ambiguous-{index}",
            200 + index,
            HypothesisMode.QUERY,
            text,
            ENTITY_PROTOTYPES,
            ATTRIBUTE_PROTOTYPES,
            provenance=(f"probe:ambiguous:{index}",),
        )
        abstained += int(not result.emitted)
        ambiguous_records.append(
            {
                "text": text,
                "emitted": result.emitted,
                "reasons": list(result.reason_codes),
                "matches": [match.to_dict() for match in result.matches],
            }
        )
    abstain_rate = abstained / len(AMBIGUOUS_QUERIES)
    gates.append(
        gate(
            "P7_AMBIGUOUS_HONESTY",
            abstain_rate >= 0.80,
            f"Ambiguous-query abstention={abstain_rate:.1%} "
            f"({abstained}/{len(AMBIGUOUS_QUERIES)}).",
            {
                "abstained": abstained,
                "total": len(AMBIGUOUS_QUERIES),
                "abstain_rate": abstain_rate,
            },
        )
    )

    all_pass = all(item["passed"] for item in gates)
    verdict = {
        "verdict": gates,
        "reference": {
            "checkpoint": str(checkpoint),
            "device": str(device),
            "backend": backend.backend_id,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "margin_threshold": MARGIN_THRESHOLD,
            "scope": "Frozen F5-like producer probe; not official F1/F3/F5 evaluation.",
            "claim_status": "MEASURED in one local environment; not PROVEN.",
            "all_pass": all_pass,
            "novel_query_records": query_records,
            "ambiguous_query_records": ambiguous_records,
        },
    }
    verdict_path = results_dir / "verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(SEP, flush=True)
    print("[INFO] D_Cortex v15.7b conservative semantic producer probe", flush=True)
    print(f"[INFO] Checkpoint: {checkpoint}", flush=True)
    print(f"[INFO] Device: {device}", flush=True)
    print(SEP, flush=True)
    for item in gates:
        label = "✓ PASS" if item["passed"] else "✗ FAIL"
        print(f"{label} [{item['criterion_id']}] {item['evidence']}", flush=True)
    print(SEP, flush=True)
    print(f"[INFO] Overall: {'ALL GATES PASS' if all_pass else 'GATE FAILURE'}", flush=True)
    print(f"[INFO] Verdict: {verdict_path}", flush=True)
    return 0 if all_pass else 1


def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    default_checkpoint = (
        REPO_ROOT.parent / "D_Cortex-main" / "runs" / "warmstart" / "warmstarted_init.pt"
    )
    parser = argparse.ArgumentParser(description="D_Cortex semantic producer probe")
    parser.add_argument("--checkpoint", type=str, default=str(default_checkpoint))
    parser.add_argument(
        "--run-dir", type=str, default=str(REPO_ROOT / "runs" / "semantic_producer")
    )
    return parser


def main() -> int:
    """CLI entry point."""
    return run(build_argparser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
