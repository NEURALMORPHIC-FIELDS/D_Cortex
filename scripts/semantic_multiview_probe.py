# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b multi-view likelihood evidence-fusion probe.

import argparse
import ast
import hashlib
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from dcortex.semantic_adapter import (
    ConservativeSemanticAdapter,
    DecisionStatus,
    HypothesisMode,
    RequestedDestination,
)
from dcortex.semantic_producer import (
    ConservativeMultiViewLikelihoodQueryProducer,
    ConservativePrototypeProducer,
    DCortexCausalLikelihoodBackend,
    DCortexTokenEmbeddingBackend,
    MultiViewProducerResult,
)
from scripts.semantic_likelihood_probe import (
    ATTRIBUTE_CANDIDATES,
    CALIBRATION_AMBIGUOUS,
    CALIBRATION_QUERIES,
    ENC,
    EOT,
    EVALUATION_AMBIGUOUS,
    EVALUATION_QUERIES,
    SEALED_SHA256,
    SEALED_SOURCE,
    TOKEN_MEAN_ATTRIBUTE_PROTOTYPES,
    gate,
    load_model,
    model_buffer_hash,
    text_set_hash,
)

SEP = "=" * 70
F5_SEED = 20261215
F5_TRIALS = 500
AMBIGUOUS_SEED = 20261216
AMBIGUOUS_TRIALS = 100
ENTITY_PROMPTS = (
    "Question: {source_text}\nAnswer entity:",
    "Question: {source_text}\nThis question is about",
)
ATTRIBUTE_PROMPTS = (
    "Question: {source_text}\nAnswer type:",
    "Question: {source_text}\nThis question asks about the object's",
    "Question: {source_text}\nThe requested attribute is",
    "Question: {source_text}\nRequested property:",
)
AMBIGUOUS_TEMPLATES = (
    "Give me a general overview of the {entity}.",
    "What should be remembered about the {entity}?",
    "Describe the {entity} without focusing on one property.",
    "Tell me any relevant fact about the {entity}.",
    "Summarize available information concerning the {entity}.",
    "What is noteworthy about the {entity}?",
    "Discuss the {entity} in broad terms.",
    "Provide an unrestricted description of the {entity}.",
    "What can be recalled regarding the {entity}?",
    "Share something about the {entity}.",
)


def extract_sealed_f5_definitions() -> Dict[str, Any]:
    """Extract only the frozen F5 generator definitions from sealed Pas 7a."""
    source = SEALED_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    required = {
        "HOLDOUT_ENTITIES_SINGLE",
        "HOLDOUT_COLORS",
        "HOLDOUT_SIZES",
        "HOLDOUT_LOCATIONS",
        "HOLDOUT_STATES",
        "HOLDOUT_ATTR_VALUES",
        "HOLDOUT_ATTR_TYPES",
        "F5_QUERY_FORMS",
    }
    selected: List[ast.stmt] = []
    found = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = {
            target.id for target in node.targets if isinstance(target, ast.Name)
        }
        if names.intersection(required):
            selected.append(node)
            found.update(names.intersection(required))
    missing = sorted(required.difference(found))
    if missing:
        raise RuntimeError(f"Sealed F5 definitions missing: {missing}")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: Dict[str, Any] = {}
    exec(compile(module, str(SEALED_SOURCE), "exec"), namespace)  # noqa: S102
    return {name: namespace[name] for name in required}


def generate_f5_queries(
    definitions: Mapping[str, Any],
) -> Tuple[Tuple[str, str, str, str], ...]:
    """Generate the frozen F5 query-intent verdict set."""
    rng = random.Random(F5_SEED)
    entities = definitions["HOLDOUT_ENTITIES_SINGLE"]
    attributes = definitions["HOLDOUT_ATTR_TYPES"]
    values = definitions["HOLDOUT_ATTR_VALUES"]
    forms = definitions["F5_QUERY_FORMS"]
    records: List[Tuple[str, str, str, str]] = []
    for _ in range(F5_TRIALS):
        entity = rng.choice(entities)
        attribute = rng.choice(attributes)
        value = rng.choice(values[attribute])
        query = rng.choice(forms[attribute])(entity, value)
        records.append((query, entity, attribute, value))
    return tuple(records)


def generate_ambiguous_queries(entities: Sequence[str]) -> Tuple[str, ...]:
    """Generate the frozen attribute-unspecified ambiguity verdict set."""
    rng = random.Random(AMBIGUOUS_SEED)
    return tuple(
        rng.choice(AMBIGUOUS_TEMPLATES).format(entity=rng.choice(entities))
        for _ in range(AMBIGUOUS_TRIALS)
    )


def build_multiview_producer(
    backend: DCortexCausalLikelihoodBackend,
    entity_candidates: Mapping[str, str],
    attribute_margin_threshold: float,
) -> ConservativeMultiViewLikelihoodQueryProducer:
    """Build a fresh producer with the frozen multi-view policy."""
    del entity_candidates
    return ConservativeMultiViewLikelihoodQueryProducer(
        backend=backend,
        adapter=ConservativeSemanticAdapter(),
        entity_prompts=ENTITY_PROMPTS,
        attribute_prompts=ATTRIBUTE_PROMPTS,
        entity_margin_threshold=0.0,
        attribute_margin_threshold=attribute_margin_threshold,
        entity_minimum_consensus=2,
        attribute_minimum_consensus=2,
    )


def attribute_margin(result: MultiViewProducerResult) -> float:
    """Extract the aggregated attribute probability margin."""
    return next(score.margin for score in result.scores if score.axis == "attribute")


def calibrate_margin(
    backend: DCortexCausalLikelihoodBackend,
    entity_candidates: Mapping[str, str],
) -> Dict[str, Any]:
    """Choose the smallest aggregate margin yielding 80% development abstention."""
    development_ambiguous = tuple(CALIBRATION_AMBIGUOUS) + tuple(EVALUATION_AMBIGUOUS)
    producer = build_multiview_producer(backend, entity_candidates, 0.0)
    margins: List[float] = []
    for index, text in enumerate(development_ambiguous):
        result = producer.produce(
            f"calibration-{index}",
            index,
            text,
            entity_candidates,
            ATTRIBUTE_CANDIDATES,
            provenance=(f"calibration:multiview:{index}",),
        )
        margins.append(attribute_margin(result))
    ordered = sorted(margins)
    target_abstentions = math.ceil(0.80 * len(ordered))
    threshold = min(1.0, ordered[target_abstentions - 1] + 0.000001)
    achieved = sum(margin < threshold for margin in margins) / len(margins)
    return {
        "threshold": threshold,
        "target_abstention": 0.80,
        "achieved_margin_only_abstention": achieved,
        "development_ambiguous_count": len(development_ambiguous),
        "ambiguous_margins": margins,
    }


def run(args: argparse.Namespace) -> int:
    """Run the first frozen sealed-F5 multi-view producer verdict."""
    checkpoint = Path(args.checkpoint)
    results_dir = Path(args.run_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    actual_sealed_hash = hashlib.sha256(SEALED_SOURCE.read_bytes()).hexdigest()
    definitions = extract_sealed_f5_definitions()
    f5_records = generate_f5_queries(definitions)
    entities = tuple(definitions["HOLDOUT_ENTITIES_SINGLE"])
    entity_candidates = {entity: entity for entity in entities}
    token_entity_prototypes = {entity: (entity,) for entity in entities}
    ambiguous_queries = generate_ambiguous_queries(entities)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, device)
    backend = DCortexCausalLikelihoodBackend(
        model=model,
        tokenizer=ENC.encode_ordinary,
        pad_token_id=EOT,
        max_seq_len=128,
        backend_version=checkpoint.name,
    )
    buffers_before = model_buffer_hash(model)
    calibration = calibrate_margin(backend, entity_candidates)
    threshold = float(calibration["threshold"])

    records: List[Dict[str, Any]] = []
    emitted = 0
    accepted = 0
    query_only = 0
    correct = 0
    wrong = 0
    consensus_ok = 0
    for index, (text, expected_entity, expected_attribute, value) in enumerate(
        f5_records
    ):
        result = build_multiview_producer(
            backend, entity_candidates, threshold
        ).produce(
            f"f5-{index}",
            1000 + index,
            text,
            entity_candidates,
            ATTRIBUTE_CANDIDATES,
            provenance=(
                f"sealed_f5_sha256:{actual_sealed_hash}",
                f"seed:{F5_SEED}",
                f"trial:{index}",
            ),
        )
        emitted += int(result.emitted)
        accepted += int(
            result.adapter_decision is not None
            and result.adapter_decision.status == DecisionStatus.ACCEPT_QUERY
        )
        query_only += int(
            result.hypothesis is not None
            and result.hypothesis.requested_destination == RequestedDestination.QUERY_ONLY
        )
        prediction = None
        is_correct = False
        if result.hypothesis is not None:
            prediction = {
                "entity": result.hypothesis.entity_id,
                "attribute": result.hypothesis.attr_type,
            }
            is_correct = (
                result.hypothesis.entity_id == expected_entity
                and result.hypothesis.attr_type == expected_attribute
            )
            correct += int(is_correct)
            wrong += int(not is_correct)
            consensus_ok += int(
                all(
                    score.consensus_count >= score.minimum_consensus
                    for score in result.scores
                )
            )
        records.append(
            {
                "text": text,
                "value": value,
                "expected_entity": expected_entity,
                "expected_attribute": expected_attribute,
                "prediction": prediction,
                "correct": is_correct,
                "emitted": result.emitted,
                "reasons": list(result.reason_codes),
                "scores": [score.to_dict() for score in result.scores],
            }
        )
    accuracy = correct / len(f5_records)
    wrong_rate = wrong / len(f5_records)

    ambiguous_records: List[Dict[str, Any]] = []
    abstained = 0
    for index, text in enumerate(ambiguous_queries):
        result = build_multiview_producer(
            backend, entity_candidates, threshold
        ).produce(
            f"ambiguous-{index}",
            2000 + index,
            text,
            entity_candidates,
            ATTRIBUTE_CANDIDATES,
            provenance=(f"ambiguous_seed:{AMBIGUOUS_SEED}", f"trial:{index}"),
        )
        abstained += int(not result.emitted)
        ambiguous_records.append(
            {
                "text": text,
                "emitted": result.emitted,
                "reasons": list(result.reason_codes),
                "scores": [score.to_dict() for score in result.scores],
            }
        )
    abstain_rate = abstained / len(ambiguous_queries)

    deterministic_a = build_multiview_producer(
        backend, entity_candidates, threshold
    ).produce(
        "deterministic",
        3000,
        f5_records[0][0],
        entity_candidates,
        ATTRIBUTE_CANDIDATES,
        provenance=("deterministic",),
    ).to_dict()
    deterministic_b = build_multiview_producer(
        backend, entity_candidates, threshold
    ).produce(
        "deterministic",
        3000,
        f5_records[0][0],
        entity_candidates,
        ATTRIBUTE_CANDIDATES,
        provenance=("deterministic",),
    ).to_dict()
    deterministic = deterministic_a == deterministic_b

    token_backend = DCortexTokenEmbeddingBackend(
        model=model,
        tokenizer=ENC.encode_ordinary,
        max_seq_len=128,
        backend_version=checkpoint.name,
    )
    token_correct = 0
    token_emitted = 0
    for index, (text, expected_entity, expected_attribute, _) in enumerate(f5_records):
        baseline = ConservativePrototypeProducer(
            token_backend,
            ConservativeSemanticAdapter(),
            similarity_threshold=0.55,
            margin_threshold=0.025,
        ).produce(
            f"baseline-{index}",
            4000 + index,
            HypothesisMode.QUERY,
            text,
            token_entity_prototypes,
            TOKEN_MEAN_ATTRIBUTE_PROTOTYPES,
            provenance=(f"baseline:f5:{index}",),
        )
        token_emitted += int(baseline.emitted)
        token_correct += int(
            baseline.hypothesis is not None
            and baseline.hypothesis.entity_id == expected_entity
            and baseline.hypothesis.attr_type == expected_attribute
        )
    token_accuracy = token_correct / len(f5_records)
    uplift = accuracy - token_accuracy

    development_texts = (
        [text for text, _, _ in CALIBRATION_QUERIES]
        + list(CALIBRATION_AMBIGUOUS)
        + [text for text, _, _ in EVALUATION_QUERIES]
        + list(EVALUATION_AMBIGUOUS)
    )
    verdict_texts = [text for text, _, _, _ in f5_records] + list(ambiguous_queries)
    overlap = sorted(set(development_texts).intersection(verdict_texts))
    producer_source = (REPO_ROOT / "dcortex" / "semantic_producer.py").read_text(
        encoding="utf-8"
    )
    ast.parse(producer_source)
    forbidden = ["ALIAS_MAP", "SYNONYM_MAP", "HARDCODED_SYNONYM"]
    present = [item for item in forbidden if item in producer_source]
    buffers_after = model_buffer_hash(model)

    gates = [
        gate(
            "R0_ADAPTER_REQUIRED",
            accepted == emitted,
            f"Adapter accepted {accepted}/{emitted} emitted interpretations.",
            {"emitted": emitted, "accepted": accepted},
        ),
        gate(
            "R1_NO_DIRECT_COMMIT",
            query_only == emitted,
            f"Query-only destinations {query_only}/{emitted} emitted interpretations.",
            {"emitted": emitted, "query_only": query_only},
        ),
        gate(
            "R2_SEALED_F5_SOURCE",
            actual_sealed_hash == SEALED_SHA256 and len(f5_records) == F5_TRIALS,
            f"Pas 7a hash match={actual_sealed_hash == SEALED_SHA256}; "
            f"AST-generated F5 trials={len(f5_records)}.",
            {
                "actual_hash": actual_sealed_hash,
                "frozen_hash": SEALED_SHA256,
                "seed": F5_SEED,
                "trials": len(f5_records),
                "entities": len(entities),
                "attributes": list(definitions["HOLDOUT_ATTR_TYPES"]),
            },
        ),
        gate(
            "R3_CALIBRATION_SEPARATION",
            not overlap,
            f"Exact development/verdict text overlap count={len(overlap)}.",
            {
                "overlap": overlap,
                "development_hash": text_set_hash(development_texts),
                "verdict_hash": text_set_hash(verdict_texts),
            },
        ),
        gate(
            "R4_DETERMINISTIC",
            deterministic,
            f"Repeated result exact={deterministic}.",
            {"exact": deterministic},
        ),
        gate(
            "R5_NO_SYNONYM_MAP",
            not present,
            f"Forbidden lexical-map identifiers present={present}.",
            {"present": present},
        ),
        gate(
            "R6_F5_QUERY_INTERPRETATION",
            accuracy >= 0.85,
            f"Sealed-F5 query interpretation accuracy={accuracy:.1%} "
            f"({correct}/{len(f5_records)}), emitted={emitted}.",
            {
                "correct": correct,
                "total": len(f5_records),
                "accuracy": accuracy,
                "emitted": emitted,
            },
        ),
        gate(
            "R7_WRONG_INTERPRETATION",
            wrong_rate <= 0.02,
            f"Wrong emitted interpretations={wrong}/{len(f5_records)} "
            f"({wrong_rate:.1%}).",
            {"wrong": wrong, "total": len(f5_records), "wrong_rate": wrong_rate},
        ),
        gate(
            "R8_AMBIGUOUS_HONESTY",
            abstain_rate >= 0.80,
            f"Attribute-unspecified abstention={abstain_rate:.1%} "
            f"({abstained}/{len(ambiguous_queries)}).",
            {
                "abstained": abstained,
                "total": len(ambiguous_queries),
                "abstain_rate": abstain_rate,
            },
        ),
        gate(
            "R9_TOKEN_MEAN_UPLIFT",
            uplift >= 0.30,
            f"Multi-view accuracy={accuracy:.1%}; frozen token-mean accuracy="
            f"{token_accuracy:.1%}; uplift={uplift:+.1%}.",
            {
                "multiview_accuracy": accuracy,
                "token_mean_accuracy": token_accuracy,
                "token_mean_emitted": token_emitted,
                "uplift": uplift,
            },
        ),
        gate(
            "R10_MULTIVIEW_CONSENSUS",
            consensus_ok == emitted,
            f"Consensus policy satisfied by {consensus_ok}/{emitted} emissions.",
            {"emitted": emitted, "consensus_ok": consensus_ok},
        ),
        gate(
            "R11_READ_ONLY_MODEL_STATE",
            buffers_before == buffers_after,
            f"All model buffers byte-identical={buffers_before == buffers_after}.",
            {"before": buffers_before, "after": buffers_after},
        ),
        gate(
            "R12_SEALED_UNTOUCHED",
            actual_sealed_hash == SEALED_SHA256,
            f"Pas 7a SHA-256 actual={actual_sealed_hash}, frozen={SEALED_SHA256}.",
            {"actual": actual_sealed_hash, "frozen": SEALED_SHA256},
        ),
    ]
    all_pass = all(item["passed"] for item in gates)
    verdict = {
        "verdict": gates,
        "reference": {
            "run_timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "checkpoint": str(checkpoint),
            "checkpoint_mtime": datetime.fromtimestamp(
                checkpoint.stat().st_mtime
            ).astimezone().isoformat(timespec="seconds"),
            "device": str(device),
            "backend": backend.backend_id,
            "backend_version": backend.backend_version,
            "entity_prompts": list(ENTITY_PROMPTS),
            "attribute_prompts": list(ATTRIBUTE_PROMPTS),
            "entity_minimum_consensus": 2,
            "attribute_minimum_consensus": 2,
            "calibration": calibration,
            "f5_records": records,
            "ambiguous_records": ambiguous_records,
            "scope": (
                "Query-intent interpretation of sealed Pas 7a F5 forms. It is "
                "not an end-to-end F5 commit result, fact internalization result, "
                "or semantic-memory advantage measurement."
            ),
            "claim_status": "MEASURED in one local environment; not PROVEN.",
            "all_pass": all_pass,
        },
    }
    verdict_path = results_dir / "verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(SEP, flush=True)
    print("[INFO] D_Cortex v15.7b multi-view sealed-F5 probe", flush=True)
    print(f"[INFO] Checkpoint: {checkpoint}", flush=True)
    print(f"[INFO] Device: {device}", flush=True)
    print(f"[INFO] Attribute aggregate margin threshold: {threshold:.6f}", flush=True)
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
    parser = argparse.ArgumentParser(description="D_Cortex multi-view F5 probe")
    parser.add_argument("--checkpoint", type=str, default=str(default_checkpoint))
    parser.add_argument(
        "--run-dir",
        type=str,
        default=str(REPO_ROOT / "runs" / "semantic_multiview"),
    )
    return parser


def main() -> int:
    """CLI entry point."""
    return run(build_argparser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
