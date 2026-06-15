# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b-F2 frozen attribute-conditioned fact verdict.

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Type

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import tiktoken

from dcortex.semantic_adapter import ConservativeSemanticAdapter, DecisionStatus
from dcortex.semantic_fact_producer import (
    ConservativeAttributeConditionedFactProducer,
    ConservativeTrainedFactProducer,
    SemanticFactHead,
)
from dcortex.semantic_producer import DCortexContextualFeatureBackend
from scripts.semantic_contextual_curriculum import load_contextual_model
from scripts.semantic_fact_curriculum import (
    ATTRIBUTE_MARGIN,
    ENTITY_MARGIN,
    FOLDS,
    SEALED_FILES,
    UNKNOWN_ATTRIBUTE,
    UNKNOWN_ENTITY,
    UNKNOWN_VALUE,
    VALUE_MARGIN,
    CachedFactBackend,
    build_fold_data,
    extract_definitions,
    records_hash,
    sha256_file,
    state_tensor_hash,
    value_id,
)
from scripts.semantic_likelihood_probe import gate

SEP = "=" * 70
ENC = tiktoken.get_encoding("gpt2")
TRIALS_PER_FOLD = 500
HELDOUT_SEED_BASE = 20261450
AMBIGUITY_SEED = 20261460
PREDECESSOR_VERDICT = (
    REPO_ROOT / "runs" / "semantic_fact_curriculum" / "results" / "verdict.json"
)
PREDECESSOR_SHA256 = "185e8f102449c69b2a1bad08afde475ac8319aed23da15dae41cd712499185d3"


def generate_new_heldout(
    definitions: Dict[str, Any],
    form_index: int,
) -> Tuple[Tuple[str, str, str, str], ...]:
    """Generate a new frozen held-out F1 sample."""
    rng = random.Random(HELDOUT_SEED_BASE + form_index)
    records: List[Tuple[str, str, str, str]] = []
    for _ in range(TRIALS_PER_FOLD):
        attribute = rng.choice(definitions["HOLDOUT_ATTR_TYPES"])
        entity = rng.choice(definitions["HOLDOUT_ENTITIES_SINGLE"])
        value = rng.choice(definitions["HOLDOUT_ATTR_VALUES"][attribute])
        records.append(
            (
                definitions["F1_FACT_CONSTRUCTIONS"][attribute][form_index](
                    entity, value
                ),
                entity,
                attribute,
                value_id(attribute, value),
            )
        )
    return tuple(records)


def generate_new_ambiguity(
    definitions: Dict[str, Any],
) -> Tuple[Tuple[str, str, str, str], ...]:
    """Generate a new frozen ambiguity sample."""
    rng = random.Random(AMBIGUITY_SEED)
    records: List[Tuple[str, str, str, str]] = []
    entities = definitions["HOLDOUT_ENTITIES_SINGLE"]
    attributes = definitions["HOLDOUT_ATTR_TYPES"]
    values = definitions["HOLDOUT_ATTR_VALUES"]
    distractors = definitions["F4_DISTRACTOR_SENTENCES"]
    for index in range(200):
        attribute = rng.choice(attributes)
        entity_a, entity_b = rng.sample(entities, 2)
        value_a, value_b = rng.sample(values[attribute], 2)
        records.extend(
            (
                (
                    f"No consensus exists on whether the {entity_a} is {value_a} or {value_b}.",
                    UNKNOWN_ENTITY,
                    UNKNOWN_ATTRIBUTE,
                    UNKNOWN_VALUE,
                ),
                (
                    f"The statement may concern the {entity_a} or the {entity_b}: one is {value_a}.",
                    UNKNOWN_ENTITY,
                    UNKNOWN_ATTRIBUTE,
                    UNKNOWN_VALUE,
                ),
                (
                    f"General observation {index}: {rng.choice(distractors)}",
                    UNKNOWN_ENTITY,
                    UNKNOWN_ATTRIBUTE,
                    UNKNOWN_VALUE,
                ),
            )
        )
    return tuple(records)


def load_backend(
    head_path: Path,
    feature_by_text: Dict[str, torch.Tensor],
    device: torch.device,
) -> CachedFactBackend:
    """Load one predecessor head over the new frozen feature cache."""
    payload = torch.load(head_path, map_location="cpu", weights_only=False)
    head = SemanticFactHead(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        entity_classes=len(payload["entity_ids"]),
        attribute_classes=len(payload["attribute_ids"]),
        value_classes=len(payload["value_ids"]),
        dropout=0.1,
    ).to(device)
    head.load_state_dict(payload["head"], strict=True)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return CachedFactBackend(
        feature_by_text,
        head,
        payload["entity_ids"],
        payload["attribute_ids"],
        payload["value_ids"],
        head_path.name,
    )


def evaluate_known(
    name: str,
    records: Sequence[Tuple[str, str, str, str]],
    backend: CachedFactBackend,
    producer_type: Type[
        ConservativeTrainedFactProducer | ConservativeAttributeConditionedFactProducer
    ],
) -> Dict[str, Any]:
    """Evaluate one fact producer on known facts."""
    adapter = ConservativeSemanticAdapter()
    producer = producer_type(
        backend, adapter, ENTITY_MARGIN, ATTRIBUTE_MARGIN, VALUE_MARGIN
    )
    correct = emitted = wrong = accepted = 0
    for index, (text, entity, attribute, qualified_value) in enumerate(records):
        result = producer.produce(
            f"{name}-{index}",
            index,
            text,
            provenance=(f"family:{name}", f"trial:{index}"),
        )
        emitted += int(result.emitted)
        accepted += int(
            result.adapter_decision is not None
            and result.adapter_decision.status == DecisionStatus.ACCEPT_PROVISIONAL
        )
        expected_value = qualified_value.split(":", 1)[1]
        is_correct = (
            result.hypothesis is not None
            and result.hypothesis.entity_id == entity
            and result.hypothesis.attr_type == attribute
            and result.hypothesis.value_id == expected_value
        )
        correct += int(is_correct)
        wrong += int(result.emitted and not is_correct)
    total = len(records)
    return {
        "name": name,
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "emitted": emitted,
        "wrong": wrong,
        "wrong_rate": wrong / total,
        "adapter_accepted": accepted,
    }


def evaluate_ambiguity(
    records: Sequence[Tuple[str, str, str, str]],
    backend: CachedFactBackend,
) -> Dict[str, Any]:
    """Evaluate conditioned-producer abstention."""
    producer = ConservativeAttributeConditionedFactProducer(
        backend,
        ConservativeSemanticAdapter(),
        ENTITY_MARGIN,
        ATTRIBUTE_MARGIN,
        VALUE_MARGIN,
    )
    emitted = sum(
        producer.produce(
            f"ambiguity-{index}",
            100000 + index,
            record[0],
            provenance=("family:ambiguity", f"trial:{index}"),
        ).emitted
        for index, record in enumerate(records)
    )
    total = len(records)
    return {
        "total": total,
        "emitted": emitted,
        "abstained": total - emitted,
        "abstain_rate": (total - emitted) / total,
    }


def combine(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine fold summaries."""
    total = sum(item["total"] for item in results)
    correct = sum(item["correct"] for item in results)
    emitted = sum(item["emitted"] for item in results)
    wrong = sum(item["wrong"] for item in results)
    accepted = sum(item["adapter_accepted"] for item in results)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "emitted": emitted,
        "wrong": wrong,
        "wrong_rate": wrong / total,
        "adapter_accepted": accepted,
    }


def run(args: argparse.Namespace) -> int:
    """Run the frozen attribute-conditioned fact verdict."""
    checkpoint = Path(args.checkpoint)
    heads_dir = Path(args.heads_dir)
    results_dir = Path(args.run_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Frozen conditioned fact verdict requires CUDA")
    definitions = extract_definitions()
    samples_a = {fold: generate_new_heldout(definitions, fold) for fold in FOLDS}
    samples_b = {fold: generate_new_heldout(definitions, fold) for fold in FOLDS}
    ambiguity_a = generate_new_ambiguity(definitions)
    ambiguity_b = generate_new_ambiguity(definitions)
    fold_data = {fold: build_fold_data(definitions, fold) for fold in FOLDS}
    holdout_checks = [
        not (
            {record[0] for record in fold_data[fold]["base"]}
            & {record[0] for record in samples_a[fold]}
        )
        for fold in FOLDS
    ]
    deterministic = all(
        records_hash(samples_a[fold]) == records_hash(samples_b[fold])
        for fold in FOLDS
    ) and records_hash(ambiguity_a) == records_hash(ambiguity_b)

    print(SEP, flush=True)
    print("[INFO] Loading frozen contextual substrate and predecessor heads", flush=True)
    model, state, payload = load_contextual_model(checkpoint, device)
    substrate_hash_before = state_tensor_hash(state)
    feature_backend = DCortexContextualFeatureBackend(
        model=model,
        tokenizer=ENC.encode_ordinary,
        max_seq_len=128,
        batch_size=128,
        backend_version=checkpoint.name,
    )
    all_texts = sorted(
        {record[0] for fold in FOLDS for record in samples_a[fold]}
        | {record[0] for record in ambiguity_a}
    )
    features = feature_backend.features(all_texts).cpu()
    feature_by_text = {text: features[index] for index, text in enumerate(all_texts)}
    unconstrained_results: List[Dict[str, Any]] = []
    conditioned_results: List[Dict[str, Any]] = []
    ambiguity_results: List[Dict[str, Any]] = []
    for fold in FOLDS:
        backend = load_backend(
            heads_dir / f"fold_{fold}_best_head.pt", feature_by_text, device
        )
        unconstrained_results.append(
            evaluate_known(
                f"unconstrained-fold-{fold}",
                samples_a[fold],
                backend,
                ConservativeTrainedFactProducer,
            )
        )
        conditioned_results.append(
            evaluate_known(
                f"conditioned-fold-{fold}",
                samples_a[fold],
                backend,
                ConservativeAttributeConditionedFactProducer,
            )
        )
        ambiguity_results.append(evaluate_ambiguity(ambiguity_a, backend))
    unconstrained = combine(unconstrained_results)
    conditioned = combine(conditioned_results)
    ambiguity_total = sum(item["total"] for item in ambiguity_results)
    ambiguity_emitted = sum(item["emitted"] for item in ambiguity_results)
    ambiguity = {
        "total": ambiguity_total,
        "emitted": ambiguity_emitted,
        "abstained": ambiguity_total - ambiguity_emitted,
        "abstain_rate": (ambiguity_total - ambiguity_emitted) / ambiguity_total,
        "per_fold": ambiguity_results,
    }
    substrate_hash_after = state_tensor_hash(state)
    trainable_substrate = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    predecessor_preserved = (
        PREDECESSOR_VERDICT.exists()
        and sha256_file(PREDECESSOR_VERDICT) == PREDECESSOR_SHA256
    )
    actual_seals = {
        name: sha256_file(path) for name, (path, _) in SEALED_FILES.items()
    }
    expected_seals = {
        name: expected for name, (_, expected) in SEALED_FILES.items()
    }
    seals_untouched = actual_seals == expected_seals
    provisional_only = conditioned["adapter_accepted"] == conditioned["emitted"]
    probe_backend = load_backend(
        heads_dir / "fold_0_best_head.pt", feature_by_text, device
    )
    probe = ConservativeAttributeConditionedFactProducer(
        probe_backend,
        ConservativeSemanticAdapter(),
        ENTITY_MARGIN,
        ATTRIBUTE_MARGIN,
        VALUE_MARGIN,
    )
    forbidden = ("commit", "committed", "bank", "writer", "consolidator")
    no_direct_commit = not any(hasattr(probe, item) for item in forbidden)
    uplift = conditioned["accuracy"] - unconstrained["accuracy"]
    gates = [
        gate(
            "K0_PREDECESSOR_PRESERVED",
            predecessor_preserved,
            f"Predecessor verdict artifact unchanged={predecessor_preserved}.",
            {
                "expected": PREDECESSOR_SHA256,
                "actual": (
                    sha256_file(PREDECESSOR_VERDICT)
                    if PREDECESSOR_VERDICT.exists()
                    else None
                ),
            },
        ),
        gate(
            "K1_NEW_SAMPLE_HOLDOUT",
            all(holdout_checks) and deterministic,
            f"New sample held out={all(holdout_checks)}; deterministic={deterministic}.",
            {
                "holdout_checks": holdout_checks,
                "sample_hashes": {
                    str(fold): records_hash(samples_a[fold]) for fold in FOLDS
                },
                "ambiguity_hash": records_hash(ambiguity_a),
            },
        ),
        gate(
            "K2_CONDITIONED_ACCURACY",
            conditioned["accuracy"] >= 0.85,
            f"Conditioned accuracy={conditioned['accuracy']:.1%} "
            f"({conditioned['correct']}/{conditioned['total']}).",
            conditioned,
        ),
        gate(
            "K3_WRONG_PROVISIONAL",
            conditioned["wrong_rate"] <= 0.02,
            f"Conditioned wrong provisional={conditioned['wrong_rate']:.1%} "
            f"({conditioned['wrong']}/{conditioned['total']}).",
            conditioned,
        ),
        gate(
            "K4_AMBIGUOUS_HONESTY",
            ambiguity["abstain_rate"] >= 0.95,
            f"Conditioned ambiguity abstention={ambiguity['abstain_rate']:.1%} "
            f"({ambiguity['abstained']}/{ambiguity['total']}).",
            ambiguity,
        ),
        gate(
            "K5_COVERAGE_UPLIFT",
            uplift >= 0.15,
            f"Conditioned accuracy uplift={uplift:+.1%} "
            f"({unconstrained['accuracy']:.1%}->{conditioned['accuracy']:.1%}).",
            {"unconstrained": unconstrained, "conditioned": conditioned, "uplift": uplift},
        ),
        gate(
            "K6_ADAPTER_PROVISIONAL_ONLY",
            provisional_only,
            f"Every conditioned emission adapter-accepted provisional-only={provisional_only}.",
            {
                "emitted": conditioned["emitted"],
                "adapter_accepted": conditioned["adapter_accepted"],
            },
        ),
        gate(
            "K7_FROZEN_SUBSTRATE",
            substrate_hash_before == substrate_hash_after and trainable_substrate == 0,
            f"Substrate byte-identical={substrate_hash_before == substrate_hash_after}; "
            f"trainable parameters={trainable_substrate}.",
            {
                "before": substrate_hash_before,
                "after": substrate_hash_after,
                "trainable_substrate_parameters": trainable_substrate,
            },
        ),
        gate(
            "K8_SEALS_UNTOUCHED",
            seals_untouched,
            f"Pas 7a/query-side seals unchanged={seals_untouched}.",
            {"expected": expected_seals, "actual": actual_seals},
        ),
        gate(
            "K9_NO_DIRECT_COMMIT",
            no_direct_commit,
            f"Conditioned producer exposes no direct commit path={no_direct_commit}.",
            {"forbidden_attributes": list(forbidden)},
        ),
    ]
    all_pass = all(item["passed"] for item in gates)
    verdict = {
        "verdict": gates,
        "reference": {
            "run_timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "checkpoint": str(checkpoint),
            "checkpoint_mtime": datetime.fromtimestamp(checkpoint.stat().st_mtime)
            .astimezone()
            .isoformat(timespec="seconds"),
            "device": str(device),
            "unconstrained_per_fold": unconstrained_results,
            "conditioned_per_fold": conditioned_results,
            "unconstrained": unconstrained,
            "conditioned": conditioned,
            "ambiguity": ambiguity,
            "scope": (
                "Attribute-conditioned architecture regression on new samples "
                "from the same sealed F1 form families. Provisional-only; no "
                "Pas 7a ingestion or committed-memory result."
            ),
            "claim_status": "MEASURED in one local environment; not PROVEN.",
            "all_pass": all_pass,
        },
    }
    verdict_path = results_dir / "verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    del payload

    print(SEP, flush=True)
    print("[INFO] D_Cortex v15.7b-F2 conditioned fact verdict", flush=True)
    for item in gates:
        label = "✓ PASS" if item["passed"] else "✗ FAIL"
        print(f"{label} [{item['criterion_id']}] {item['evidence']}", flush=True)
    print(SEP, flush=True)
    print(f"[INFO] Overall: {'ALL GATES PASS' if all_pass else 'GATE FAILURE'}", flush=True)
    print(f"[INFO] Verdict: {verdict_path}", flush=True)
    return 0 if all_pass else 1


def build_argparser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    source_root = REPO_ROOT.parent / "D_Cortex-main"
    parser = argparse.ArgumentParser(description="D_Cortex conditioned fact verdict")
    parser.add_argument(
        "--checkpoint",
        default=str(source_root / "runs" / "warmstart" / "warmstarted_init.pt"),
    )
    parser.add_argument(
        "--heads-dir",
        default=str(REPO_ROOT / "runs" / "semantic_fact_curriculum" / "results"),
    )
    parser.add_argument(
        "--run-dir",
        default=str(REPO_ROOT / "runs" / "semantic_fact_conditioned"),
    )
    return parser


def main() -> int:
    """CLI entry point."""
    return run(build_argparser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
