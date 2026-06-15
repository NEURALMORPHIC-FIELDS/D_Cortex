# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b leave-one-form-out semantic curriculum cross-validation.

import argparse
import hashlib
import json
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
import tiktoken

from dcortex.semantic_adapter import (
    ConservativeSemanticAdapter,
    DecisionStatus,
    RequestedDestination,
)
from dcortex.semantic_producer import (
    ConservativeTrainedQueryProducer,
    DCortexPooledFeatureBackend,
    PooledSemanticClassificationBackend,
    SemanticFeatureBackend,
)
from scripts.semantic_likelihood_probe import SEALED_SHA256, SEALED_SOURCE, gate
from scripts.train_semantic_internalizer import (
    ATTRIBUTE_MARGIN,
    ENTITY_MARGIN,
    HEAD_HIDDEN_DIM,
    TRAINING_SEED,
    UNKNOWN_ATTRIBUTE,
    UNKNOWN_ENTITY,
    TokenEmbeddingModel,
    build_base_training_examples,
    extract_definitions,
    generate_family,
    sample_hash,
    seed_everything,
    split_and_balance,
    state_tensor_hash,
    train_head,
)

SEP = "=" * 70
FOLDS = (0, 1, 2, 3)
AMBIGUOUS_SEED = 20261236
AMBIGUOUS_TRIALS = 200
AMBIGUOUS_TEMPLATES = (
    "Give a broad account of the {entity} without choosing an aspect.",
    "What general knowledge concerns the {entity}?",
    "Explain the {entity} in an unrestricted manner.",
    "Provide any overview of the {entity}.",
    "What is generally known regarding the {entity}?",
    "Speak about the {entity} without a specific question.",
    "Offer background on the {entity}.",
    "Describe whatever matters about the {entity}.",
    "Recall general information about the {entity}.",
    "Summarize the {entity} broadly.",
)


def build_curriculum_data(
    definitions: Mapping[str, Any],
    entity_ids: Sequence[str],
    heldout_form_index: int | None,
) -> Dict[str, Any]:
    """Build one deterministic curriculum fold or the all-form final dataset."""
    examples = set(build_base_training_examples(definitions, entity_ids))
    for source_name in ("F1_QUERY_CONSTRUCTIONS", "F3_NOVEL_ALIAS_QUERIES"):
        for attribute, forms in definitions[source_name].items():
            for form_index, builder in enumerate(forms):
                if heldout_form_index is not None and form_index == heldout_form_index:
                    continue
                for entity in entity_ids:
                    examples.add(
                        (
                            builder(entity),
                            entity,
                            attribute,
                            f"{source_name}_FORM_{form_index}",
                        )
                    )
    base = tuple(sorted(examples))
    train, validation = split_and_balance(base)
    return {
        "base": base,
        "train": train,
        "validation": validation,
        "base_hash": sample_hash(base),
        "train_hash": sample_hash(train),
        "validation_hash": sample_hash(validation),
    }


def generate_heldout_form_records(
    definitions: Mapping[str, Any],
    source_name: str,
    form_index: int,
) -> Tuple[Tuple[str, str, str, str], ...]:
    """Generate exhaustive held-out-form records for one family and fold."""
    records: List[Tuple[str, str, str, str]] = []
    for attribute in definitions["HOLDOUT_ATTR_TYPES"]:
        builder = definitions[source_name][attribute][form_index]
        for entity in definitions["HOLDOUT_ENTITIES_SINGLE"]:
            records.append((builder(entity), entity, attribute, ""))
    return tuple(records)


def generate_ambiguous(entity_ids: Sequence[str]) -> Tuple[Tuple[str, str], ...]:
    """Generate the frozen new ambiguity set."""
    rng = random.Random(AMBIGUOUS_SEED)
    records: List[Tuple[str, str]] = []
    for _ in range(AMBIGUOUS_TRIALS):
        entity = rng.choice(entity_ids)
        records.append((rng.choice(AMBIGUOUS_TEMPLATES).format(entity=entity), entity))
    return tuple(records)


def evaluate_records(
    family: str,
    records: Sequence[Tuple[str, str, str, str]],
    backend: PooledSemanticClassificationBackend,
) -> Dict[str, Any]:
    """Evaluate trained query production on one record set."""
    correct = 0
    emitted = 0
    wrong = 0
    accepted = 0
    query_only = 0
    details: List[Dict[str, Any]] = []
    for index, (text, expected_entity, expected_attribute, value) in enumerate(records):
        result = ConservativeTrainedQueryProducer(
            backend,
            ConservativeSemanticAdapter(),
            entity_margin_threshold=ENTITY_MARGIN,
            attribute_margin_threshold=ATTRIBUTE_MARGIN,
        ).produce(
            f"{family}-{index}",
            index,
            text,
            provenance=(f"family:{family}", f"trial:{index}"),
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
        details.append(
            {
                "text": text,
                "value": value,
                "expected_entity": expected_entity,
                "expected_attribute": expected_attribute,
                "prediction": prediction,
                "correct": is_correct,
                "emitted": result.emitted,
                "reason_codes": list(result.reason_codes),
                "scores": [score.to_dict() for score in result.scores],
            }
        )
    total = len(records)
    return {
        "family": family,
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "emitted": emitted,
        "wrong": wrong,
        "wrong_rate": wrong / total,
        "adapter_accepted": accepted,
        "query_only": query_only,
        "details": details,
    }


def combine_results(family: str, results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate out-of-fold family results."""
    total = sum(int(result["total"]) for result in results)
    correct = sum(int(result["correct"]) for result in results)
    emitted = sum(int(result["emitted"]) for result in results)
    wrong = sum(int(result["wrong"]) for result in results)
    accepted = sum(int(result["adapter_accepted"]) for result in results)
    query_only = sum(int(result["query_only"]) for result in results)
    return {
        "family": family,
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "emitted": emitted,
        "wrong": wrong,
        "wrong_rate": wrong / total,
        "adapter_accepted": accepted,
        "query_only": query_only,
    }


def train_backend(
    feature_backend: SemanticFeatureBackend,
    data: Mapping[str, Any],
    entity_ids: Sequence[str],
    attribute_ids: Sequence[str],
    device: torch.device,
    output_path: Path,
    seed: int,
) -> Tuple[PooledSemanticClassificationBackend, Dict[str, Any]]:
    """Train one fold head and wrap it as a classification backend."""
    seed_everything(seed)
    head, optimization = train_head(
        feature_backend,
        data["train"],
        data["validation"],
        entity_ids,
        attribute_ids,
        device,
        output_path,
    )
    backend = PooledSemanticClassificationBackend(
        feature_backend=feature_backend,
        head=head,
        entity_ids=entity_ids,
        attribute_ids=attribute_ids,
        unknown_entity_id=UNKNOWN_ENTITY,
        unknown_attribute_id=UNKNOWN_ATTRIBUTE,
        backend_version=f"curriculum:{output_path.name}",
    )
    return backend, optimization


def run(args: argparse.Namespace) -> int:
    """Run the frozen leave-one-form-out semantic curriculum."""
    checkpoint = Path(args.checkpoint)
    results_dir = Path(args.run_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint}")
    definitions = extract_definitions()
    entities = tuple(definitions["HOLDOUT_ENTITIES_SINGLE"])
    entity_ids = tuple(sorted(entities)) + (UNKNOWN_ENTITY,)
    attribute_ids = tuple(definitions["HOLDOUT_ATTR_TYPES"]) + (UNKNOWN_ATTRIBUTE,)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = payload.get("model", payload)
    substrate_hash_before = state_tensor_hash(state)
    model = TokenEmbeddingModel(state["shared_token_emb.weight"].detach().float()).to(
        device
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    feature_backend = DCortexPooledFeatureBackend(
        model=model,
        tokenizer=tiktoken.get_encoding("gpt2").encode_ordinary,
        max_seq_len=128,
        batch_size=256,
        backend_version=checkpoint.name,
    )

    fold_reports: List[Dict[str, Any]] = []
    f1_results: List[Dict[str, Any]] = []
    f3_results: List[Dict[str, Any]] = []
    deterministic_checks: List[bool] = []
    holdout_checks: List[bool] = []
    optimization_checks: List[bool] = []
    for fold in FOLDS:
        print(SEP, flush=True)
        print(f"[INFO] Curriculum fold {fold}/3", flush=True)
        data_a = build_curriculum_data(definitions, entities, fold)
        data_b = build_curriculum_data(definitions, entities, fold)
        deterministic = all(
            data_a[key] == data_b[key]
            for key in ("base_hash", "train_hash", "validation_hash")
        )
        deterministic_checks.append(deterministic)
        heldout_f1 = generate_heldout_form_records(
            definitions, "F1_QUERY_CONSTRUCTIONS", fold
        )
        heldout_f3 = generate_heldout_form_records(
            definitions, "F3_NOVEL_ALIAS_QUERIES", fold
        )
        training_texts = {item[0] for item in data_a["base"]}
        heldout_texts = {item[0] for item in heldout_f1 + heldout_f3}
        held_source_names = {
            f"F1_QUERY_CONSTRUCTIONS_FORM_{fold}",
            f"F3_NOVEL_ALIAS_QUERIES_FORM_{fold}",
        }
        sources = {item[3] for item in data_a["base"]}
        holdout_ok = not training_texts.intersection(heldout_texts) and not sources.intersection(
            held_source_names
        )
        holdout_checks.append(holdout_ok)
        backend, optimization = train_backend(
            feature_backend,
            data_a,
            entity_ids,
            attribute_ids,
            device,
            results_dir / f"fold_{fold}_best_head.pt",
            TRAINING_SEED + 100 + fold,
        )
        optimization_ok = (
            optimization["loss_drop"] >= 0.20
            and optimization["best_joint_accuracy"] >= 0.95
        )
        optimization_checks.append(optimization_ok)
        f1 = evaluate_records(f"F1_fold_{fold}", heldout_f1, backend)
        f3 = evaluate_records(f"F3_fold_{fold}", heldout_f3, backend)
        f1_results.append(f1)
        f3_results.append(f3)
        fold_reports.append(
            {
                "fold": fold,
                "data": {
                    "base_count": len(data_a["base"]),
                    "train_count": len(data_a["train"]),
                    "validation_count": len(data_a["validation"]),
                    "base_hash": data_a["base_hash"],
                    "train_hash": data_a["train_hash"],
                    "validation_hash": data_a["validation_hash"],
                },
                "deterministic": deterministic,
                "holdout_ok": holdout_ok,
                "optimization": optimization,
                "F1": f1,
                "F3": f3,
            }
        )
    f1_aggregate = combine_results("F1_out_of_fold", f1_results)
    f3_aggregate = combine_results("F3_out_of_fold", f3_results)

    print(SEP, flush=True)
    print("[INFO] Training final all-curriculum head", flush=True)
    final_data_a = build_curriculum_data(definitions, entities, None)
    final_data_b = build_curriculum_data(definitions, entities, None)
    final_deterministic = all(
        final_data_a[key] == final_data_b[key]
        for key in ("base_hash", "train_hash", "validation_hash")
    )
    deterministic_checks.append(final_deterministic)
    final_backend, final_optimization = train_backend(
        feature_backend,
        final_data_a,
        entity_ids,
        attribute_ids,
        device,
        results_dir / "final_best_head.pt",
        TRAINING_SEED + 200,
    )
    optimization_checks.append(
        final_optimization["loss_drop"] >= 0.20
        and final_optimization["best_joint_accuracy"] >= 0.95
    )
    f5_records = generate_family(definitions, "F5")
    f5_result = evaluate_records("F5_final", f5_records, final_backend)

    ambiguous_records = generate_ambiguous(entities)
    ambiguous_abstained = 0
    ambiguous_details: List[Dict[str, Any]] = []
    for index, (text, expected_entity) in enumerate(ambiguous_records):
        result = ConservativeTrainedQueryProducer(
            final_backend,
            ConservativeSemanticAdapter(),
            entity_margin_threshold=ENTITY_MARGIN,
            attribute_margin_threshold=ATTRIBUTE_MARGIN,
        ).produce(
            f"ambiguous-{index}",
            30000 + index,
            text,
            provenance=(f"ambiguous_seed:{AMBIGUOUS_SEED}", f"trial:{index}"),
        )
        ambiguous_abstained += int(not result.emitted)
        ambiguous_details.append(
            {
                "text": text,
                "expected_entity": expected_entity,
                "emitted": result.emitted,
                "reason_codes": list(result.reason_codes),
                "scores": [score.to_dict() for score in result.scores],
            }
        )
    ambiguous_rate = ambiguous_abstained / len(ambiguous_records)

    substrate_hash_after = state_tensor_hash(state)
    substrate_trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    del payload
    all_adapter = all(
        result["adapter_accepted"] == result["emitted"]
        and result["query_only"] == result["emitted"]
        for result in f1_results + f3_results + [f5_result]
    )
    gates = [
        gate(
            "C0_FORM_HOLDOUT",
            all(holdout_checks),
            f"All four folds have exact held-form separation={all(holdout_checks)}.",
            {"checks": holdout_checks},
        ),
        gate(
            "C1_REAL_OPTIMIZATION",
            all(optimization_checks),
            f"All fold/final optimization checks pass={all(optimization_checks)}.",
            {"checks": optimization_checks},
        ),
        gate(
            "C2_F1_OUT_OF_FOLD",
            f1_aggregate["accuracy"] >= 0.85,
            f"F1 out-of-fold accuracy={f1_aggregate['accuracy']:.1%} "
            f"({f1_aggregate['correct']}/{f1_aggregate['total']}).",
            f1_aggregate,
        ),
        gate(
            "C3_F3_OUT_OF_FOLD",
            f3_aggregate["accuracy"] >= 0.85,
            f"F3 out-of-fold accuracy={f3_aggregate['accuracy']:.1%} "
            f"({f3_aggregate['correct']}/{f3_aggregate['total']}).",
            f3_aggregate,
        ),
        gate(
            "C4_OUT_OF_FOLD_WRONG",
            f1_aggregate["wrong_rate"] <= 0.02
            and f3_aggregate["wrong_rate"] <= 0.02,
            f"Out-of-fold wrong rates: F1={f1_aggregate['wrong_rate']:.1%}, "
            f"F3={f3_aggregate['wrong_rate']:.1%}.",
            {
                "F1_wrong_rate": f1_aggregate["wrong_rate"],
                "F3_wrong_rate": f3_aggregate["wrong_rate"],
            },
        ),
        gate(
            "C5_F5_FINAL",
            f5_result["accuracy"] >= 0.85 and f5_result["wrong_rate"] <= 0.02,
            f"Final-head F5 accuracy={f5_result['accuracy']:.1%}; "
            f"wrong={f5_result['wrong_rate']:.1%}.",
            {key: value for key, value in f5_result.items() if key != "details"},
        ),
        gate(
            "C6_AMBIGUOUS_HONESTY",
            ambiguous_rate >= 0.80,
            f"New ambiguous abstention={ambiguous_rate:.1%} "
            f"({ambiguous_abstained}/{len(ambiguous_records)}).",
            {
                "abstained": ambiguous_abstained,
                "total": len(ambiguous_records),
                "abstain_rate": ambiguous_rate,
            },
        ),
        gate(
            "C7_ADAPTER_REQUIRED",
            all_adapter,
            f"All emitted interpretations routed query-only through adapter={all_adapter}.",
            {"all_adapter": all_adapter},
        ),
        gate(
            "C8_FROZEN_SUBSTRATE",
            substrate_hash_before == substrate_hash_after and substrate_trainable == 0,
            f"Substrate state byte-identical={substrate_hash_before == substrate_hash_after}; "
            f"trainable substrate parameters={substrate_trainable}.",
            {
                "before": substrate_hash_before,
                "after": substrate_hash_after,
                "trainable_substrate_parameters": substrate_trainable,
            },
        ),
        gate(
            "C9_DETERMINISTIC",
            all(deterministic_checks),
            f"All curriculum dataset reconstructions exact={all(deterministic_checks)}.",
            {"checks": deterministic_checks},
        ),
        gate(
            "C10_SEALED_UNTOUCHED",
            hashlib.sha256(SEALED_SOURCE.read_bytes()).hexdigest() == SEALED_SHA256,
            f"Pas 7a SHA-256 remains {SEALED_SHA256}.",
            {
                "actual": hashlib.sha256(SEALED_SOURCE.read_bytes()).hexdigest(),
                "frozen": SEALED_SHA256,
            },
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
            "folds": list(FOLDS),
            "attribute_margin": ATTRIBUTE_MARGIN,
            "entity_margin": ENTITY_MARGIN,
            "head_hidden_dim": HEAD_HIDDEN_DIM,
            "fold_reports": fold_reports,
            "F1_aggregate": f1_aggregate,
            "F3_aggregate": f3_aggregate,
            "final_data": {
                "base_count": len(final_data_a["base"]),
                "train_count": len(final_data_a["train"]),
                "validation_count": len(final_data_a["validation"]),
                "base_hash": final_data_a["base_hash"],
                "train_hash": final_data_a["train_hash"],
                "validation_hash": final_data_a["validation_hash"],
                "optimization": final_optimization,
            },
            "F5_final": f5_result,
            "ambiguous_details": ambiguous_details,
            "scope": (
                "Leave-one-form-out query-side semantic curriculum over frozen "
                "D_Cortex token embeddings. Not open-domain semantics, fact-side "
                "internalization, or end-to-end memory improvement."
            ),
            "claim_status": "MEASURED in one local environment; not PROVEN.",
            "all_pass": all_pass,
        },
    }
    verdict_path = results_dir / "verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(SEP, flush=True)
    print("[INFO] D_Cortex v15.7b semantic curriculum cross-validation", flush=True)
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
    parser = argparse.ArgumentParser(description="D_Cortex semantic curriculum")
    parser.add_argument("--checkpoint", type=str, default=str(default_checkpoint))
    parser.add_argument(
        "--run-dir",
        type=str,
        default=str(REPO_ROOT / "runs" / "semantic_curriculum"),
    )
    return parser


def main() -> int:
    """CLI entry point."""
    return run(build_argparser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
