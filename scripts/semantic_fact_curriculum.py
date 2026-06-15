# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b-F frozen leave-one-form-out semantic fact curriculum.

import argparse
import ast
import copy
import hashlib
import json
import os
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
import torch.nn.functional as F
import tiktoken

from dcortex.semantic_adapter import (
    ConservativeSemanticAdapter,
    DecisionStatus,
    RequestedDestination,
)
from dcortex.semantic_fact_producer import (
    ConservativeTrainedFactProducer,
    PooledSemanticFactClassificationBackend,
    SemanticFactClassificationBackend,
    SemanticFactHead,
)
from dcortex.semantic_producer import DCortexContextualFeatureBackend
from scripts.semantic_contextual_curriculum import load_contextual_model
from scripts.semantic_likelihood_probe import SEALED_SHA256, SEALED_SOURCE, gate
from scripts.train_semantic_internalizer import state_tensor_hash

SEP = "=" * 70
ENC = tiktoken.get_encoding("gpt2")
FOLDS = (0, 1, 2, 3)
TRAINING_SEED = 20261340
EVAL_TRIALS_PER_FOLD = 500
AMBIGUITY_TRIALS_PER_TYPE = 200
ENTITY_MARGIN = 0.0
ATTRIBUTE_MARGIN = 0.4
VALUE_MARGIN = 0.4
HEAD_HIDDEN_DIM = 256
UNKNOWN_ENTITY = "UNKNOWN_ENTITY"
UNKNOWN_ATTRIBUTE = "UNKNOWN"
UNKNOWN_VALUE = "UNKNOWN_VALUE"

SEALED_FILES = {
    "pas7a": (
        SEALED_SOURCE,
        SEALED_SHA256,
    ),
    "semantic_adapter": (
        REPO_ROOT / "dcortex" / "semantic_adapter.py",
        "719afbe8359af4ee98006e0fa19f169a0a40f22ee0cc41e5dc1459d6e07e605e",
    ),
    "semantic_query_producer": (
        REPO_ROOT / "dcortex" / "semantic_producer.py",
        "24797a121c90f78161518362f0401bbc1b161546e87bac194adb4144f4aa67d0",
    ),
    "semantic_contextual_evaluator": (
        REPO_ROOT / "scripts" / "semantic_contextual_curriculum.py",
        "bab8ec1268d5000d5a9da5ae5946d29946822efa9a6b0b3ce53be59052631b57",
    ),
    "semantic_query_bridge": (
        REPO_ROOT / "dcortex" / "semantic_query_bridge.py",
        "403d4d724a1bffee61ab9cdfa469adb0c4fb3afb75c04ad4d65ad3e7c86e1b43",
    ),
}


class CachedFactBackend(SemanticFactClassificationBackend):
    """Fact backend using precomputed frozen contextual features."""

    backend_id = "dcortex_cached_contextual_semantic_fact_classifier"

    def __init__(
        self,
        feature_by_text: Mapping[str, torch.Tensor],
        head: SemanticFactHead,
        entity_ids: Sequence[str],
        attribute_ids: Sequence[str],
        value_ids: Sequence[str],
        backend_version: str,
    ) -> None:
        self.feature_by_text = feature_by_text
        self.head = head
        self.entity_ids = tuple(entity_ids)
        self.attribute_ids = tuple(attribute_ids)
        self.value_ids = tuple(value_ids)
        self.unknown_entity_id = UNKNOWN_ENTITY
        self.unknown_attribute_id = UNKNOWN_ATTRIBUTE
        self.unknown_value_id = UNKNOWN_VALUE
        self.backend_version = backend_version

    def classify(
        self, texts: Sequence[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Classify texts from the frozen feature cache."""
        features = torch.stack([self.feature_by_text[text] for text in texts])
        device = next(self.head.parameters()).device
        with torch.inference_mode():
            entity, attribute, value = self.head(features.to(device))
        return (
            torch.softmax(entity.float(), dim=-1).cpu(),
            torch.softmax(attribute.float(), dim=-1).cpu(),
            torch.softmax(value.float(), dim=-1).cpu(),
        )


def sha256_file(path: Path) -> str:
    """Return lowercase SHA-256 for one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records_hash(records: Sequence[Tuple[str, str, str, str]]) -> str:
    """Return deterministic record hash."""
    payload = json.dumps(
        list(records),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def extract_definitions() -> Dict[str, Any]:
    """Extract the frozen fact definitions from sealed Pas 7a."""
    required = {
        "V15_FACT_TEMPLATES",
        "HOLDOUT_ENTITIES_SINGLE",
        "HOLDOUT_COLORS",
        "HOLDOUT_SIZES",
        "HOLDOUT_LOCATIONS",
        "HOLDOUT_STATES",
        "HOLDOUT_ATTR_VALUES",
        "HOLDOUT_ATTR_TYPES",
        "F1_FACT_CONSTRUCTIONS",
        "F4_DISTRACTOR_SENTENCES",
    }
    tree = ast.parse(SEALED_SOURCE.read_text(encoding="utf-8"))
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
    missing = required - found
    if missing:
        raise RuntimeError(f"Missing sealed definitions: {sorted(missing)}")
    namespace: Dict[str, Any] = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SEALED_SOURCE), "exec"), namespace)
    return {name: namespace[name] for name in required}


def value_id(attribute: str, value: str) -> str:
    """Return attribute-qualified value identity."""
    return f"{attribute}:{value}"


def build_unknown_training(
    definitions: Mapping[str, Any],
) -> Tuple[Tuple[str, str, str, str], ...]:
    """Build deterministic training-only unknown/ambiguous examples."""
    rng = random.Random(TRAINING_SEED + 1)
    entities = definitions["HOLDOUT_ENTITIES_SINGLE"]
    attributes = definitions["HOLDOUT_ATTR_TYPES"]
    values = definitions["HOLDOUT_ATTR_VALUES"]
    distractors = definitions["F4_DISTRACTOR_SENTENCES"]
    records = set()
    for index in range(1600):
        attribute = rng.choice(attributes)
        entity = rng.choice(entities)
        value_a, value_b = rng.sample(values[attribute], 2)
        entity_b = rng.choice([item for item in entities if item != entity])
        form = index % 4
        if form == 0:
            text = f"The {entity} is both {value_a} and {value_b}."
        elif form == 1:
            text = f"The {entity} and the {entity_b} are {value_a}."
        elif form == 2:
            text = (
                f"A report mentioned {entity} and {value_a} without confirming "
                f"anything, note {index}."
            )
        else:
            text = f"{rng.choice(distractors)} Observation {index}."
        records.add((text, UNKNOWN_ENTITY, UNKNOWN_ATTRIBUTE, UNKNOWN_VALUE))
    return tuple(sorted(records))


def build_fold_data(
    definitions: Mapping[str, Any],
    heldout_form: int,
) -> Dict[str, Any]:
    """Build one deterministic leave-one-form-out fact curriculum fold."""
    examples = set(build_unknown_training(definitions))
    for attribute in definitions["HOLDOUT_ATTR_TYPES"]:
        for entity in definitions["HOLDOUT_ENTITIES_SINGLE"]:
            for value in definitions["HOLDOUT_ATTR_VALUES"][attribute]:
                qualified = value_id(attribute, value)
                for template in definitions["V15_FACT_TEMPLATES"][attribute]:
                    examples.add(
                        (template.format(e=entity, v=value), entity, attribute, qualified)
                    )
                for form_index, builder in enumerate(
                    definitions["F1_FACT_CONSTRUCTIONS"][attribute]
                ):
                    if form_index == heldout_form:
                        continue
                    examples.add((builder(entity, value), entity, attribute, qualified))
    base = tuple(sorted(examples))
    train: List[Tuple[str, str, str, str]] = []
    validation: List[Tuple[str, str, str, str]] = []
    for record in base:
        bucket = int(hashlib.sha256(record[0].encode("utf-8")).hexdigest()[:8], 16) % 10
        (validation if bucket < 2 else train).append(record)
    return {
        "base": base,
        "train": tuple(train),
        "validation": tuple(validation),
        "base_hash": records_hash(base),
        "train_hash": records_hash(train),
        "validation_hash": records_hash(validation),
    }


def generate_heldout_records(
    definitions: Mapping[str, Any],
    form_index: int,
) -> Tuple[Tuple[str, str, str, str], ...]:
    """Generate the frozen held-out F1 fact records for one fold."""
    rng = random.Random(TRAINING_SEED + 100 + form_index)
    records: List[Tuple[str, str, str, str]] = []
    for _ in range(EVAL_TRIALS_PER_FOLD):
        attribute = rng.choice(definitions["HOLDOUT_ATTR_TYPES"])
        entity = rng.choice(definitions["HOLDOUT_ENTITIES_SINGLE"])
        value = rng.choice(definitions["HOLDOUT_ATTR_VALUES"][attribute])
        text = definitions["F1_FACT_CONSTRUCTIONS"][attribute][form_index](
            entity, value
        )
        records.append((text, entity, attribute, value_id(attribute, value)))
    return tuple(records)


def generate_ambiguity_records(
    definitions: Mapping[str, Any],
) -> Tuple[Tuple[str, str, str, str], ...]:
    """Generate frozen evaluation-only conflict, multi-entity, and non-facts."""
    rng = random.Random(TRAINING_SEED + 200)
    entities = definitions["HOLDOUT_ENTITIES_SINGLE"]
    attributes = definitions["HOLDOUT_ATTR_TYPES"]
    values = definitions["HOLDOUT_ATTR_VALUES"]
    distractors = definitions["F4_DISTRACTOR_SENTENCES"]
    records: List[Tuple[str, str, str, str]] = []
    for index in range(AMBIGUITY_TRIALS_PER_TYPE):
        attribute = rng.choice(attributes)
        entity = rng.choice(entities)
        value_a, value_b = rng.sample(values[attribute], 2)
        records.append(
            (
                f"Reports disagree: the {entity} is {value_a}, yet witnesses call it {value_b}.",
                UNKNOWN_ENTITY,
                UNKNOWN_ATTRIBUTE,
                UNKNOWN_VALUE,
            )
        )
        entity_b = rng.choice([item for item in entities if item != entity])
        records.append(
            (
                f"Either the {entity} or the {entity_b} is {value_a}, but the record is unclear.",
                UNKNOWN_ENTITY,
                UNKNOWN_ATTRIBUTE,
                UNKNOWN_VALUE,
            )
        )
        records.append(
            (
                f"{rng.choice(distractors)} No factual conclusion {index}.",
                UNKNOWN_ENTITY,
                UNKNOWN_ATTRIBUTE,
                UNKNOWN_VALUE,
            )
        )
    return tuple(records)


def atomic_save(payload: Dict[str, Any], path: Path) -> None:
    """Atomically save one Torch artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def labels_for(
    records: Sequence[Tuple[str, str, str, str]],
    entity_to_index: Mapping[str, int],
    attribute_to_index: Mapping[str, int],
    value_to_index: Mapping[str, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tensorize three-axis labels."""
    return (
        torch.tensor([entity_to_index[record[1]] for record in records]),
        torch.tensor([attribute_to_index[record[2]] for record in records]),
        torch.tensor([value_to_index[record[3]] for record in records]),
    )


def train_head(
    feature_by_text: Mapping[str, torch.Tensor],
    train_records: Sequence[Tuple[str, str, str, str]],
    validation_records: Sequence[Tuple[str, str, str, str]],
    entity_ids: Sequence[str],
    attribute_ids: Sequence[str],
    value_ids: Sequence[str],
    device: torch.device,
    output_path: Path,
    seed: int,
) -> Tuple[SemanticFactHead, Dict[str, Any]]:
    """Train one fact head over frozen cached features."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    entity_to_index = {value: index for index, value in enumerate(entity_ids)}
    attribute_to_index = {value: index for index, value in enumerate(attribute_ids)}
    value_to_index = {value: index for index, value in enumerate(value_ids)}
    train_features = torch.stack([feature_by_text[item[0]] for item in train_records])
    validation_features = torch.stack(
        [feature_by_text[item[0]] for item in validation_records]
    )
    train_labels = labels_for(
        train_records, entity_to_index, attribute_to_index, value_to_index
    )
    validation_labels = labels_for(
        validation_records, entity_to_index, attribute_to_index, value_to_index
    )
    head = SemanticFactHead(
        input_dim=train_features.shape[1],
        hidden_dim=HEAD_HIDDEN_DIM,
        entity_classes=len(entity_ids),
        attribute_classes=len(attribute_ids),
        value_classes=len(value_ids),
        dropout=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-3)
    batch_size = 256
    patience = 12
    best_loss = float("inf")
    best_accuracy = 0.0
    best_epoch = -1
    best_state: Dict[str, torch.Tensor] = {}
    stale = 0
    history: List[Dict[str, float]] = []

    def evaluate() -> Tuple[float, float]:
        head.eval()
        with torch.inference_mode():
            logits = head(validation_features.to(device))
            losses = [
                F.cross_entropy(logit, label.to(device))
                for logit, label in zip(logits, validation_labels)
            ]
            joint = torch.ones(len(validation_records), dtype=torch.bool, device=device)
            for logit, label in zip(logits, validation_labels):
                joint &= logit.argmax(dim=1) == label.to(device)
        return float(sum(losses).item()), float(joint.float().mean().item())

    initial_loss, _ = evaluate()
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(100):
        head.train()
        order = torch.randperm(len(train_records), generator=generator)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            logits = head(train_features[batch].to(device))
            loss = sum(
                F.cross_entropy(logit, label[batch].to(device))
                for logit, label in zip(logits, train_labels)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
        validation_loss, joint_accuracy = evaluate()
        history.append(
            {
                "epoch": epoch,
                "validation_loss": validation_loss,
                "joint_accuracy": joint_accuracy,
            }
        )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_accuracy = joint_accuracy
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience and epoch >= 15:
            break
    head.load_state_dict(best_state, strict=True)
    head.eval()
    atomic_save(
        {
            "head": best_state,
            "entity_ids": list(entity_ids),
            "attribute_ids": list(attribute_ids),
            "value_ids": list(value_ids),
            "input_dim": int(train_features.shape[1]),
            "hidden_dim": HEAD_HIDDEN_DIM,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "best_joint_accuracy": best_accuracy,
        },
        output_path,
    )
    return head, {
        "initial_validation_loss": initial_loss,
        "best_validation_loss": best_loss,
        "loss_drop": (initial_loss - best_loss) / initial_loss,
        "best_joint_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "history": history,
    }


def evaluate_records(
    family: str,
    records: Sequence[Tuple[str, str, str, str]],
    backend: CachedFactBackend,
) -> Dict[str, Any]:
    """Evaluate provisional fact production on known facts."""
    adapter = ConservativeSemanticAdapter()
    producer = ConservativeTrainedFactProducer(
        backend,
        adapter,
        ENTITY_MARGIN,
        ATTRIBUTE_MARGIN,
        VALUE_MARGIN,
    )
    correct = emitted = wrong = accepted = provisional_only = 0
    details: List[Dict[str, Any]] = []
    for index, (text, entity, attribute, qualified_value) in enumerate(records):
        result = producer.produce(
            f"{family}-{index}",
            index,
            text,
            provenance=(f"family:{family}", f"trial:{index}"),
        )
        emitted += int(result.emitted)
        accepted += int(
            result.adapter_decision is not None
            and result.adapter_decision.status == DecisionStatus.ACCEPT_PROVISIONAL
        )
        provisional_only += int(
            result.hypothesis is not None
            and result.hypothesis.requested_destination
            == RequestedDestination.PROVISIONAL_ONLY
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
        details.append(
            {
                "text": text,
                "expected": [entity, attribute, expected_value],
                "emitted": result.emitted,
                "prediction": (
                    None
                    if result.hypothesis is None
                    else [
                        result.hypothesis.entity_id,
                        result.hypothesis.attr_type,
                        result.hypothesis.value_id,
                    ]
                ),
                "reason_codes": list(result.reason_codes),
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
        "provisional_only": provisional_only,
        "details": details,
    }


def evaluate_ambiguity(
    records: Sequence[Tuple[str, str, str, str]],
    backend: CachedFactBackend,
) -> Dict[str, Any]:
    """Evaluate abstention on frozen ambiguous/non-fact inputs."""
    producer = ConservativeTrainedFactProducer(
        backend,
        ConservativeSemanticAdapter(),
        ENTITY_MARGIN,
        ATTRIBUTE_MARGIN,
        VALUE_MARGIN,
    )
    emitted = 0
    details: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        result = producer.produce(
            f"ambiguity-{index}",
            100000 + index,
            record[0],
            provenance=("family:ambiguity", f"trial:{index}"),
        )
        emitted += int(result.emitted)
        details.append(
            {
                "text": record[0],
                "emitted": result.emitted,
                "reason_codes": list(result.reason_codes),
            }
        )
    total = len(records)
    return {
        "total": total,
        "abstained": total - emitted,
        "abstain_rate": (total - emitted) / total,
        "emitted": emitted,
        "details": details,
    }


def combine_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine fold results without duplicating detail in the summary."""
    total = sum(item["total"] for item in results)
    correct = sum(item["correct"] for item in results)
    emitted = sum(item["emitted"] for item in results)
    wrong = sum(item["wrong"] for item in results)
    accepted = sum(item["adapter_accepted"] for item in results)
    provisional_only = sum(item["provisional_only"] for item in results)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "emitted": emitted,
        "wrong": wrong,
        "wrong_rate": wrong / total,
        "adapter_accepted": accepted,
        "provisional_only": provisional_only,
    }


def run(args: argparse.Namespace) -> int:
    """Run the frozen fact-side provisional curriculum verdict."""
    checkpoint = Path(args.checkpoint)
    results_dir = Path(args.run_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Frozen fact curriculum requires CUDA")
    definitions = extract_definitions()
    entity_ids = tuple(sorted(definitions["HOLDOUT_ENTITIES_SINGLE"])) + (
        UNKNOWN_ENTITY,
    )
    attribute_ids = tuple(definitions["HOLDOUT_ATTR_TYPES"]) + (UNKNOWN_ATTRIBUTE,)
    value_ids = tuple(
        sorted(
            value_id(attribute, value)
            for attribute in definitions["HOLDOUT_ATTR_TYPES"]
            for value in definitions["HOLDOUT_ATTR_VALUES"][attribute]
        )
    ) + (UNKNOWN_VALUE,)

    fold_data_a = {fold: build_fold_data(definitions, fold) for fold in FOLDS}
    fold_data_b = {fold: build_fold_data(definitions, fold) for fold in FOLDS}
    heldout_records = {
        fold: generate_heldout_records(definitions, fold) for fold in FOLDS
    }
    ambiguity_a = generate_ambiguity_records(definitions)
    ambiguity_b = generate_ambiguity_records(definitions)
    holdout_checks = []
    deterministic_checks = []
    for fold in FOLDS:
        training_texts = {item[0] for item in fold_data_a[fold]["base"]}
        heldout_texts = {item[0] for item in heldout_records[fold]}
        holdout_checks.append(not training_texts.intersection(heldout_texts))
        deterministic_checks.append(
            all(
                fold_data_a[fold][key] == fold_data_b[fold][key]
                for key in ("base_hash", "train_hash", "validation_hash")
            )
        )
    deterministic_checks.append(records_hash(ambiguity_a) == records_hash(ambiguity_b))

    print(SEP, flush=True)
    print("[INFO] Loading frozen contextual substrate", flush=True)
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
        {
            record[0]
            for fold in FOLDS
            for key in ("train", "validation")
            for record in fold_data_a[fold][key]
        }
        | {
            record[0]
            for fold in FOLDS
            for record in heldout_records[fold]
        }
        | {record[0] for record in ambiguity_a}
    )
    print(f"[INFO] Extracting contextual features for {len(all_texts)} texts", flush=True)
    all_features = feature_backend.features(all_texts).cpu()
    feature_by_text = {
        text: all_features[index] for index, text in enumerate(all_texts)
    }

    fold_reports: List[Dict[str, Any]] = []
    fold_results: List[Dict[str, Any]] = []
    ambiguity_results: List[Dict[str, Any]] = []
    optimization_checks: List[bool] = []
    final_backend: CachedFactBackend | None = None
    for fold in FOLDS:
        print(SEP, flush=True)
        print(f"[INFO] Training fact curriculum fold {fold}/3", flush=True)
        head, optimization = train_head(
            feature_by_text,
            fold_data_a[fold]["train"],
            fold_data_a[fold]["validation"],
            entity_ids,
            attribute_ids,
            value_ids,
            device,
            results_dir / f"fold_{fold}_best_head.pt",
            TRAINING_SEED + fold,
        )
        backend = CachedFactBackend(
            feature_by_text,
            head,
            entity_ids,
            attribute_ids,
            value_ids,
            f"fact-fold-{fold}",
        )
        result = evaluate_records(f"F1_fact_fold_{fold}", heldout_records[fold], backend)
        ambiguity_result = evaluate_ambiguity(ambiguity_a, backend)
        fold_results.append(result)
        ambiguity_results.append(ambiguity_result)
        optimization_ok = (
            optimization["loss_drop"] >= 0.20
            and optimization["best_joint_accuracy"] >= 0.95
        )
        optimization_checks.append(optimization_ok)
        fold_reports.append(
            {
                "fold": fold,
                "holdout_ok": holdout_checks[fold],
                "deterministic": deterministic_checks[fold],
                "optimization": optimization,
                "result": result,
                "ambiguity": ambiguity_result,
            }
        )
        final_backend = backend
    aggregate = combine_results(fold_results)
    assert final_backend is not None
    ambiguity_total = sum(item["total"] for item in ambiguity_results)
    ambiguity_emitted = sum(item["emitted"] for item in ambiguity_results)
    ambiguity = {
        "total": ambiguity_total,
        "abstained": ambiguity_total - ambiguity_emitted,
        "abstain_rate": (ambiguity_total - ambiguity_emitted) / ambiguity_total,
        "emitted": ambiguity_emitted,
        "per_fold": [
            {key: value for key, value in item.items() if key != "details"}
            for item in ambiguity_results
        ],
        "details": [
            detail
            for item in ambiguity_results
            for detail in item["details"]
        ],
    }

    anti_adapter = ConservativeSemanticAdapter()
    anti_producer = ConservativeTrainedFactProducer(
        final_backend,
        anti_adapter,
        ENTITY_MARGIN,
        ATTRIBUTE_MARGIN,
        VALUE_MARGIN,
    )
    anti_record = next(
        record
        for record in heldout_records[FOLDS[-1]]
        if anti_producer.produce(
            "anti-probe", 1, record[0], provenance=("anti:probe",)
        ).emitted
    )
    anti_adapter = ConservativeSemanticAdapter()
    anti_producer = ConservativeTrainedFactProducer(
        final_backend,
        anti_adapter,
        ENTITY_MARGIN,
        ATTRIBUTE_MARGIN,
        VALUE_MARGIN,
    )
    first = anti_producer.produce("anti-1", 1, anti_record[0], provenance=("anti",))
    second = anti_producer.produce("anti-2", 1, anti_record[0], provenance=("anti",))
    third = anti_producer.produce("anti-3", 2, anti_record[0], provenance=("anti",))
    assert first.hypothesis is not None
    same_episode_count = anti_adapter.confirmation_count(
        str(first.hypothesis.entity_id),
        str(first.hypothesis.attr_type),
        str(first.hypothesis.value_id),
    )
    anti_inflation = (
        first.emitted and second.emitted and third.emitted and same_episode_count == 2
    )

    substrate_hash_after = state_tensor_hash(state)
    trainable_substrate = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    actual_seal_hashes = {
        name: sha256_file(path) for name, (path, _) in SEALED_FILES.items()
    }
    frozen_seal_hashes = {
        name: expected for name, (_, expected) in SEALED_FILES.items()
    }
    seals_untouched = actual_seal_hashes == frozen_seal_hashes
    all_adapter = (
        aggregate["adapter_accepted"] == aggregate["emitted"]
        and aggregate["provisional_only"] == aggregate["emitted"]
    )
    forbidden = ("commit", "committed", "bank", "writer", "consolidator")
    no_direct_commit = not any(
        hasattr(anti_producer, attribute) for attribute in forbidden
    ) and all(
        result.hypothesis is None
        or result.hypothesis.requested_destination
        == RequestedDestination.PROVISIONAL_ONLY
        for result in (first, second, third)
    )
    gates = [
        gate(
            "F0_FORM_HOLDOUT",
            all(holdout_checks),
            f"All evaluated fact forms absent from fold training={all(holdout_checks)}.",
            {"checks": holdout_checks},
        ),
        gate(
            "F1_REAL_OPTIMIZATION",
            all(optimization_checks),
            f"Every fold optimization gate passes={all(optimization_checks)}.",
            {"checks": optimization_checks},
        ),
        gate(
            "F2_FACT_OUT_OF_FOLD",
            aggregate["accuracy"] >= 0.85,
            f"F1 fact out-of-fold accuracy={aggregate['accuracy']:.1%} "
            f"({aggregate['correct']}/{aggregate['total']}).",
            aggregate,
        ),
        gate(
            "F3_WRONG_PROVISIONAL",
            aggregate["wrong_rate"] <= 0.02,
            f"Wrong provisional rate={aggregate['wrong_rate']:.1%} "
            f"({aggregate['wrong']}/{aggregate['total']}).",
            aggregate,
        ),
        gate(
            "F4_AMBIGUOUS_HONESTY",
            ambiguity["abstain_rate"] >= 0.95,
            f"Ambiguous/non-fact abstention={ambiguity['abstain_rate']:.1%} "
            f"({ambiguity['abstained']}/{ambiguity['total']}).",
            {key: value for key, value in ambiguity.items() if key != "details"},
        ),
        gate(
            "F5_ADAPTER_PROVISIONAL_ONLY",
            all_adapter,
            f"Every emission adapter-accepted provisional-only={all_adapter}.",
            {
                "emitted": aggregate["emitted"],
                "accepted": aggregate["adapter_accepted"],
                "provisional_only": aggregate["provisional_only"],
            },
        ),
        gate(
            "F6_NO_DIRECT_COMMIT",
            no_direct_commit,
            f"Producer exposes no direct commit path={no_direct_commit}.",
            {"forbidden_attributes": list(forbidden)},
        ),
        gate(
            "F7_FROZEN_SUBSTRATE",
            substrate_hash_before == substrate_hash_after and trainable_substrate == 0,
            f"Substrate byte-identical={substrate_hash_before == substrate_hash_after}; "
            f"trainable substrate parameters={trainable_substrate}.",
            {
                "before": substrate_hash_before,
                "after": substrate_hash_after,
                "trainable_substrate_parameters": trainable_substrate,
            },
        ),
        gate(
            "F8_DETERMINISTIC",
            all(deterministic_checks),
            f"Repeated curriculum and ambiguity construction exact={all(deterministic_checks)}.",
            {"checks": deterministic_checks},
        ),
        gate(
            "F9_SEALS_UNTOUCHED",
            seals_untouched,
            f"All Pas 7a/query-side sealed source hashes unchanged={seals_untouched}.",
            {"frozen": frozen_seal_hashes, "actual": actual_seal_hashes},
        ),
        gate(
            "F10_ANTI_INFLATION",
            anti_inflation,
            f"Same-episode repeats count once; distinct episode count={same_episode_count}.",
            {
                "same_episode_then_distinct_confirmation_count": same_episode_count,
                "all_emitted": first.emitted and second.emitted and third.emitted,
            },
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
            "fold_reports": fold_reports,
            "aggregate": aggregate,
            "ambiguity": ambiguity,
            "curriculum_hashes": {
                str(fold): {
                    key: fold_data_a[fold][key]
                    for key in ("base_hash", "train_hash", "validation_hash")
                }
                for fold in FOLDS
            },
            "ambiguity_hash": records_hash(ambiguity_a),
            "scope": (
                "Fact-side provisional semantic producer on sealed F1 forms. "
                "No Pas 7a ingestion, promotion, or committed-memory result."
            ),
            "claim_status": "MEASURED in one local environment; not PROVEN.",
            "all_pass": all_pass,
        },
    }
    verdict_path = results_dir / "verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    del payload

    print(SEP, flush=True)
    print("[INFO] D_Cortex v15.7b-F fact provisional verdict", flush=True)
    for item in gates:
        label = "✓ PASS" if item["passed"] else "✗ FAIL"
        print(f"{label} [{item['criterion_id']}] {item['evidence']}", flush=True)
    print(SEP, flush=True)
    print(f"[INFO] Overall: {'ALL GATES PASS' if all_pass else 'GATE FAILURE'}", flush=True)
    print(f"[INFO] Verdict: {verdict_path}", flush=True)
    return 0 if all_pass else 1


def build_argparser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    default_checkpoint = (
        REPO_ROOT.parent / "D_Cortex-main" / "runs" / "warmstart" / "warmstarted_init.pt"
    )
    parser = argparse.ArgumentParser(description="D_Cortex fact provisional curriculum")
    parser.add_argument("--checkpoint", default=str(default_checkpoint))
    parser.add_argument(
        "--run-dir",
        default=str(REPO_ROOT / "runs" / "semantic_fact_curriculum"),
    )
    return parser


def main() -> int:
    """CLI entry point."""
    return run(build_argparser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
