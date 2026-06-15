# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b trained pooled semantic-query internalizer.

import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Mapping, Sequence, Tuple

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
import torch.nn.functional as F

from dcortex.semantic_adapter import (
    ConservativeSemanticAdapter,
    DecisionStatus,
    HypothesisMode,
    RequestedDestination,
)
from dcortex.semantic_producer import (
    ConservativePrototypeProducer,
    ConservativeTrainedQueryProducer,
    DCortexPooledFeatureBackend,
    DCortexTokenEmbeddingBackend,
    PooledSemanticClassificationBackend,
    SemanticFeatureBackend,
    SemanticQueryHead,
)
from scripts.semantic_likelihood_probe import (
    ATTRIBUTE_CANDIDATES,
    CALIBRATION_AMBIGUOUS,
    ENC,
    EVALUATION_AMBIGUOUS,
    SEALED_SHA256,
    SEALED_SOURCE,
    TOKEN_MEAN_ATTRIBUTE_PROTOTYPES,
    gate,
    text_set_hash,
)

SEP = "=" * 70
TRAINING_SEED = 20261220
FAMILY_SPECS = {
    "F1": {"definition": "F1_QUERY_CONSTRUCTIONS", "seed": 20261221, "trials": 500},
    "F3": {"definition": "F3_NOVEL_ALIAS_QUERIES", "seed": 20261223, "trials": 500},
    "F5": {"definition": "F5_QUERY_FORMS", "seed": 20261225, "trials": 500},
}
AMBIGUOUS_SEED = 20261226
AMBIGUOUS_TRIALS = 200
ATTRIBUTE_MARGIN = 0.40
ENTITY_MARGIN = 0.00
HEAD_HIDDEN_DIM = 256
UNKNOWN_ENTITY = "UNKNOWN_ENTITY"
UNKNOWN_ATTRIBUTE = "UNKNOWN"
AMBIGUOUS_TEMPLATES = (
    "Offer unrestricted information concerning the {entity}.",
    "What broad facts are available about the {entity}?",
    "Recall the {entity} without selecting a particular trait.",
    "Give a neutral summary of the {entity}.",
    "Speak generally about the {entity}.",
    "What can be reported overall about the {entity}?",
    "Provide background concerning the {entity}.",
    "Mention anything known regarding the {entity}.",
    "Describe the {entity} in an open-ended way.",
    "Share general knowledge about the {entity}.",
)
GENERIC_UNKNOWN_TEXTS = (
    "Time passed quietly.",
    "A storm gathered nearby.",
    "Tell me a story.",
    "What happened next?",
    "Give me an overview.",
    "Describe this generally.",
)


class TokenEmbeddingModel(nn.Module):
    """Minimal frozen wrapper exposing the checkpoint token embeddings."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.shared_token_emb = nn.Embedding.from_pretrained(weight, freeze=True)


def seed_everything(seed: int) -> None:
    """Seed Python and Torch deterministically."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_tensor_hash(state: Mapping[str, torch.Tensor]) -> str:
    """Return a byte-level hash over all checkpoint model-state tensors."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        data = (
            tensor.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
        )
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(data)
    return digest.hexdigest()


def extract_definitions() -> Dict[str, Any]:
    """Extract the frozen training and verdict definitions from sealed Pas 7a."""
    required = {
        "V15_FACT_TEMPLATES",
        "V15_QUERY_TEMPLATES",
        "F4_DISTRACTOR_SENTENCES",
        "HOLDOUT_ENTITIES_SINGLE",
        "HOLDOUT_COLORS",
        "HOLDOUT_SIZES",
        "HOLDOUT_LOCATIONS",
        "HOLDOUT_STATES",
        "HOLDOUT_ATTR_VALUES",
        "HOLDOUT_ATTR_TYPES",
        "F1_QUERY_CONSTRUCTIONS",
        "F3_NOVEL_ALIAS_QUERIES",
        "F5_QUERY_FORMS",
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
    missing = sorted(required.difference(found))
    if missing:
        raise RuntimeError(f"Sealed definitions missing: {missing}")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: Dict[str, Any] = {}
    exec(compile(module, str(SEALED_SOURCE), "exec"), namespace)  # noqa: S102
    return {name: namespace[name] for name in required}


def find_entity(text: str, entity_ids: Sequence[str]) -> str:
    """Find the unique canonical entity occurring as a whole word."""
    found = [
        entity
        for entity in entity_ids
        if re.search(rf"\b{re.escape(entity)}\b", text.lower())
    ]
    return found[0] if len(found) == 1 else UNKNOWN_ENTITY


def sample_hash(examples: Sequence[Tuple[str, str, str, str]]) -> str:
    """Return a stable dataset hash."""
    payload = json.dumps(
        sorted(examples), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_base_training_examples(
    definitions: Mapping[str, Any],
    entity_ids: Sequence[str],
) -> Tuple[Tuple[str, str, str, str], ...]:
    """Build unique non-F1/F3/F5 semantic internalizer examples."""
    examples = set()
    for attribute, templates in definitions["V15_QUERY_TEMPLATES"].items():
        for entity in entity_ids:
            for template in templates:
                examples.add(
                    (template.format(e=entity), entity, attribute, "V15_QUERY_TEMPLATES")
                )
    for attribute, templates in definitions["V15_FACT_TEMPLATES"].items():
        for entity in entity_ids:
            for value in definitions["HOLDOUT_ATTR_VALUES"][attribute]:
                for template in templates:
                    examples.add(
                        (
                            template.format(e=entity, v=value),
                            entity,
                            attribute,
                            "V15_FACT_TEMPLATES",
                        )
                    )
    for text in tuple(CALIBRATION_AMBIGUOUS) + tuple(EVALUATION_AMBIGUOUS):
        examples.add(
            (
                text,
                find_entity(text, entity_ids),
                UNKNOWN_ATTRIBUTE,
                "OBSERVED_AMBIGUOUS_DEVELOPMENT",
            )
        )
    for text in definitions["F4_DISTRACTOR_SENTENCES"]:
        examples.add(
            (text, UNKNOWN_ENTITY, UNKNOWN_ATTRIBUTE, "F4_DISTRACTOR_SENTENCES")
        )
    for text in GENERIC_UNKNOWN_TEXTS:
        examples.add((text, UNKNOWN_ENTITY, UNKNOWN_ATTRIBUTE, "GENERIC_UNKNOWN"))
    return tuple(sorted(examples))


def split_and_balance(
    base: Sequence[Tuple[str, str, str, str]],
) -> Tuple[
    Tuple[Tuple[str, str, str, str], ...],
    Tuple[Tuple[str, str, str, str], ...],
]:
    """Hash-split unique data, then balance training attribute classes."""
    train_unique: List[Tuple[str, str, str, str]] = []
    validation: List[Tuple[str, str, str, str]] = []
    for example in base:
        digest = hashlib.sha256(
            f"{TRAINING_SEED}:{example[0]}".encode("utf-8")
        ).digest()
        (validation if int.from_bytes(digest[:4], "big") % 10 == 0 else train_unique).append(
            example
        )
    by_attribute: DefaultDict[str, List[Tuple[str, str, str, str]]] = defaultdict(list)
    for example in train_unique:
        by_attribute[example[2]].append(example)
    target = max(len(items) for items in by_attribute.values())
    balanced: List[Tuple[str, str, str, str]] = []
    for attribute in sorted(by_attribute):
        items = sorted(by_attribute[attribute])
        for index in range(target):
            balanced.append(items[index % len(items)])
    rng = random.Random(TRAINING_SEED)
    rng.shuffle(balanced)
    return tuple(balanced), tuple(sorted(validation))


def build_training_data(
    definitions: Mapping[str, Any],
    entity_ids: Sequence[str],
) -> Dict[str, Any]:
    """Build deterministic train/validation datasets."""
    base = build_base_training_examples(definitions, entity_ids)
    train, validation = split_and_balance(base)
    return {
        "base": base,
        "train": train,
        "validation": validation,
        "base_hash": sample_hash(base),
        "train_hash": sample_hash(train),
        "validation_hash": sample_hash(validation),
    }


def generate_family(
    definitions: Mapping[str, Any],
    family: str,
) -> Tuple[Tuple[str, str, str, str], ...]:
    """Generate one frozen AST-sourced external query family."""
    spec = FAMILY_SPECS[family]
    forms = definitions[spec["definition"]]
    rng = random.Random(spec["seed"])
    records: List[Tuple[str, str, str, str]] = []
    for _ in range(spec["trials"]):
        entity = rng.choice(definitions["HOLDOUT_ENTITIES_SINGLE"])
        attribute = rng.choice(definitions["HOLDOUT_ATTR_TYPES"])
        value = rng.choice(definitions["HOLDOUT_ATTR_VALUES"][attribute])
        builder = rng.choice(forms[attribute])
        text = builder(entity, value) if family == "F5" else builder(entity)
        records.append((text, entity, attribute, value))
    return tuple(records)


def generate_ambiguous(entity_ids: Sequence[str]) -> Tuple[Tuple[str, str], ...]:
    """Generate the frozen final attribute-unspecified query set."""
    rng = random.Random(AMBIGUOUS_SEED)
    return tuple(
        (
            rng.choice(AMBIGUOUS_TEMPLATES).format(
                entity=(entity := rng.choice(entity_ids))
            ),
            entity,
        )
        for _ in range(AMBIGUOUS_TRIALS)
    )


def tensorize_labels(
    examples: Sequence[Tuple[str, str, str, str]],
    entity_to_index: Mapping[str, int],
    attribute_to_index: Mapping[str, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert entity and attribute labels to tensors."""
    entity = torch.tensor([entity_to_index[item[1]] for item in examples])
    attribute = torch.tensor([attribute_to_index[item[2]] for item in examples])
    return entity, attribute


def atomic_save(payload: Dict[str, Any], path: Path) -> None:
    """Atomically save a Torch artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_head(
    feature_backend: SemanticFeatureBackend,
    train_examples: Sequence[Tuple[str, str, str, str]],
    validation_examples: Sequence[Tuple[str, str, str, str]],
    entity_ids: Sequence[str],
    attribute_ids: Sequence[str],
    device: torch.device,
    output_path: Path,
) -> Tuple[SemanticQueryHead, Dict[str, Any]]:
    """Train only the semantic head and return measured optimization history."""
    entity_to_index = {value: index for index, value in enumerate(entity_ids)}
    attribute_to_index = {value: index for index, value in enumerate(attribute_ids)}
    print("[INFO] Extracting frozen pooled features...", flush=True)
    train_features = feature_backend.features([item[0] for item in train_examples]).cpu()
    validation_features = feature_backend.features(
        [item[0] for item in validation_examples]
    ).cpu()
    train_entity, train_attribute = tensorize_labels(
        train_examples, entity_to_index, attribute_to_index
    )
    validation_entity, validation_attribute = tensorize_labels(
        validation_examples, entity_to_index, attribute_to_index
    )
    head = SemanticQueryHead(
        input_dim=feature_backend.output_dim,
        hidden_dim=HEAD_HIDDEN_DIM,
        entity_classes=len(entity_ids),
        attribute_classes=len(attribute_ids),
        dropout=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-3)
    batch_size = 256
    patience = 15
    best_loss = float("inf")
    best_accuracy = 0.0
    best_epoch = -1
    stale = 0
    history: List[Dict[str, float]] = []

    def evaluate() -> Tuple[float, float, float, float]:
        head.eval()
        with torch.no_grad():
            entity_logits, attribute_logits = head(validation_features.to(device))
            entity_loss = F.cross_entropy(entity_logits, validation_entity.to(device))
            attribute_loss = F.cross_entropy(
                attribute_logits, validation_attribute.to(device)
            )
            entity_accuracy = (
                entity_logits.argmax(dim=1) == validation_entity.to(device)
            ).float().mean()
            attribute_accuracy = (
                attribute_logits.argmax(dim=1) == validation_attribute.to(device)
            ).float().mean()
            joint_accuracy = (
                (entity_logits.argmax(dim=1) == validation_entity.to(device))
                & (attribute_logits.argmax(dim=1) == validation_attribute.to(device))
            ).float().mean()
        return (
            float(attribute_loss + 0.5 * entity_loss),
            float(entity_accuracy),
            float(attribute_accuracy),
            float(joint_accuracy),
        )

    initial_loss, _, _, _ = evaluate()
    generator = torch.Generator(device="cpu").manual_seed(TRAINING_SEED)
    for epoch in range(100):
        head.train()
        permutation = torch.randperm(len(train_examples), generator=generator)
        for start in range(0, len(train_examples), batch_size):
            indices = permutation[start : start + batch_size]
            features = train_features[indices].to(device)
            entity_target = train_entity[indices].to(device)
            attribute_target = train_attribute[indices].to(device)
            entity_logits, attribute_logits = head(features)
            loss = F.cross_entropy(
                attribute_logits, attribute_target, label_smoothing=0.02
            ) + 0.5 * F.cross_entropy(
                entity_logits, entity_target, label_smoothing=0.02
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        val_loss, entity_acc, attribute_acc, joint_acc = evaluate()
        history.append(
            {
                "epoch": epoch,
                "validation_loss": val_loss,
                "entity_accuracy": entity_acc,
                "attribute_accuracy": attribute_acc,
                "joint_accuracy": joint_acc,
            }
        )
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_accuracy = joint_acc
            best_epoch = epoch
            stale = 0
            atomic_save(
                {
                    "head": {key: value.detach().cpu() for key, value in head.state_dict().items()},
                    "entity_ids": list(entity_ids),
                    "attribute_ids": list(attribute_ids),
                    "input_dim": feature_backend.output_dim,
                    "hidden_dim": HEAD_HIDDEN_DIM,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                    "best_joint_accuracy": best_accuracy,
                },
                output_path,
            )
        else:
            stale += 1
        if epoch % 5 == 0 or stale >= patience:
            print(
                f"[INFO] Epoch {epoch:03d} | val_loss={val_loss:.4f} | "
                f"entity={entity_acc:.1%} | attr={attribute_acc:.1%} | "
                f"joint={joint_acc:.1%} | stale={stale}",
                flush=True,
            )
        if stale >= patience:
            break

    saved = torch.load(output_path, map_location="cpu", weights_only=False)
    head.load_state_dict(saved["head"])
    head.eval()
    final_loss, entity_acc, attribute_acc, joint_acc = evaluate()
    metrics = {
        "initial_validation_loss": initial_loss,
        "best_validation_loss": best_loss,
        "reloaded_validation_loss": final_loss,
        "loss_drop": (initial_loss - best_loss) / max(initial_loss, 1e-12),
        "best_epoch": best_epoch,
        "best_joint_accuracy": best_accuracy,
        "reloaded_entity_accuracy": entity_acc,
        "reloaded_attribute_accuracy": attribute_acc,
        "reloaded_joint_accuracy": joint_acc,
        "epochs_ran": len(history),
        "history": history,
    }
    return head, metrics


def evaluate_family(
    family: str,
    records: Sequence[Tuple[str, str, str, str]],
    producer_backend: PooledSemanticClassificationBackend,
    token_backend: DCortexTokenEmbeddingBackend,
    token_entity_prototypes: Mapping[str, Tuple[str, ...]],
) -> Dict[str, Any]:
    """Evaluate one frozen external query family."""
    correct = 0
    emitted = 0
    wrong = 0
    accepted = 0
    query_only = 0
    details: List[Dict[str, Any]] = []
    baseline_correct = 0
    baseline_emitted = 0
    for index, (text, expected_entity, expected_attribute, value) in enumerate(records):
        result = ConservativeTrainedQueryProducer(
            producer_backend,
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
        baseline = ConservativePrototypeProducer(
            token_backend,
            ConservativeSemanticAdapter(),
            similarity_threshold=0.55,
            margin_threshold=0.025,
        ).produce(
            f"baseline-{family}-{index}",
            10000 + index,
            HypothesisMode.QUERY,
            text,
            token_entity_prototypes,
            TOKEN_MEAN_ATTRIBUTE_PROTOTYPES,
            provenance=(f"baseline:{family}:{index}",),
        )
        baseline_emitted += int(baseline.emitted)
        baseline_correct += int(
            baseline.hypothesis is not None
            and baseline.hypothesis.entity_id == expected_entity
            and baseline.hypothesis.attr_type == expected_attribute
        )
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
        "baseline_correct": baseline_correct,
        "baseline_accuracy": baseline_correct / total,
        "baseline_emitted": baseline_emitted,
        "uplift": (correct - baseline_correct) / total,
        "details": details,
    }


def run(args: argparse.Namespace) -> int:
    """Train and evaluate the frozen semantic internalizer cycle."""
    checkpoint = Path(args.checkpoint)
    results_dir = Path(args.run_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint}")
    seed_everything(TRAINING_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    definitions = extract_definitions()
    actual_sealed_hash = hashlib.sha256(SEALED_SOURCE.read_bytes()).hexdigest()
    entity_ids = tuple(sorted(definitions["HOLDOUT_ENTITIES_SINGLE"])) + (
        UNKNOWN_ENTITY,
    )
    attribute_ids = tuple(definitions["HOLDOUT_ATTR_TYPES"]) + (UNKNOWN_ATTRIBUTE,)
    training_data_a = build_training_data(
        definitions, definitions["HOLDOUT_ENTITIES_SINGLE"]
    )
    training_data_b = build_training_data(
        definitions, definitions["HOLDOUT_ENTITIES_SINGLE"]
    )
    deterministic_data = {
        key: training_data_a[key] == training_data_b[key]
        for key in ("base_hash", "train_hash", "validation_hash")
    }
    family_records = {
        family: generate_family(definitions, family) for family in FAMILY_SPECS
    }
    ambiguous_records = generate_ambiguous(definitions["HOLDOUT_ENTITIES_SINGLE"])
    training_texts = {item[0] for item in training_data_a["base"]}
    verdict_texts = {
        text
        for records in family_records.values()
        for text, _, _, _ in records
    }.union(text for text, _ in ambiguous_records)
    overlap = sorted(training_texts.intersection(verdict_texts))
    forbidden_sources = sorted(
        {
            item[3]
            for item in training_data_a["base"]
            if item[3] in {"F1_QUERY_CONSTRUCTIONS", "F3_NOVEL_ALIAS_QUERIES", "F5_QUERY_FORMS"}
        }
    )

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
        tokenizer=ENC.encode_ordinary,
        max_seq_len=128,
        batch_size=256,
        backend_version=checkpoint.name,
    )
    head_path = results_dir / "best_head.pt"
    head, optimization = train_head(
        feature_backend,
        training_data_a["train"],
        training_data_a["validation"],
        entity_ids,
        attribute_ids,
        device,
        head_path,
    )
    classification_backend = PooledSemanticClassificationBackend(
        feature_backend=feature_backend,
        head=head,
        entity_ids=entity_ids,
        attribute_ids=attribute_ids,
        unknown_entity_id=UNKNOWN_ENTITY,
        unknown_attribute_id=UNKNOWN_ATTRIBUTE,
        backend_version=f"trained:{head_path.name}",
    )
    token_backend = DCortexTokenEmbeddingBackend(
        model=model,
        tokenizer=ENC.encode_ordinary,
        max_seq_len=128,
        backend_version=checkpoint.name,
    )
    token_entity_prototypes = {
        entity: (entity,) for entity in definitions["HOLDOUT_ENTITIES_SINGLE"]
    }
    family_results = {
        family: evaluate_family(
            family,
            records,
            classification_backend,
            token_backend,
            token_entity_prototypes,
        )
        for family, records in family_records.items()
    }

    ambiguous_details: List[Dict[str, Any]] = []
    abstained = 0
    for index, (text, expected_entity) in enumerate(ambiguous_records):
        result = ConservativeTrainedQueryProducer(
            classification_backend,
            ConservativeSemanticAdapter(),
            entity_margin_threshold=ENTITY_MARGIN,
            attribute_margin_threshold=ATTRIBUTE_MARGIN,
        ).produce(
            f"ambiguous-{index}",
            20000 + index,
            text,
            provenance=(f"ambiguous_seed:{AMBIGUOUS_SEED}", f"trial:{index}"),
        )
        abstained += int(not result.emitted)
        ambiguous_details.append(
            {
                "text": text,
                "expected_entity": expected_entity,
                "emitted": result.emitted,
                "reason_codes": list(result.reason_codes),
                "scores": [score.to_dict() for score in result.scores],
            }
        )
    ambiguous_abstention = abstained / len(ambiguous_records)
    substrate_hash_after = state_tensor_hash(state)
    substrate_trainable = sum(parameter.requires_grad for parameter in model.parameters())
    head_trainable = sum(parameter.requires_grad for parameter in head.parameters())
    del payload

    all_family_accuracy = all(
        result["accuracy"] >= 0.85 for result in family_results.values()
    )
    all_family_wrong = all(
        result["wrong_rate"] <= 0.02 for result in family_results.values()
    )
    all_adapter = all(
        result["adapter_accepted"] == result["emitted"]
        and result["query_only"] == result["emitted"]
        for result in family_results.values()
    )
    all_uplift = all(result["uplift"] >= 0.30 for result in family_results.values())
    gates = [
        gate(
            "T0_TRAINING_SEPARATION",
            not forbidden_sources and not overlap,
            f"Forbidden training sources={forbidden_sources}; exact train/verdict "
            f"text overlap={len(overlap)}.",
            {
                "forbidden_sources": forbidden_sources,
                "overlap": overlap,
                "training_text_hash": text_set_hash(sorted(training_texts)),
                "verdict_text_hash": text_set_hash(sorted(verdict_texts)),
            },
        ),
        gate(
            "T1_REAL_OPTIMIZATION",
            optimization["loss_drop"] >= 0.20
            and optimization["best_joint_accuracy"] >= 0.95,
            f"Validation loss drop={optimization['loss_drop']:.1%}; best joint "
            f"accuracy={optimization['best_joint_accuracy']:.1%}.",
            optimization,
        ),
        gate(
            "T2_DETERMINISTIC_DATA",
            all(deterministic_data.values()),
            f"Repeated dataset hashes exact={all(deterministic_data.values())}.",
            {
                "checks": deterministic_data,
                "base_count": len(training_data_a["base"]),
                "train_count": len(training_data_a["train"]),
                "validation_count": len(training_data_a["validation"]),
                "base_hash": training_data_a["base_hash"],
                "train_hash": training_data_a["train_hash"],
                "validation_hash": training_data_a["validation_hash"],
            },
        ),
        gate(
            "T3_F1_QUERY_INTERPRETATION",
            family_results["F1"]["accuracy"] >= 0.85,
            f"F1 total accuracy={family_results['F1']['accuracy']:.1%} "
            f"({family_results['F1']['correct']}/{family_results['F1']['total']}).",
            {key: value for key, value in family_results["F1"].items() if key != "details"},
        ),
        gate(
            "T4_F3_QUERY_INTERPRETATION",
            family_results["F3"]["accuracy"] >= 0.85,
            f"F3 total accuracy={family_results['F3']['accuracy']:.1%} "
            f"({family_results['F3']['correct']}/{family_results['F3']['total']}).",
            {key: value for key, value in family_results["F3"].items() if key != "details"},
        ),
        gate(
            "T5_F5_QUERY_INTERPRETATION",
            family_results["F5"]["accuracy"] >= 0.85,
            f"F5 total accuracy={family_results['F5']['accuracy']:.1%} "
            f"({family_results['F5']['correct']}/{family_results['F5']['total']}).",
            {key: value for key, value in family_results["F5"].items() if key != "details"},
        ),
        gate(
            "T6_WRONG_INTERPRETATION",
            all_family_wrong,
            "Wrong emitted interpretation rates: "
            + ", ".join(
                f"{family}={result['wrong_rate']:.1%}"
                for family, result in family_results.items()
            ),
            {
                family: {"wrong": result["wrong"], "wrong_rate": result["wrong_rate"]}
                for family, result in family_results.items()
            },
        ),
        gate(
            "T7_AMBIGUOUS_HONESTY",
            ambiguous_abstention >= 0.80,
            f"Attribute-unspecified abstention={ambiguous_abstention:.1%} "
            f"({abstained}/{len(ambiguous_records)}).",
            {
                "abstained": abstained,
                "total": len(ambiguous_records),
                "abstain_rate": ambiguous_abstention,
            },
        ),
        gate(
            "T8_ADAPTER_REQUIRED",
            all_adapter,
            f"Every family routed all emissions through query-only adapter={all_adapter}.",
            {
                family: {
                    "emitted": result["emitted"],
                    "accepted": result["adapter_accepted"],
                    "query_only": result["query_only"],
                }
                for family, result in family_results.items()
            },
        ),
        gate(
            "T9_FROZEN_SUBSTRATE",
            substrate_hash_before == substrate_hash_after
            and substrate_trainable == 0
            and head_trainable > 0,
            f"Substrate state byte-identical={substrate_hash_before == substrate_hash_after}; "
            f"substrate trainable={substrate_trainable}; head trainable={head_trainable}.",
            {
                "before": substrate_hash_before,
                "after": substrate_hash_after,
                "substrate_trainable_parameters": substrate_trainable,
                "head_trainable_parameters": head_trainable,
            },
        ),
        gate(
            "T10_BASELINE_UPLIFT",
            all_uplift,
            "Internalizer minus frozen token-mean accuracy: "
            + ", ".join(
                f"{family}={result['uplift']:+.1%}"
                for family, result in family_results.items()
            ),
            {
                family: {
                    "internalizer_accuracy": result["accuracy"],
                    "baseline_accuracy": result["baseline_accuracy"],
                    "uplift": result["uplift"],
                }
                for family, result in family_results.items()
            },
        ),
        gate(
            "T11_SEALED_UNTOUCHED",
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
            "training_seed": TRAINING_SEED,
            "entity_ids": list(entity_ids),
            "attribute_ids": list(attribute_ids),
            "entity_margin": ENTITY_MARGIN,
            "attribute_margin": ATTRIBUTE_MARGIN,
            "head_hidden_dim": HEAD_HIDDEN_DIM,
            "training_data": {
                "base_count": len(training_data_a["base"]),
                "train_count": len(training_data_a["train"]),
                "validation_count": len(training_data_a["validation"]),
                "base_hash": training_data_a["base_hash"],
                "train_hash": training_data_a["train_hash"],
                "validation_hash": training_data_a["validation_hash"],
            },
            "optimization": optimization,
            "family_results": family_results,
            "ambiguous_details": ambiguous_details,
            "scope": (
                "Trained query-side semantic internalizer over frozen D_Cortex "
                "token embeddings. Not fact-side internalization, end-to-end "
                "F1/F3/F5 commit improvement, or semantic-memory advantage."
            ),
            "claim_status": "MEASURED in one local environment; not PROVEN.",
            "all_pass": all_pass,
        },
    }
    verdict_path = results_dir / "verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(SEP, flush=True)
    print("[INFO] D_Cortex v15.7b trained pooled semantic internalizer", flush=True)
    print(f"[INFO] Checkpoint: {checkpoint}", flush=True)
    print(f"[INFO] Device: {device}", flush=True)
    print(SEP, flush=True)
    for item in gates:
        label = "✓ PASS" if item["passed"] else "✗ FAIL"
        print(f"{label} [{item['criterion_id']}] {item['evidence']}", flush=True)
    print(SEP, flush=True)
    print(f"[INFO] Overall: {'ALL GATES PASS' if all_pass else 'GATE FAILURE'}", flush=True)
    print(f"[INFO] Head: {head_path}", flush=True)
    print(f"[INFO] Verdict: {verdict_path}", flush=True)
    return 0 if all_pass else 1


def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    default_checkpoint = (
        REPO_ROOT.parent / "D_Cortex-main" / "runs" / "warmstart" / "warmstarted_init.pt"
    )
    parser = argparse.ArgumentParser(description="Train D_Cortex semantic internalizer")
    parser.add_argument("--checkpoint", type=str, default=str(default_checkpoint))
    parser.add_argument(
        "--run-dir",
        type=str,
        default=str(REPO_ROOT / "runs" / "semantic_internalizer"),
    )
    return parser


def main() -> int:
    """CLI entry point."""
    return run(build_argparser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
