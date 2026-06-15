# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b-RB1 frozen conservative learned role-binder verdict.

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
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

import tiktoken
import torch
import torch.nn.functional as F

from dcortex.semantic_adapter import (
    ConservativeSemanticAdapter,
    DecisionStatus,
    RequestedDestination,
)
from dcortex.semantic_producer import DCortexContextualFeatureBackend
from dcortex.semantic_role_binder import (
    ASSIGNMENT_ORDER,
    ConservativeLearnedRoleBinder,
    RoleBindingAssignment,
    RoleBindingScoringBackend,
    RoleBindingScoringHead,
    candidate_views,
    expected_assignment,
)
from scripts.semantic_contextual_curriculum import load_contextual_model
from scripts.semantic_likelihood_probe import SEALED_SHA256, SEALED_SOURCE, gate
from scripts.semantic_role_binding_benchmark import (
    BASELINES,
    RoleBindingRecord,
    sample_hash,
    score_baseline,
    sha256_file,
)
from scripts.train_semantic_internalizer import seed_everything, state_tensor_hash

SEP = "=" * 70
TRAINING_SEED = 20261530
HEAD_HIDDEN_DIM = 256
MARGIN_CANDIDATES = tuple(round(index * 0.05, 2) for index in range(11))
RB0_SAMPLE_SHA256 = "2c4a2dd117535b6ee7929bbb1c9882eddd95ab50fd97c4b1a343a9c196fd3625"
RB0_VERDICT_SHA256 = "c4dcd47d471d679fa20e78e178943599d2cda383e9fbcc23b5bcde39fe1bb876"
RB0_SEMANTIC_SAMPLE_HASH = "7e1d681c84ceb728fa92cf04e7c463605fd1e9a2af720e01875de45d843a956a"
QUERY_SIDE_SEALS = {
    "dcortex/semantic_adapter.py": "719afbe8359af4ee98006e0fa19f169a0a40f22ee0cc41e5dc1459d6e07e605e",
    "dcortex/semantic_producer.py": "24797a121c90f78161518362f0401bbc1b161546e87bac194adb4144f4aa67d0",
    "scripts/semantic_contextual_curriculum.py": "bab8ec1268d5000d5a9da5ae5946d29946822efa9a6b0b3ce53be59052631b57",
    "dcortex/semantic_query_bridge.py": "403d4d724a1bffee61ab9cdfa469adb0c4fb3afb75c04ad4d65ad3e7c86e1b43",
}


class FixedRecordScoreBackend(RoleBindingScoringBackend):
    """Evaluation backend carrying one measured three-candidate score vector."""

    backend_id = "measured_contextual_role_binding_scores"
    backend_version = "RB1"

    def __init__(self, scores: torch.Tensor) -> None:
        self.scores = scores.detach().float().cpu()

    def score(self, views: Sequence[str]) -> torch.Tensor:
        """Return the measured score vector for the current record."""
        if len(views) != len(ASSIGNMENT_ORDER):
            raise ValueError("exactly three candidate views are required")
        return self.scores


def record_from_dict(data: Mapping[str, Any]) -> RoleBindingRecord:
    """Reconstruct one sealed RB0 role-binding record."""
    return RoleBindingRecord(
        family=str(data["family"]),
        index=int(data["index"]),
        text=str(data["text"]),
        attribute=str(data["attribute"]),
        entities=tuple(str(item) for item in data["entities"]),
        values=tuple(str(item) for item in data["values"]),
        expected=tuple(tuple(str(part) for part in item) for item in data["expected"]),
        ambiguous=bool(data["ambiguous"]),
    )


def load_rb0_sample(path: Path) -> Tuple[RoleBindingRecord, ...]:
    """Load the sealed predecessor sample without rebuilding it."""
    return tuple(
        record_from_dict(item)
        for item in json.loads(path.read_text(encoding="utf-8"))
    )


def split_name(text: str) -> str:
    """Return the frozen deterministic 70/15/15 text-hash split name."""
    bucket = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big") % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def split_records(
    records: Sequence[RoleBindingRecord],
) -> Dict[str, Tuple[RoleBindingRecord, ...]]:
    """Split records deterministically by exact source-text hash."""
    result: Dict[str, List[RoleBindingRecord]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for record in records:
        result[split_name(record.text)].append(record)
    return {name: tuple(items) for name, items in result.items()}


def assignment_index(record: RoleBindingRecord) -> int:
    """Return the fixed candidate index matching one record's truth."""
    assignment = expected_assignment(
        record.attribute,
        record.entities,
        record.values,
        record.expected,
    )
    return ASSIGNMENT_ORDER.index(assignment)


def build_feature_texts(
    records: Sequence[RoleBindingRecord],
) -> Tuple[str, ...]:
    """Flatten the three fixed candidate views for each record."""
    return tuple(
        view
        for record in records
        for view in candidate_views(
            record.text, record.attribute, record.entities, record.values
        )
    )


def atomic_save(payload: Dict[str, Any], path: Path) -> None:
    """Atomically save a Torch artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_head(
    all_features: torch.Tensor,
    records: Sequence[RoleBindingRecord],
    splits: Mapping[str, Sequence[RoleBindingRecord]],
    record_to_index: Mapping[str, int],
    device: torch.device,
    output_path: Path,
) -> Tuple[RoleBindingScoringHead, Dict[str, Any]]:
    """Train only the scalar role-binding head and report real optimization."""
    seed_everything(TRAINING_SEED)
    labels = torch.tensor([assignment_index(record) for record in records], dtype=torch.long)
    train_indices = torch.tensor(
        [record_to_index[record.text] for record in splits["train"]], dtype=torch.long
    )
    validation_indices = torch.tensor(
        [record_to_index[record.text] for record in splits["validation"]], dtype=torch.long
    )
    train_features = all_features[train_indices]
    validation_features = all_features[validation_indices]
    train_labels = labels[train_indices]
    validation_labels = labels[validation_indices]
    head = RoleBindingScoringHead(
        input_dim=all_features.shape[-1],
        hidden_dim=HEAD_HIDDEN_DIM,
        dropout=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-3)
    batch_size = 128
    patience = 15
    best_loss = float("inf")
    best_accuracy = 0.0
    best_epoch = -1
    stale = 0
    history: List[Dict[str, float]] = []

    def logits_for(features: torch.Tensor) -> torch.Tensor:
        flat = features.reshape(-1, features.shape[-1]).to(device)
        return head(flat).reshape(-1, len(ASSIGNMENT_ORDER))

    def evaluate() -> Tuple[float, float]:
        head.eval()
        with torch.no_grad():
            logits = logits_for(validation_features)
            loss = F.cross_entropy(logits, validation_labels.to(device))
            accuracy = (logits.argmax(dim=1) == validation_labels.to(device)).float().mean()
        return float(loss), float(accuracy)

    initial_loss, initial_accuracy = evaluate()
    generator = torch.Generator(device="cpu").manual_seed(TRAINING_SEED)
    for epoch in range(100):
        head.train()
        permutation = torch.randperm(len(train_indices), generator=generator)
        for start in range(0, len(train_indices), batch_size):
            indices = permutation[start : start + batch_size]
            logits = logits_for(train_features[indices])
            loss = F.cross_entropy(
                logits,
                train_labels[indices].to(device),
                label_smoothing=0.02,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
        validation_loss, validation_accuracy = evaluate()
        history.append(
            {
                "epoch": epoch,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_accuracy = validation_accuracy
            best_epoch = epoch
            stale = 0
            atomic_save(
                {
                    "head": {
                        key: value.detach().cpu()
                        for key, value in head.state_dict().items()
                    },
                    "input_dim": int(all_features.shape[-1]),
                    "hidden_dim": HEAD_HIDDEN_DIM,
                    "training_seed": TRAINING_SEED,
                    "best_epoch": best_epoch,
                    "validation_loss": best_loss,
                    "validation_accuracy": best_accuracy,
                },
                output_path,
            )
        else:
            stale += 1
            if stale >= patience:
                break
    artifact = torch.load(output_path, map_location="cpu", weights_only=False)
    head.load_state_dict(artifact["head"])
    head.eval()
    loss_drop = (initial_loss - best_loss) / max(initial_loss, 1e-12)
    return head, {
        "initial_validation_loss": initial_loss,
        "initial_validation_accuracy": initial_accuracy,
        "best_validation_loss": best_loss,
        "best_validation_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "loss_drop": loss_drop,
        "history": history,
        "artifact": str(output_path),
        "artifact_sha256": sha256_file(output_path),
    }


def score_all(
    head: RoleBindingScoringHead,
    features: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return measured three-candidate logits for every record."""
    head.eval()
    with torch.inference_mode():
        logits = head(features.reshape(-1, features.shape[-1]).to(device))
    return logits.reshape(-1, len(ASSIGNMENT_ORDER)).detach().float().cpu()


def select_margin_threshold(
    records: Sequence[RoleBindingRecord],
    logits: torch.Tensor,
) -> Dict[str, Any]:
    """Select the smallest frozen-grid margin meeting validation honesty."""
    probabilities = torch.softmax(logits.float(), dim=1)
    top_probability, top_index = probabilities.max(dim=1)
    second_probability = probabilities.topk(2, dim=1).values[:, 1]
    margins = top_probability - second_probability
    reports = []
    selected = None
    for threshold in MARGIN_CANDIDATES:
        ambiguous_indices = [
            index for index, record in enumerate(records) if record.ambiguous
        ]
        overcommit = sum(
            int(
                int(top_index[index]) != ASSIGNMENT_ORDER.index(RoleBindingAssignment.UNRESOLVED)
                and float(margins[index]) >= threshold
            )
            for index in ambiguous_indices
        )
        rate = overcommit / len(ambiguous_indices)
        reports.append(
            {
                "threshold": threshold,
                "ambiguous_n": len(ambiguous_indices),
                "ambiguous_overcommit": overcommit,
                "ambiguous_overcommit_rate": rate,
            }
        )
        if selected is None and rate <= 0.02:
            selected = threshold
    return {
        "threshold": MARGIN_CANDIDATES[-1] if selected is None else selected,
        "target_met": selected is not None,
        "grid": list(MARGIN_CANDIDATES),
        "reports": reports,
    }


def evaluate_split(
    records: Sequence[RoleBindingRecord],
    logits: torch.Tensor,
    margin_threshold: float,
    episode_offset: int,
) -> Dict[str, Any]:
    """Evaluate the conservative binder through the semantic adapter."""
    known = 0
    known_exact = 0
    known_wrong = 0
    known_abstained = 0
    ambiguous = 0
    ambiguous_abstained = 0
    emitted_mappings = 0
    emitted_facts = 0
    adapter_accepted = 0
    provisional_only = 0
    details = []
    per_family: Dict[str, Dict[str, int]] = {}
    for index, (record, record_logits) in enumerate(zip(records, logits)):
        result = ConservativeLearnedRoleBinder(
            FixedRecordScoreBackend(record_logits),
            ConservativeSemanticAdapter(),
            margin_threshold=margin_threshold,
        ).produce(
            f"rb1-{episode_offset + index}",
            episode_offset + index,
            record.text,
            record.attribute,
            record.entities,
            record.values,
            provenance=(f"sealed_rb0:{record.family}:{record.index}",),
        )
        family = per_family.setdefault(
            record.family,
            {"n": 0, "exact": 0, "wrong": 0, "abstained": 0, "emitted": 0},
        )
        family["n"] += 1
        emitted_mappings += int(result.emitted)
        emitted_facts += len(result.hypotheses)
        adapter_accepted += sum(
            decision.status == DecisionStatus.ACCEPT_PROVISIONAL
            for decision in result.adapter_decisions
        )
        provisional_only += sum(
            hypothesis.requested_destination == RequestedDestination.PROVISIONAL_ONLY
            for hypothesis in result.hypotheses
        )
        exact = result.emitted and result.facts == record.expected
        if record.ambiguous:
            ambiguous += 1
            ambiguous_abstained += int(not result.emitted)
            family["abstained"] += int(not result.emitted)
        else:
            known += 1
            known_exact += int(exact)
            known_wrong += int(result.emitted and not exact)
            known_abstained += int(not result.emitted)
            family["exact"] += int(exact)
            family["wrong"] += int(result.emitted and not exact)
            family["abstained"] += int(not result.emitted)
        family["emitted"] += int(result.emitted)
        details.append(
            {
                "family": record.family,
                "index": record.index,
                "text": record.text,
                "ambiguous": record.ambiguous,
                "expected": [list(item) for item in record.expected],
                "result": result.to_dict(),
                "exact": exact,
            }
        )
    return {
        "known_n": known,
        "known_exact": known_exact,
        "known_exact_rate": known_exact / known,
        "known_wrong": known_wrong,
        "known_wrong_rate": known_wrong / known,
        "known_abstained": known_abstained,
        "ambiguous_n": ambiguous,
        "ambiguous_abstained": ambiguous_abstained,
        "ambiguous_abstention_rate": ambiguous_abstained / ambiguous,
        "emitted_mappings": emitted_mappings,
        "emitted_facts": emitted_facts,
        "adapter_accepted": adapter_accepted,
        "provisional_only": provisional_only,
        "per_family": per_family,
        "details": details,
    }


def split_summary(splits: Mapping[str, Sequence[RoleBindingRecord]]) -> Dict[str, Any]:
    """Return auditable split counts and text hashes."""
    result: Dict[str, Any] = {}
    for name, records in splits.items():
        result[name] = {
            "n": len(records),
            "family_counts": dict(Counter(record.family for record in records)),
            "text_hash": hashlib.sha256(
                json.dumps(
                    sorted(record.text for record in records),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
    return result


def run(args: argparse.Namespace) -> int:
    """Run the frozen conservative learned role-binder verdict."""
    checkpoint = Path(args.checkpoint)
    sample_path = Path(args.sample)
    rb0_verdict_path = Path(args.rb0_verdict)
    results_dir = Path(args.run_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint}")
    if not sample_path.exists() or not rb0_verdict_path.exists():
        raise RuntimeError("Sealed RB0 predecessor artifacts are required")

    records = load_rb0_sample(sample_path)
    splits = split_records(records)
    split_texts = {
        name: {record.text for record in subset} for name, subset in splits.items()
    }
    split_separation = (
        not split_texts["train"].intersection(split_texts["validation"])
        and not split_texts["train"].intersection(split_texts["test"])
        and not split_texts["validation"].intersection(split_texts["test"])
        and sum(len(items) for items in splits.values()) == len(records)
    )
    record_to_index = {record.text: index for index, record in enumerate(records)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, state, payload = load_contextual_model(checkpoint, device)
    substrate_hash_before = state_tensor_hash(state)
    feature_backend = DCortexContextualFeatureBackend(
        model=model,
        tokenizer=tiktoken.get_encoding("gpt2").encode_ordinary,
        max_seq_len=128,
        batch_size=128,
        backend_version=checkpoint.name,
    )
    print("[INFO] Extracting frozen contextual candidate-view features...", flush=True)
    flat_features = feature_backend.features(build_feature_texts(records)).cpu()
    all_features = flat_features.reshape(
        len(records), len(ASSIGNMENT_ORDER), feature_backend.output_dim
    )
    head, optimization = train_head(
        all_features,
        records,
        splits,
        record_to_index,
        device,
        results_dir / "best_role_binding_head.pt",
    )
    all_logits = score_all(head, all_features, device)
    validation_indices = [record_to_index[item.text] for item in splits["validation"]]
    test_indices = [record_to_index[item.text] for item in splits["test"]]
    calibration = select_margin_threshold(
        splits["validation"], all_logits[validation_indices]
    )
    margin_threshold = float(calibration["threshold"])
    test_result = evaluate_split(
        splits["test"],
        all_logits[test_indices],
        margin_threshold,
        episode_offset=210000,
    )
    test_baselines = {
        name: score_baseline(splits["test"], baseline)
        for name, baseline in BASELINES.items()
    }
    best_test_lexical = max(
        test_baselines[name]["known_exact_rate"]
        for name in (
            "ordered_first_occurrence",
            "minimum_distance",
            "lexical_cartesian",
        )
    )
    lexical_uplift = test_result["known_exact_rate"] - best_test_lexical
    substrate_hash_after = state_tensor_hash(state)
    trainable_substrate = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    rb0_hashes = {
        "sample_file_actual": sha256_file(sample_path),
        "sample_file_expected": RB0_SAMPLE_SHA256,
        "verdict_file_actual": sha256_file(rb0_verdict_path),
        "verdict_file_expected": RB0_VERDICT_SHA256,
        "semantic_sample_actual": sample_hash(records),
        "semantic_sample_expected": RB0_SEMANTIC_SAMPLE_HASH,
    }
    rb0_preserved = (
        rb0_hashes["sample_file_actual"] == rb0_hashes["sample_file_expected"]
        and rb0_hashes["verdict_file_actual"] == rb0_hashes["verdict_file_expected"]
        and rb0_hashes["semantic_sample_actual"] == rb0_hashes["semantic_sample_expected"]
    )
    seal_hashes = {
        path: {
            "expected": expected,
            "actual": sha256_file(REPO_ROOT / path),
        }
        for path, expected in QUERY_SIDE_SEALS.items()
    }
    seal_hashes[str(SEALED_SOURCE.relative_to(REPO_ROOT))] = {
        "expected": SEALED_SHA256,
        "actual": sha256_file(SEALED_SOURCE),
    }
    seals_untouched = all(
        item["actual"] == item["expected"] for item in seal_hashes.values()
    )
    adapter_safe = (
        test_result["adapter_accepted"] == test_result["emitted_facts"]
        and test_result["provisional_only"] == test_result["emitted_facts"]
        and test_result["emitted_facts"] == 2 * test_result["emitted_mappings"]
    )
    forbidden_paths = ("commit", "write", "consolidate", "promote")
    no_direct_commit = (
        not any(hasattr(ConservativeLearnedRoleBinder, name) for name in forbidden_paths)
        and adapter_safe
    )
    del payload

    gates = [
        gate(
            "J0_RB0_PRESERVED",
            rb0_preserved,
            f"Sealed RB0 predecessor artifacts and semantic sample unchanged={rb0_preserved}.",
            rb0_hashes,
        ),
        gate(
            "J1_SPLIT_SEPARATION",
            split_separation,
            f"Deterministic train/validation/test texts disjoint={split_separation}.",
            split_summary(splits),
        ),
        gate(
            "J2_REAL_OPTIMIZATION",
            optimization["loss_drop"] >= 0.20,
            f"Validation loss drop={optimization['loss_drop']:.1%}; "
            f"{optimization['initial_validation_loss']:.4f} -> "
            f"{optimization['best_validation_loss']:.4f}.",
            optimization,
        ),
        gate(
            "J3_TEST_EXACT_BINDING",
            test_result["known_exact_rate"] >= 0.70,
            f"Known test exact binding={test_result['known_exact_rate']:.1%} "
            f"({test_result['known_exact']}/{test_result['known_n']}).",
            {key: value for key, value in test_result.items() if key != "details"},
        ),
        gate(
            "J4_TEST_WRONG_MAPPING",
            test_result["known_wrong_rate"] <= 0.05,
            f"Wrong emitted mapping rate={test_result['known_wrong_rate']:.1%} "
            f"({test_result['known_wrong']}/{test_result['known_n']}).",
            {
                "known_wrong": test_result["known_wrong"],
                "known_n": test_result["known_n"],
                "known_wrong_rate": test_result["known_wrong_rate"],
            },
        ),
        gate(
            "J5_AMBIGUOUS_HONESTY",
            test_result["ambiguous_abstention_rate"] >= 0.95,
            f"Ambiguous test abstention={test_result['ambiguous_abstention_rate']:.1%} "
            f"({test_result['ambiguous_abstained']}/{test_result['ambiguous_n']}); "
            f"validation calibration target met={calibration['target_met']}.",
            {
                "calibration": calibration,
                "test_ambiguous_n": test_result["ambiguous_n"],
                "test_ambiguous_abstained": test_result["ambiguous_abstained"],
                "test_ambiguous_abstention_rate": test_result[
                    "ambiguous_abstention_rate"
                ],
            },
        ),
        gate(
            "J6_LEXICAL_UPLIFT",
            lexical_uplift >= 0.25,
            f"Known test exact={test_result['known_exact_rate']:.1%}; best same-test "
            f"lexical={best_test_lexical:.1%}; uplift={lexical_uplift:+.1%}.",
            {
                "producer_exact_rate": test_result["known_exact_rate"],
                "best_test_lexical": best_test_lexical,
                "uplift": lexical_uplift,
                "test_baselines": test_baselines,
            },
        ),
        gate(
            "J7_ADAPTER_PROVISIONAL_ONLY",
            adapter_safe,
            f"Adapter-accepted provisional facts={test_result['adapter_accepted']}/"
            f"{test_result['emitted_facts']}; exactly two per emitted mapping="
            f"{test_result['emitted_facts'] == 2 * test_result['emitted_mappings']}.",
            {
                "emitted_mappings": test_result["emitted_mappings"],
                "emitted_facts": test_result["emitted_facts"],
                "adapter_accepted": test_result["adapter_accepted"],
                "provisional_only": test_result["provisional_only"],
            },
        ),
        gate(
            "J8_FROZEN_SUBSTRATE",
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
            "J9_NO_DIRECT_COMMIT",
            no_direct_commit,
            f"No direct write/commit/consolidation API and all emissions provisional-only="
            f"{no_direct_commit}.",
            {
                "forbidden_paths": list(forbidden_paths),
                "present": [
                    name
                    for name in forbidden_paths
                    if hasattr(ConservativeLearnedRoleBinder, name)
                ],
                "adapter_safe": adapter_safe,
            },
        ),
        gate(
            "J10_SEALS_UNTOUCHED",
            seals_untouched,
            f"Pas 7a and query-side seal hashes unchanged={seals_untouched}.",
            seal_hashes,
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
            "head_hidden_dim": HEAD_HIDDEN_DIM,
            "feature_backend": feature_backend.backend_id,
            "split": split_summary(splits),
            "optimization": optimization,
            "calibration": calibration,
            "test": test_result,
            "best_test_lexical": best_test_lexical,
            "lexical_uplift": lexical_uplift,
            "scope": (
                "Controlled, development-exposed role-binding measurement on "
                "held-out identifiers/texts with seen syntax families. Not "
                "unseen-syntax, open-domain, Pas 7a ingestion, committed-memory, "
                "or end-to-end memory improvement proof."
            ),
            "claim_status": "MEASURED in one local environment; not PROVEN.",
            "all_pass": all_pass,
        },
    }
    verdict_path = results_dir / "verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(SEP, flush=True)
    print("[INFO] D_Cortex v15.7b-RB1 conservative learned role-binder verdict", flush=True)
    print(f"[INFO] Device: {device}", flush=True)
    print(f"[INFO] Frozen validation-selected margin: {margin_threshold:.2f}", flush=True)
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
    rb0_results = REPO_ROOT / "runs" / "semantic_role_binding_benchmark" / "results"
    parser = argparse.ArgumentParser(description="D_Cortex learned role-binder verdict")
    parser.add_argument("--checkpoint", default=str(default_checkpoint))
    parser.add_argument("--sample", default=str(rb0_results / "sample.json"))
    parser.add_argument("--rb0-verdict", default=str(rb0_results / "verdict.json"))
    parser.add_argument(
        "--run-dir",
        default=str(REPO_ROOT / "runs" / "semantic_role_binder"),
    )
    return parser


def main() -> int:
    """CLI entry point."""
    return run(build_argparser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
