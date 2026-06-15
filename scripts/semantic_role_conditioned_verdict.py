# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b-RB2 frozen token-level role-conditioned binder verdict.

import argparse
import hashlib
import inspect
import json
import os
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

import tiktoken
import torch
import torch.nn.functional as F

from dcortex.semantic_role_binder import (
    ASSIGNMENT_ORDER,
    ConservativeLearnedRoleBinder,
)
from dcortex.semantic_role_conditioned import (
    DCortexTokenContextBackend,
    RoleConditionedSequenceScoringHead,
    build_role_masks,
)
from scripts.semantic_contextual_curriculum import load_contextual_model
from scripts.semantic_likelihood_probe import SEALED_SHA256, SEALED_SOURCE, gate
from scripts.semantic_role_binder_verdict import (
    QUERY_SIDE_SEALS,
    RB0_SAMPLE_SHA256,
    RB0_SEMANTIC_SAMPLE_HASH,
    RB0_VERDICT_SHA256,
    assignment_index,
    atomic_save,
    evaluate_split,
    load_rb0_sample,
    select_margin_threshold,
    sha256_file,
    split_records,
    split_summary,
)
from scripts.semantic_role_binding_benchmark import BASELINES, sample_hash, score_baseline
from scripts.train_semantic_internalizer import seed_everything, state_tensor_hash

SEP = "=" * 70
TRAINING_SEED = 20261540
PROJECTION_DIM = 128
ROLE_EMBEDDING_DIM = 32
RECURRENT_HIDDEN_DIM = 128
RB1_VERDICT_SHA256 = "92035e5d8148ead5d03d3ba5fac571bc646103ee1acd8a2677216e07d30c0b6f"
RB1_KNOWN_EXACT_RATE = 0.484375


def build_role_tensor(
    records: Sequence[Any],
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: Any,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Build padded candidate role masks and report integrity evidence."""
    role_masks = []
    audits = []
    for record, ids, mask in zip(records, token_ids, attention_mask):
        length = int(mask.sum())
        roles, audit = build_role_masks(
            ids[:length].tolist(),
            record.entities,
            record.values,
            tokenizer,
        )
        padded = torch.zeros((len(ASSIGNMENT_ORDER), ids.shape[0]), dtype=torch.long)
        padded[:, :length] = roles
        role_masks.append(padded)
        audits.append(audit)
    complete = sum(audit.complete for audit in audits)
    truth_not_input = "expected" not in inspect.signature(build_role_masks).parameters
    return torch.stack(role_masks), {
        "records": len(records),
        "complete": complete,
        "complete_rate": complete / len(records),
        "truth_not_input": truth_not_input,
        "all_unresolved_empty": all(audit.unresolved_marked_tokens == 0 for audit in audits),
        "all_entities_unchanged": all(audit.entity_roles_unchanged for audit in audits),
        "all_values_swapped": all(audit.value_roles_swapped for audit in audits),
        "failed_indices": [
            index for index, audit in enumerate(audits) if not audit.complete
        ],
    }


def train_head(
    contextual_tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    role_masks: torch.Tensor,
    records: Sequence[Any],
    splits: Mapping[str, Sequence[Any]],
    record_to_index: Mapping[str, int],
    device: torch.device,
    output_path: Path,
) -> Tuple[RoleConditionedSequenceScoringHead, Dict[str, Any]]:
    """Train only the role-conditioned sequence head."""
    seed_everything(TRAINING_SEED)
    labels = torch.tensor([assignment_index(record) for record in records], dtype=torch.long)
    train_indices = torch.tensor(
        [record_to_index[record.text] for record in splits["train"]], dtype=torch.long
    )
    validation_indices = torch.tensor(
        [record_to_index[record.text] for record in splits["validation"]], dtype=torch.long
    )
    train_labels = labels[train_indices]
    validation_labels = labels[validation_indices]
    head = RoleConditionedSequenceScoringHead(
        context_dim=contextual_tokens.shape[-1],
        projection_dim=PROJECTION_DIM,
        role_embedding_dim=ROLE_EMBEDDING_DIM,
        recurrent_hidden_dim=RECURRENT_HIDDEN_DIM,
        dropout=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-3)
    batch_size = 64
    patience = 15
    best_loss = float("inf")
    best_accuracy = 0.0
    best_epoch = -1
    stale = 0
    history: List[Dict[str, float]] = []

    def logits_for(indices: torch.Tensor) -> torch.Tensor:
        context = contextual_tokens[indices].to(device)
        roles = role_masks[indices].to(device)
        mask = attention_mask[indices].to(device)
        batch, candidates, width = roles.shape
        expanded_context = context.unsqueeze(1).expand(
            batch, candidates, width, context.shape[-1]
        )
        expanded_mask = mask.unsqueeze(1).expand(batch, candidates, width)
        scores = head(
            expanded_context.reshape(-1, width, context.shape[-1]),
            roles.reshape(-1, width),
            expanded_mask.reshape(-1, width),
        )
        return scores.reshape(batch, candidates)

    def evaluate() -> Tuple[float, float]:
        head.eval()
        losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for start in range(0, len(validation_indices), batch_size):
                indices = validation_indices[start : start + batch_size]
                logits = logits_for(indices)
                targets = validation_labels[start : start + batch_size].to(device)
                losses.append(
                    float(F.cross_entropy(logits, targets)) * len(indices)
                )
                correct += int((logits.argmax(dim=1) == targets).sum())
                total += len(indices)
        return sum(losses) / total, correct / total

    initial_loss, initial_accuracy = evaluate()
    generator = torch.Generator(device="cpu").manual_seed(TRAINING_SEED)
    for epoch in range(100):
        head.train()
        permutation = torch.randperm(len(train_indices), generator=generator)
        for start in range(0, len(train_indices), batch_size):
            local = permutation[start : start + batch_size]
            indices = train_indices[local]
            logits = logits_for(indices)
            loss = F.cross_entropy(
                logits,
                train_labels[local].to(device),
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
                    "context_dim": int(contextual_tokens.shape[-1]),
                    "projection_dim": PROJECTION_DIM,
                    "role_embedding_dim": ROLE_EMBEDDING_DIM,
                    "recurrent_hidden_dim": RECURRENT_HIDDEN_DIM,
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
    return head, {
        "initial_validation_loss": initial_loss,
        "initial_validation_accuracy": initial_accuracy,
        "best_validation_loss": best_loss,
        "best_validation_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "loss_drop": (initial_loss - best_loss) / max(initial_loss, 1e-12),
        "history": history,
        "artifact": str(output_path),
        "artifact_sha256": sha256_file(output_path),
    }


def score_all(
    head: RoleConditionedSequenceScoringHead,
    contextual_tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    role_masks: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return measured three-candidate logits for every record."""
    head.eval()
    batches = []
    batch_size = 64
    with torch.inference_mode():
        for start in range(0, len(contextual_tokens), batch_size):
            context = contextual_tokens[start : start + batch_size].to(device)
            roles = role_masks[start : start + batch_size].to(device)
            mask = attention_mask[start : start + batch_size].to(device)
            batch, candidates, width = roles.shape
            expanded_context = context.unsqueeze(1).expand(
                batch, candidates, width, context.shape[-1]
            )
            expanded_mask = mask.unsqueeze(1).expand(batch, candidates, width)
            scores = head(
                expanded_context.reshape(-1, width, context.shape[-1]),
                roles.reshape(-1, width),
                expanded_mask.reshape(-1, width),
            ).reshape(batch, candidates)
            batches.append(scores.detach().float().cpu())
    return torch.cat(batches)


def run(args: argparse.Namespace) -> int:
    """Run the frozen token-level role-conditioned binder verdict."""
    checkpoint = Path(args.checkpoint)
    sample_path = Path(args.sample)
    rb0_verdict_path = Path(args.rb0_verdict)
    rb1_verdict_path = Path(args.rb1_verdict)
    results_dir = Path(args.run_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for required in (checkpoint, sample_path, rb0_verdict_path, rb1_verdict_path):
        if not required.exists():
            raise RuntimeError(f"Required predecessor artifact not found: {required}")

    records = load_rb0_sample(sample_path)
    splits = split_records(records)
    record_to_index = {record.text: index for index, record in enumerate(records)}
    split_texts = {
        name: {record.text for record in subset} for name, subset in splits.items()
    }
    split_separation = (
        not split_texts["train"].intersection(split_texts["validation"])
        and not split_texts["train"].intersection(split_texts["test"])
        and not split_texts["validation"].intersection(split_texts["test"])
        and sum(len(items) for items in splits.values()) == len(records)
    )
    predecessor_hashes = {
        "rb0_sample_actual": sha256_file(sample_path),
        "rb0_sample_expected": RB0_SAMPLE_SHA256,
        "rb0_verdict_actual": sha256_file(rb0_verdict_path),
        "rb0_verdict_expected": RB0_VERDICT_SHA256,
        "rb0_semantic_sample_actual": sample_hash(records),
        "rb0_semantic_sample_expected": RB0_SEMANTIC_SAMPLE_HASH,
        "rb1_verdict_actual": sha256_file(rb1_verdict_path),
        "rb1_verdict_expected": RB1_VERDICT_SHA256,
    }
    predecessors_preserved = all(
        predecessor_hashes[key.replace("_actual", "_expected")]
        == value
        for key, value in predecessor_hashes.items()
        if key.endswith("_actual")
    )
    rb1_verdict = json.loads(rb1_verdict_path.read_text(encoding="utf-8"))
    rb1_exact_rate = float(rb1_verdict["reference"]["test"]["known_exact_rate"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, state, payload = load_contextual_model(checkpoint, device)
    substrate_hash_before = state_tensor_hash(state)
    tokenizer = tiktoken.get_encoding("gpt2").encode_ordinary
    token_backend = DCortexTokenContextBackend(
        model=model,
        tokenizer=tokenizer,
        max_seq_len=128,
        batch_size=128,
        backend_version=checkpoint.name,
    )
    print("[INFO] Extracting frozen contextual source-token states...", flush=True)
    context = token_backend.features([record.text for record in records])
    role_masks, role_integrity = build_role_tensor(
        records,
        context.token_ids,
        context.attention_mask,
        tokenizer,
    )
    head, optimization = train_head(
        context.hidden,
        context.attention_mask,
        role_masks,
        records,
        splits,
        record_to_index,
        device,
        results_dir / "best_role_conditioned_head.pt",
    )
    all_logits = score_all(
        head,
        context.hidden,
        context.attention_mask,
        role_masks,
        device,
    )
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
        episode_offset=220000,
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
    rb1_uplift = test_result["known_exact_rate"] - rb1_exact_rate
    lexical_uplift = test_result["known_exact_rate"] - best_test_lexical
    adapter_safe = (
        test_result["adapter_accepted"] == test_result["emitted_facts"]
        and test_result["provisional_only"] == test_result["emitted_facts"]
        and test_result["emitted_facts"] == 2 * test_result["emitted_mappings"]
    )
    substrate_hash_after = state_tensor_hash(state)
    trainable_substrate = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    source = (REPO_ROOT / "dcortex" / "semantic_role_conditioned.py").read_text(
        encoding="utf-8"
    )
    forbidden_identifiers = (
        "RELATION_LEXICON",
        "RELATION_RULES",
        "SYNONYM_MAP",
        "ALIAS_MAP",
    )
    forbidden_present = [
        identifier for identifier in forbidden_identifiers if identifier in source
    ]
    forbidden_paths = ("commit", "write", "consolidate", "promote")
    direct_paths_present = [
        name for name in forbidden_paths if hasattr(ConservativeLearnedRoleBinder, name)
    ]
    no_handwritten_relation_or_commit = not forbidden_present and not direct_paths_present
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
    del payload

    gates = [
        gate(
            "L0_PREDECESSORS_PRESERVED",
            predecessors_preserved and rb1_exact_rate == RB1_KNOWN_EXACT_RATE,
            f"RB0/RB1 predecessor artifacts unchanged={predecessors_preserved}; "
            f"RB1 exact rate={rb1_exact_rate:.1%}.",
            predecessor_hashes,
        ),
        gate(
            "L1_SPLIT_SEPARATION",
            split_separation,
            f"Unchanged deterministic RB1 split texts disjoint={split_separation}.",
            split_summary(splits),
        ),
        gate(
            "L2_ROLE_MASK_INTEGRITY",
            (
                role_integrity["complete_rate"] == 1.0
                and role_integrity["truth_not_input"]
                and role_integrity["all_unresolved_empty"]
                and role_integrity["all_entities_unchanged"]
                and role_integrity["all_values_swapped"]
            ),
            f"Complete source-mention role masks={role_integrity['complete']}/"
            f"{role_integrity['records']}; truth label absent from mask builder="
            f"{role_integrity['truth_not_input']}.",
            role_integrity,
        ),
        gate(
            "L3_REAL_OPTIMIZATION",
            optimization["loss_drop"] >= 0.20,
            f"Validation loss drop={optimization['loss_drop']:.1%}; "
            f"{optimization['initial_validation_loss']:.4f} -> "
            f"{optimization['best_validation_loss']:.4f}.",
            optimization,
        ),
        gate(
            "L4_TEST_EXACT_BINDING",
            test_result["known_exact_rate"] >= 0.75,
            f"Known test exact binding={test_result['known_exact_rate']:.1%} "
            f"({test_result['known_exact']}/{test_result['known_n']}).",
            {key: value for key, value in test_result.items() if key != "details"},
        ),
        gate(
            "L5_TEST_WRONG_MAPPING",
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
            "L6_AMBIGUOUS_HONESTY",
            test_result["ambiguous_abstention_rate"] >= 0.95,
            f"Ambiguous test abstention={test_result['ambiguous_abstention_rate']:.1%} "
            f"({test_result['ambiguous_abstained']}/{test_result['ambiguous_n']}).",
            {
                "calibration": calibration,
                "ambiguous_n": test_result["ambiguous_n"],
                "ambiguous_abstained": test_result["ambiguous_abstained"],
                "ambiguous_abstention_rate": test_result[
                    "ambiguous_abstention_rate"
                ],
            },
        ),
        gate(
            "L7_RB1_UPLIFT",
            rb1_uplift >= 0.20,
            f"Role-conditioned exact={test_result['known_exact_rate']:.1%}; "
            f"frozen RB1={rb1_exact_rate:.1%}; uplift={rb1_uplift:+.1%}.",
            {
                "role_conditioned_exact": test_result["known_exact_rate"],
                "rb1_exact": rb1_exact_rate,
                "uplift": rb1_uplift,
            },
        ),
        gate(
            "L8_LEXICAL_UPLIFT",
            lexical_uplift >= 0.30,
            f"Known exact={test_result['known_exact_rate']:.1%}; best same-test "
            f"lexical={best_test_lexical:.1%}; uplift={lexical_uplift:+.1%}.",
            {
                "role_conditioned_exact": test_result["known_exact_rate"],
                "best_test_lexical": best_test_lexical,
                "uplift": lexical_uplift,
                "test_baselines": test_baselines,
            },
        ),
        gate(
            "L9_ADAPTER_PROVISIONAL_ONLY",
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
            "L10_FROZEN_SUBSTRATE",
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
            "L11_NO_HANDWRITTEN_RELATION_OR_COMMIT",
            no_handwritten_relation_or_commit,
            f"Forbidden relation-map identifiers={forbidden_present}; direct mutation "
            f"paths={direct_paths_present}.",
            {
                "forbidden_identifiers": list(forbidden_identifiers),
                "present_identifiers": forbidden_present,
                "direct_paths_present": direct_paths_present,
            },
        ),
        gate(
            "L12_SEALS_UNTOUCHED",
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
            "projection_dim": PROJECTION_DIM,
            "role_embedding_dim": ROLE_EMBEDDING_DIM,
            "recurrent_hidden_dim": RECURRENT_HIDDEN_DIM,
            "token_backend": token_backend.backend_id,
            "split": split_summary(splits),
            "role_integrity": role_integrity,
            "optimization": optimization,
            "calibration": calibration,
            "test": test_result,
            "rb1_exact_rate": rb1_exact_rate,
            "rb1_uplift": rb1_uplift,
            "best_test_lexical": best_test_lexical,
            "lexical_uplift": lexical_uplift,
            "scope": (
                "Controlled, development-exposed token-level role-binding "
                "measurement on held-out texts and identifiers with seen syntax "
                "families. Not unseen-syntax, open-domain, Pas 7a ingestion, "
                "committed-memory, or end-to-end memory improvement proof."
            ),
            "claim_status": "MEASURED in one local environment; not PROVEN.",
            "all_pass": all_pass,
        },
    }
    verdict_path = results_dir / "verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(SEP, flush=True)
    print("[INFO] D_Cortex v15.7b-RB2 token-level role-conditioned verdict", flush=True)
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
    parser = argparse.ArgumentParser(description="D_Cortex role-conditioned verdict")
    parser.add_argument("--checkpoint", default=str(default_checkpoint))
    parser.add_argument("--sample", default=str(rb0_results / "sample.json"))
    parser.add_argument("--rb0-verdict", default=str(rb0_results / "verdict.json"))
    parser.add_argument(
        "--rb1-verdict",
        default=str(REPO_ROOT / "runs" / "semantic_role_binder" / "results" / "verdict.json"),
    )
    parser.add_argument(
        "--run-dir",
        default=str(REPO_ROOT / "runs" / "semantic_role_conditioned"),
    )
    return parser


def main() -> int:
    """CLI entry point."""
    return run(build_argparser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
