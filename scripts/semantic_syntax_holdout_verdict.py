# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b-RB3 frozen leave-one-syntax-family-out verdict.

import argparse
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

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

from dcortex.semantic_role_binder import ConservativeLearnedRoleBinder
from dcortex.semantic_role_conditioned import DCortexTokenContextBackend, build_role_masks
from scripts.semantic_contextual_curriculum import load_contextual_model
from scripts.semantic_likelihood_probe import SEALED_SHA256, SEALED_SOURCE, gate
from scripts.semantic_role_binder_verdict import (
    QUERY_SIDE_SEALS,
    RB0_SAMPLE_SHA256,
    RB0_SEMANTIC_SAMPLE_HASH,
    RB0_VERDICT_SHA256,
    evaluate_split,
    load_rb0_sample,
    select_margin_threshold,
    sha256_file,
    split_records,
)
from scripts.semantic_role_binding_benchmark import (
    BASELINES,
    KNOWN_FAMILIES,
    sample_hash,
    score_baseline,
)
from scripts.semantic_role_conditioned_verdict import (
    build_role_tensor,
    score_all,
    train_head,
)
from scripts.train_semantic_internalizer import state_tensor_hash

SEP = "=" * 70
RB1_VERDICT_SHA256 = "92035e5d8148ead5d03d3ba5fac571bc646103ee1acd8a2677216e07d30c0b6f"
RB2_VERDICT_SHA256 = "e370695d2bce6c6843e8b4514a31e3e37c3e14a78e8eabe7879d9250a1b54705"


def build_fold_splits(
    records: Sequence[Any],
    heldout_family: str,
) -> Dict[str, Tuple[Any, ...]]:
    """Build one syntax-family holdout fold from the frozen RB1 text split."""
    if heldout_family not in KNOWN_FAMILIES:
        raise ValueError(f"unknown heldout family: {heldout_family}")
    original = split_records(records)
    train = tuple(
        record
        for record in original["train"]
        if record.family == "RB5" or record.family != heldout_family
    )
    validation = tuple(
        record
        for record in original["validation"]
        if record.family == "RB5" or record.family != heldout_family
    )
    test = tuple(
        [record for record in records if record.family == heldout_family]
        + [record for record in original["test"] if record.family == "RB5"]
    )
    return {"train": train, "validation": validation, "test": test}


def fold_separation_report(
    folds: Mapping[str, Mapping[str, Sequence[Any]]],
) -> Dict[str, Any]:
    """Report syntax-family and text separation for every fold."""
    reports: Dict[str, Any] = {}
    for heldout, split in folds.items():
        train_families = sorted({record.family for record in split["train"]})
        validation_families = sorted(
            {record.family for record in split["validation"]}
        )
        train_texts = {record.text for record in split["train"]}
        validation_texts = {record.text for record in split["validation"]}
        test_texts = {record.text for record in split["test"]}
        reports[heldout] = {
            "train_n": len(split["train"]),
            "validation_n": len(split["validation"]),
            "test_n": len(split["test"]),
            "heldout_absent_train": heldout not in train_families,
            "heldout_absent_validation": heldout not in validation_families,
            "train_validation_disjoint": not train_texts.intersection(validation_texts),
            "train_test_disjoint": not train_texts.intersection(test_texts),
            "validation_test_disjoint": not validation_texts.intersection(test_texts),
            "heldout_test_n": sum(
                record.family == heldout for record in split["test"]
            ),
            "ambiguous_test_n": sum(
                record.family == "RB5" for record in split["test"]
            ),
            "train_families": train_families,
            "validation_families": validation_families,
        }
    return reports


def run(args: argparse.Namespace) -> int:
    """Run the frozen leave-one-syntax-family-out verdict."""
    checkpoint = Path(args.checkpoint)
    sample_path = Path(args.sample)
    rb0_verdict_path = Path(args.rb0_verdict)
    rb1_verdict_path = Path(args.rb1_verdict)
    rb2_verdict_path = Path(args.rb2_verdict)
    results_dir = Path(args.run_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for required in (
        checkpoint,
        sample_path,
        rb0_verdict_path,
        rb1_verdict_path,
        rb2_verdict_path,
    ):
        if not required.exists():
            raise RuntimeError(f"Required predecessor artifact not found: {required}")

    records = load_rb0_sample(sample_path)
    record_to_index = {record.text: index for index, record in enumerate(records)}
    folds = {
        family: build_fold_splits(records, family) for family in KNOWN_FAMILIES
    }
    separation = fold_separation_report(folds)
    syntax_holdout_ok = all(
        report["heldout_absent_train"]
        and report["heldout_absent_validation"]
        and report["train_validation_disjoint"]
        and report["train_test_disjoint"]
        and report["validation_test_disjoint"]
        and report["heldout_test_n"] == 400
        and report["ambiguous_test_n"] > 0
        for report in separation.values()
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
        "rb2_verdict_actual": sha256_file(rb2_verdict_path),
        "rb2_verdict_expected": RB2_VERDICT_SHA256,
    }
    predecessors_preserved = all(
        predecessor_hashes[key.replace("_actual", "_expected")] == value
        for key, value in predecessor_hashes.items()
        if key.endswith("_actual")
    )

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
    print("[INFO] Extracting shared frozen contextual source-token states...", flush=True)
    context = token_backend.features([record.text for record in records])
    role_masks, role_integrity = build_role_tensor(
        records,
        context.token_ids,
        context.attention_mask,
        tokenizer,
    )

    fold_reports: Dict[str, Any] = {}
    optimization_ok = True
    aggregate_known_n = 0
    aggregate_exact = 0
    aggregate_wrong = 0
    aggregate_emitted_mappings = 0
    aggregate_emitted_facts = 0
    aggregate_adapter_accepted = 0
    aggregate_provisional_only = 0
    per_family_exact: Dict[str, float] = {}
    ambiguity_rates = []
    for fold_index, heldout in enumerate(KNOWN_FAMILIES):
        print(SEP, flush=True)
        print(f"[INFO] Training syntax holdout fold: {heldout}", flush=True)
        split = folds[heldout]
        head, optimization = train_head(
            context.hidden,
            context.attention_mask,
            role_masks,
            records,
            split,
            record_to_index,
            device,
            results_dir / f"{heldout.lower()}_best_head.pt",
        )
        optimization_ok &= optimization["loss_drop"] >= 0.20
        logits = score_all(
            head,
            context.hidden,
            context.attention_mask,
            role_masks,
            device,
        )
        validation_indices = [
            record_to_index[record.text] for record in split["validation"]
        ]
        test_indices = [record_to_index[record.text] for record in split["test"]]
        calibration = select_margin_threshold(
            split["validation"], logits[validation_indices]
        )
        evaluation = evaluate_split(
            split["test"],
            logits[test_indices],
            float(calibration["threshold"]),
            episode_offset=230000 + fold_index * 10000,
        )
        aggregate_known_n += evaluation["known_n"]
        aggregate_exact += evaluation["known_exact"]
        aggregate_wrong += evaluation["known_wrong"]
        aggregate_emitted_mappings += evaluation["emitted_mappings"]
        aggregate_emitted_facts += evaluation["emitted_facts"]
        aggregate_adapter_accepted += evaluation["adapter_accepted"]
        aggregate_provisional_only += evaluation["provisional_only"]
        per_family_exact[heldout] = evaluation["known_exact_rate"]
        ambiguity_rates.append(evaluation["ambiguous_abstention_rate"])
        fold_reports[heldout] = {
            "separation": separation[heldout],
            "optimization": optimization,
            "calibration": calibration,
            "evaluation": evaluation,
        }

    aggregate_exact_rate = aggregate_exact / aggregate_known_n
    aggregate_wrong_rate = aggregate_wrong / aggregate_known_n
    min_family_exact = min(per_family_exact.values())
    min_ambiguity = min(ambiguity_rates)
    baseline_scores = {
        name: score_baseline(records, baseline)
        for name, baseline in BASELINES.items()
    }
    best_lexical = max(
        baseline_scores[name]["known_exact_rate"]
        for name in (
            "ordered_first_occurrence",
            "minimum_distance",
            "lexical_cartesian",
        )
    )
    lexical_uplift = aggregate_exact_rate - best_lexical
    adapter_safe = (
        aggregate_adapter_accepted == aggregate_emitted_facts
        and aggregate_provisional_only == aggregate_emitted_facts
        and aggregate_emitted_facts == 2 * aggregate_emitted_mappings
    )
    substrate_hash_after = state_tensor_hash(state)
    trainable_substrate = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    role_source = (REPO_ROOT / "dcortex" / "semantic_role_conditioned.py").read_text(
        encoding="utf-8"
    )
    forbidden_identifiers = (
        "RELATION_LEXICON",
        "RELATION_RULES",
        "SYNONYM_MAP",
        "ALIAS_MAP",
    )
    forbidden_present = [
        identifier for identifier in forbidden_identifiers if identifier in role_source
    ]
    direct_paths_present = [
        name
        for name in ("commit", "write", "consolidate", "promote")
        if hasattr(ConservativeLearnedRoleBinder, name)
    ]
    no_relation_or_commit = not forbidden_present and not direct_paths_present
    truth_not_input = "expected" not in inspect.signature(build_role_masks).parameters
    role_integrity_ok = (
        role_integrity["complete_rate"] == 1.0
        and role_integrity["truth_not_input"]
        and truth_not_input
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
    del payload

    gates = [
        gate(
            "M0_PREDECESSORS_PRESERVED",
            predecessors_preserved,
            f"RB0, RB1, and RB2 predecessor artifacts unchanged={predecessors_preserved}.",
            predecessor_hashes,
        ),
        gate(
            "M1_SYNTAX_FAMILY_HOLDOUT",
            syntax_holdout_ok,
            f"Held-out syntax family absent from train and validation in every fold="
            f"{syntax_holdout_ok}.",
            separation,
        ),
        gate(
            "M2_ROLE_MASK_INTEGRITY",
            role_integrity_ok,
            f"Complete candidate-inventory role masks={role_integrity['complete']}/"
            f"{role_integrity['records']}; truth input absent={truth_not_input}.",
            role_integrity,
        ),
        gate(
            "M3_REAL_OPTIMIZATION",
            optimization_ok,
            f"Validation loss drop >=20% in every fold={optimization_ok}.",
            {
                family: report["optimization"]
                for family, report in fold_reports.items()
            },
        ),
        gate(
            "M4_UNSEEN_SYNTAX_BINDING",
            aggregate_exact_rate >= 0.70 and min_family_exact >= 0.55,
            f"Aggregate unseen-syntax exact={aggregate_exact_rate:.1%} "
            f"({aggregate_exact}/{aggregate_known_n}); minimum family exact="
            f"{min_family_exact:.1%}.",
            {
                "aggregate_known_n": aggregate_known_n,
                "aggregate_exact": aggregate_exact,
                "aggregate_exact_rate": aggregate_exact_rate,
                "per_family_exact": per_family_exact,
                "minimum_family_exact": min_family_exact,
            },
        ),
        gate(
            "M5_WRONG_MAPPING",
            aggregate_wrong_rate <= 0.10,
            f"Aggregate wrong emitted mapping={aggregate_wrong_rate:.1%} "
            f"({aggregate_wrong}/{aggregate_known_n}).",
            {
                "aggregate_wrong": aggregate_wrong,
                "aggregate_known_n": aggregate_known_n,
                "aggregate_wrong_rate": aggregate_wrong_rate,
            },
        ),
        gate(
            "M6_AMBIGUOUS_HONESTY",
            min_ambiguity >= 0.95,
            f"Minimum fold ambiguous abstention={min_ambiguity:.1%}.",
            {
                family: report["evaluation"]["ambiguous_abstention_rate"]
                for family, report in fold_reports.items()
            },
        ),
        gate(
            "M7_LEXICAL_UPLIFT",
            lexical_uplift >= 0.25,
            f"Aggregate unseen-syntax exact={aggregate_exact_rate:.1%}; best "
            f"aggregate lexical={best_lexical:.1%}; uplift={lexical_uplift:+.1%}.",
            {
                "aggregate_exact_rate": aggregate_exact_rate,
                "best_lexical": best_lexical,
                "uplift": lexical_uplift,
                "baseline_scores": baseline_scores,
            },
        ),
        gate(
            "M8_ADAPTER_PROVISIONAL_ONLY",
            adapter_safe,
            f"Adapter-accepted provisional facts={aggregate_adapter_accepted}/"
            f"{aggregate_emitted_facts}; exactly two per emitted mapping="
            f"{aggregate_emitted_facts == 2 * aggregate_emitted_mappings}.",
            {
                "emitted_mappings": aggregate_emitted_mappings,
                "emitted_facts": aggregate_emitted_facts,
                "adapter_accepted": aggregate_adapter_accepted,
                "provisional_only": aggregate_provisional_only,
            },
        ),
        gate(
            "M9_FROZEN_SUBSTRATE",
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
            "M10_NO_HANDWRITTEN_RELATION_OR_COMMIT",
            no_relation_or_commit,
            f"Forbidden relation-map identifiers={forbidden_present}; direct mutation "
            f"paths={direct_paths_present}.",
            {
                "forbidden_identifiers": list(forbidden_identifiers),
                "present_identifiers": forbidden_present,
                "direct_paths_present": direct_paths_present,
            },
        ),
        gate(
            "M11_SEALS_UNTOUCHED",
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
            "fold_reports": fold_reports,
            "aggregate": {
                "known_n": aggregate_known_n,
                "exact": aggregate_exact,
                "exact_rate": aggregate_exact_rate,
                "wrong": aggregate_wrong,
                "wrong_rate": aggregate_wrong_rate,
                "per_family_exact": per_family_exact,
                "minimum_family_exact": min_family_exact,
                "minimum_ambiguity_abstention": min_ambiguity,
                "best_lexical": best_lexical,
                "lexical_uplift": lexical_uplift,
            },
            "scope": (
                "Leave-one-construction-family-out generalization over four "
                "controlled synthetic role-binding families. Not open-domain "
                "language understanding, Pas 7a ingestion, committed-memory, "
                "or end-to-end memory improvement proof."
            ),
            "claim_status": "MEASURED in one local environment; not PROVEN.",
            "all_pass": all_pass,
        },
    }
    verdict_path = results_dir / "verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(SEP, flush=True)
    print("[INFO] D_Cortex v15.7b-RB3 syntax-family holdout verdict", flush=True)
    print(f"[INFO] Device: {device}", flush=True)
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
    parser = argparse.ArgumentParser(description="D_Cortex syntax holdout verdict")
    parser.add_argument("--checkpoint", default=str(default_checkpoint))
    parser.add_argument("--sample", default=str(rb0_results / "sample.json"))
    parser.add_argument("--rb0-verdict", default=str(rb0_results / "verdict.json"))
    parser.add_argument(
        "--rb1-verdict",
        default=str(REPO_ROOT / "runs" / "semantic_role_binder" / "results" / "verdict.json"),
    )
    parser.add_argument(
        "--rb2-verdict",
        default=str(REPO_ROOT / "runs" / "semantic_role_conditioned" / "results" / "verdict.json"),
    )
    parser.add_argument(
        "--run-dir",
        default=str(REPO_ROOT / "runs" / "semantic_syntax_holdout"),
    )
    return parser


def main() -> int:
    """CLI entry point."""
    return run(build_argparser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
