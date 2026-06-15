# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b frozen-contextual semantic curriculum regression.

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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

from dcortex.backbone.transformer import StandardTransformerBlock
from dcortex.config import DCortexConfig
from dcortex.semantic_adapter import ConservativeSemanticAdapter
from dcortex.semantic_producer import (
    ConservativeTrainedQueryProducer,
    DCortexContextualFeatureBackend,
)
from scripts.semantic_curriculum_crossval import (
    AMBIGUOUS_SEED,
    FOLDS,
    build_curriculum_data,
    combine_results,
    evaluate_records,
    generate_ambiguous,
    generate_heldout_form_records,
    train_backend,
)
from scripts.semantic_likelihood_probe import SEALED_SHA256, SEALED_SOURCE, gate
from scripts.train_semantic_internalizer import (
    ATTRIBUTE_MARGIN,
    ENTITY_MARGIN,
    HEAD_HIDDEN_DIM,
    TRAINING_SEED,
    UNKNOWN_ATTRIBUTE,
    UNKNOWN_ENTITY,
    extract_definitions,
    generate_family,
    seed_everything,
    state_tensor_hash,
)

SEP = "=" * 70


class FrozenContextualDecoder(nn.Module):
    """Minimal frozen D_Cortex decoder-standard path with no memory modules."""

    def __init__(self, config: DCortexConfig) -> None:
        super().__init__()
        self.shared_token_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.shared_pos_emb = nn.Embedding(config.max_seq_len, config.hidden_dim)
        self.dec_emb_norm = nn.LayerNorm(config.hidden_dim)
        self.dec_emb_drop = nn.Dropout(config.dropout)
        self.dec_standard_blocks = nn.ModuleList(
            [StandardTransformerBlock(config) for _ in range(config.n_dec_standard_layers)]
        )


def load_contextual_model(
    checkpoint: Path,
    device: torch.device,
) -> Tuple[FrozenContextualDecoder, Dict[str, torch.Tensor], Dict[str, Any]]:
    """Load only the frozen decoder-standard state required for context features."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if "model" not in payload or "config_model" not in payload:
        raise RuntimeError("Checkpoint lacks model/config_model")
    state = payload["model"]
    config = DCortexConfig(**payload["config_model"])
    model = FrozenContextualDecoder(config)
    selected = {
        "shared_token_emb.weight": state["shared_token_emb.weight"],
        "shared_pos_emb.weight": state["shared_pos_emb.weight"],
        "dec_emb_norm.weight": state["dec_emb_norm.weight"],
        "dec_emb_norm.bias": state["dec_emb_norm.bias"],
    }
    for key, value in state.items():
        if key.startswith("dec_standard_blocks."):
            selected[key] = value
    model.load_state_dict(selected, strict=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, state, payload


def run(args: argparse.Namespace) -> int:
    """Run the frozen contextual semantic curriculum regression."""
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
    model, state, payload = load_contextual_model(checkpoint, device)
    substrate_hash_before = state_tensor_hash(state)
    feature_backend = DCortexContextualFeatureBackend(
        model=model,
        tokenizer=tiktoken.get_encoding("gpt2").encode_ordinary,
        max_seq_len=128,
        batch_size=128,
        backend_version=checkpoint.name,
    )
    forbidden_memory_attributes = (
        "state_mem",
        "episode_obj_mem",
        "conflict_mem",
        "archive_mem",
        "working_mem",
        "decode",
        "encode",
        "_bank_dict",
    )
    memory_bypass = not any(hasattr(model, name) for name in forbidden_memory_attributes)

    fold_reports: List[Dict[str, Any]] = []
    f1_results: List[Dict[str, Any]] = []
    f3_results: List[Dict[str, Any]] = []
    holdout_checks: List[bool] = []
    optimization_checks: List[bool] = []
    deterministic_checks: List[bool] = []
    for fold in FOLDS:
        print(SEP, flush=True)
        print(f"[INFO] Contextual curriculum fold {fold}/3", flush=True)
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
        holdout_ok = not training_texts.intersection(heldout_texts)
        holdout_checks.append(holdout_ok)
        backend, optimization = train_backend(
            feature_backend,
            data_a,
            entity_ids,
            attribute_ids,
            device,
            results_dir / f"fold_{fold}_best_head.pt",
            TRAINING_SEED + 300 + fold,
        )
        optimization_checks.append(
            optimization["loss_drop"] >= 0.20
            and optimization["best_joint_accuracy"] >= 0.95
        )
        f1 = evaluate_records(f"contextual_F1_fold_{fold}", heldout_f1, backend)
        f3 = evaluate_records(f"contextual_F3_fold_{fold}", heldout_f3, backend)
        f1_results.append(f1)
        f3_results.append(f3)
        fold_reports.append(
            {
                "fold": fold,
                "holdout_ok": holdout_ok,
                "deterministic": deterministic,
                "optimization": optimization,
                "F1": f1,
                "F3": f3,
            }
        )
    f1_aggregate = combine_results("contextual_F1_out_of_fold", f1_results)
    f3_aggregate = combine_results("contextual_F3_out_of_fold", f3_results)

    print(SEP, flush=True)
    print("[INFO] Training final contextual all-curriculum head", flush=True)
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
        TRAINING_SEED + 400,
    )
    optimization_checks.append(
        final_optimization["loss_drop"] >= 0.20
        and final_optimization["best_joint_accuracy"] >= 0.95
    )
    f5_result = evaluate_records(
        "contextual_F5_final", generate_family(definitions, "F5"), final_backend
    )

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
            f"contextual-ambiguous-{index}",
            50000 + index,
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
    trainable_substrate = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    del payload
    all_adapter = all(
        result["adapter_accepted"] == result["emitted"]
        and result["query_only"] == result["emitted"]
        for result in f1_results + f3_results + [f5_result]
    )
    all_wrong = (
        f1_aggregate["wrong_rate"] <= 0.02
        and f3_aggregate["wrong_rate"] <= 0.02
        and f5_result["wrong_rate"] <= 0.02
    )
    gates = [
        gate(
            "D0_FORM_HOLDOUT",
            all(holdout_checks),
            f"All contextual folds preserve exact form holdout={all(holdout_checks)}.",
            {"checks": holdout_checks},
        ),
        gate(
            "D1_F1_CONTEXTUAL_OUT_OF_FOLD",
            f1_aggregate["accuracy"] >= 0.85,
            f"Contextual F1 out-of-fold accuracy={f1_aggregate['accuracy']:.1%} "
            f"({f1_aggregate['correct']}/{f1_aggregate['total']}).",
            f1_aggregate,
        ),
        gate(
            "D2_F3_CONTEXTUAL_OUT_OF_FOLD",
            f3_aggregate["accuracy"] >= 0.85,
            f"Contextual F3 out-of-fold accuracy={f3_aggregate['accuracy']:.1%} "
            f"({f3_aggregate['correct']}/{f3_aggregate['total']}).",
            f3_aggregate,
        ),
        gate(
            "D3_F5_CONTEXTUAL_FINAL",
            f5_result["accuracy"] >= 0.85,
            f"Contextual final F5 accuracy={f5_result['accuracy']:.1%} "
            f"({f5_result['correct']}/{f5_result['total']}).",
            {key: value for key, value in f5_result.items() if key != "details"},
        ),
        gate(
            "D4_WRONG_INTERPRETATION",
            all_wrong,
            f"Wrong rates F1={f1_aggregate['wrong_rate']:.1%}, "
            f"F3={f3_aggregate['wrong_rate']:.1%}, F5={f5_result['wrong_rate']:.1%}.",
            {
                "F1": f1_aggregate["wrong_rate"],
                "F3": f3_aggregate["wrong_rate"],
                "F5": f5_result["wrong_rate"],
            },
        ),
        gate(
            "D5_AMBIGUOUS_HONESTY",
            ambiguous_rate >= 0.80,
            f"Contextual ambiguity abstention={ambiguous_rate:.1%} "
            f"({ambiguous_abstained}/{len(ambiguous_records)}).",
            {
                "abstained": ambiguous_abstained,
                "total": len(ambiguous_records),
                "abstain_rate": ambiguous_rate,
            },
        ),
        gate(
            "D6_ADAPTER_REQUIRED",
            all_adapter,
            f"All contextual emissions routed query-only through adapter={all_adapter}.",
            {"all_adapter": all_adapter},
        ),
        gate(
            "D7_FROZEN_SUBSTRATE",
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
            "D8_DETERMINISTIC",
            all(deterministic_checks) and all(optimization_checks),
            f"Dataset reconstruction exact={all(deterministic_checks)}; "
            f"optimization checks pass={all(optimization_checks)}.",
            {
                "dataset_checks": deterministic_checks,
                "optimization_checks": optimization_checks,
            },
        ),
        gate(
            "D9_MEMORY_BYPASS",
            memory_bypass,
            f"Minimal contextual model exposes no memory/read/write API={memory_bypass}.",
            {
                "forbidden_attributes": list(forbidden_memory_attributes),
                "present": [
                    name for name in forbidden_memory_attributes if hasattr(model, name)
                ],
            },
        ),
        gate(
            "D10_SEALED_UNTOUCHED",
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
            "feature_backend": feature_backend.backend_id,
            "attribute_margin": ATTRIBUTE_MARGIN,
            "entity_margin": ENTITY_MARGIN,
            "head_hidden_dim": HEAD_HIDDEN_DIM,
            "fold_reports": fold_reports,
            "F1_aggregate": f1_aggregate,
            "F3_aggregate": f3_aggregate,
            "final_optimization": final_optimization,
            "F5_final": f5_result,
            "ambiguous_details": ambiguous_details,
            "scope": (
                "Architecture regression of frozen contextual query features on "
                "previously observed leave-one-form-out definitions. Not an "
                "independent open-domain holdout or end-to-end memory result."
            ),
            "claim_status": "MEASURED in one local environment; not PROVEN.",
            "all_pass": all_pass,
        },
    }
    verdict_path = results_dir / "verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(SEP, flush=True)
    print("[INFO] D_Cortex v15.7b contextual semantic curriculum", flush=True)
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
    parser = argparse.ArgumentParser(description="D_Cortex contextual curriculum")
    parser.add_argument("--checkpoint", type=str, default=str(default_checkpoint))
    parser.add_argument(
        "--run-dir",
        type=str,
        default=str(REPO_ROOT / "runs" / "semantic_contextual"),
    )
    return parser


def main() -> int:
    """CLI entry point."""
    return run(build_argparser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
