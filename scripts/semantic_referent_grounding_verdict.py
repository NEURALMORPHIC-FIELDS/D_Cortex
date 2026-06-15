# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b-G frozen explicit-referent grounding verdict.

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from dcortex.semantic_grounded_reader import (
    ExplicitReferentGroundingGate,
    GroundedSemanticObjectReader,
    ReferentGroundingStatus,
)
from dcortex.semantic_object_reader import (
    DirectSemanticObjectReader,
    ObjectMemorySnapshot,
    ObjectReadStatus,
    SemanticObjectReadResult,
)
from scripts.semantic_likelihood_probe import gate
from scripts.semantic_object_read_verdict import (
    FAMILIES,
    KNOWN_FAMILIES,
    SEALED_FILES,
    ObjectReadEpisode,
    build_decisions,
    canonical_query,
    extract_definitions,
    sample_hash,
    sha256_file,
)

SEP = "=" * 70
TRIALS_PER_FAMILY = 200
SAMPLE_SEED = 20261500
PREDECESSOR_VERDICT = (
    REPO_ROOT / "runs" / "semantic_object_read" / "results" / "verdict.json"
)
PREDECESSOR_SHA256 = (
    "13d0df32d6d4de17446c7a09dddf048108866cc051a5a95fd7058ff5eb63efa2"
)


def build_sample(definitions: Mapping[str, Any]) -> List[ObjectReadEpisode]:
    """Build the newly frozen referent-grounding sample."""
    rng = random.Random(SAMPLE_SEED)
    entities = definitions["HOLDOUT_ENTITIES_SINGLE"]
    attributes = definitions["HOLDOUT_ATTR_TYPES"]
    values = definitions["HOLDOUT_ATTR_VALUES"]
    episodes: List[ObjectReadEpisode] = []
    for family in FAMILIES:
        for index in range(TRIALS_PER_FAMILY):
            attribute = rng.choice(attributes)
            if family in KNOWN_FAMILIES:
                entity = rng.choice(entities)
                value = rng.choice(values[attribute])
                form_index = rng.randrange(4)
                if family == "F1":
                    query = definitions["F1_QUERY_CONSTRUCTIONS"][attribute][
                        form_index
                    ](entity)
                elif family == "F3":
                    query = definitions["F3_NOVEL_ALIAS_QUERIES"][attribute][
                        form_index
                    ](entity)
                else:
                    query = definitions["F5_QUERY_FORMS"][attribute][form_index](
                        entity, value
                    )
                episodes.append(
                    ObjectReadEpisode(
                        family=family,
                        index=index,
                        query=query,
                        entity=entity,
                        attribute=attribute,
                        target_value=value,
                        target_unknown=False,
                        form_index=form_index,
                        snapshot=ObjectMemorySnapshot(
                            committed=((entity, attribute, value),)
                        ),
                    )
                )
            elif family == "S5":
                entity = rng.choice(entities)
                value_a, value_b = rng.sample(values[attribute], 2)
                episodes.append(
                    ObjectReadEpisode(
                        family=family,
                        index=index,
                        query=canonical_query(entity, attribute),
                        entity=entity,
                        attribute=attribute,
                        target_value=None,
                        target_unknown=True,
                        form_index=None,
                        snapshot=ObjectMemorySnapshot(
                            provisional=(
                                (entity, attribute, value_a),
                                (entity, attribute, value_b),
                            )
                        ),
                    )
                )
            else:
                entity_a, entity_b = rng.sample(entities, 2)
                value_a = rng.choice(values[attribute])
                value_b = rng.choice(values[attribute])
                pronoun_query = {
                    "color": "What color is it? It is",
                    "size": "What size is it? It is",
                    "location": "Where is it? It is in the",
                    "state": "What state is it in? It is",
                }[attribute]
                episodes.append(
                    ObjectReadEpisode(
                        family=family,
                        index=index,
                        query=pronoun_query,
                        entity=None,
                        attribute=attribute,
                        target_value=None,
                        target_unknown=True,
                        form_index=None,
                        snapshot=ObjectMemorySnapshot(
                            committed=(
                                (entity_a, attribute, value_a),
                                (entity_b, attribute, value_b),
                            )
                        ),
                    )
                )
    return episodes


def summarize_results(
    episodes: Sequence[ObjectReadEpisode],
    results: Sequence[SemanticObjectReadResult],
) -> Dict[str, Dict[str, Any]]:
    """Summarize one reader's outcomes per family."""
    output: Dict[str, Dict[str, Any]] = {}
    for family in FAMILIES:
        indices = [
            index for index, episode in enumerate(episodes) if episode.family == family
        ]
        if family in KNOWN_FAMILIES:
            correct = sum(
                results[index].status == ObjectReadStatus.FOUND_COMMITTED
                and results[index].pred_value == episodes[index].target_value
                for index in indices
            )
            wrong = sum(
                results[index].pred_value is not None
                and results[index].pred_value != episodes[index].target_value
                for index in indices
            )
            output[family] = {
                "n": len(indices),
                "correct": correct,
                "correct_rate": correct / len(indices),
                "wrong": wrong,
                "wrong_rate": wrong / len(indices),
            }
        else:
            overcommit = sum(results[index].pred_value is not None for index in indices)
            disputed = sum(
                results[index].status == ObjectReadStatus.FOUND_DISPUTED
                for index in indices
            )
            refused = sum(
                results[index].status == ObjectReadStatus.REFUSED_INPUT
                for index in indices
            )
            output[family] = {
                "n": len(indices),
                "honest": len(indices) - overcommit,
                "honesty_rate": (len(indices) - overcommit) / len(indices),
                "overcommit": overcommit,
                "overcommit_rate": overcommit / len(indices),
                "disputed": disputed,
                "refused": refused,
            }
    return output


def run_comparison(episodes: Sequence[ObjectReadEpisode]) -> Dict[str, Any]:
    """Run predecessor and grounded successor over identical snapshots."""
    predecessor = DirectSemanticObjectReader()
    successor = GroundedSemanticObjectReader()
    baseline_results: List[SemanticObjectReadResult] = []
    grounded_results: List[SemanticObjectReadResult] = []
    details: List[Dict[str, Any]] = []
    immutable = True
    deterministic = True
    exact_grounding = True
    grounded_nonrefused = True
    for episode in episodes:
        before = episode.snapshot.fingerprint
        baseline = predecessor.read(
            episode.snapshot, episode.hypothesis, episode.decision
        )
        first = successor.read(episode.snapshot, episode.hypothesis, episode.decision)
        second = successor.read(episode.snapshot, episode.hypothesis, episode.decision)
        after = episode.snapshot.fingerprint
        immutable &= (
            before
            == after
            == baseline.snapshot_fingerprint
            == first.read.snapshot_fingerprint
        )
        deterministic &= first.to_json() == second.to_json()
        if first.grounding.status == ReferentGroundingStatus.GROUNDED:
            span = first.grounding.matched_span
            exact_grounding &= (
                span is not None
                and first.grounding.source_tokens[span[0] : span[1]]
                == first.grounding.entity_tokens
            )
        if first.read.status != ObjectReadStatus.REFUSED_INPUT:
            grounded_nonrefused &= (
                first.grounding.status == ReferentGroundingStatus.GROUNDED
            )
        baseline_results.append(baseline)
        grounded_results.append(first.read)
        details.append(
            {
                "family": episode.family,
                "index": episode.index,
                "query": episode.query,
                "hypothesis_entity": (
                    None if episode.hypothesis is None else episode.hypothesis.entity_id
                ),
                "baseline": baseline.to_dict(),
                "grounded": first.to_dict(),
            }
        )
    return {
        "baseline_results": baseline_results,
        "grounded_results": grounded_results,
        "baseline_summary": summarize_results(episodes, baseline_results),
        "grounded_summary": summarize_results(episodes, grounded_results),
        "snapshot_immutable": immutable,
        "repeated_grounded_reads_exact": deterministic,
        "exact_grounding": exact_grounding,
        "all_nonrefused_grounded": grounded_nonrefused,
        "details": details,
    }


def no_mutation_check() -> Dict[str, Any]:
    """Verify neither grounding nor grounded reader exposes mutation paths."""
    forbidden = (
        "write",
        "commit",
        "consolidate",
        "promote",
        "retrograde",
        "prune",
        "update",
    )
    objects = (ExplicitReferentGroundingGate(), GroundedSemanticObjectReader())
    present = {
        type(instance).__name__: [
            name for name in forbidden if hasattr(instance, name)
        ]
        for instance in objects
    }
    return {"forbidden_attributes": list(forbidden), "present": present}


def run(args: argparse.Namespace) -> int:
    """Run the frozen referent-grounding verdict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Frozen referent-grounding verdict requires CUDA")
    checkpoint = Path(args.query_checkpoint)
    heads_dir = Path(args.heads_dir)
    results_dir = Path(args.run_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    definitions = extract_definitions()
    episodes_a = build_sample(definitions)
    episodes_b = build_sample(definitions)
    deterministic_sample = sample_hash(episodes_a) == sample_hash(episodes_b)

    print(SEP, flush=True)
    print("[INFO] Building frozen semantic query decisions", flush=True)
    substrate = build_decisions(episodes_a, checkpoint, heads_dir, device)
    comparison = run_comparison(episodes_a)
    baseline = comparison["baseline_summary"]
    grounded = comparison["grounded_summary"]
    predecessor_preserved = (
        PREDECESSOR_VERDICT.exists()
        and sha256_file(PREDECESSOR_VERDICT) == PREDECESSOR_SHA256
    )
    no_aliases = not hasattr(ExplicitReferentGroundingGate(), "aliases")
    explicit_only = (
        comparison["exact_grounding"]
        and comparison["all_nonrefused_grounded"]
        and no_aliases
    )
    s6_uplift = grounded["S6"]["honesty_rate"] - baseline["S6"]["honesty_rate"]
    known_no_regression = all(
        grounded[family]["correct_rate"] >= baseline[family]["correct_rate"] - 0.005
        and grounded[family]["wrong_rate"] <= 0.01
        for family in KNOWN_FAMILIES
    )
    s5_preserved = (
        grounded["S5"]["honesty_rate"] >= 0.95
        and grounded["S5"]["honesty_rate"] >= baseline["S5"]["honesty_rate"] - 0.005
    )
    no_mutation = no_mutation_check()
    actual_seals = {
        name: sha256_file(path) for name, (path, _) in SEALED_FILES.items()
    }
    expected_seals = {
        name: expected for name, (_, expected) in SEALED_FILES.items()
    }
    seals_untouched = actual_seals == expected_seals
    gates = [
        gate(
            "H0_PREDECESSOR_PRESERVED",
            predecessor_preserved,
            f"Step 18 verdict artifact unchanged={predecessor_preserved}.",
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
            "H1_EXPLICIT_EVIDENCE_ONLY",
            explicit_only,
            f"Every non-refused successor read explicitly grounded={comparison['all_nonrefused_grounded']}; "
            f"exact token evidence={comparison['exact_grounding']}; aliases absent={no_aliases}.",
            {
                "all_nonrefused_grounded": comparison["all_nonrefused_grounded"],
                "exact_grounding": comparison["exact_grounding"],
                "aliases_absent": no_aliases,
            },
        ),
        gate(
            "H2_S6_OVERCOMMIT",
            grounded["S6"]["overcommit_rate"] <= 0.01,
            f"Grounded S6 overcommit={grounded['S6']['overcommit_rate']:.1%} "
            f"({grounded['S6']['overcommit']}/{grounded['S6']['n']}).",
            grounded["S6"],
        ),
        gate(
            "H3_S6_UPLIFT",
            s6_uplift >= 0.05,
            f"S6 honesty uplift={s6_uplift:+.1%} "
            f"({baseline['S6']['honesty_rate']:.1%}->{grounded['S6']['honesty_rate']:.1%}).",
            {
                "baseline": baseline["S6"],
                "grounded": grounded["S6"],
                "uplift": s6_uplift,
            },
        ),
        gate(
            "H4_KNOWN_NO_REGRESSION",
            known_no_regression,
            "Known-family grounded correct rates: "
            + ", ".join(
                f"{family} {baseline[family]['correct_rate']:.1%}->"
                f"{grounded[family]['correct_rate']:.1%}, wrong "
                f"{grounded[family]['wrong_rate']:.1%}"
                for family in KNOWN_FAMILIES
            )
            + ".",
            {
                family: {
                    "baseline": baseline[family],
                    "grounded": grounded[family],
                }
                for family in KNOWN_FAMILIES
            },
        ),
        gate(
            "H5_S5_PRESERVED",
            s5_preserved,
            f"S5 honesty {baseline['S5']['honesty_rate']:.1%}->"
            f"{grounded['S5']['honesty_rate']:.1%}; grounded overcommit="
            f"{grounded['S5']['overcommit_rate']:.1%}.",
            {"baseline": baseline["S5"], "grounded": grounded["S5"]},
        ),
        gate(
            "H6_IMMUTABLE_DETERMINISTIC",
            deterministic_sample
            and comparison["snapshot_immutable"]
            and comparison["repeated_grounded_reads_exact"],
            f"Sample deterministic={deterministic_sample}; snapshots immutable="
            f"{comparison['snapshot_immutable']}; repeated grounded reads exact="
            f"{comparison['repeated_grounded_reads_exact']}.",
            {
                "sample_hash": sample_hash(episodes_a),
                "snapshot_immutable": comparison["snapshot_immutable"],
                "repeated_grounded_reads_exact": comparison[
                    "repeated_grounded_reads_exact"
                ],
            },
        ),
        gate(
            "H7_NO_MUTATION_PATH",
            all(not values for values in no_mutation["present"].values()),
            f"Mutation capabilities present={no_mutation['present']}.",
            no_mutation,
        ),
        gate(
            "H8_SEALS_UNTOUCHED",
            seals_untouched
            and substrate["substrate_unchanged"]
            and substrate["trainable_substrate_parameters"] == 0,
            f"Seals unchanged={seals_untouched}; substrate byte-identical="
            f"{substrate['substrate_unchanged']}; trainable substrate parameters="
            f"{substrate['trainable_substrate_parameters']}.",
            {
                "expected_seals": expected_seals,
                "actual_seals": actual_seals,
                "substrate": substrate,
            },
        ),
    ]
    all_pass = all(item["passed"] for item in gates)
    verdict = {
        "verdict": gates,
        "reference": {
            "run_timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "query_checkpoint": str(checkpoint),
            "heads_dir": str(heads_dir),
            "sample_seed": SAMPLE_SEED,
            "sample_hash": sample_hash(episodes_a),
            "trials_per_family": TRIALS_PER_FAMILY,
            "baseline_summary": baseline,
            "grounded_summary": grounded,
            "details": comparison["details"],
            "scope": (
                "Explicit-referent grounding safety over direct semantic object "
                "reads. Does not target or close F1 semantic coverage and is not "
                "general coreference resolution or Pas 7a runtime integration."
            ),
            "claim_status": "MEASURED in one local environment; not PROVEN.",
            "all_pass": all_pass,
        },
    }
    verdict_path = results_dir / "verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(SEP, flush=True)
    print("[INFO] D_Cortex v15.7b-G explicit referent-grounding verdict", flush=True)
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
    parser = argparse.ArgumentParser(description="D_Cortex referent grounding verdict")
    parser.add_argument(
        "--query-checkpoint",
        default=str(source_root / "runs" / "warmstart" / "warmstarted_init.pt"),
    )
    parser.add_argument(
        "--heads-dir",
        default=str(REPO_ROOT / "runs" / "semantic_contextual" / "results"),
    )
    parser.add_argument(
        "--run-dir",
        default=str(REPO_ROOT / "runs" / "semantic_referent_grounding"),
    )
    return parser


def main() -> int:
    """CLI entry point."""
    return run(build_argparser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
