# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b-O frozen direct semantic object-read verdict.

import argparse
import gc
import hashlib
import inspect
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
    AdapterDecision,
    ConservativeSemanticAdapter,
    DecisionStatus,
    HypothesisMode,
    RequestedDestination,
    SemanticHypothesis,
)
from dcortex.semantic_object_reader import (
    DirectSemanticObjectReader,
    ObjectMemorySnapshot,
    ObjectReadStatus,
    SemanticObjectReadResult,
)
from dcortex.semantic_producer import (
    ConservativeTrainedQueryProducer,
    DCortexContextualFeatureBackend,
    PooledSemanticClassificationBackend,
)
from scripts.semantic_bridge_end_to_end import (
    extract_definitions,
    load_semantic_backend,
    sha256_file,
)
from scripts.semantic_contextual_curriculum import load_contextual_model
from scripts.semantic_likelihood_probe import SEALED_SHA256, SEALED_SOURCE, gate
from scripts.train_semantic_internalizer import (
    ATTRIBUTE_MARGIN,
    ENTITY_MARGIN,
    UNKNOWN_ATTRIBUTE,
    UNKNOWN_ENTITY,
    state_tensor_hash,
)

SEP = "=" * 70
ENC = tiktoken.get_encoding("gpt2")
TRIALS_PER_FAMILY = 200
SAMPLE_SEED = 20261480
FAMILIES = ("F1", "F3", "F5", "S5", "S6")
KNOWN_FAMILIES = ("F1", "F3", "F5")

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


@dataclass
class ObjectReadEpisode:
    """One frozen direct semantic object-read trial."""

    family: str
    index: int
    query: str
    entity: Optional[str]
    attribute: str
    target_value: Optional[str]
    target_unknown: bool
    form_index: Optional[int]
    snapshot: ObjectMemorySnapshot
    hypothesis: Optional[SemanticHypothesis] = None
    decision: Optional[AdapterDecision] = None
    result: Optional[SemanticObjectReadResult] = None

    def sample_dict(self) -> Dict[str, Any]:
        """Return deterministic sample-only data."""
        return {
            "family": self.family,
            "index": self.index,
            "query": self.query,
            "entity": self.entity,
            "attribute": self.attribute,
            "target_value": self.target_value,
            "target_unknown": self.target_unknown,
            "form_index": self.form_index,
            "snapshot": self.snapshot.to_dict(),
        }


def sample_hash(episodes: Sequence[ObjectReadEpisode]) -> str:
    """Return deterministic SHA-256 for a complete sample."""
    payload = json.dumps(
        [episode.sample_dict() for episode in episodes],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_query(entity: str, attribute: str) -> str:
    """Return a standard control query for one semantic coordinate."""
    templates = {
        "color": "What color is the {entity}? The {entity} is",
        "size": "What size is the {entity}? The {entity} is",
        "location": "Where is the {entity}? The {entity} is in the",
        "state": "What state is the {entity} in? The {entity} is",
    }
    return templates[attribute].format(entity=entity)


def build_sample(definitions: Mapping[str, Any]) -> List[ObjectReadEpisode]:
    """Build the newly frozen direct object-read sample."""
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
                snapshot = ObjectMemorySnapshot(
                    committed=((entity, attribute, value),)
                )
                episodes.append(
                    ObjectReadEpisode(
                        family,
                        index,
                        query,
                        entity,
                        attribute,
                        value,
                        False,
                        form_index,
                        snapshot,
                    )
                )
            elif family == "S5":
                entity = rng.choice(entities)
                value_a, value_b = rng.sample(values[attribute], 2)
                snapshot = ObjectMemorySnapshot(
                    provisional=(
                        (entity, attribute, value_a),
                        (entity, attribute, value_b),
                    )
                )
                episodes.append(
                    ObjectReadEpisode(
                        family,
                        index,
                        canonical_query(entity, attribute),
                        entity,
                        attribute,
                        None,
                        True,
                        None,
                        snapshot,
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
                snapshot = ObjectMemorySnapshot(
                    committed=(
                        (entity_a, attribute, value_a),
                        (entity_b, attribute, value_b),
                    )
                )
                episodes.append(
                    ObjectReadEpisode(
                        family,
                        index,
                        pronoun_query,
                        None,
                        attribute,
                        None,
                        True,
                        None,
                        snapshot,
                    )
                )
    return episodes


def produce_batch(
    episodes: Sequence[ObjectReadEpisode],
    backend: PooledSemanticClassificationBackend,
    provenance: str,
) -> None:
    """Produce adapter-approved semantic coordinates without textual routing."""
    if not episodes:
        return
    entity_probabilities, attribute_probabilities = backend.classify(
        [episode.query for episode in episodes]
    )
    for row, episode in enumerate(episodes):
        scores = (
            ConservativeTrainedQueryProducer._score_axis(
                "entity",
                entity_probabilities[row],
                backend.entity_ids,
                backend.unknown_entity_id,
                ENTITY_MARGIN,
            ),
            ConservativeTrainedQueryProducer._score_axis(
                "attribute",
                attribute_probabilities[row],
                backend.attribute_ids,
                backend.unknown_attribute_id,
                ATTRIBUTE_MARGIN,
            ),
        )
        if not all(score.passed for score in scores):
            continue
        selected = {score.axis: score.selected_id for score in scores}
        confidence = min(score.top_probability for score in scores)
        hypothesis = SemanticHypothesis(
            hypothesis_id=f"{episode.family}-{episode.index}",
            episode_id=episode.index,
            mode=HypothesisMode.QUERY,
            source_text=episode.query,
            producer="conservative_trained_query_producer",
            producer_version="1.0",
            provenance=(provenance, f"trial:{episode.index}"),
            confidence=confidence,
            uncertainty=1.0 - confidence,
            requested_destination=RequestedDestination.QUERY_ONLY,
            entity_id=selected["entity"],
            attr_type=selected["attribute"],
        )
        episode.hypothesis = hypothesis
        episode.decision = ConservativeSemanticAdapter().submit(hypothesis)


def build_decisions(
    episodes: Sequence[ObjectReadEpisode],
    checkpoint: Path,
    heads_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Build frozen query decisions and verify the substrate remains frozen."""
    model, state, payload = load_contextual_model(checkpoint, device)
    state_before = state_tensor_hash(state)
    feature_backend = DCortexContextualFeatureBackend(
        model=model,
        tokenizer=ENC.encode_ordinary,
        max_seq_len=128,
        batch_size=128,
        backend_version=checkpoint.name,
    )
    backends = {
        f"fold_{fold}": load_semantic_backend(
            feature_backend, heads_dir / f"fold_{fold}_best_head.pt", device
        )
        for fold in range(4)
    }
    backends["final"] = load_semantic_backend(
        feature_backend, heads_dir / "final_best_head.pt", device
    )
    for family in ("F1", "F3"):
        for fold in range(4):
            produce_batch(
                [
                    episode
                    for episode in episodes
                    if episode.family == family and episode.form_index == fold
                ],
                backends[f"fold_{fold}"],
                f"family:{family}:heldout_fold:{fold}",
            )
    for family in ("F5", "S5", "S6"):
        produce_batch(
            [episode for episode in episodes if episode.family == family],
            backends["final"],
            f"family:{family}:final_head",
        )
    state_after = state_tensor_hash(state)
    trainable_substrate = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    output = {
        "substrate_hash_before": state_before,
        "substrate_hash_after": state_after,
        "substrate_unchanged": state_before == state_after,
        "trainable_substrate_parameters": trainable_substrate,
        "checkpoint_mtime": datetime.fromtimestamp(checkpoint.stat().st_mtime)
        .astimezone()
        .isoformat(timespec="seconds"),
    }
    del backends, feature_backend, model, state, payload
    gc.collect()
    torch.cuda.empty_cache()
    return output


def run_reads(episodes: Sequence[ObjectReadEpisode]) -> Dict[str, Any]:
    """Run every direct coordinate read twice and verify immutability."""
    reader = DirectSemanticObjectReader()
    immutable = True
    deterministic = True
    details: List[Dict[str, Any]] = []
    for episode in episodes:
        before = episode.snapshot.fingerprint
        first = reader.read(episode.snapshot, episode.hypothesis, episode.decision)
        middle = episode.snapshot.fingerprint
        second = reader.read(episode.snapshot, episode.hypothesis, episode.decision)
        after = episode.snapshot.fingerprint
        immutable &= before == middle == after == first.snapshot_fingerprint
        deterministic &= first.to_json() == second.to_json()
        episode.result = first
        details.append(
            {
                "family": episode.family,
                "index": episode.index,
                "target_value": episode.target_value,
                "hypothesis_entity": (
                    None if episode.hypothesis is None else episode.hypothesis.entity_id
                ),
                "hypothesis_attribute": (
                    None if episode.hypothesis is None else episode.hypothesis.attr_type
                ),
                "result": first.to_dict(),
            }
        )
    return {
        "snapshot_immutable": immutable,
        "repeated_reads_exact": deterministic,
        "details": details,
    }


def summarize(episodes: Sequence[ObjectReadEpisode]) -> Dict[str, Dict[str, Any]]:
    """Summarize direct object-read outcomes per family."""
    summaries: Dict[str, Dict[str, Any]] = {}
    for family in FAMILIES:
        subset = [episode for episode in episodes if episode.family == family]
        assert all(episode.result is not None for episode in subset)
        if family in KNOWN_FAMILIES:
            correct = sum(
                episode.result is not None
                and episode.result.status == ObjectReadStatus.FOUND_COMMITTED
                and episode.result.pred_value == episode.target_value
                for episode in subset
            )
            wrong = sum(
                episode.result is not None
                and episode.result.pred_value is not None
                and episode.result.pred_value != episode.target_value
                for episode in subset
            )
            summaries[family] = {
                "n": len(subset),
                "correct": correct,
                "correct_rate": correct / len(subset),
                "wrong_committed_read": wrong,
                "wrong_rate": wrong / len(subset),
                "refused_or_missing": len(subset) - correct - wrong,
            }
        else:
            overcommit = sum(
                episode.result is not None and episode.result.pred_value is not None
                for episode in subset
            )
            disputed = sum(
                episode.result is not None
                and episode.result.status == ObjectReadStatus.FOUND_DISPUTED
                for episode in subset
            )
            refused = sum(
                episode.result is not None
                and episode.result.status == ObjectReadStatus.REFUSED_INPUT
                for episode in subset
            )
            summaries[family] = {
                "n": len(subset),
                "honest_no_committed_value": len(subset) - overcommit,
                "honesty_rate": (len(subset) - overcommit) / len(subset),
                "overcommit": overcommit,
                "overcommit_rate": overcommit / len(subset),
                "disputed": disputed,
                "refused": refused,
                "disputed_or_refused_rate": (disputed + refused) / len(subset),
            }
    return summaries


def accepted_query_only_check(episodes: Sequence[ObjectReadEpisode]) -> bool:
    """Verify every non-refused read came from an accepted query-only decision."""
    for episode in episodes:
        assert episode.result is not None
        if episode.result.status == ObjectReadStatus.REFUSED_INPUT:
            continue
        if (
            episode.hypothesis is None
            or episode.decision is None
            or episode.decision.status != DecisionStatus.ACCEPT_QUERY
            or episode.hypothesis.mode != HypothesisMode.QUERY
            or episode.hypothesis.requested_destination
            != RequestedDestination.QUERY_ONLY
        ):
            return False
    reader = DirectSemanticObjectReader()
    snapshot = ObjectMemorySnapshot(committed=(("dragon", "color", "red"),))
    return reader.read(snapshot, None, None).status == ObjectReadStatus.REFUSED_INPUT


def direct_coordinate_check() -> Dict[str, Any]:
    """Verify the reader API has no raw-text or routing parameter."""
    parameters = tuple(inspect.signature(DirectSemanticObjectReader.read).parameters)
    forbidden = ("query", "text", "parser", "route", "token")
    return {
        "parameters": parameters,
        "direct": parameters == ("self", "snapshot", "hypothesis", "decision"),
        "forbidden_parameter_present": any(
            marker in parameter.lower()
            for parameter in parameters
            for marker in forbidden
        ),
    }


def no_mutation_check() -> Dict[str, Any]:
    """Verify the reader exposes no mutation operation."""
    reader = DirectSemanticObjectReader()
    forbidden = (
        "write",
        "commit",
        "consolidate",
        "promote",
        "retrograde",
        "prune",
        "update",
    )
    present = [name for name in forbidden if hasattr(reader, name)]
    return {"forbidden_attributes": list(forbidden), "present": present}


def run(args: argparse.Namespace) -> int:
    """Run the frozen direct semantic object-read verdict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Frozen direct object-read verdict requires CUDA")
    checkpoint = Path(args.query_checkpoint)
    heads_dir = Path(args.heads_dir)
    results_dir = Path(args.run_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    definitions = extract_definitions()
    episodes_a = build_sample(definitions)
    episodes_b = build_sample(definitions)
    sample_hash_a = sample_hash(episodes_a)
    sample_hash_b = sample_hash(episodes_b)
    sample_deterministic = sample_hash_a == sample_hash_b

    print(SEP, flush=True)
    print("[INFO] Building frozen semantic query decisions", flush=True)
    substrate = build_decisions(episodes_a, checkpoint, heads_dir, device)
    read_checks = run_reads(episodes_a)
    summaries = summarize(episodes_a)
    accepted_only = accepted_query_only_check(episodes_a)
    direct_coordinate = direct_coordinate_check()
    no_mutation = no_mutation_check()
    actual_seals = {
        name: sha256_file(path) for name, (path, _) in SEALED_FILES.items()
    }
    expected_seals = {
        name: expected for name, (_, expected) in SEALED_FILES.items()
    }
    seals_untouched = actual_seals == expected_seals
    known_correct = all(
        summaries[family]["correct_rate"] >= 0.85 for family in KNOWN_FAMILIES
    )
    known_wrong = all(
        summaries[family]["wrong_rate"] <= 0.01 for family in KNOWN_FAMILIES
    )
    s5_honest = (
        summaries["S5"]["disputed_or_refused_rate"] >= 0.95
        and summaries["S5"]["overcommit_rate"] <= 0.02
    )
    s6_honest = (
        summaries["S6"]["honesty_rate"] >= 0.95
        and summaries["S6"]["overcommit_rate"] <= 0.02
    )
    gates = [
        gate(
            "O0_ACCEPTED_QUERY_ONLY",
            accepted_only,
            f"Every non-refused read came from accepted QUERY_ONLY={accepted_only}.",
            {"accepted_query_only": accepted_only},
        ),
        gate(
            "O1_DIRECT_COORDINATE",
            direct_coordinate["direct"]
            and not direct_coordinate["forbidden_parameter_present"],
            "Object reader accepts snapshot+hypothesis+decision only; "
            f"forbidden raw-text parameter={direct_coordinate['forbidden_parameter_present']}.",
            direct_coordinate,
        ),
        gate(
            "O2_SNAPSHOT_IMMUTABLE",
            read_checks["snapshot_immutable"],
            f"Every read preserved snapshot fingerprint={read_checks['snapshot_immutable']}.",
            {"snapshot_immutable": read_checks["snapshot_immutable"]},
        ),
        gate(
            "O3_F1_F3_F5_CORRECT",
            known_correct,
            "Direct correct reads: "
            + ", ".join(
                f"{family}={summaries[family]['correct_rate']:.1%}"
                for family in KNOWN_FAMILIES
            )
            + ".",
            {family: summaries[family] for family in KNOWN_FAMILIES},
        ),
        gate(
            "O4_WRONG_READ",
            known_wrong,
            "Wrong committed reads: "
            + ", ".join(
                f"{family}={summaries[family]['wrong_rate']:.1%}"
                for family in KNOWN_FAMILIES
            )
            + ".",
            {family: summaries[family] for family in KNOWN_FAMILIES},
        ),
        gate(
            "O5_S5_DISPUTE_HONESTY",
            s5_honest,
            f"S5 disputed/refused={summaries['S5']['disputed_or_refused_rate']:.1%}; "
            f"overcommit={summaries['S5']['overcommit_rate']:.1%}.",
            summaries["S5"],
        ),
        gate(
            "O6_S6_REFERENT_HONESTY",
            s6_honest,
            f"S6 no-committed-value honesty={summaries['S6']['honesty_rate']:.1%}; "
            f"overcommit={summaries['S6']['overcommit_rate']:.1%}.",
            summaries["S6"],
        ),
        gate(
            "O7_NO_MUTATION_PATH",
            not no_mutation["present"],
            f"Reader mutation capabilities present={no_mutation['present']}.",
            no_mutation,
        ),
        gate(
            "O8_DETERMINISTIC",
            sample_deterministic and read_checks["repeated_reads_exact"],
            f"Sample deterministic={sample_deterministic}; repeated reads exact="
            f"{read_checks['repeated_reads_exact']}.",
            {
                "sample_hash_a": sample_hash_a,
                "sample_hash_b": sample_hash_b,
                "repeated_reads_exact": read_checks["repeated_reads_exact"],
            },
        ),
        gate(
            "O9_SEALS_UNTOUCHED",
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
            "sample_hash": sample_hash_a,
            "sample_seed": SAMPLE_SEED,
            "trials_per_family": TRIALS_PER_FAMILY,
            "summaries": summaries,
            "substrate": substrate,
            "details": read_checks["details"],
            "scope": (
                "Direct semantic-coordinate reads over immutable epistemic object "
                "snapshots. Known-family snapshots are pre-populated, so this "
                "isolates query-side reading. Not Pas 7a runtime integration or "
                "fact-side internalization."
            ),
            "claim_status": "MEASURED in one local environment; not PROVEN.",
            "all_pass": all_pass,
        },
    }
    verdict_path = results_dir / "verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(SEP, flush=True)
    print("[INFO] D_Cortex v15.7b-O direct semantic object-read verdict", flush=True)
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
    parser = argparse.ArgumentParser(description="D_Cortex direct object-read verdict")
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
        default=str(REPO_ROOT / "runs" / "semantic_object_read"),
    )
    return parser


def main() -> int:
    """CLI entry point."""
    return run(build_argparser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
