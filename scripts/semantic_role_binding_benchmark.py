# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b-RB0 frozen non-trivial role-binding benchmark.

import argparse
import hashlib
import itertools
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.semantic_bridge_end_to_end import extract_definitions, sha256_file
from scripts.semantic_likelihood_probe import SEALED_SHA256, SEALED_SOURCE, gate

SEP = "=" * 70
RECORDS_PER_FAMILY = 400
SAMPLE_SEED = 20261520
KNOWN_FAMILIES = ("RB1", "RB2", "RB3", "RB4")
AMBIGUOUS_FAMILIES = ("RB5",)
FAMILIES = KNOWN_FAMILIES + AMBIGUOUS_FAMILIES
TupleFact = Tuple[str, str, str]
Baseline = Callable[["RoleBindingRecord"], Tuple[TupleFact, ...]]


@dataclass(frozen=True)
class RoleBindingRecord:
    """One controlled two-entity, two-value role-binding record."""

    family: str
    index: int
    text: str
    attribute: str
    entities: Tuple[str, str]
    values: Tuple[str, str]
    expected: Tuple[TupleFact, ...]
    ambiguous: bool

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the benchmark record."""
        data = asdict(self)
        data["entities"] = list(self.entities)
        data["values"] = list(self.values)
        data["expected"] = [list(item) for item in self.expected]
        return data


def _tokens(text: str) -> Tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text.lower()))


def _first_phrase_position(text: str, phrase: str) -> int:
    source = _tokens(text)
    target = _tokens(phrase)
    for start in range(0, len(source) - len(target) + 1):
        if source[start : start + len(target)] == target:
            return start
    raise ValueError(f"phrase {phrase!r} not found in text {text!r}")


def _facts(
    attribute: str, assignments: Sequence[Tuple[str, str]]
) -> Tuple[TupleFact, ...]:
    return tuple(sorted((entity, attribute, value) for entity, value in assignments))


def build_sample(
    definitions: Mapping[str, Any],
    records_per_family: int = RECORDS_PER_FAMILY,
    seed: int = SAMPLE_SEED,
) -> Tuple[RoleBindingRecord, ...]:
    """Build a deterministic role-binding benchmark sample."""
    if records_per_family < 1:
        raise ValueError("records_per_family must be positive")
    rng = random.Random(seed)
    entities = tuple(definitions["HOLDOUT_ENTITIES_SINGLE"])
    attributes = tuple(definitions["HOLDOUT_ATTR_TYPES"])
    values = definitions["HOLDOUT_ATTR_VALUES"]
    records: List[RoleBindingRecord] = []
    seen_texts = set()
    for family in FAMILIES:
        family_records: List[RoleBindingRecord] = []
        attempts = 0
        while len(family_records) < records_per_family:
            attempts += 1
            if attempts > records_per_family * 100:
                raise RuntimeError(f"unable to build unique records for {family}")
            attribute = rng.choice(attributes)
            entity_a, entity_b = rng.sample(entities, 2)
            value_a, value_b = rng.sample(values[attribute], 2)
            if family == "RB1":
                text = (
                    f"The {entity_a} is {value_a}, while the {entity_b} is "
                    f"{value_b}."
                )
                expected = _facts(
                    attribute, ((entity_a, value_a), (entity_b, value_b))
                )
                ambiguous = False
            elif family == "RB2":
                text = (
                    f"The {entity_a} is not {value_a} but {value_b}; the "
                    f"{entity_b} is not {value_b} but {value_a}."
                )
                expected = _facts(
                    attribute, ((entity_a, value_b), (entity_b, value_a))
                )
                ambiguous = False
            elif family == "RB3":
                text = (
                    f"The value {value_a} was rejected for the {entity_a} and "
                    f"assigned to the {entity_b}; the {entity_a} instead "
                    f"received {value_b}."
                )
                expected = _facts(
                    attribute, ((entity_a, value_b), (entity_b, value_a))
                )
                ambiguous = False
            elif family == "RB4":
                text = (
                    f"Between the {entity_a} and the {entity_b}, {value_a} "
                    f"describes the latter and {value_b} the former."
                )
                expected = _facts(
                    attribute, ((entity_a, value_b), (entity_b, value_a))
                )
                ambiguous = False
            else:
                text = (
                    f"Either the {entity_a} or the {entity_b} is {value_a}; "
                    f"the other is {value_b}."
                )
                expected = ()
                ambiguous = True
            if text in seen_texts:
                continue
            seen_texts.add(text)
            family_records.append(
                RoleBindingRecord(
                    family=family,
                    index=len(family_records),
                    text=text,
                    attribute=attribute,
                    entities=tuple(sorted((entity_a, entity_b))),
                    values=tuple(sorted((value_a, value_b))),
                    expected=expected,
                    ambiguous=ambiguous,
                )
            )
        records.extend(family_records)
    return tuple(records)


def sample_hash(records: Sequence[RoleBindingRecord]) -> str:
    """Return deterministic SHA-256 for the complete sample."""
    payload = json.dumps(
        [record.to_dict() for record in records],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ordered_baseline(record: RoleBindingRecord) -> Tuple[TupleFact, ...]:
    """Pair entities and values by first textual occurrence."""
    entities = sorted(record.entities, key=lambda item: _first_phrase_position(record.text, item))
    values = sorted(record.values, key=lambda item: _first_phrase_position(record.text, item))
    return _facts(record.attribute, tuple(zip(entities, values)))


def distance_baseline(record: RoleBindingRecord) -> Tuple[TupleFact, ...]:
    """Choose the one-to-one assignment with minimum first-position distance."""
    entities = tuple(sorted(record.entities))
    positions_entity = {
        entity: _first_phrase_position(record.text, entity) for entity in entities
    }
    positions_value = {
        value: _first_phrase_position(record.text, value) for value in record.values
    }
    ranked = []
    for values in itertools.permutations(sorted(record.values)):
        cost = sum(
            abs(positions_entity[entity] - positions_value[value])
            for entity, value in zip(entities, values)
        )
        ranked.append((cost, values))
    _, selected = min(ranked)
    return _facts(record.attribute, tuple(zip(entities, selected)))


def cartesian_baseline(record: RoleBindingRecord) -> Tuple[TupleFact, ...]:
    """Emit every lexical entity-value combination without binding."""
    return _facts(
        record.attribute,
        tuple((entity, value) for entity in record.entities for value in record.values),
    )


def safe_abstain_baseline(record: RoleBindingRecord) -> Tuple[TupleFact, ...]:
    """Emit no facts."""
    del record
    return ()


BASELINES: Dict[str, Baseline] = {
    "ordered_first_occurrence": ordered_baseline,
    "minimum_distance": distance_baseline,
    "lexical_cartesian": cartesian_baseline,
    "safe_abstain": safe_abstain_baseline,
}


def score_baseline(
    records: Sequence[RoleBindingRecord], baseline: Baseline
) -> Dict[str, Any]:
    """Score one lexical baseline on known and ambiguous records."""
    known = [record for record in records if not record.ambiguous]
    ambiguous = [record for record in records if record.ambiguous]
    exact = sum(baseline(record) == record.expected for record in known)
    covered = sum(bool(baseline(record)) for record in known)
    overcommit = sum(bool(baseline(record)) for record in ambiguous)
    per_family = {}
    for family in FAMILIES:
        subset = [record for record in records if record.family == family]
        per_family[family] = {
            "n": len(subset),
            "exact": sum(baseline(record) == record.expected for record in subset),
            "nonempty": sum(bool(baseline(record)) for record in subset),
        }
        per_family[family]["exact_rate"] = (
            per_family[family]["exact"] / len(subset)
        )
    return {
        "known_n": len(known),
        "known_exact": exact,
        "known_exact_rate": exact / len(known),
        "known_coverage": covered,
        "known_coverage_rate": covered / len(known),
        "ambiguous_n": len(ambiguous),
        "ambiguous_overcommit": overcommit,
        "ambiguous_overcommit_rate": overcommit / len(ambiguous),
        "per_family": per_family,
    }


def structure_checks(
    records: Sequence[RoleBindingRecord], definitions: Mapping[str, Any]
) -> Dict[str, Any]:
    """Verify benchmark truth structure and ontology-only sourcing."""
    ontology_entities = set(definitions["HOLDOUT_ENTITIES_SINGLE"])
    ontology_attributes = set(definitions["HOLDOUT_ATTR_TYPES"])
    ontology_values = {
        value
        for attribute in definitions["HOLDOUT_ATTR_TYPES"]
        for value in definitions["HOLDOUT_ATTR_VALUES"][attribute]
    }
    known_ok = True
    ambiguous_ok = True
    ontology_only = True
    inventory_ok = True
    for record in records:
        ontology_only &= (
            set(record.entities) <= ontology_entities
            and record.attribute in ontology_attributes
            and set(record.values) <= ontology_values
        )
        inventory_ok &= all(
            _first_phrase_position(record.text, item) >= 0
            for item in record.entities + record.values
        )
        if record.ambiguous:
            ambiguous_ok &= len(record.expected) == 0
        else:
            expected_entities = {item[0] for item in record.expected}
            expected_values = {item[2] for item in record.expected}
            known_ok &= (
                len(record.entities) == 2
                and len(set(record.entities)) == 2
                and len(record.values) == 2
                and len(set(record.values)) == 2
                and len(record.expected) == 2
                and expected_entities == set(record.entities)
                and expected_values == set(record.values)
                and all(item[1] == record.attribute for item in record.expected)
            )
    return {
        "known_structure": known_ok,
        "ambiguous_structure": ambiguous_ok,
        "ontology_only": ontology_only,
        "lexical_inventory_present": inventory_ok,
    }


def run(args: argparse.Namespace) -> int:
    """Run the frozen role-binding benchmark verdict."""
    results_dir = Path(args.run_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    definitions = extract_definitions()
    sample_a = build_sample(definitions)
    sample_b = build_sample(definitions)
    hash_a = sample_hash(sample_a)
    hash_b = sample_hash(sample_b)
    deterministic = hash_a == hash_b
    structure = structure_checks(sample_a, definitions)
    duplicate_count = len(sample_a) - len({record.text for record in sample_a})
    scores = {
        name: score_baseline(sample_a, baseline)
        for name, baseline in BASELINES.items()
    }
    non_abstaining = [
        scores[name]["known_exact_rate"]
        for name in ("ordered_first_occurrence", "minimum_distance", "lexical_cartesian")
    ]
    best_lexical = max(non_abstaining)
    no_trivial_safe_win = all(
        not (
            score["known_exact_rate"] >= 0.50
            and score["ambiguous_overcommit_rate"] <= 0.02
        )
        for score in scores.values()
    )
    safe_abstain_cost = (
        scores["safe_abstain"]["ambiguous_overcommit_rate"] == 0.0
        and scores["safe_abstain"]["known_coverage_rate"] == 0.0
    )
    sealed_untouched = sha256_file(SEALED_SOURCE) == SEALED_SHA256
    gates = [
        gate(
            "RB0_DETERMINISTIC",
            deterministic,
            f"Repeated sample construction exact={deterministic}; hash={hash_a}.",
            {"hash_a": hash_a, "hash_b": hash_b},
        ),
        gate(
            "RB1_STRUCTURE",
            structure["known_structure"] and structure["lexical_inventory_present"],
            f"Known one-to-one structure={structure['known_structure']}; "
            f"lexical inventory present={structure['lexical_inventory_present']}.",
            structure,
        ),
        gate(
            "RB2_AMBIGUITY",
            structure["ambiguous_structure"],
            f"Every ambiguous record has no committed truth mapping="
            f"{structure['ambiguous_structure']}.",
            structure,
        ),
        gate(
            "RB3_NO_DUPLICATE_TEXT",
            duplicate_count == 0,
            f"Duplicate record texts={duplicate_count}.",
            {"duplicates": duplicate_count, "total": len(sample_a)},
        ),
        gate(
            "RB4_LEXICAL_NON_TRIVIAL",
            best_lexical < 0.50,
            f"Best non-abstaining lexical baseline exact-known={best_lexical:.1%}.",
            {
                name: scores[name]
                for name in (
                    "ordered_first_occurrence",
                    "minimum_distance",
                    "lexical_cartesian",
                )
            },
        ),
        gate(
            "RB5_NO_TRIVIAL_SAFE_WIN",
            no_trivial_safe_win,
            f"No baseline combines >=50% known exact with <=2% ambiguity "
            f"overcommit={no_trivial_safe_win}.",
            scores,
        ),
        gate(
            "RB6_SAFE_ABSTAIN_COST",
            safe_abstain_cost,
            f"Safe abstain ambiguity overcommit="
            f"{scores['safe_abstain']['ambiguous_overcommit_rate']:.1%}; "
            f"known coverage={scores['safe_abstain']['known_coverage_rate']:.1%}.",
            scores["safe_abstain"],
        ),
        gate(
            "RB7_SEALED_SOURCE",
            sealed_untouched and structure["ontology_only"],
            f"Pas 7a sealed source unchanged={sealed_untouched}; ontology-only "
            f"sourcing={structure['ontology_only']}.",
            {
                "expected_hash": SEALED_SHA256,
                "actual_hash": sha256_file(SEALED_SOURCE),
                "ontology_only": structure["ontology_only"],
            },
        ),
    ]
    all_pass = all(item["passed"] for item in gates)
    sample_path = results_dir / "sample.json"
    sample_path.write_text(
        json.dumps([record.to_dict() for record in sample_a], indent=2),
        encoding="utf-8",
    )
    verdict = {
        "verdict": gates,
        "reference": {
            "run_timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "sample_seed": SAMPLE_SEED,
            "records_per_family": RECORDS_PER_FAMILY,
            "sample_hash": hash_a,
            "sample_path": str(sample_path),
            "scores": scores,
            "structure": structure,
            "scope": (
                "Benchmark and lexical baselines only. Passing establishes "
                "controlled role-binding non-triviality, not a semantic fact "
                "internalizer or memory improvement."
            ),
            "claim_status": "MEASURED in one local environment; not PROVEN.",
            "all_pass": all_pass,
        },
    }
    verdict_path = results_dir / "verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(SEP, flush=True)
    print("[INFO] D_Cortex v15.7b-RB0 role-binding benchmark verdict", flush=True)
    for item in gates:
        label = "✓ PASS" if item["passed"] else "✗ FAIL"
        print(f"{label} [{item['criterion_id']}] {item['evidence']}", flush=True)
    print(SEP, flush=True)
    print(f"[INFO] Overall: {'ALL GATES PASS' if all_pass else 'GATE FAILURE'}", flush=True)
    print(f"[INFO] Verdict: {verdict_path}", flush=True)
    return 0 if all_pass else 1


def build_argparser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="D_Cortex role-binding benchmark")
    parser.add_argument(
        "--run-dir",
        default=str(REPO_ROOT / "runs" / "semantic_role_binding_benchmark"),
    )
    return parser


def main() -> int:
    """CLI entry point."""
    return run(build_argparser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
