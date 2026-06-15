# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# D_Cortex v15.7b-RB4 frozen independent role-binding corpus audit.

import argparse
import hashlib
import itertools
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
SEP = "=" * 70

TupleFact = Tuple[str, str, str]
Baseline = Callable[["IndependentRoleRecord"], Tuple[TupleFact, ...]]

ALLOWED_SPLITS = ("train", "validation", "evaluation")
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "validation": "validation",
    "val": "validation",
    "eval": "evaluation",
    "evaluation": "evaluation",
    "test": "evaluation",
}

RESERVED_DEVELOPMENT_FAMILIES = {"RB0", "RB1", "RB2", "RB3", "RB4", "RB5"}
BAD_PROVENANCE_TOKENS = {
    "",
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
    "synthetic",
    "generated",
    "template",
}

MIN_TOTAL_RECORDS = 120
MIN_RECORDS_PER_SPLIT = 20
MIN_EVALUATION_KNOWN_RECORDS = 40
MIN_EVALUATION_AMBIGUOUS_RECORDS = 10
MAX_POSITION_BASELINE_EXACT_RATE = 0.50
MAX_SAFE_AMBIGUITY_OVERCOMMIT_RATE = 0.02
MAX_DUPLICATE_EXAMPLES = 20

PREDECESSOR_ARTIFACTS = {
    "runs/semantic_role_binding_benchmark/results/sample.json": (
        "2c4a2dd117535b6ee7929bbb1c9882eddd95ab50fd97c4b1a343a9c196fd3625"
    ),
    "runs/semantic_role_binding_benchmark/results/verdict.json": (
        "c4dcd47d471d679fa20e78e178943599d2cda383e9fbcc23b5bcde39fe1bb876"
    ),
    "runs/semantic_role_binder/results/verdict.json": (
        "92035e5d8148ead5d03d3ba5fac571bc646103ee1acd8a2677216e07d30c0b6f"
    ),
    "runs/semantic_role_conditioned/results/verdict.json": (
        "e370695d2bce6c6843e8b4514a31e3e37c3e14a78e8eabe7879d9250a1b54705"
    ),
    "runs/semantic_syntax_holdout/results/verdict.json": (
        "a6551156e655b714ceB7360c9792c55a69ffde4de20944c0757011dba8cd22c4"
    ).lower(),
}

SEALED_ARTIFACTS = {
    "steps/13_v15_7a_consolidation/code.py": (
        "25b4906ecc05a6b51b10902e54332a0ec2b26c4c622aa4e6ee74bd4961369aa3"
    ),
    "dcortex/semantic_adapter.py": (
        "719afbe8359af4ee98006e0fa19f169a0a40f22ee0cc41e5dc1459d6e07e605e"
    ),
    "dcortex/semantic_producer.py": (
        "24797a121c90f78161518362f0401bbc1b161546e87bac194adb4144f4aa67d0"
    ),
    "scripts/semantic_contextual_curriculum.py": (
        "bab8ec1268d5000d5a9da5ae5946d29946822efa9a6b0b3ce53be59052631b57"
    ),
    "dcortex/semantic_query_bridge.py": (
        "403d4d724a1bffee61ab9cdfa469adb0c4fb3afb75c04ad4d65ad3e7c86e1b43"
    ),
}


@dataclass(frozen=True)
class IndependentRoleRecord:
    """One RB4 independent two-entity, two-value role-binding record."""

    record_id: str
    split: str
    construction_family: str
    source_text: str
    attribute: str
    entities: Tuple[str, str]
    values: Tuple[str, str]
    expected: Tuple[TupleFact, ...]
    ambiguous: bool
    provenance: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the record into a stable dictionary."""
        data = asdict(self)
        data["entities"] = list(self.entities)
        data["values"] = list(self.values)
        data["expected"] = [list(item) for item in self.expected]
        return data


def gate(
    criterion_id: str,
    passed: bool,
    evidence: str,
    distribution: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build one frozen RB4 gate record."""
    return {
        "criterion_id": criterion_id,
        "passed": bool(passed),
        "evidence": evidence,
        "distribution": dict(distribution),
    }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hash of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_records_hash(records: Sequence[IndependentRoleRecord]) -> str:
    """Return a stable hash for the parsed RB4 corpus records."""
    payload = json.dumps(
        [record.to_dict() for record in records],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_split(value: Any) -> str:
    """Normalize one split label into the frozen split vocabulary."""
    split = str(value).strip().lower()
    if split not in SPLIT_ALIASES:
        raise ValueError(f"unknown split {value!r}; expected train/validation/evaluation")
    return SPLIT_ALIASES[split]


def normalize_space(text: str) -> str:
    """Lowercase and normalize whitespace for audit comparisons."""
    return re.sub(r"\s+", " ", text.strip().lower())


def tokens(text: str) -> Tuple[str, ...]:
    """Tokenize text for simple lexical baseline positioning."""
    return tuple(re.findall(r"[a-z0-9]+", text.lower()))


def phrase_position(text: str, phrase: str) -> int | None:
    """Return the first token position of a phrase, or None if absent."""
    source = tokens(text)
    target = tokens(phrase)
    if not target:
        return None
    for start in range(0, len(source) - len(target) + 1):
        if source[start : start + len(target)] == target:
            return start
    return None


def phrase_present(text: str, phrase: str) -> bool:
    """Return whether a phrase is lexically present in source text."""
    return phrase_position(text, phrase) is not None


def coerce_string_tuple(value: Any, field_name: str) -> Tuple[str, str]:
    """Coerce a two-item string sequence from input data."""
    if not isinstance(value, list) and not isinstance(value, tuple):
        raise ValueError(f"{field_name} must be a two-item list")
    items = tuple(str(item).strip() for item in value)
    if len(items) != 2:
        raise ValueError(f"{field_name} must contain exactly two items")
    if not all(items):
        raise ValueError(f"{field_name} contains an empty item")
    if len(set(item.lower() for item in items)) != 2:
        raise ValueError(f"{field_name} items must be distinct")
    return items  # type: ignore[return-value]


def coerce_provenance(value: Any) -> str:
    """Serialize provenance into a stable audit string."""
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value).strip()


def provenance_is_auditable(provenance: str) -> bool:
    """Return whether provenance is non-trivial enough for local audit."""
    lower = provenance.strip().lower()
    if lower in BAD_PROVENANCE_TOKENS:
        return False
    if any(token in lower for token in ("rb0", "rb1", "rb2", "rb3", "rb4", "rb5")):
        return False
    if len(lower) < 12:
        return False
    markers = ("doi", "isbn", "http", "file:", "source", "document", "citation")
    return any(marker in lower for marker in markers)


def parse_expected_fact(item: Any, default_attribute: str) -> TupleFact:
    """Parse one expected fact from a list or dictionary."""
    if isinstance(item, dict):
        entity = str(item.get("entity", "")).strip()
        attribute = str(item.get("attribute", default_attribute)).strip()
        value = str(item.get("value", "")).strip()
    elif isinstance(item, (list, tuple)) and len(item) == 3:
        entity = str(item[0]).strip()
        attribute = str(item[1]).strip()
        value = str(item[2]).strip()
    else:
        raise ValueError("expected facts must be dicts or three-item lists")
    if not entity or not attribute or not value:
        raise ValueError("expected fact contains an empty field")
    return entity, attribute, value


def facts(attribute: str, assignments: Sequence[Tuple[str, str]]) -> Tuple[TupleFact, ...]:
    """Build a stable sorted tuple of role-binding facts."""
    return tuple(sorted((entity, attribute, value) for entity, value in assignments))


def coerce_expected(value: Any, attribute: str) -> Tuple[TupleFact, ...]:
    """Coerce the expected fact list from input data."""
    if value is None:
        return ()
    if not isinstance(value, list) and not isinstance(value, tuple):
        raise ValueError("expected must be a list")
    return tuple(sorted(parse_expected_fact(item, attribute) for item in value))


def first_present(data: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """Return the first present input field from a list of aliases."""
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(keys[0])


def record_from_dict(data: Mapping[str, Any], fallback_index: int) -> IndependentRoleRecord:
    """Parse one independent corpus record from input JSON data."""
    record_id = str(data.get("record_id", data.get("id", fallback_index))).strip()
    split = normalize_split(first_present(data, ("split",)))
    construction_family = str(
        first_present(data, ("construction_family", "family", "source_family"))
    ).strip()
    source_text = str(first_present(data, ("source_text", "text"))).strip()
    attribute = str(first_present(data, ("attribute",))).strip()
    entities = coerce_string_tuple(first_present(data, ("entities",)), "entities")
    values = coerce_string_tuple(first_present(data, ("values",)), "values")
    ambiguous = bool(first_present(data, ("ambiguous",)))
    expected = coerce_expected(data.get("expected", ()), attribute)
    provenance = coerce_provenance(first_present(data, ("provenance", "source")))
    if not record_id:
        raise ValueError("record_id must be non-empty")
    if not construction_family:
        raise ValueError("construction_family must be non-empty")
    if not source_text:
        raise ValueError("source_text must be non-empty")
    if not attribute:
        raise ValueError("attribute must be non-empty")
    return IndependentRoleRecord(
        record_id=record_id,
        split=split,
        construction_family=construction_family,
        source_text=source_text,
        attribute=attribute,
        entities=entities,
        values=values,
        expected=expected,
        ambiguous=ambiguous,
        provenance=provenance,
    )


def read_raw_records(path: Path) -> List[Any]:
    """Read raw JSON or JSONL records from disk."""
    if not path.exists():
        raise RuntimeError(
            "Independent RB4 corpus not found: "
            f"{path}. Provide a JSON/JSONL corpus before running RB4. "
            "No model training or memory integration is permitted before this audit passes."
        )
    if path.suffix.lower() == ".jsonl":
        records = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at line {line_number}: {exc}") from exc
        return records
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return list(payload["records"])
    if isinstance(payload, list):
        return payload
    raise RuntimeError("corpus JSON must be a list or an object with a records list")


def load_corpus(path: Path) -> Tuple[Tuple[IndependentRoleRecord, ...], List[Dict[str, Any]]]:
    """Load an independent corpus and return parsed records plus parse errors."""
    raw_records = read_raw_records(path)
    records: List[IndependentRoleRecord] = []
    errors: List[Dict[str, Any]] = []
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            errors.append({"index": index, "error": "record is not an object"})
            continue
        try:
            records.append(record_from_dict(raw, index))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append({"index": index, "error": str(exc)})
    return tuple(records), errors


def expected_is_one_to_one(record: IndependentRoleRecord) -> bool:
    """Return whether a known record has exactly one one-to-one mapping."""
    if record.ambiguous:
        return not record.expected
    if len(record.expected) != 2:
        return False
    fact_entities = tuple(item[0] for item in record.expected)
    fact_attributes = tuple(item[1] for item in record.expected)
    fact_values = tuple(item[2] for item in record.expected)
    return (
        set(fact_entities) == set(record.entities)
        and set(fact_values) == set(record.values)
        and all(attribute == record.attribute for attribute in fact_attributes)
        and len(set(fact_entities)) == 2
        and len(set(fact_values)) == 2
    )


def inventory_present(record: IndependentRoleRecord) -> bool:
    """Return whether all declared entities and values occur in source text."""
    phrases = tuple(record.entities) + tuple(record.values)
    return all(phrase_present(record.source_text, phrase) for phrase in phrases)


def label_structure_report(
    records: Sequence[IndependentRoleRecord],
    parse_errors: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Audit label structure for all parsed records."""
    bad_records = []
    known = [record for record in records if not record.ambiguous]
    ambiguous = [record for record in records if record.ambiguous]
    for record in records:
        issues = []
        if not expected_is_one_to_one(record):
            issues.append("expected_mapping_not_one_to_one")
        if not inventory_present(record):
            issues.append("inventory_phrase_missing_from_text")
        if issues:
            bad_records.append({"record_id": record.record_id, "issues": issues})
    return {
        "records": len(records),
        "parse_errors": list(parse_errors)[:MAX_DUPLICATE_EXAMPLES],
        "parse_error_count": len(parse_errors),
        "known_records": len(known),
        "ambiguous_records": len(ambiguous),
        "bad_record_count": len(bad_records),
        "bad_records": bad_records[:MAX_DUPLICATE_EXAMPLES],
        "all_ok": len(parse_errors) == 0 and len(bad_records) == 0,
    }


def split_report(records: Sequence[IndependentRoleRecord]) -> Dict[str, Any]:
    """Report split counts and construction-family separation."""
    per_split: Dict[str, Dict[str, Any]] = {}
    for split in ALLOWED_SPLITS:
        subset = [record for record in records if record.split == split]
        families = sorted({record.construction_family for record in subset})
        reserved = sorted(
            family for family in families if family.upper() in RESERVED_DEVELOPMENT_FAMILIES
        )
        per_split[split] = {
            "records": len(subset),
            "known": sum(not record.ambiguous for record in subset),
            "ambiguous": sum(record.ambiguous for record in subset),
            "families": families,
            "reserved_development_families": reserved,
        }
    family_sets = {split: set(per_split[split]["families"]) for split in ALLOWED_SPLITS}
    overlaps = {
        "train_validation": sorted(family_sets["train"].intersection(family_sets["validation"])),
        "train_evaluation": sorted(family_sets["train"].intersection(family_sets["evaluation"])),
        "validation_evaluation": sorted(
            family_sets["validation"].intersection(family_sets["evaluation"])
        ),
    }
    split_counts_ok = (
        len(records) >= MIN_TOTAL_RECORDS
        and all(per_split[split]["records"] >= MIN_RECORDS_PER_SPLIT for split in ALLOWED_SPLITS)
        and per_split["evaluation"]["known"] >= MIN_EVALUATION_KNOWN_RECORDS
        and per_split["evaluation"]["ambiguous"] >= MIN_EVALUATION_AMBIGUOUS_RECORDS
    )
    pairwise_disjoint = all(not value for value in overlaps.values())
    reserved_absent = all(
        not per_split[split]["reserved_development_families"] for split in ALLOWED_SPLITS
    )
    return {
        "per_split": per_split,
        "family_overlaps": overlaps,
        "split_counts_ok": split_counts_ok,
        "pairwise_family_disjoint": pairwise_disjoint,
        "reserved_development_families_absent": reserved_absent,
        "all_ok": split_counts_ok and pairwise_disjoint and reserved_absent,
    }


def structural_signature(record: IndependentRoleRecord) -> str:
    """Return a normalized template signature with entities and values masked."""
    text = normalize_space(record.source_text)
    replacements: List[Tuple[str, str]] = []
    replacements.extend((item.lower(), "<ENTITY>") for item in record.entities)
    replacements.extend((item.lower(), "<VALUE>") for item in record.values)
    replacements.append((record.attribute.lower(), "<ATTRIBUTE>"))
    for phrase, token in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        if phrase:
            text = re.sub(re.escape(phrase), token, text)
    text = re.sub(r"\b\d+\b", "<NUMBER>", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def duplicate_report(records: Sequence[IndependentRoleRecord]) -> Dict[str, Any]:
    """Audit exact and structural duplicate leakage across splits."""
    exact_map: Dict[str, List[IndependentRoleRecord]] = {}
    structural_map: Dict[str, List[IndependentRoleRecord]] = {}
    for record in records:
        exact_map.setdefault(normalize_space(record.source_text), []).append(record)
        structural_map.setdefault(structural_signature(record), []).append(record)

    def cross_split_duplicates(
        mapping: Mapping[str, Sequence[IndependentRoleRecord]]
    ) -> List[Dict[str, Any]]:
        duplicates = []
        for signature, grouped in mapping.items():
            splits = sorted({record.split for record in grouped})
            if len(splits) > 1:
                duplicates.append(
                    {
                        "signature": signature,
                        "splits": splits,
                        "record_ids": [record.record_id for record in grouped],
                    }
                )
        return duplicates

    exact = cross_split_duplicates(exact_map)
    structural = cross_split_duplicates(structural_map)
    return {
        "exact_cross_split_count": len(exact),
        "structural_cross_split_count": len(structural),
        "exact_cross_split_examples": exact[:MAX_DUPLICATE_EXAMPLES],
        "structural_cross_split_examples": structural[:MAX_DUPLICATE_EXAMPLES],
        "all_ok": not exact and not structural,
    }


def ordered_first_occurrence(record: IndependentRoleRecord) -> Tuple[TupleFact, ...]:
    """Pair entities and values by first occurrence in source text."""
    entities = sorted(
        record.entities,
        key=lambda item: (
            phrase_position(record.source_text, item) is None,
            phrase_position(record.source_text, item) or 10**9,
            item.lower(),
        ),
    )
    values = sorted(
        record.values,
        key=lambda item: (
            phrase_position(record.source_text, item) is None,
            phrase_position(record.source_text, item) or 10**9,
            item.lower(),
        ),
    )
    return facts(record.attribute, tuple(zip(entities, values)))


def minimum_distance(record: IndependentRoleRecord) -> Tuple[TupleFact, ...]:
    """Choose the one-to-one assignment with minimum phrase distance."""
    entities = tuple(sorted(record.entities))
    entity_positions = {
        entity: phrase_position(record.source_text, entity) or 10**9 for entity in entities
    }
    value_positions = {
        value: phrase_position(record.source_text, value) or 10**9 for value in record.values
    }
    ranked = []
    for values in itertools.permutations(sorted(record.values)):
        cost = sum(
            abs(entity_positions[entity] - value_positions[value])
            for entity, value in zip(entities, values)
        )
        ranked.append((cost, values))
    _, selected = min(ranked)
    return facts(record.attribute, tuple(zip(entities, selected)))


def lexical_cartesian(record: IndependentRoleRecord) -> Tuple[TupleFact, ...]:
    """Emit every entity-value combination without role binding."""
    return facts(
        record.attribute,
        tuple((entity, value) for entity in record.entities for value in record.values),
    )


def safe_abstain(record: IndependentRoleRecord) -> Tuple[TupleFact, ...]:
    """Emit no facts."""
    del record
    return ()


BASELINES: Dict[str, Baseline] = {
    "ordered_first_occurrence": ordered_first_occurrence,
    "minimum_distance": minimum_distance,
    "lexical_cartesian": lexical_cartesian,
    "safe_abstain": safe_abstain,
}


def score_baseline(
    records: Sequence[IndependentRoleRecord],
    baseline: Baseline,
) -> Dict[str, Any]:
    """Score one baseline on known and ambiguous records."""
    known = [record for record in records if not record.ambiguous]
    ambiguous = [record for record in records if record.ambiguous]
    known_exact = 0
    known_wrong = 0
    known_emitted = 0
    for record in known:
        prediction = baseline(record)
        emitted = bool(prediction)
        exact = prediction == record.expected
        known_exact += int(exact)
        known_wrong += int(emitted and not exact)
        known_emitted += int(emitted)
    ambiguous_overcommit = sum(bool(baseline(record)) for record in ambiguous)
    return {
        "known_n": len(known),
        "known_exact": known_exact,
        "known_exact_rate": known_exact / len(known) if known else 0.0,
        "known_emitted": known_emitted,
        "known_emitted_rate": known_emitted / len(known) if known else 0.0,
        "known_wrong": known_wrong,
        "known_wrong_rate": known_wrong / len(known) if known else 0.0,
        "ambiguous_n": len(ambiguous),
        "ambiguous_overcommit": ambiguous_overcommit,
        "ambiguous_overcommit_rate": (
            ambiguous_overcommit / len(ambiguous) if ambiguous else 1.0
        ),
    }


def score_baselines(
    records: Sequence[IndependentRoleRecord],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Score all trivial baselines on all records and the evaluation split."""
    evaluation = [record for record in records if record.split == "evaluation"]
    return {
        "all_records": {
            name: score_baseline(records, baseline) for name, baseline in BASELINES.items()
        },
        "evaluation": {
            name: score_baseline(evaluation, baseline)
            for name, baseline in BASELINES.items()
        },
    }


def baseline_nontrivial_report(
    baseline_scores: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> Dict[str, Any]:
    """Return whether trivial baselines are below the frozen exact threshold."""
    position_names = ("ordered_first_occurrence", "minimum_distance", "lexical_cartesian")
    evaluation_scores = baseline_scores["evaluation"]
    all_scores = baseline_scores["all_records"]
    best_evaluation_exact = max(
        evaluation_scores[name]["known_exact_rate"] for name in position_names
    )
    best_all_exact = max(all_scores[name]["known_exact_rate"] for name in position_names)
    safe_candidates = [
        score
        for score in evaluation_scores.values()
        if score["ambiguous_overcommit_rate"] <= MAX_SAFE_AMBIGUITY_OVERCOMMIT_RATE
    ]
    best_safe_exact = (
        max(score["known_exact_rate"] for score in safe_candidates)
        if safe_candidates
        else 0.0
    )
    all_ok = (
        best_evaluation_exact < MAX_POSITION_BASELINE_EXACT_RATE
        and best_all_exact < MAX_POSITION_BASELINE_EXACT_RATE
        and best_safe_exact < MAX_POSITION_BASELINE_EXACT_RATE
    )
    return {
        "best_evaluation_position_exact": best_evaluation_exact,
        "best_all_position_exact": best_all_exact,
        "best_safe_ambiguity_exact": best_safe_exact,
        "max_position_baseline_exact_rate": MAX_POSITION_BASELINE_EXACT_RATE,
        "max_safe_ambiguity_overcommit_rate": MAX_SAFE_AMBIGUITY_OVERCOMMIT_RATE,
        "all_ok": all_ok,
    }


def provenance_report(records: Sequence[IndependentRoleRecord]) -> Dict[str, Any]:
    """Audit provenance and local independence evidence."""
    bad = [
        {
            "record_id": record.record_id,
            "construction_family": record.construction_family,
            "provenance": record.provenance,
        }
        for record in records
        if not provenance_is_auditable(record.provenance)
    ]
    return {
        "records": len(records),
        "bad_provenance_count": len(bad),
        "bad_provenance_examples": bad[:MAX_DUPLICATE_EXAMPLES],
        "all_ok": len(records) > 0 and not bad,
    }


def ambiguity_report(records: Sequence[IndependentRoleRecord]) -> Dict[str, Any]:
    """Audit ambiguous-record consistency and distribution."""
    per_split: Dict[str, Dict[str, int]] = {}
    bad = []
    for split in ALLOWED_SPLITS:
        subset = [record for record in records if record.split == split]
        per_split[split] = {
            "known": sum(not record.ambiguous for record in subset),
            "ambiguous": sum(record.ambiguous for record in subset),
        }
    for record in records:
        if record.ambiguous and record.expected:
            bad.append({"record_id": record.record_id, "issue": "ambiguous_has_expected"})
        if record.ambiguous and (len(record.entities) != 2 or len(record.values) != 2):
            bad.append({"record_id": record.record_id, "issue": "ambiguous_inventory_invalid"})
    distribution_ok = (
        all(per_split[split]["ambiguous"] > 0 for split in ALLOWED_SPLITS)
        and per_split["evaluation"]["ambiguous"] >= MIN_EVALUATION_AMBIGUOUS_RECORDS
    )
    return {
        "per_split": per_split,
        "bad_ambiguous_count": len(bad),
        "bad_ambiguous_examples": bad[:MAX_DUPLICATE_EXAMPLES],
        "distribution_ok": distribution_ok,
        "all_ok": distribution_ok and not bad,
    }


def artifact_hash_report(artifacts: Mapping[str, str]) -> Dict[str, Any]:
    """Hash a set of frozen artifacts and compare against expectations."""
    results = {}
    for relative_path, expected in artifacts.items():
        path = REPO_ROOT / relative_path
        actual = sha256_file(path) if path.exists() else None
        results[relative_path] = {
            "exists": path.exists(),
            "expected": expected,
            "actual": actual,
            "matches": actual == expected,
        }
    return {
        "artifacts": results,
        "all_ok": all(item["matches"] for item in results.values()),
    }


def data_only_report() -> Dict[str, Any]:
    """Audit that this RB4 gate script is data-only."""
    source = Path(__file__).read_text(encoding="utf-8")
    forbidden_snippets = (
        "import " + "torch",
        "from " + "torch",
        ".back" + "ward(",
        "optimizer." + "step(",
        "torch." + "optim",
        "train_" + "head(",
        "RoleConditionedSequenceScoring" + "Head",
    )
    hits = [snippet for snippet in forbidden_snippets if snippet in source]
    return {
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "forbidden_snippets": list(forbidden_snippets),
        "hits": hits,
        "all_ok": not hits,
    }


def build_verdict(
    corpus_path: Path,
    records: Sequence[IndependentRoleRecord],
    parse_errors: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build the complete frozen RB4 corpus-audit verdict."""
    predecessors = artifact_hash_report(PREDECESSOR_ARTIFACTS)
    seals = artifact_hash_report(SEALED_ARTIFACTS)
    provenance = provenance_report(records)
    splits = split_report(records)
    labels = label_structure_report(records, parse_errors)
    duplicates = duplicate_report(records)
    baselines = score_baselines(records)
    nontrivial = baseline_nontrivial_report(baselines)
    ambiguity = ambiguity_report(records)
    data_only = data_only_report()
    corpus_hash = canonical_records_hash(records)

    gates = [
        gate(
            "N0_PREDECESSORS_PRESERVED",
            predecessors["all_ok"],
            f"RB0-RB3 predecessor hashes preserved={predecessors['all_ok']}.",
            predecessors,
        ),
        gate(
            "N1_PROVENANCE",
            provenance["all_ok"],
            "Every parsed record has non-trivial auditable provenance="
            f"{provenance['all_ok']} ({provenance['bad_provenance_count']} bad).",
            provenance,
        ),
        gate(
            "N2_CONSTRUCTION_SEPARATION",
            splits["all_ok"],
            "Split sizes, reserved-family exclusion, and pairwise construction-family "
            f"separation pass={splits['all_ok']}.",
            splits,
        ),
        gate(
            "N3_LABEL_STRUCTURE",
            labels["all_ok"],
            "Known records have one exact one-to-one mapping and ambiguous records have "
            f"none={labels['all_ok']} ({labels['bad_record_count']} bad, "
            f"{labels['parse_error_count']} parse errors).",
            labels,
        ),
        gate(
            "N4_NO_DUPLICATE_LEAKAGE",
            duplicates["all_ok"],
            "No exact or normalized structural duplicate crosses splits="
            f"{duplicates['all_ok']}.",
            duplicates,
        ),
        gate(
            "N5_NON_TRIVIAL_BASELINES",
            nontrivial["all_ok"],
            "Best evaluation position/lexical baseline exact="
            f"{nontrivial['best_evaluation_position_exact']:.1%}; "
            f"threshold <{MAX_POSITION_BASELINE_EXACT_RATE:.0%}; "
            f"pass={nontrivial['all_ok']}.",
            {"summary": nontrivial, "scores": baselines},
        ),
        gate(
            "N6_AMBIGUITY_AUDIT",
            ambiguity["all_ok"],
            "Ambiguous records are internally consistent and present in every split="
            f"{ambiguity['all_ok']}.",
            ambiguity,
        ),
        gate(
            "N7_DATA_ONLY",
            data_only["all_ok"],
            f"RB4 audit script has no model-training snippets={data_only['all_ok']}.",
            data_only,
        ),
        gate(
            "N8_SEALS_UNTOUCHED",
            seals["all_ok"],
            f"Pas 7a and sealed semantic artifacts untouched={seals['all_ok']}.",
            seals,
        ),
    ]
    all_pass = all(item["passed"] for item in gates)
    return {
        "verdict": gates,
        "reference": {
            "run_timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "corpus_path": str(corpus_path),
            "corpus_mtime": (
                datetime.fromtimestamp(corpus_path.stat().st_mtime)
                .astimezone()
                .isoformat(timespec="seconds")
                if corpus_path.exists()
                else None
            ),
            "corpus_sha256": sha256_file(corpus_path) if corpus_path.exists() else None,
            "parsed_corpus_hash": corpus_hash,
            "records": len(records),
            "thresholds": {
                "min_total_records": MIN_TOTAL_RECORDS,
                "min_records_per_split": MIN_RECORDS_PER_SPLIT,
                "min_evaluation_known_records": MIN_EVALUATION_KNOWN_RECORDS,
                "min_evaluation_ambiguous_records": MIN_EVALUATION_AMBIGUOUS_RECORDS,
                "max_position_baseline_exact_rate": MAX_POSITION_BASELINE_EXACT_RATE,
                "max_safe_ambiguity_overcommit_rate": (
                    MAX_SAFE_AMBIGUITY_OVERCOMMIT_RATE
                ),
            },
            "scope": (
                "RB4 data-only independent role-binding corpus audit. This verdict "
                "does not train a model, does not integrate RB2/RB3 into memory, and "
                "does not prove role-binding generalization."
            ),
            "claim_status": "DATA AUDIT ONLY; not a model-performance claim.",
            "all_pass": all_pass,
        },
    }


def write_verdict(verdict: Mapping[str, Any], run_dir: Path) -> Path:
    """Write the RB4 audit verdict JSON."""
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    verdict_path = results_dir / "verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return verdict_path


def run(args: argparse.Namespace) -> int:
    """Run the independent role-binding corpus audit."""
    corpus_path = Path(args.corpus)
    records, parse_errors = load_corpus(corpus_path)
    verdict = build_verdict(corpus_path, records, parse_errors)
    verdict_path = write_verdict(verdict, Path(args.run_dir))
    print(SEP, flush=True)
    print("[INFO] D_Cortex v15.7b-RB4 independent corpus audit", flush=True)
    print(f"[INFO] Corpus: {corpus_path}", flush=True)
    print(f"[INFO] Parsed records: {len(records)}", flush=True)
    for item in verdict["verdict"]:
        label = "✓ PASS" if item["passed"] else "✗ FAIL"
        print(f"{label} [{item['criterion_id']}] {item['evidence']}", flush=True)
    print(SEP, flush=True)
    print(
        f"[INFO] Overall: {'ALL GATES PASS' if verdict['reference']['all_pass'] else 'GATE FAILURE'}",
        flush=True,
    )
    print(f"[INFO] Verdict: {verdict_path}", flush=True)
    return 0 if verdict["reference"]["all_pass"] else 1


def build_argparser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="D_Cortex RB4 independent role-binding corpus audit"
    )
    parser.add_argument(
        "--corpus",
        default=str(REPO_ROOT / "data" / "rb4" / "independent_role_corpus.jsonl"),
    )
    parser.add_argument(
        "--run-dir",
        default=str(REPO_ROOT / "runs" / "independent_role_corpus_audit"),
    )
    return parser


def main() -> int:
    """CLI entry point."""
    try:
        return run(build_argparser().parse_args())
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
