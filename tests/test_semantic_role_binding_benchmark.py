"""Regression tests for the non-trivial role-binding benchmark."""

from scripts.semantic_role_binding_benchmark import (
    BASELINES,
    build_sample,
    extract_definitions,
    sample_hash,
    score_baseline,
    structure_checks,
)


def test_role_binding_sample_is_deterministic_and_structured() -> None:
    definitions = extract_definitions()
    first = build_sample(definitions, records_per_family=20, seed=91)
    second = build_sample(definitions, records_per_family=20, seed=91)
    assert sample_hash(first) == sample_hash(second)
    checks = structure_checks(first, definitions)
    assert all(checks.values())


def test_lexical_baselines_do_not_solve_role_binding() -> None:
    records = build_sample(extract_definitions(), records_per_family=40, seed=92)
    scores = {
        name: score_baseline(records, baseline)
        for name, baseline in BASELINES.items()
    }
    assert max(
        scores[name]["known_exact_rate"]
        for name in (
            "ordered_first_occurrence",
            "minimum_distance",
            "lexical_cartesian",
        )
    ) < 0.50
    assert scores["safe_abstain"]["known_coverage_rate"] == 0.0
    assert scores["safe_abstain"]["ambiguous_overcommit_rate"] == 0.0
