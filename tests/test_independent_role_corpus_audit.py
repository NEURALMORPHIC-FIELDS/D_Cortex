"""Regression tests for the RB4 independent corpus audit harness."""

from scripts.independent_role_corpus_audit import (
    baseline_nontrivial_report,
    data_only_report,
    duplicate_report,
    expected_is_one_to_one,
    inventory_present,
    record_from_dict,
    score_baselines,
    split_report,
)


def _known_record(
    record_id: str,
    split: str,
    family: str,
    entity_a: str,
    entity_b: str,
    value_a: str,
    value_b: str,
) -> dict:
    return {
        "record_id": record_id,
        "split": split,
        "construction_family": family,
        "source_text": (
            f"Document source states {entity_a} received {value_a}, "
            f"whereas {entity_b} received {value_b}."
        ),
        "attribute": "color",
        "entities": [entity_a, entity_b],
        "values": [value_a, value_b],
        "expected": [
            [entity_a, "color", value_a],
            [entity_b, "color", value_b],
        ],
        "ambiguous": False,
        "provenance": {"source_id": f"document-{record_id}", "citation": "unit source"},
    }


def test_record_parser_preserves_one_to_one_known_mapping() -> None:
    record = record_from_dict(
        _known_record("k1", "evaluation", "IndependentSyntaxA", "alpha", "beta", "red", "blue"),
        0,
    )
    assert record.split == "evaluation"
    assert expected_is_one_to_one(record)
    assert inventory_present(record)
    assert record.expected == (
        ("alpha", "color", "red"),
        ("beta", "color", "blue"),
    )


def test_structural_duplicate_leakage_crosses_splits() -> None:
    train = record_from_dict(
        _known_record("d1", "train", "IndependentSyntaxA", "alpha", "beta", "red", "blue"),
        0,
    )
    validation = record_from_dict(
        _known_record(
            "d2",
            "validation",
            "IndependentSyntaxB",
            "gamma",
            "delta",
            "green",
            "yellow",
        ),
        1,
    )
    report = duplicate_report((train, validation))
    assert report["structural_cross_split_count"] == 1
    assert not report["all_ok"]


def test_reserved_development_family_blocks_split_gate() -> None:
    records = (
        record_from_dict(_known_record("s1", "train", "RB1", "alpha", "beta", "red", "blue"), 0),
        record_from_dict(
            _known_record("s2", "validation", "IndependentSyntaxB", "gamma", "delta", "red", "blue"),
            1,
        ),
        record_from_dict(
            _known_record("s3", "evaluation", "IndependentSyntaxC", "theta", "iota", "red", "blue"),
            2,
        ),
    )
    report = split_report(records)
    assert report["per_split"]["train"]["reserved_development_families"] == ["RB1"]
    assert not report["all_ok"]


def test_position_baseline_detects_lexically_solvable_corpus() -> None:
    records = tuple(
        record_from_dict(
            _known_record(
                f"b{index}",
                "evaluation",
                "IndependentSyntaxC",
                f"entity{index}a",
                f"entity{index}b",
                f"value{index}a",
                f"value{index}b",
            ),
            index,
        )
        for index in range(4)
    )
    scores = score_baselines(records)
    report = baseline_nontrivial_report(scores)
    assert scores["evaluation"]["ordered_first_occurrence"]["known_exact_rate"] == 1.0
    assert not report["all_ok"]


def test_rb4_audit_script_is_data_only() -> None:
    report = data_only_report()
    assert report["all_ok"]
    assert report["hits"] == []
