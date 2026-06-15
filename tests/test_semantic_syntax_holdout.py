"""Regression tests for leave-one-syntax-family-out split construction."""

from scripts.semantic_role_binder_verdict import load_rb0_sample
from scripts.semantic_role_binding_benchmark import KNOWN_FAMILIES, REPO_ROOT
from scripts.semantic_syntax_holdout_verdict import (
    build_fold_splits,
    fold_separation_report,
)


def test_each_known_family_is_held_out_exactly_once() -> None:
    records = load_rb0_sample(
        REPO_ROOT / "runs" / "semantic_role_binding_benchmark" / "results" / "sample.json"
    )
    folds = {family: build_fold_splits(records, family) for family in KNOWN_FAMILIES}
    reports = fold_separation_report(folds)
    assert sum(report["heldout_test_n"] for report in reports.values()) == 1600
    for family, report in reports.items():
        assert report["heldout_absent_train"]
        assert report["heldout_absent_validation"]
        assert report["heldout_test_n"] == 400
        assert report["ambiguous_test_n"] > 0
        assert report["train_test_disjoint"]
        assert report["validation_test_disjoint"]
