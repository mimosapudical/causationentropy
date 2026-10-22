"""Check that the review audit measures truth separately from algorithm agreement."""

import numpy as np
import pytest

pytest.importorskip("benchmarks")

from benchmarks.path_b_screening import audit_target, containment, run_audit, summarize


def test_full_support_agreement_is_not_parent_containment():
    counts = containment({0, 1}, {0}, {0})
    row = {
        "family": "gaussian",
        "T": 80,
        "retention_target": 0.4,
        "method": "conditional_rescue",
        "seed": 0,
        "retained_fraction": 0.4,
        "TP": 1,
        "FP": 0,
        "FN": 1,
        "full_TP": 1,
        "full_FP": 0,
        "full_FN": 1,
        "cmi_calls": 5,
        "full_cmi_calls": 10,
        "measured_seconds": 1.0,
        "full_seconds": 2.0,
        **counts,
    }
    summary = summarize([row])[0]
    assert summary["full_support_coverage"] == 1.0
    assert summary["parent_recall"] == 0.5
    assert summary["parent_complete_target_rate"] == 0.0


def test_production_full_retention_matches_full_ocse():
    rng = np.random.default_rng(81)
    X = rng.normal(size=(80, 4))
    Z = rng.normal(size=(80, 1))
    Y = 2 * X[:, [1]] + 0.1 * rng.normal(size=(80, 1))
    rows = audit_target(X, Y, Z, {1}, 81, [1.0], "gaussian", 9, True)
    by_method = {row["method"]: row for row in rows}
    production = by_method["conditional_rescue"]
    assert production["screen"] == [0, 1, 2, 3]
    assert set(production["final"]) == set(production["full_support"])
    assert production["parent_complete"]
    assert len(by_method["marginal_fill_matched"]["screen"]) == 4


def test_screen_only_audit_does_not_invent_full_support_metrics(tmp_path):
    output = tmp_path / "audit"
    summary = run_audit("cancellation", 5, [40], [0], [0.4], 9, output)
    assert len(summary) == 3
    assert all(row["full_support_coverage"] is None for row in summary)
    assert all("final_F1" not in row for row in summary)
    assert (output / "targets.jsonl").is_file()
    with pytest.raises(FileExistsError):
        run_audit("cancellation", 5, [40], [0], [0.4], 9, output)
