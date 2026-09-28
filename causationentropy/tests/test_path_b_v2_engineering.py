import json
import subprocess
import sys
from unittest.mock import patch

import numpy as np

from causationentropy.core.discovery import (
    _candidate_cmi_values,
    shuffle_test,
    standard_forward,
)
from causationentropy.core.information.conditional_mutual_information import (
    conditional_mutual_information,
    gaussian_conditional_mutual_information,
    prepare_gaussian_cmi_context,
)
from experiments.path_b_benchmark_v2 import logistic_case
from experiments.path_b_screen_frontier_v2 import (
    endpoint_plus_conditional_rescue,
)


def test_standard_forward_reuses_observed_scores_when_z_is_unchanged():
    X = np.tile(np.array([[0.0, 1.0, 2.0]]), (20, 1))
    Y = np.arange(20, dtype=float).reshape(-1, 1)
    Z = np.ones((20, 1))

    def fake_cmi(Xj, Y_arg, Z_arg, **kwargs):
        return {0: 3.0, 1: 2.0, 2: 1.0}[int(Xj[0, 0])]

    shuffle_results = [
        {"Pass": False},
        {"Pass": True},
        {"Pass": False},
    ]

    with patch(
        "causationentropy.core.discovery.conditional_mutual_information",
        side_effect=fake_cmi,
    ) as cmi, patch(
        "causationentropy.core.discovery.shuffle_test",
        side_effect=shuffle_results,
    ) as shuffle:
        selected = standard_forward(
            X,
            Y,
            Z,
            np.random.default_rng(0),
            n_shuffles=3,
        )

    assert selected == [1]
    assert cmi.call_count == 4
    assert shuffle.call_count == 3


def test_gaussian_context_reuse_is_numerically_equivalent():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 1))
    Y = 0.4 * X + rng.normal(size=(120, 1))
    Z = rng.normal(size=(120, 3))

    expected = conditional_mutual_information(X, Y, Z, method="gaussian")
    context = prepare_gaussian_cmi_context(Y, Z)
    got = gaussian_conditional_mutual_information(X, Y, Z, context=context)

    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_candidate_gaussian_context_reuse_matches_direct_scores():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 5))
    Y = rng.normal(size=(80, 1))
    Z = rng.normal(size=(80, 2))
    candidates = list(range(X.shape[1]))

    expected = np.asarray(
        [
            conditional_mutual_information(
                X[:, [j]], Y, Z, method="gaussian"
            )
            for j in candidates
        ]
    )
    got = _candidate_cmi_values(
        X,
        candidates,
        Y,
        Z,
        "gaussian",
        "euclidean",
        5,
        "silverman",
    )

    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_shuffle_is_reproducible_with_internal_gaussian_reuse():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 1))
    Z = rng.normal(size=(80, 2))
    Y = 0.3 * X + 0.2 * Z[:, [0]] + rng.normal(size=(80, 1))
    observed = conditional_mutual_information(X, Y, Z, method="gaussian")

    first = shuffle_test(
        X,
        Y,
        Z,
        observed,
        rng=123,
        n_shuffles=25,
        information="gaussian",
    )
    second = shuffle_test(
        X,
        Y,
        Z,
        observed,
        rng=123,
        n_shuffles=25,
        information="gaussian",
    )

    assert first == second


def test_conditional_rescue_adds_best_excluded_candidate():
    X = np.column_stack(
        [
            np.arange(20, dtype=float),
            np.arange(20, dtype=float) + 1,
            np.arange(20, dtype=float) + 2,
            np.arange(20, dtype=float) + 3,
        ]
    )
    Y = np.arange(20, dtype=float).reshape(-1, 1)

    def fake_cmi(Xj, Y_arg, Z_arg, **kwargs):
        candidate = int(Xj[0, 0])
        return {0: 9.0, 2: 2.0, 3: 1.0}.get(candidate, 0.0)

    with patch(
        "experiments.path_b_screen_frontier_v2."
        "information_lasso_optimal_causation_entropy",
        return_value=[1],
    ), patch(
        "experiments.path_b_screen_frontier_v2.conditional_mutual_information",
        side_effect=fake_cmi,
    ):
        selected, diagnostics = endpoint_plus_conditional_rescue(
            X,
            Y,
            np.random.default_rng(0),
            retention=0.5,
        )

    assert selected == [0, 1]
    assert diagnostics == {"endpoint_size": 1, "rescued": 1}


def test_logistic_benchmark_case_is_finite_and_bounded():
    data, truth, information, metadata = logistic_case(seed=0)

    assert information == "kde"
    assert metadata["benchmark_role"] == "primary"
    assert metadata["generator"] == "logisic_dynamics"
    assert metadata["r"] == 3.9
    assert metadata["sigma"] == 0.01
    assert metadata["finite"] is True
    assert metadata["within_unit_interval"] is True
    assert np.isfinite(data).all()
    assert np.min(data) >= 0.0
    assert np.max(data) <= 1.0
    assert isinstance(truth, set)


def test_summary_schema_matches_v2_outputs(tmp_path):
    frontier = {
        "gaussian_frontier": {
            "summary": {
                "rescue_0.40": {
                    "mean_retention": 0.4,
                    "mean_full_forward_recall": 0.98,
                    "mean_full_support_recall": 1.0,
                    "targets": 2,
                }
            }
        },
        "stress": {
            "hidden_parent": {
                "endpoint_parent_recall": 0.8,
                "rescue_parent_recall": 1.0,
            }
        },
    }
    (tmp_path / "03_screen_frontier.json").write_text(
        json.dumps(frontier), encoding="utf-8"
    )
    (tmp_path / "04_common_random_numbers.json").write_text(
        json.dumps(
            {
                "summary": {
                    "0.40": {
                        "mean_actual_retention": 0.4,
                        "mean_forward_closure_recall": 0.98,
                        "mean_final_support_screen_recall": 1.0,
                        "mean_restricted_final_recall": 1.0,
                        "seeds": 1,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    gaussian = {
        "gaussian": {
            "config": {"n_nodes": 20, "benchmark_role": "primary"},
            "standard": {
                "runtime_seconds": 2.0,
                "f1": 0.9,
                "shuffle_cmi_evaluations": 1000,
                "total_cmi_evaluations": 1200,
            },
            "path_b_v2": {
                "runtime_seconds": 1.0,
                "candidate_retention": 0.4,
                "screen_full_support_recall": 1.0,
                "refined_full_support_recall": 1.0,
                "f1": 0.9,
                "shuffle_cmi_evaluations": 500,
                "screen_marginal_cmi_scores": 100,
                "screen_rescue_cmi_scores": 50,
                "total_cmi_evaluations": 800,
            },
        }
    }
    (tmp_path / "05_gaussian_seed_0.json").write_text(
        json.dumps(gaussian), encoding="utf-8"
    )
    (tmp_path / "06_scale_n20_seed_0.json").write_text(
        json.dumps(gaussian), encoding="utf-8"
    )

    logistic = {
        "logistic": {
            "config": {"benchmark_role": "primary"},
            "standard": {"f1": 0.8},
            "path_b_v2": {"f1": 0.8, "screen_full_support_recall": 1.0},
        }
    }
    (tmp_path / "07_logistic_seed_0.json").write_text(
        json.dumps(logistic), encoding="utf-8"
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.summarize_path_b_v2",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Screening frontier" in completed.stdout
    assert "Gaussian scaling" in completed.stdout
    assert "20" in completed.stdout
    assert "hidden_parent" in completed.stdout
