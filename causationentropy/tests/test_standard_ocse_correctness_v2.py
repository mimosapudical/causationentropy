"""Regression tests for the corrected standard-oCSE baseline used by Path B."""

from unittest.mock import patch

import numpy as np

from causationentropy.core.discovery import backward, discover_network


@patch(
    "causationentropy.core.discovery.standard_optimal_causation_entropy"
)
def test_standard_excludes_conditioned_target_history(mock_standard):
    """Columns already present in Z_init are not causal candidates."""
    mock_standard.return_value = []
    data = np.arange(120, dtype=float).reshape(40, 3)

    discover_network(
        data,
        method="standard",
        max_lag=2,
        n_shuffles=5,
    )

    assert mock_standard.call_count == 3
    for call in mock_standard.call_args_list:
        X_candidates = call.args[0]
        Z_init = call.args[2]
        assert X_candidates.shape == (38, 4)
        assert Z_init.shape == (38, 2)
        for candidate in X_candidates.T:
            assert not any(
                np.array_equal(candidate, conditioned)
                for conditioned in Z_init.T
            )


def test_backward_retains_initial_conditioning(monkeypatch):
    """Standard backward tests condition on Z_init plus other selected predictors."""
    rng = np.random.default_rng(7)
    X_full = rng.normal(size=(50, 3))
    Y = rng.normal(size=(50, 1))
    Z_init = rng.normal(size=(50, 2))
    seen = []

    def fake_cmi(X, Y, Z, **kwargs):
        seen.append(Z.copy())
        return 1.0

    def fake_shuffle(*args, **kwargs):
        return {
            "Threshold": 0.0,
            "Value": 1.0,
            "Pass": True,
            "P_value": 0.0,
        }

    monkeypatch.setattr(
        "causationentropy.core.discovery.conditional_mutual_information",
        fake_cmi,
    )
    monkeypatch.setattr(
        "causationentropy.core.discovery.shuffle_test",
        fake_shuffle,
    )

    selected = backward(
        X_full,
        Y,
        [0, 1],
        rng=np.random.default_rng(11),
        n_shuffles=5,
        information="kde",
        Z_init=Z_init,
    )

    assert selected == [0, 1]
    assert len(seen) == 2
    for Z in seen:
        assert Z.shape == (50, 3)
        np.testing.assert_array_equal(Z[:, :2], Z_init)


@patch("causationentropy.core.discovery.shuffle_test")
@patch("causationentropy.core.discovery.conditional_mutual_information")
@patch(
    "causationentropy.core.discovery.standard_optimal_causation_entropy"
)
def test_standard_edge_reporting_retains_initial_history(
    mock_standard,
    mock_cmi,
    mock_shuffle,
):
    """Final standard edge CMI/p-values use the same initial conditioning."""
    mock_standard.return_value = [0]
    mock_cmi.side_effect = lambda X, Y, Z, **kwargs: 0.5
    mock_shuffle.return_value = {
        "Threshold": 0.1,
        "Value": 0.5,
        "Pass": True,
        "P_value": 0.01,
    }
    data = np.random.default_rng(12).normal(size=(40, 2))

    graph = discover_network(
        data,
        method="standard",
        max_lag=1,
        n_shuffles=5,
    )

    assert graph.number_of_edges() == 2
    assert mock_cmi.call_count == 2
    np.testing.assert_array_equal(
        mock_cmi.call_args_list[0].args[2],
        data[:-1, [0]],
    )
    np.testing.assert_array_equal(
        mock_cmi.call_args_list[1].args[2],
        data[:-1, [1]],
    )


@patch("causationentropy.core.discovery.shuffle_test")
@patch("causationentropy.core.discovery.conditional_mutual_information")
@patch(
    "causationentropy.core.discovery.standard_optimal_causation_entropy"
)
def test_standard_report_all_uses_actual_candidate_universe(
    mock_standard,
    mock_cmi,
    mock_shuffle,
):
    """Report-all excludes target-history columns that standard never tests."""
    mock_standard.return_value = []
    mock_cmi.return_value = 0.0
    mock_shuffle.return_value = {
        "Threshold": 0.1,
        "Value": 0.0,
        "Pass": False,
        "P_value": 1.0,
    }
    data = np.random.default_rng(13).normal(size=(30, 3))

    graph = discover_network(
        data,
        method="standard",
        max_lag=2,
        n_shuffles=5,
        only_return_significant=False,
    )

    assert graph.number_of_edges() == 3 * 2 * 2
    assert all(source != target for source, target in graph.edges())
