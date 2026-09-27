import json

from experiments.path_b_benchmark_v1 import run_benchmark


def test_path_b_benchmark_smoke():
    """Exercise the reusable benchmark end-to-end on a deliberately tiny case."""
    result = run_benchmark(
        n_nodes=4,
        T=100,
        edge_probability=0.3,
        rho=0.7,
        max_lag=1,
        alpha=0.1,
        n_shuffles=20,
        seed=7,
    )

    print("PATH_B_BENCHMARK_SMOKE=" + json.dumps(result, sort_keys=True))

    assert set(result) == {
        "config",
        "standard",
        "lasso",
        "information_lasso",
        "path_b_v1",
    }
    assert result["config"]["n_nodes"] == 4
    assert 0.0 <= result["path_b_v1"]["screen_parent_recall"] <= 1.0
    assert 0.0 <= result["path_b_v1"]["candidate_retention"] <= 1.0
    assert 0.0 <= result["path_b_v1"]["candidate_reduction"] <= 1.0

    for method in ("standard", "lasso", "information_lasso", "path_b_v1"):
        assert 0.0 <= result[method]["precision"] <= 1.0
        assert 0.0 <= result[method]["recall"] <= 1.0
        assert 0.0 <= result[method]["f1"] <= 1.0
        assert result[method]["runtime_seconds"] >= 0.0
