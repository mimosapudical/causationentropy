"""Path-B v2 screening diagnostics.

This experiment answers one narrow question: how much of the exact oCSE support
can a cheap Information-LASSO screen retain at a given candidate budget?

It compares:
- the standalone Path-A endpoint support,
- a relaxed Information-LASSO/LARS entry path,
- Path-A endpoint plus one conditional-CMI rescue pass.

The rescue pass performs no shuffle tests. It only ranks candidates excluded by
Path A using I(X_j; Y | X_PathA), then admits the strongest candidates until the
requested retention budget is reached.
"""

import argparse
import json
import math
import warnings

import networkx as nx
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import lars_path

from causationentropy.core.discovery import (
    information_lasso_optimal_causation_entropy,
    standard_optimal_causation_entropy,
)
from causationentropy.core.information.conditional_mutual_information import (
    conditional_mutual_information,
)
from causationentropy.datasets.synthetic import linear_stochastic_gaussian_process

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def lagged_design(data, max_lag=1):
    series = np.asarray(data)
    T, n = series.shape
    X = []
    feature_names = []
    for source in range(n):
        for lag in range(1, max_lag + 1):
            X.append(series[max_lag - lag : T - lag, source])
            feature_names.append((source, lag))
    return np.column_stack(X), series[max_lag:, :], feature_names, series


def information_values(
    X,
    Y,
    information="gaussian",
    metric="euclidean",
    k_means=5,
    bandwidth="silverman",
):
    values = np.empty(X.shape[1], dtype=float)
    for j in range(X.shape[1]):
        value = conditional_mutual_information(
            X[:, [j]],
            Y,
            None,
            method=information,
            metric=metric,
            k=k_means,
            bandwidth=bandwidth,
        )
        if not np.isfinite(value):
            raise ValueError(f"non-finite marginal information for candidate {j}")
        values[j] = max(0.0, value)
    return values


def information_lars_order(X, Y, values):
    """Return LARS entry order using numerically stable relative information weights."""
    peak = float(np.max(values)) if values.size else 0.0
    if peak <= 0:
        return list(range(X.shape[1]))

    # Dividing by max rather than sum multiplies all Path-A feature weights by
    # one common constant. That only rescales lambda in the weighted-L1 path,
    # while avoiding unnecessarily tiny design-matrix columns.
    weights = values / peak
    X_weighted = X * weights.reshape(1, -1)
    X_centered = X_weighted - X_weighted.mean(axis=0, keepdims=True)
    y = Y.reshape(-1)
    y_centered = y - y.mean()

    try:
        _, active, _ = lars_path(X_centered, y_centered, method="lasso")
        order = [int(j) for j in active]
    except (ValueError, np.linalg.LinAlgError):
        # This path is diagnostic only. If LARS becomes numerically singular,
        # keep the experiment running with the information ranking rather than
        # changing the production Path-A implementation.
        order = []

    order.extend(int(j) for j in np.argsort(-values) if int(j) not in order)
    return order


def endpoint_plus_conditional_rescue(
    X,
    Y,
    rng,
    retention=0.30,
    min_rescue=1,
    information="gaussian",
    metric="euclidean",
    k_means=5,
    bandwidth="silverman",
):
    """Path-B v2 screen: Path-A endpoint plus one cheap conditional rescue pass."""
    endpoint = information_lasso_optimal_causation_entropy(
        X,
        Y,
        rng,
        information=information,
        metric=metric,
        k_means=k_means,
        bandwidth=bandwidth,
    )
    endpoint = [int(j) for j in endpoint]
    selected = set(endpoint)

    target_size = max(
        int(math.ceil(retention * X.shape[1])),
        len(selected) + (min_rescue if len(selected) < X.shape[1] else 0),
    )
    target_size = min(X.shape[1], target_size)
    if len(selected) >= target_size:
        return sorted(selected), {"endpoint_size": len(endpoint), "rescued": 0}

    Z = X[:, endpoint] if endpoint else None
    scored = []
    for j in range(X.shape[1]):
        if j in selected:
            continue
        score = conditional_mutual_information(
            X[:, [j]],
            Y,
            Z,
            method=information,
            metric=metric,
            k=k_means,
            bandwidth=bandwidth,
        )
        if not np.isfinite(score):
            score = -np.inf
        scored.append((float(score), int(j)))

    scored.sort(key=lambda item: (-item[0], item[1]))
    rescue_count = target_size - len(selected)
    selected.update(j for _, j in scored[:rescue_count])
    return sorted(selected), {
        "endpoint_size": len(endpoint),
        "rescued": rescue_count,
    }


def exact_standard_support(
    X,
    Y,
    Z_init,
    rng,
    alpha=0.05,
    n_shuffles=100,
    information="gaussian",
):
    return set(
        int(j)
        for j in standard_optimal_causation_entropy(
            X,
            Y,
            Z_init,
            rng,
            alpha1=alpha,
            alpha2=alpha,
            n_shuffles=n_shuffles,
            information=information,
        )
    )


def _coverage(reference, selected):
    return len(reference.intersection(selected)) / len(reference) if reference else 1.0


def run_gaussian_frontier(
    n_nodes=12,
    T=300,
    edge_probability=0.12,
    rho=0.7,
    seeds=5,
    n_shuffles=100,
    retentions=(0.10, 0.20, 0.30, 0.40, 0.50),
):
    rows = []
    for seed in range(seeds):
        graph_true = nx.erdos_renyi_graph(
            n_nodes, edge_probability, seed=seed, directed=True
        )
        data, _ = linear_stochastic_gaussian_process(
            rho=rho,
            n=n_nodes,
            T=T,
            p=edge_probability,
            seed=seed,
            G=graph_true,
        )
        X, Y_all, feature_names, series = lagged_design(data, max_lag=1)

        for target in range(n_nodes):
            Y = Y_all[:, [target]]
            Z_init = series[:-1, [target]]
            rng_full = np.random.default_rng(seed * 10000 + target)
            full = exact_standard_support(
                X, Y, Z_init, rng_full, n_shuffles=n_shuffles
            )

            rng_a = np.random.default_rng(seed * 10000 + target)
            endpoint = set(
                int(j)
                for j in information_lasso_optimal_causation_entropy(X, Y, rng_a)
            )
            values = information_values(X, Y)
            order = information_lars_order(X, Y, values)

            rows.append(
                {
                    "seed": seed,
                    "target": target,
                    "screen": "endpoint",
                    "retention": len(endpoint) / X.shape[1],
                    "full_support_recall": _coverage(full, endpoint),
                }
            )

            for retention in retentions:
                budget = max(1, int(math.ceil(retention * X.shape[1])))
                path_selected = set(order[:budget])
                rows.append(
                    {
                        "seed": seed,
                        "target": target,
                        "screen": f"path_{retention:.2f}",
                        "retention": len(path_selected) / X.shape[1],
                        "full_support_recall": _coverage(full, path_selected),
                    }
                )

                rng_rescue = np.random.default_rng(seed * 10000 + target)
                rescued, _ = endpoint_plus_conditional_rescue(
                    X, Y, rng_rescue, retention=retention
                )
                rescued = set(rescued)
                rows.append(
                    {
                        "seed": seed,
                        "target": target,
                        "screen": f"rescue_{retention:.2f}",
                        "retention": len(rescued) / X.shape[1],
                        "full_support_recall": _coverage(full, rescued),
                    }
                )

    summary = {}
    for name in sorted({row["screen"] for row in rows}):
        subset = [row for row in rows if row["screen"] == name]
        summary[name] = {
            "mean_retention": float(np.mean([row["retention"] for row in subset])),
            "mean_full_support_recall": float(
                np.mean([row["full_support_recall"] for row in subset])
            ),
            "targets": len(subset),
        }
    return {"rows": rows, "summary": summary}


def hidden_parent_stress(seed=0, T=1000):
    """Two true parents where one has near-zero marginal information in population."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=T)
    innovation = rng.normal(size=T)
    innovation -= x1 * (np.dot(x1, innovation) / np.dot(x1, x1))
    x1 -= x1.mean()
    innovation -= innovation.mean()
    x2 = x1 + innovation

    y = np.zeros(T)
    y[1:] = x2[:-1] - x1[:-1] + 0.05 * rng.normal(size=T - 1)
    d1 = rng.normal(size=T)
    d2 = rng.normal(size=T)
    X = np.column_stack([x1[:-1], x2[:-1], d1[:-1], d2[:-1]])
    Y = y[1:, None]
    return X, Y, {0, 1}


def correlated_lag_stress(seed=0, T=1200, phi=0.95):
    """Two true lags from a highly autocorrelated source plus distractors."""
    rng = np.random.default_rng(seed)
    source = np.zeros(T)
    for t in range(1, T):
        source[t] = phi * source[t - 1] + 0.3 * rng.normal()

    y = np.zeros(T)
    for t in range(3, T):
        y[t] = (
            0.8 * source[t - 1]
            + 0.8 * source[t - 3]
            + 0.2 * rng.normal()
        )
    d1 = rng.normal(size=T)
    d2 = rng.normal(size=T)
    X = np.column_stack(
        [source[2:-1], source[1:-2], source[:-3], d1[2:-1], d2[2:-1]]
    )
    Y = y[3:, None]
    return X, Y, {0, 2}


def run_stress_suite(seeds=20, retention=0.30):
    output = {}
    for name, generator in (
        ("hidden_parent", hidden_parent_stress),
        ("correlated_lag", correlated_lag_stress),
    ):
        endpoint_recalls = []
        rescue_recalls = []
        for seed in range(seeds):
            X, Y, truth = generator(seed)
            rng = np.random.default_rng(seed)
            endpoint = set(
                information_lasso_optimal_causation_entropy(X, Y, rng)
            )
            rescued, _ = endpoint_plus_conditional_rescue(
                X,
                Y,
                np.random.default_rng(seed),
                retention=retention,
            )
            endpoint_recalls.append(_coverage(truth, endpoint))
            rescue_recalls.append(_coverage(truth, set(rescued)))
        output[name] = {
            "endpoint_parent_recall": float(np.mean(endpoint_recalls)),
            "rescue_parent_recall": float(np.mean(rescue_recalls)),
            "retention_target": retention,
            "seeds": seeds,
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-shuffles", type=int, default=100)
    args = parser.parse_args()
    result = {
        "gaussian_frontier": run_gaussian_frontier(
            seeds=args.seeds, n_shuffles=args.n_shuffles
        ),
        "stress": run_stress_suite(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
