"""Budget-matched screening baselines for Path-B.

This experiment asks a stricter question than the main benchmark:
is Path-B's screen better than cheaper screens when every method receives
exactly the same per-target candidate budget?

For each target, Path-B is run first and its realized screen size d_t is used
as the budget for:
  1. marginal Gaussian-CMI top-k screening;
  2. greedy forward-regression / OMP-style screening;
  3. deterministic random screening.

Each screen is then followed by the same restricted standard oCSE refinement.
The experiment reports parent containment, final graph accuracy, and runtime.
"""

import argparse
import json
import math
import time

import numpy as np

from causationentropy.core.discovery import standard_optimal_causation_entropy
from experiments.path_b_benchmark_v2 import (
    candidate_ids_for_target,
    gaussian_case,
    metrics,
)
from experiments.path_b_screen_frontier_v2 import (
    endpoint_plus_conditional_rescue,
    information_values,
    lagged_design,
)


def _top_marginal_screen(X, Y, budget):
    vals = np.asarray(
        information_values(X, Y, information="gaussian"),
        dtype=float,
    )
    vals = np.nan_to_num(vals, nan=-np.inf, neginf=-np.inf, posinf=np.inf)
    order = sorted(range(X.shape[1]), key=lambda j: (-vals[j], j))
    return order[:budget]


def _forward_regression_screen(X, Y, budget):
    """Greedy forward regression using residual correlation."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(Y, dtype=float).reshape(-1)
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    norms = np.linalg.norm(Xc, axis=0)
    norms[norms == 0] = np.inf

    selected = []
    available = set(range(X.shape[1]))
    residual = yc.copy()

    for _ in range(min(budget, X.shape[1])):
        scores = np.full(X.shape[1], -np.inf, dtype=float)
        if available:
            idx = np.fromiter(sorted(available), dtype=int)
            scores[idx] = np.abs(Xc[:, idx].T @ residual) / norms[idx]
        j = int(np.argmax(scores))
        if not np.isfinite(scores[j]):
            # Fill deterministically if all remaining columns are degenerate.
            selected.extend(sorted(available)[: budget - len(selected)])
            break
        selected.append(j)
        available.remove(j)

        Xs = Xc[:, selected]
        coef, *_ = np.linalg.lstsq(Xs, yc, rcond=None)
        residual = yc - Xs @ coef

    return selected[:budget]


def _random_screen(n_features, budget, seed):
    rng = np.random.default_rng(seed)
    return sorted(
        int(j) for j in rng.choice(n_features, size=budget, replace=False)
    )


def _truth_local(candidate_ids, feature_names, truth_edges, target):
    result = set()
    for local_idx, global_idx in enumerate(candidate_ids):
        source, lag = feature_names[global_idx]
        if (int(source), int(target), int(lag)) in truth_edges:
            result.add(int(local_idx))
    return result


def _refine(
    X,
    Y,
    Z_init,
    candidate_ids,
    screen_local,
    feature_names,
    target,
    seed,
    alpha,
    n_shuffles,
):
    start = time.perf_counter()
    if not screen_local:
        return set(), time.perf_counter() - start

    screened_global = [candidate_ids[int(j)] for j in screen_local]
    rng = np.random.default_rng(seed * 10000 + target)
    refined_local = standard_optimal_causation_entropy(
        X[:, screened_global],
        Y,
        Z_init,
        rng,
        alpha1=alpha,
        alpha2=alpha,
        n_shuffles=n_shuffles,
        information="gaussian",
    )
    edges = set()
    for j in refined_local:
        global_idx = screened_global[int(j)]
        source, lag = feature_names[global_idx]
        edges.add((int(source), int(target), int(lag)))
    return edges, time.perf_counter() - start


def run_seed(
    seed,
    n_nodes=20,
    T=300,
    retention=0.40,
    alpha=0.05,
    n_shuffles=20,
):
    data, truth, _information, _metadata = gaussian_case(
        seed, n_nodes=n_nodes, T=T
    )
    X, Y_all, feature_names, series = lagged_design(data, max_lag=1)

    methods = ("path_b", "marginal_topk", "forward_regression", "random")
    predicted = {name: set() for name in methods}
    screen_seconds = {name: 0.0 for name in methods}
    refine_seconds = {name: 0.0 for name in methods}
    parent_complete = {name: [] for name in methods}
    parent_recall = {name: [] for name in methods}
    realized_retention = {name: [] for name in methods}

    for target in range(n_nodes):
        Y = Y_all[:, [target]]
        candidate_ids = candidate_ids_for_target(feature_names, target)
        Xc = X[:, candidate_ids]
        truth_local = _truth_local(
            candidate_ids, feature_names, truth, target
        )
        Z_init = series[:-1, [target]]

        start = time.perf_counter()
        path_local, _diag = endpoint_plus_conditional_rescue(
            Xc,
            Y,
            np.random.default_rng(seed * 10000 + target),
            retention=retention,
            information="gaussian",
        )
        screen_seconds["path_b"] += time.perf_counter() - start
        path_local = [int(j) for j in path_local]
        budget = len(path_local)

        start = time.perf_counter()
        marginal_local = _top_marginal_screen(Xc, Y, budget)
        screen_seconds["marginal_topk"] += time.perf_counter() - start

        start = time.perf_counter()
        forward_local = _forward_regression_screen(Xc, Y, budget)
        screen_seconds["forward_regression"] += time.perf_counter() - start

        start = time.perf_counter()
        random_local = _random_screen(
            Xc.shape[1], budget, seed * 10000 + target + 17
        )
        screen_seconds["random"] += time.perf_counter() - start

        screens = {
            "path_b": path_local,
            "marginal_topk": marginal_local,
            "forward_regression": forward_local,
            "random": random_local,
        }

        for name, screen_local in screens.items():
            screen_set = set(int(j) for j in screen_local)
            tp = len(truth_local & screen_set)
            parent_complete[name].append(truth_local.issubset(screen_set))
            parent_recall[name].append(
                tp / len(truth_local) if truth_local else 1.0
            )
            realized_retention[name].append(
                len(screen_set) / Xc.shape[1]
            )
            edges, elapsed = _refine(
                X,
                Y,
                Z_init,
                candidate_ids,
                screen_local,
                feature_names,
                target,
                seed,
                alpha,
                n_shuffles,
            )
            predicted[name].update(edges)
            refine_seconds[name] += elapsed

    result = {
        "config": {
            "seed": seed,
            "n_nodes": n_nodes,
            "T": T,
            "retention_target": retention,
            "alpha": alpha,
            "n_shuffles": n_shuffles,
            "truth_edges": len(truth),
        },
        "methods": {},
    }
    for name in methods:
        result["methods"][name] = {
            **metrics(truth, predicted[name], n_nodes, 1),
            "parent_complete_target_rate": float(
                np.mean(parent_complete[name])
            ),
            "mean_parent_recall": float(np.mean(parent_recall[name])),
            "mean_candidate_retention": float(
                np.mean(realized_retention[name])
            ),
            "screen_runtime_seconds": screen_seconds[name],
            "refine_runtime_seconds": refine_seconds[name],
            "total_runtime_seconds": (
                screen_seconds[name] + refine_seconds[name]
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", nargs="+", type=int, default=[20, 50])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--retention", type=float, default=0.40)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-shuffles", type=int, default=20)
    args = parser.parse_args()

    output = {"runs": []}
    for n_nodes in args.nodes:
        T = max(args.T, 5 * n_nodes)
        for seed in range(args.seeds):
            output["runs"].append(
                run_seed(
                    seed=seed,
                    n_nodes=n_nodes,
                    T=T,
                    retention=args.retention,
                    alpha=args.alpha,
                    n_shuffles=args.n_shuffles,
                )
            )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
