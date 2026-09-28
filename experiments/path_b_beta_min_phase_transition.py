"""Beta-min controlled Gaussian VAR benchmark for Path-B theory.

This benchmark intentionally removes the random near-zero-edge artifact of the
repository's default Gaussian generator.  It uses a random DAG VAR(1), so the
coefficient matrix is nilpotent (hence stable) and every nonzero coefficient
has magnitude at least beta_min by construction.

The goal is not to replace the repository benchmark.  It is a theorem-facing
diagnostic: does parent containment exhibit the expected transition as
beta_min * sqrt(n_samples / log(p)) increases?

Metrics are separated for:
- marginal Information-LASSO endpoint,
- endpoint + one conditional rescue pass (Path-B screen),
- full keyed oCSE,
- restricted keyed oCSE on the Path-B screen.

The benchmark also records the minimum marginal Gaussian information among true
parents.  This exposes cases where the weighted screen's marginal-faithfulness
condition is violated even though beta_min is nonzero.
"""

import argparse
import json
import math
import time

import networkx as nx
import numpy as np

from causationentropy.core.discovery import (
    information_lasso_optimal_causation_entropy,
)
from causationentropy.core.information.conditional_mutual_information import (
    conditional_mutual_information,
)
from experiments.path_b_benchmark_v2 import candidate_ids_for_target
from experiments.path_b_common_random_numbers import keyed_ocse
from experiments.path_b_screen_frontier_v2 import (
    endpoint_plus_conditional_rescue,
    lagged_design,
)


def controlled_dag_var(
    n,
    T,
    edge_probability,
    beta_min,
    beta_max_ratio=2.0,
    noise_scale=0.1,
    seed=0,
):
    rng = np.random.default_rng(seed)

    # Random topological order, then only forward edges in that order.
    order = rng.permutation(n)
    position = np.empty(n, dtype=int)
    position[order] = np.arange(n)

    A = np.zeros((n, n), dtype=float)
    for source in range(n):
        for target in range(n):
            if source == target:
                continue
            if position[source] >= position[target]:
                continue
            if rng.random() >= edge_probability:
                continue
            magnitude = rng.uniform(
                beta_min, beta_max_ratio * beta_min
            )
            sign = -1.0 if rng.random() < 0.5 else 1.0
            # Generator convention: X_t[target] = A[target, source] X_{t-1}[source].
            A[target, source] = sign * magnitude

    # A is permutation-similar to a strictly triangular matrix, so rho(A)=0.
    spectral_radius = float(
        np.max(np.abs(np.linalg.eigvals(A))) if n else 0.0
    )

    series = np.zeros((T, n), dtype=float)
    series[0] = noise_scale * rng.standard_normal(n)
    for t in range(1, T):
        series[t] = A @ series[t - 1] + noise_scale * rng.standard_normal(n)

    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    for target in range(n):
        for source in np.flatnonzero(A[target] != 0):
            graph.add_edge(int(source), int(target))

    return series, A, graph, spectral_radius


def _truth_for_target(A, target):
    return {
        int(source)
        for source in np.flatnonzero(A[target] != 0)
        if int(source) != int(target)
    }


def _globalize(candidate_ids, local_ids):
    return {
        int(candidate_ids[int(local_idx)])
        for local_idx in local_ids
    }


def _set_metrics(truth, predicted):
    truth = set(truth)
    predicted = set(predicted)
    tp = len(truth & predicted)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": tp / len(truth) if truth else 1.0,
        "precision": tp / len(predicted) if predicted else 1.0,
        "contains_truth": truth.issubset(predicted),
    }


def _true_parent_min_marginal_info(X, Y, truth):
    if not truth:
        return None
    values = []
    for source in sorted(truth):
        value = conditional_mutual_information(
            X[:, [source]],
            Y,
            None,
            method="gaussian",
        )
        values.append(max(0.0, float(value)))
    return float(min(values))


def _target_run(
    X,
    Y_all,
    feature_names,
    series,
    A,
    seed,
    target,
    retention,
    n_shuffles,
):
    Y = Y_all[:, [target]]
    Z_init = series[:-1, [target]]
    truth = _truth_for_target(A, target)
    candidate_ids = candidate_ids_for_target(feature_names, target)
    X_candidates = X[:, candidate_ids]

    t0 = time.perf_counter()
    endpoint_local = information_lasso_optimal_causation_entropy(
        X_candidates,
        Y,
        np.random.default_rng(seed * 10000 + target),
        information="gaussian",
    )
    endpoint_time = time.perf_counter() - t0
    endpoint = _globalize(candidate_ids, endpoint_local)

    t0 = time.perf_counter()
    screen_local, screen_diag = endpoint_plus_conditional_rescue(
        X_candidates,
        Y,
        np.random.default_rng(seed * 10000 + target),
        retention=retention,
        information="gaussian",
    )
    screen_time = time.perf_counter() - t0
    screen = _globalize(candidate_ids, screen_local)

    t0 = time.perf_counter()
    full_forward_local, full_final_local = keyed_ocse(
        X_candidates,
        Y,
        Z_init,
        candidate_ids,
        seed,
        target,
        n_shuffles=n_shuffles,
    )
    full_time = time.perf_counter() - t0
    full_forward = _globalize(candidate_ids, full_forward_local)
    full_final = _globalize(candidate_ids, full_final_local)

    restricted_ids = sorted(screen)
    if restricted_ids:
        restricted_local_X = X[:, restricted_ids]
        t0 = time.perf_counter()
        _, restricted_final_local = keyed_ocse(
            restricted_local_X,
            Y,
            Z_init,
            restricted_ids,
            seed,
            target,
            n_shuffles=n_shuffles,
        )
        restricted_time = time.perf_counter() - t0
        restricted_final = _globalize(
            restricted_ids, restricted_final_local
        )
    else:
        restricted_time = 0.0
        restricted_final = set()

    return {
        "target": target,
        "true_parent_count": len(truth),
        "truth": sorted(truth),
        "min_true_parent_marginal_info": _true_parent_min_marginal_info(
            X, Y, truth
        ),
        "endpoint": _set_metrics(truth, endpoint),
        "screen": _set_metrics(truth, screen),
        "full_final": _set_metrics(truth, full_final),
        "restricted_final": _set_metrics(truth, restricted_final),
        "full_forward_contains_truth": truth.issubset(full_forward),
        "screen_size": len(screen),
        "candidate_count": len(candidate_ids),
        "screen_retention": len(screen) / len(candidate_ids),
        "screen_endpoint_size": int(screen_diag["endpoint_size"]),
        "screen_rescued": int(screen_diag["rescued"]),
        "endpoint_time_sec": endpoint_time,
        "screen_time_sec": screen_time,
        "full_time_sec": full_time,
        "restricted_time_sec": restricted_time,
    }


def _aggregate(rows):
    nonempty = [row for row in rows if row["true_parent_count"] > 0]

    def mean(path):
        vals = []
        for row in nonempty:
            obj = row
            for key in path:
                obj = obj[key]
            vals.append(float(obj))
        return float(np.mean(vals)) if vals else None

    min_infos = [
        row["min_true_parent_marginal_info"]
        for row in nonempty
        if row["min_true_parent_marginal_info"] is not None
    ]

    return {
        "targets": len(rows),
        "targets_with_parents": len(nonempty),
        "endpoint_parent_containment_rate": mean(
            ["endpoint", "contains_truth"]
        ),
        "screen_parent_containment_rate": mean(
            ["screen", "contains_truth"]
        ),
        "full_parent_containment_rate": mean(
            ["full_final", "contains_truth"]
        ),
        "restricted_parent_containment_rate": mean(
            ["restricted_final", "contains_truth"]
        ),
        "endpoint_true_parent_recall": mean(["endpoint", "recall"]),
        "screen_true_parent_recall": mean(["screen", "recall"]),
        "full_true_parent_recall": mean(["full_final", "recall"]),
        "restricted_true_parent_recall": mean(
            ["restricted_final", "recall"]
        ),
        "endpoint_precision": mean(["endpoint", "precision"]),
        "full_precision": mean(["full_final", "precision"]),
        "restricted_precision": mean(["restricted_final", "precision"]),
        "mean_screen_retention": float(
            np.mean([row["screen_retention"] for row in rows])
        ),
        "median_min_true_parent_marginal_info": (
            float(np.median(min_infos)) if min_infos else None
        ),
        "p10_min_true_parent_marginal_info": (
            float(np.quantile(min_infos, 0.10)) if min_infos else None
        ),
        "mean_full_time_sec": float(
            np.mean([row["full_time_sec"] for row in rows])
        ),
        "mean_restricted_time_sec": float(
            np.mean([row["restricted_time_sec"] for row in rows])
        ),
        "wallclock_ratio_restricted_over_full": (
            sum(row["restricted_time_sec"] for row in rows)
            / sum(row["full_time_sec"] for row in rows)
            if sum(row["full_time_sec"] for row in rows) > 0
            else None
        ),
        "wallclock_ratio_total_path_b_over_full": (
            sum(
                row["screen_time_sec"] + row["restricted_time_sec"]
                for row in rows
            )
            / sum(row["full_time_sec"] for row in rows)
            if sum(row["full_time_sec"] for row in rows) > 0
            else None
        ),
        "end_to_end_speedup": (
            sum(row["full_time_sec"] for row in rows)
            / sum(
                row["screen_time_sec"] + row["restricted_time_sec"]
                for row in rows
            )
            if sum(
                row["screen_time_sec"] + row["restricted_time_sec"]
                for row in rows
            ) > 0
            else None
        ),
    }


def run_benchmark(
    n_nodes,
    T_values,
    beta_values,
    seeds,
    edge_probability,
    retention,
    n_shuffles,
    noise_scale,
):
    result = {
        "config": {
            "n_nodes": n_nodes,
            "T_values": list(T_values),
            "beta_values": list(beta_values),
            "seeds": seeds,
            "edge_probability": edge_probability,
            "retention": retention,
            "n_shuffles": n_shuffles,
            "noise_scale": noise_scale,
        },
        "cells": [],
    }

    p_candidates = max(2, n_nodes - 1)

    for T in T_values:
        for beta_min in beta_values:
            rows = []
            spectral_radii = []
            min_realized_weights = []

            for seed in range(seeds):
                series, A, _graph, spectral_radius = controlled_dag_var(
                    n=n_nodes,
                    T=T,
                    edge_probability=edge_probability,
                    beta_min=beta_min,
                    noise_scale=noise_scale,
                    seed=seed,
                )
                spectral_radii.append(spectral_radius)
                nonzero = np.abs(A[A != 0])
                if nonzero.size:
                    min_realized_weights.append(float(nonzero.min()))

                X, Y_all, feature_names, original_series = lagged_design(
                    series, max_lag=1
                )
                for target in range(n_nodes):
                    rows.append(
                        _target_run(
                            X,
                            Y_all,
                            feature_names,
                            original_series,
                            A,
                            seed,
                            target,
                            retention,
                            n_shuffles,
                        )
                    )

            scale = beta_min * math.sqrt(
                max(1, T - 1) / math.log(p_candidates)
            )
            result["cells"].append(
                {
                    "T": T,
                    "beta_min": beta_min,
                    "theory_scale_beta_sqrt_n_over_logp": scale,
                    "max_spectral_radius": float(max(spectral_radii)),
                    "min_realized_abs_weight": (
                        float(min(min_realized_weights))
                        if min_realized_weights
                        else None
                    ),
                    "summary": _aggregate(rows),
                    "rows": rows,
                }
            )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nodes", type=int, default=20)
    parser.add_argument(
        "--T-values", type=int, nargs="+", default=[150, 300, 600]
    )
    parser.add_argument(
        "--beta-values",
        type=float,
        nargs="+",
        default=[0.02, 0.04, 0.08, 0.16],
    )
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--edge-probability", type=float, default=0.10)
    parser.add_argument("--retention", type=float, default=0.40)
    parser.add_argument("--n-shuffles", type=int, default=10)
    parser.add_argument("--noise-scale", type=float, default=0.1)
    args = parser.parse_args()

    result = run_benchmark(
        n_nodes=args.n_nodes,
        T_values=tuple(args.T_values),
        beta_values=tuple(args.beta_values),
        seeds=args.seeds,
        edge_probability=args.edge_probability,
        retention=args.retention,
        n_shuffles=args.n_shuffles,
        noise_scale=args.noise_scale,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
