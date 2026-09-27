"""Reusable benchmark harness for the first Path-B experiment.

This file intentionally lives outside the package API.  It is an experimental
harness for comparing:

1. full standard oCSE,
2. ordinary LASSO,
3. standalone Information-LASSO (Path A),
4. Information-LASSO screening followed by exact standard oCSE (Path B v1).

The Path-B implementation here is deliberately minimal: it uses the final
Information-LASSO support as the screen.  That makes it a diagnostic baseline,
not the proposed final method.  In particular, any true parent dropped by the
screen cannot be recovered by the exact oCSE refinement.
"""

import argparse
import json
import time

import networkx as nx
import numpy as np

from causationentropy import discover_network
from causationentropy.core.discovery import (
    information_lasso_optimal_causation_entropy,
    standard_optimal_causation_entropy,
)
from causationentropy.datasets.synthetic import linear_stochastic_gaussian_process


def _lagged_design(data, max_lag):
    series = np.asarray(data)
    T, n = series.shape
    if T <= max_lag + 2:
        raise ValueError("Time series too short for chosen max_lag.")

    X_lagged = []
    feature_names = []
    for j in range(n):
        for tau in range(1, max_lag + 1):
            X_lagged.append(series[max_lag - tau : T - tau, j])
            feature_names.append((j, tau))

    return (
        np.column_stack(X_lagged),
        series[max_lag:, :],
        feature_names,
        series,
    )


def _edge_set(graph):
    return {
        (int(str(u).lstrip("X")), int(str(v).lstrip("X")), int(data.get("lag", 1)))
        for u, v, data in graph.edges(data=True)
    }


def _truth_edge_set(adjacency):
    rows, cols = np.nonzero(np.asarray(adjacency))
    return {(int(src), int(dst), 1) for src, dst in zip(rows, cols)}


def _metrics(truth, predicted, n_nodes, max_lag):
    tp = len(truth & predicted)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    total_possible = n_nodes * n_nodes * max_lag
    tn = total_possible - len(truth) - fp
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
    }


def _run_discover(data, method, max_lag, alpha, n_shuffles, seed):
    start = time.perf_counter()
    graph = discover_network(
        data=data,
        method=method,
        information="gaussian",
        max_lag=max_lag,
        alpha_forward=alpha,
        alpha_backward=alpha,
        n_shuffles=n_shuffles,
        random_state=seed,
    )
    return graph, time.perf_counter() - start


def _run_path_b_v1(data, max_lag, alpha, n_shuffles, seed, truth):
    X_lagged, Y_all, feature_names, series = _lagged_design(data, max_lag)
    n_nodes = series.shape[1]
    rng = np.random.default_rng(seed)
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([f"X{i}" for i in range(n_nodes)])

    total_screened = 0
    true_screened = 0
    total_true = 0
    screen_time = 0.0

    start_total = time.perf_counter()
    for target in range(n_nodes):
        Y = Y_all[:, [target]]

        start_screen = time.perf_counter()
        screened = information_lasso_optimal_causation_entropy(
            X_lagged,
            Y,
            rng,
            information="gaussian",
        )
        screen_time += time.perf_counter() - start_screen
        total_screened += len(screened)

        true_parent_features = {
            idx
            for idx, (src, lag) in enumerate(feature_names)
            if (src, target, lag) in truth
        }
        total_true += len(true_parent_features)
        true_screened += len(true_parent_features.intersection(screened))

        if not screened:
            continue

        Z_init = np.column_stack(
            [
                series[max_lag - tau : series.shape[0] - tau, target]
                for tau in range(1, max_lag + 1)
            ]
        )
        refined_local = standard_optimal_causation_entropy(
            X_lagged[:, screened],
            Y,
            Z_init,
            rng,
            alpha1=alpha,
            alpha2=alpha,
            n_shuffles=n_shuffles,
            information="gaussian",
        )

        for local_idx in refined_local:
            original_idx = screened[local_idx]
            src, lag = feature_names[original_idx]
            graph.add_edge(f"X{src}", f"X{target}", lag=lag)

    total_time = time.perf_counter() - start_total
    total_candidates = n_nodes * X_lagged.shape[1]
    screen_parent_recall = true_screened / total_true if total_true else 1.0
    return graph, {
        "runtime_seconds": total_time,
        "screen_runtime_seconds": screen_time,
        "screen_parent_recall": screen_parent_recall,
        "candidate_retention": total_screened / total_candidates,
        "candidate_reduction": 1.0 - (total_screened / total_candidates),
        "screened_candidates": total_screened,
        "total_candidates": total_candidates,
    }


def run_benchmark(
    n_nodes=8,
    T=200,
    edge_probability=0.2,
    rho=0.7,
    max_lag=1,
    alpha=0.05,
    n_shuffles=100,
    seed=42,
):
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
    truth = {(int(src), int(dst), 1) for src, dst in graph_true.edges()}

    result = {
        "config": {
            "n_nodes": n_nodes,
            "T": T,
            "edge_probability": edge_probability,
            "rho": rho,
            "max_lag": max_lag,
            "alpha": alpha,
            "n_shuffles": n_shuffles,
            "seed": seed,
            "truth_edges": len(truth),
        }
    }

    for method in ("standard", "lasso", "information_lasso"):
        graph, runtime = _run_discover(
            data, method, max_lag, alpha, n_shuffles, seed
        )
        result[method] = {
            **_metrics(truth, _edge_set(graph), n_nodes, max_lag),
            "runtime_seconds": runtime,
        }

    graph_b, diagnostics = _run_path_b_v1(
        data, max_lag, alpha, n_shuffles, seed, truth
    )
    result["path_b_v1"] = {
        **_metrics(truth, _edge_set(graph_b), n_nodes, max_lag),
        **diagnostics,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nodes", type=int, default=8)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--edge-probability", type=float, default=0.2)
    parser.add_argument("--rho", type=float, default=0.7)
    parser.add_argument("--max-lag", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-shuffles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run_benchmark(
        n_nodes=args.n_nodes,
        T=args.T,
        edge_probability=args.edge_probability,
        rho=args.rho,
        max_lag=args.max_lag,
        alpha=args.alpha,
        n_shuffles=args.n_shuffles,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
