"""End-to-end Path-B v2 benchmark.

Path B v2 = Path-A Information-LASSO endpoint
          + one conditional-CMI rescue pass
          + exact standard oCSE refinement.

This benchmark keeps the screening rule separate from the package API and
compares it against full standard oCSE, ordinary LASSO, and Path A.
"""

import argparse
import json
import time

import networkx as nx
import numpy as np

from causationentropy import discover_network
from causationentropy.core.discovery import standard_optimal_causation_entropy
from causationentropy.datasets.synthetic import (
    linear_stochastic_gaussian_process,
    logisic_dynamics,
    poisson_coupled_oscillators,
)
from experiments.path_b_screen_frontier_v2 import (
    endpoint_plus_conditional_rescue,
    lagged_design,
)


def edge_set(graph):
    return {
        (int(str(u).lstrip("X")), int(str(v).lstrip("X")), int(data.get("lag", 1)))
        for u, v, data in graph.edges(data=True)
    }


def metrics(truth, predicted, n_nodes, max_lag):
    tp = len(truth & predicted)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    total = n_nodes * n_nodes * max_lag
    tn = max(0, total - len(truth) - fp)
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


def run_discover(data, method, information, max_lag, alpha, n_shuffles, seed, n_jobs):
    start = time.perf_counter()
    graph = discover_network(
        data=data,
        method=method,
        information=information,
        max_lag=max_lag,
        alpha_forward=alpha,
        alpha_backward=alpha,
        n_shuffles=n_shuffles,
        random_state=seed,
        n_jobs=n_jobs,
    )
    return graph, time.perf_counter() - start


def run_path_b_v2(
    data,
    information,
    retention,
    max_lag,
    alpha,
    n_shuffles,
    seed,
    n_jobs,
):
    X, Y_all, feature_names, series = lagged_design(data, max_lag=max_lag)
    n_nodes = series.shape[1]
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([f"X{i}" for i in range(n_nodes)])

    screened_total = 0
    endpoint_total = 0
    rescued_total = 0
    screen_seconds = 0.0
    start_total = time.perf_counter()

    for target in range(n_nodes):
        Y = Y_all[:, [target]]
        rng = np.random.default_rng(seed * 10000 + target)

        start_screen = time.perf_counter()
        screened, diag = endpoint_plus_conditional_rescue(
            X,
            Y,
            rng,
            retention=retention,
            information=information,
        )
        screen_seconds += time.perf_counter() - start_screen
        screened_total += len(screened)
        endpoint_total += diag["endpoint_size"]
        rescued_total += diag["rescued"]

        if not screened:
            continue

        Z_init = np.column_stack(
            [
                series[max_lag - lag : series.shape[0] - lag, target]
                for lag in range(1, max_lag + 1)
            ]
        )
        refined_local = standard_optimal_causation_entropy(
            X[:, screened],
            Y,
            Z_init,
            rng,
            alpha1=alpha,
            alpha2=alpha,
            n_shuffles=n_shuffles,
            information=information,
            n_jobs=n_jobs,
        )
        for local_idx in refined_local:
            global_idx = screened[int(local_idx)]
            source, lag = feature_names[global_idx]
            graph.add_edge(f"X{source}", f"X{target}", lag=lag)

    total_seconds = time.perf_counter() - start_total
    total_candidates = n_nodes * X.shape[1]
    return graph, {
        "runtime_seconds": total_seconds,
        "screen_runtime_seconds": screen_seconds,
        "candidate_retention": screened_total / total_candidates,
        "candidate_reduction": 1.0 - screened_total / total_candidates,
        "endpoint_candidates": endpoint_total,
        "rescued_candidates": rescued_total,
        "screened_candidates": screened_total,
        "total_candidates": total_candidates,
    }


def gaussian_case(seed, n_nodes=12, T=300, p=0.12, rho=0.7):
    graph_true = nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)
    data, _ = linear_stochastic_gaussian_process(
        rho=rho, n=n_nodes, T=T, p=p, seed=seed, G=graph_true
    )
    truth = {(int(u), int(v), 1) for u, v in graph_true.edges()}
    return data, truth, "gaussian"


def logistic_case(seed, n_nodes=8, T=250, p=0.15):
    data, adjacency = logisic_dynamics(n=n_nodes, p=p, t=T, seed=seed)
    truth = {
        (int(source), int(target), 1)
        for source, target in zip(*np.nonzero(np.asarray(adjacency)))
        if source != target
    }
    # KDE is the package's nonlinear estimator that does not rely on the open kNN path.
    return data, truth, "kde"


def poisson_case(seed, n_nodes=8, T=250, p=0.15):
    graph_true = nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)
    data, _ = poisson_coupled_oscillators(
        n=n_nodes, T=T, p=p, seed=seed, G=graph_true
    )
    truth = {(int(u), int(v), 1) for u, v in graph_true.edges()}
    return data, truth, "poisson"


def run_case(
    case,
    seed=0,
    retention=0.30,
    max_lag=1,
    alpha=0.05,
    n_shuffles=100,
    n_jobs=1,
):
    factories = {
        "gaussian": gaussian_case,
        "logistic": logistic_case,
        "poisson": poisson_case,
    }
    data, truth, information = factories[case](seed)
    n_nodes = data.shape[1]

    result = {
        "config": {
            "case": case,
            "seed": seed,
            "n_nodes": n_nodes,
            "T": int(data.shape[0]),
            "information": information,
            "retention_target": retention,
            "max_lag": max_lag,
            "alpha": alpha,
            "n_shuffles": n_shuffles,
            "n_jobs": n_jobs,
            "truth_edges": len(truth),
        }
    }

    for method in ("standard", "lasso", "information_lasso"):
        graph, runtime = run_discover(
            data,
            method,
            information,
            max_lag,
            alpha,
            n_shuffles,
            seed,
            n_jobs,
        )
        result[method] = {
            **metrics(truth, edge_set(graph), n_nodes, max_lag),
            "runtime_seconds": runtime,
        }

    graph_b, diagnostics = run_path_b_v2(
        data,
        information,
        retention,
        max_lag,
        alpha,
        n_shuffles,
        seed,
        n_jobs,
    )
    result["path_b_v2"] = {
        **metrics(truth, edge_set(graph_b), n_nodes, max_lag),
        **diagnostics,
    }

    full_edges = edge_set(
        run_discover(
            data,
            "standard",
            information,
            max_lag,
            alpha,
            n_shuffles,
            seed,
            n_jobs,
        )[0]
    )
    path_b_edges = edge_set(graph_b)
    result["path_b_v2"]["full_ocse_edge_recall"] = (
        len(full_edges & path_b_edges) / len(full_edges) if full_edges else 1.0
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("gaussian", "logistic", "poisson", "all"),
        default="gaussian",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retention", type=float, default=0.30)
    parser.add_argument("--n-shuffles", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    cases = ("gaussian", "logistic", "poisson") if args.case == "all" else (args.case,)
    output = {
        case: run_case(
            case,
            seed=args.seed,
            retention=args.retention,
            n_shuffles=args.n_shuffles,
            n_jobs=args.n_jobs,
        )
        for case in cases
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
