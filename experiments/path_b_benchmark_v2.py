"""End-to-end Path-B v2 benchmark.

Path B v2 = Path-A Information-LASSO endpoint
          + one conditional-CMI rescue pass
          + exact standard oCSE refinement.

This benchmark keeps the screening rule separate from the package API and
compares it against full standard oCSE, ordinary LASSO, and Path A.
"""

import argparse
import contextlib
import io
import json
import time
from unittest.mock import patch

import networkx as nx
import numpy as np

import causationentropy.core.discovery as discovery_module
from causationentropy import discover_network
from causationentropy.core.discovery import standard_optimal_causation_entropy
from causationentropy.core.information.conditional_mutual_information import (
    conditional_mutual_information,
)
from causationentropy.datasets.synthetic import (
    linear_stochastic_gaussian_process,
    logisic_dynamics,
    poisson_coupled_oscillators,
)
from experiments.path_b_screen_frontier_v2 import (
    endpoint_plus_conditional_rescue,
    lagged_design,
)


@contextlib.contextmanager
def discovery_work_counter():
    """Count dominant exact-oCSE work without changing statistical decisions."""
    counts = {
        "forward_observed_cmi_scores": 0,
        "backward_observed_cmi_scores": 0,
        "output_observed_cmi_scores": 0,
        "shuffle_tests": 0,
        "shuffle_cmi_evaluations": 0,
    }
    original_scores = discovery_module._candidate_cmi_values
    original_backward = discovery_module.backward
    original_shuffle = discovery_module.shuffle_test

    def counted_scores(
        X_full,
        candidates,
        Y,
        Z,
        information,
        metric,
        k_means,
        bandwidth,
    ):
        counts["forward_observed_cmi_scores"] += len(candidates)
        return original_scores(
            X_full,
            candidates,
            Y,
            Z,
            information,
            metric,
            k_means,
            bandwidth,
        )

    def counted_backward(*args, **kwargs):
        if len(args) >= 3:
            selected = args[2]
        else:
            selected = kwargs.get("S_init", [])
        counts["backward_observed_cmi_scores"] += len(selected)
        return original_backward(*args, **kwargs)

    def counted_shuffle(*args, **kwargs):
        n_shuffles = int(kwargs.get("n_shuffles", 500))
        counts["shuffle_tests"] += 1
        counts["shuffle_cmi_evaluations"] += n_shuffles
        return original_shuffle(*args, **kwargs)

    with patch.object(
        discovery_module, "_candidate_cmi_values", counted_scores
    ), patch.object(
        discovery_module, "backward", counted_backward
    ), patch.object(
        discovery_module, "shuffle_test", counted_shuffle
    ):
        yield counts


def edge_set(graph):
    return {
        (
            int(str(u).lstrip("X")),
            int(str(v).lstrip("X")),
            int(data.get("lag", 1)),
        )
        for u, v, data in graph.edges(data=True)
        if int(str(u).lstrip("X")) != int(str(v).lstrip("X"))
    }


def candidate_ids_for_target(feature_names, target):
    """External lagged predictors considered by standard oCSE."""
    return [
        idx
        for idx, (source, _lag) in enumerate(feature_names)
        if source != target
    ]


def metrics(truth, predicted, n_nodes, max_lag):
    tp = len(truth & predicted)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    total = n_nodes * (n_nodes - 1) * max_lag
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
    # discover_network prints one progress line per target. Suppress it here so
    # redirected benchmark output remains valid JSON.
    with contextlib.redirect_stdout(io.StringIO()):
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


def run_full_standard_matched(
    data,
    information,
    max_lag,
    alpha,
    n_shuffles,
    seed,
):
    """Run full standard oCSE with the same per-target RNG schedule as Path B."""
    X, Y_all, feature_names, series = lagged_design(data, max_lag=max_lag)
    n_nodes = series.shape[1]
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([f"X{i}" for i in range(n_nodes)])
    support_by_target = {}

    start_total = time.perf_counter()
    with discovery_work_counter() as work:
        for target in range(n_nodes):
            Y = Y_all[:, [target]]
            rng = np.random.default_rng(seed * 10000 + target)
            Z_init = np.column_stack(
                [
                    series[max_lag - lag : series.shape[0] - lag, target]
                    for lag in range(1, max_lag + 1)
                ]
            )
            candidate_ids = candidate_ids_for_target(
                feature_names,
                target,
            )
            X_candidates = X[:, candidate_ids]
            support_local = standard_optimal_causation_entropy(
                X_candidates,
                Y,
                Z_init,
                rng,
                alpha1=alpha,
                alpha2=alpha,
                n_shuffles=n_shuffles,
                information=information,
            )
            support = [
                int(candidate_ids[int(local_idx)])
                for local_idx in support_local
            ]
            support_by_target[target] = set(support)

            for global_idx in support:
                source, lag = feature_names[global_idx]
                others = [idx for idx in support if idx != global_idx]
                Z_selected = X[:, others] if others else None
                Z_cond = (
                    Z_init
                    if Z_selected is None
                    else np.hstack((Z_init, Z_selected))
                )
                X_predictor = X[:, [global_idx]]
                work["output_observed_cmi_scores"] += 1
                cmi = conditional_mutual_information(
                    X_predictor,
                    Y,
                    Z_cond,
                    method=information,
                )
                test_result = discovery_module.shuffle_test(
                    X_predictor,
                    Y,
                    Z_cond,
                    cmi,
                    alpha=alpha,
                    rng=rng,
                    n_shuffles=n_shuffles,
                    information=information,
                )
                graph.add_edge(
                    f"X{source}",
                    f"X{target}",
                    lag=lag,
                    cmi=cmi,
                    p_value=test_result["P_value"],
                )

    return (
        graph,
        time.perf_counter() - start_total,
        support_by_target,
        dict(work),
    )


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
    screen_by_target = {}
    support_by_target = {}
    start_total = time.perf_counter()
    screen_marginal_cmi_scores = 0
    screen_rescue_cmi_scores = 0

    with discovery_work_counter() as work:
        for target in range(n_nodes):
            Y = Y_all[:, [target]]
            rng = np.random.default_rng(seed * 10000 + target)

            candidate_ids = candidate_ids_for_target(
                feature_names,
                target,
            )
            X_candidates = X[:, candidate_ids]

            start_screen = time.perf_counter()
            screened_local, diag = endpoint_plus_conditional_rescue(
                X_candidates,
                Y,
                rng,
                retention=retention,
                information=information,
            )
            screen_seconds += time.perf_counter() - start_screen
            screened = [
                int(candidate_ids[int(local_idx)])
                for local_idx in screened_local
            ]
            screened_total += len(screened)
            endpoint_total += diag["endpoint_size"]
            rescued_total += diag["rescued"]
            screen_marginal_cmi_scores += X_candidates.shape[1]
            screen_rescue_cmi_scores += max(
                0,
                X_candidates.shape[1] - diag["endpoint_size"],
            )
            screen_by_target[target] = set(screened)

            if not screened:
                support_by_target[target] = set()
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
            )
            support_by_target[target] = {
                int(screened[int(local_idx)]) for local_idx in refined_local
            }
            for local_idx in refined_local:
                global_idx = screened[int(local_idx)]
                source, lag = feature_names[global_idx]

                # Match discover_network's output-stage work so runtime comparisons
                # do not favor Path B by omitting final edge CMI/p-value reporting.
                other_local = [
                    idx for idx in refined_local if idx != local_idx
                ]
                other_global = [screened[int(idx)] for idx in other_local]
                Z_selected = X[:, other_global] if other_global else None
                Z_cond = (
                    Z_init
                    if Z_selected is None
                    else np.hstack((Z_init, Z_selected))
                )
                X_predictor = X[:, [global_idx]]
                work["output_observed_cmi_scores"] += 1
                cmi = conditional_mutual_information(
                    X_predictor,
                    Y,
                    Z_cond,
                    method=information,
                )
                test_result = discovery_module.shuffle_test(
                    X_predictor,
                    Y,
                    Z_cond,
                    cmi,
                    alpha=alpha,
                    rng=rng,
                    n_shuffles=n_shuffles,
                    information=information,
                )
                graph.add_edge(
                    f"X{source}",
                    f"X{target}",
                    lag=lag,
                    cmi=cmi,
                    p_value=test_result["P_value"],
                )

    total_seconds = time.perf_counter() - start_total
    total_candidates = n_nodes * (n_nodes - 1) * max_lag
    return graph, {
        "runtime_seconds": total_seconds,
        "screen_runtime_seconds": screen_seconds,
        "candidate_retention": screened_total / total_candidates,
        "candidate_reduction": 1.0 - screened_total / total_candidates,
        "endpoint_candidates": endpoint_total,
        "rescued_candidates": rescued_total,
        "screened_candidates": screened_total,
        "total_candidates": total_candidates,
        "screen_marginal_cmi_scores": screen_marginal_cmi_scores,
        "screen_rescue_cmi_scores": screen_rescue_cmi_scores,
        **dict(work),
        "screen_by_target": screen_by_target,
        "support_by_target": support_by_target,
    }


def gaussian_case(seed, n_nodes=12, T=300, p=0.12, rho=0.7):
    graph_true = nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)
    data, _ = linear_stochastic_gaussian_process(
        rho=rho, n=n_nodes, T=T, p=p, seed=seed, G=graph_true
    )
    truth = {(int(u), int(v), 1) for u, v in graph_true.edges()}
    metadata = {
        "benchmark_role": "primary",
        "generator": "linear_stochastic_gaussian_process",
        "rho": rho,
        "edge_probability": p,
    }
    return data, truth, "gaussian", metadata


def logistic_case(
    seed,
    n_nodes=8,
    T=250,
    p=0.25,
    r=3.9,
    sigma=0.01,
):
    # The repository default (r=3.99, sigma=0.1) can leave the invariant
    # interval and overflow for longer trajectories. This lower-coupling
    # configuration stayed finite and in [0, 1] across the local seed scan.
    data, adjacency = logisic_dynamics(
        n=n_nodes,
        p=p,
        t=T,
        r=r,
        sigma=sigma,
        seed=seed,
    )
    if not np.isfinite(data).all():
        raise ValueError(
            "logistic benchmark produced non-finite values; "
            "do not score an unstable trajectory"
        )
    truth = {
        (int(source), int(target), 1)
        for source, target in zip(*np.nonzero(np.asarray(adjacency)))
        if source != target
    }
    metadata = {
        "benchmark_role": "primary",
        "generator": "logisic_dynamics",
        "r": r,
        "sigma": sigma,
        "edge_probability": p,
        "finite": True,
        "within_unit_interval": bool(
            np.min(data) >= 0.0 and np.max(data) <= 1.0
        ),
    }
    # KDE is the package's nonlinear estimator that does not rely on the
    # separately known kNN-estimator issues.
    return data, truth, "kde", metadata


def poisson_case(seed, n_nodes=8, T=250, p=0.15):
    graph_true = nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)
    data, _ = poisson_coupled_oscillators(
        n=n_nodes, T=T, p=p, seed=seed, G=graph_true
    )
    truth = {(int(u), int(v), 1) for u, v in graph_true.edges()}
    metadata = {
        "benchmark_role": "estimator_audit",
        "generator": "poisson_coupled_oscillators",
        "edge_probability": p,
        "note": (
            "Keep outside the primary benchmark table until the repository "
            "Poisson integration behavior is reproduced exactly."
        ),
    }
    return data, truth, "poisson", metadata


def run_case(
    case,
    seed=0,
    retention=0.40,
    max_lag=1,
    alpha=0.05,
    n_shuffles=100,
    n_jobs=1,
    n_nodes=None,
    T=None,
):
    factories = {
        "gaussian": gaussian_case,
        "logistic": logistic_case,
        "poisson": poisson_case,
    }
    factory_kwargs = {}
    if n_nodes is not None:
        factory_kwargs["n_nodes"] = n_nodes
    if T is not None:
        factory_kwargs["T"] = T
    data, truth, information, case_metadata = factories[case](seed, **factory_kwargs)
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
            **case_metadata,
        }
    }

    (
        full_graph,
        full_runtime,
        full_support_by_target,
        full_work,
    ) = run_full_standard_matched(
        data,
        information,
        max_lag,
        alpha,
        n_shuffles,
        seed,
    )
    result["standard"] = {
        **metrics(truth, edge_set(full_graph), n_nodes, max_lag),
        "runtime_seconds": full_runtime,
        **full_work,
        "total_cmi_evaluations": (
            full_work["forward_observed_cmi_scores"]
            + full_work["backward_observed_cmi_scores"]
            + full_work["output_observed_cmi_scores"]
            + full_work["shuffle_cmi_evaluations"]
        ),
    }

    for method in ("lasso", "information_lasso"):
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
    screen_by_target = diagnostics.pop("screen_by_target")
    path_b_support_by_target = diagnostics.pop("support_by_target")
    full_support_total = sum(
        len(support) for support in full_support_by_target.values()
    )
    full_support_screened = sum(
        len(full_support_by_target[target] & screen_by_target.get(target, set()))
        for target in full_support_by_target
    )
    full_support_refined = sum(
        len(
            full_support_by_target[target]
            & path_b_support_by_target.get(target, set())
        )
        for target in full_support_by_target
    )

    result["path_b_v2"] = {
        **metrics(truth, edge_set(graph_b), n_nodes, max_lag),
        **diagnostics,
        "total_cmi_evaluations": (
            diagnostics["screen_marginal_cmi_scores"]
            + diagnostics["screen_rescue_cmi_scores"]
            + diagnostics["forward_observed_cmi_scores"]
            + diagnostics["backward_observed_cmi_scores"]
            + diagnostics["output_observed_cmi_scores"]
            + diagnostics["shuffle_cmi_evaluations"]
        ),
        "screen_full_support_recall": (
            full_support_screened / full_support_total
            if full_support_total
            else 1.0
        ),
        "refined_full_support_recall": (
            full_support_refined / full_support_total
            if full_support_total
            else 1.0
        ),
    }

    full_edges = edge_set(full_graph)
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
    parser.add_argument("--retention", type=float, default=0.40)
    parser.add_argument("--n-shuffles", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-nodes", type=int, default=None)
    parser.add_argument("--T", type=int, default=None)
    args = parser.parse_args()

    cases = ("gaussian", "logistic", "poisson") if args.case == "all" else (args.case,)
    output = {
        case: run_case(
            case,
            seed=args.seed,
            retention=args.retention,
            n_shuffles=args.n_shuffles,
            n_jobs=args.n_jobs,
            n_nodes=args.n_nodes,
            T=args.T,
        )
        for case in cases
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
