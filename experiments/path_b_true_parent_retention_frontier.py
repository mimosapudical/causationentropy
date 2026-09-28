"""True-parent retention frontier for the one-shot Path-B screen.

This audit evaluates the screen against the generating causal parents rather
than against the noisy finite-sample full-oCSE support.  The Path-A endpoint and
conditional rescue scores are computed once per target; different retention
budgets use prefixes of the same deterministic rescue ranking.
"""

import argparse
import json
import math

import networkx as nx
import numpy as np

from causationentropy.core.discovery import (
    information_lasso_optimal_causation_entropy,
)
from causationentropy.core.information.conditional_mutual_information import (
    conditional_mutual_information,
)
from experiments.path_b_benchmark_v2 import candidate_ids_for_target
from experiments.path_b_beta_min_audit import floored_linear_gaussian_var
from experiments.path_b_screen_frontier_v2 import lagged_design


def _truth_local(A, target, candidate_ids, feature_names):
    return {
        int(local_idx)
        for local_idx, global_idx in enumerate(candidate_ids)
        if abs(
            float(
                A[
                    target,
                    int(feature_names[global_idx][0]),
                ]
            )
        )
        > 0
    }


def _endpoint_and_rescue_order(X, Y, seed, target):
    endpoint = set(
        int(j)
        for j in information_lasso_optimal_causation_entropy(
            X,
            Y,
            np.random.default_rng(seed * 10000 + target),
            information="gaussian",
        )
    )
    Z = X[:, sorted(endpoint)] if endpoint else None
    scored = []
    for j in range(X.shape[1]):
        if j in endpoint:
            continue
        value = conditional_mutual_information(
            X[:, [j]],
            Y,
            Z,
            method="gaussian",
        )
        score = float(value) if np.isfinite(value) else -np.inf
        scored.append((score, int(j)))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return endpoint, [j for _score, j in scored]


def _screen_for_retention(endpoint, order, p, retention):
    target_size = max(
        int(math.ceil(retention * p)),
        len(endpoint) + (1 if len(endpoint) < p else 0),
    )
    target_size = min(p, target_size)
    selected = set(endpoint)
    needed = max(0, target_size - len(selected))
    selected.update(order[:needed])
    return selected


def _summary(rows):
    parent_rows = [row for row in rows if row["parent_count"] > 0]
    edges = sum(row["parent_count"] for row in parent_rows)
    tp = sum(row["parent_tp"] for row in parent_rows)
    return {
        "targets": len(rows),
        "targets_with_parents": len(parent_rows),
        "true_parent_edges": int(edges),
        "parent_recall": tp / edges if edges else 1.0,
        "parent_complete_target_rate": (
            float(np.mean([
                row["parent_complete"] for row in parent_rows
            ]))
            if parent_rows else 1.0
        ),
        "mean_retention": float(
            np.mean([row["actual_retention"] for row in rows])
        ),
        "mean_endpoint_retention": float(
            np.mean([row["endpoint_retention"] for row in rows])
        ),
    }


def run_audit(
    n_nodes=50,
    sample_sizes=(150, 300, 600),
    weight_floors=(0.0, 0.1, 0.25, 0.5),
    retentions=(0.10, 0.20, 0.30, 0.40, 0.50),
    seeds=3,
    edge_probability=0.10,
    rho=0.7,
):
    cells = []
    for T in sample_sizes:
        for floor in weight_floors:
            by_retention = {float(r): [] for r in retentions}
            beta_values = []

            for seed in range(seeds):
                data, A, _graph = floored_linear_gaussian_var(
                    n=n_nodes,
                    T=T,
                    p=edge_probability,
                    rho=rho,
                    weight_floor=floor,
                    seed=seed,
                    burnin=200,
                )
                beta_values.extend(
                    np.abs(A[np.abs(A) > 0]).tolist()
                )
                X, Y_all, feature_names, _series = lagged_design(
                    data, max_lag=1
                )

                for target in range(n_nodes):
                    candidate_ids = candidate_ids_for_target(
                        feature_names, target
                    )
                    Xc = X[:, candidate_ids]
                    Y = Y_all[:, [target]]
                    truth = _truth_local(
                        A, target, candidate_ids, feature_names
                    )
                    endpoint, order = _endpoint_and_rescue_order(
                        Xc, Y, seed, target
                    )

                    for retention in retentions:
                        screen = _screen_for_retention(
                            endpoint,
                            order,
                            Xc.shape[1],
                            retention,
                        )
                        by_retention[float(retention)].append(
                            {
                                "seed": seed,
                                "target": target,
                                "parent_count": len(truth),
                                "parent_tp": len(truth & screen),
                                "parent_complete": truth.issubset(screen),
                                "actual_retention": (
                                    len(screen) / Xc.shape[1]
                                ),
                                "endpoint_retention": (
                                    len(endpoint) / Xc.shape[1]
                                ),
                            }
                        )

            cells.append(
                {
                    "n_nodes": n_nodes,
                    "T": T,
                    "weight_floor": floor,
                    "realized_beta_min": (
                        float(np.min(beta_values)) if beta_values else None
                    ),
                    "realized_beta_median": (
                        float(np.median(beta_values)) if beta_values else None
                    ),
                    "retentions": {
                        f"{retention:.2f}": _summary(
                            by_retention[float(retention)]
                        )
                        for retention in retentions
                    },
                }
            )

    return {
        "config": {
            "n_nodes": n_nodes,
            "sample_sizes": list(sample_sizes),
            "weight_floors": list(weight_floors),
            "retentions": list(retentions),
            "seeds": seeds,
            "edge_probability": edge_probability,
            "rho": rho,
        },
        "cells": cells,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nodes", type=int, default=50)
    parser.add_argument(
        "--sample-sizes", nargs="+", type=int, default=[150, 300, 600]
    )
    parser.add_argument(
        "--weight-floors",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.25, 0.5],
    )
    parser.add_argument(
        "--retentions",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30, 0.40, 0.50],
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--edge-probability", type=float, default=0.10)
    parser.add_argument("--rho", type=float, default=0.7)
    args = parser.parse_args()

    print(
        json.dumps(
            run_audit(
                n_nodes=args.n_nodes,
                sample_sizes=tuple(args.sample_sizes),
                weight_floors=tuple(args.weight_floors),
                retentions=tuple(args.retentions),
                seeds=args.seeds,
                edge_probability=args.edge_probability,
                rho=args.rho,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
