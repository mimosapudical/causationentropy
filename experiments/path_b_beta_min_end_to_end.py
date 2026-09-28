"""End-to-end speed/accuracy frontier in controlled beta-min Gaussian VARs.

This experiment connects the parent-sure retention frontier to actual oCSE
runtime and CMI work.  A full matched oCSE baseline is run once per
(T, weight_floor, seed) cell and reused across Path-B retention budgets.

Primary questions:
1. At which retention does Path B preserve causal accuracy in an identifiable
   beta-min regime?
2. How much wall-clock and permutation-CMI work is saved at that operating
   point?
"""

import argparse
import json

import numpy as np

from experiments.path_b_benchmark_v2 import (
    edge_set,
    metrics,
    run_full_standard_matched,
    run_path_b_v2,
)
from experiments.path_b_beta_min_audit import floored_linear_gaussian_var


def _truth_from_A(A):
    n = A.shape[0]
    return {
        (int(source), int(target), 1)
        for target in range(n)
        for source in range(n)
        if source != target and abs(float(A[target, source])) > 0
    }


def _total_cmi_full(work):
    return int(
        work["forward_observed_cmi_scores"]
        + work["backward_observed_cmi_scores"]
        + work["output_observed_cmi_scores"]
        + work["shuffle_cmi_evaluations"]
    )


def _total_cmi_pathb(diag):
    return int(
        diag["screen_marginal_cmi_scores"]
        + diag["screen_rescue_cmi_scores"]
        + diag["forward_observed_cmi_scores"]
        + diag["backward_observed_cmi_scores"]
        + diag["output_observed_cmi_scores"]
        + diag["shuffle_cmi_evaluations"]
    )


def run_audit(
    n_nodes=50,
    sample_sizes=(300, 600),
    weight_floors=(0.25, 0.5),
    retentions=(0.20, 0.30, 0.40),
    seeds=3,
    edge_probability=0.10,
    rho=0.7,
    alpha=0.05,
    n_shuffles=20,
):
    cells = []

    for T in sample_sizes:
        for floor in weight_floors:
            rows = []
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
                truth = _truth_from_A(A)

                full_graph, full_seconds, _support, full_work = (
                    run_full_standard_matched(
                        data=data,
                        information="gaussian",
                        max_lag=1,
                        alpha=alpha,
                        n_shuffles=n_shuffles,
                        seed=seed,
                    )
                )
                full_edges = edge_set(full_graph)
                full_metrics = metrics(
                    truth, full_edges, n_nodes=n_nodes, max_lag=1
                )
                full_total_cmi = _total_cmi_full(full_work)

                for retention in retentions:
                    path_graph, diag = run_path_b_v2(
                        data=data,
                        information="gaussian",
                        retention=retention,
                        max_lag=1,
                        alpha=alpha,
                        n_shuffles=n_shuffles,
                        seed=seed,
                        n_jobs=1,
                    )
                    path_edges = edge_set(path_graph)
                    path_metrics = metrics(
                        truth, path_edges, n_nodes=n_nodes, max_lag=1
                    )
                    path_total_cmi = _total_cmi_pathb(diag)

                    rows.append(
                        {
                            "seed": seed,
                            "retention_requested": retention,
                            "retention_actual": float(
                                diag["candidate_retention"]
                            ),
                            "full_runtime_seconds": float(full_seconds),
                            "path_runtime_seconds": float(
                                diag["runtime_seconds"]
                            ),
                            "runtime_speedup": float(
                                full_seconds / diag["runtime_seconds"]
                            ),
                            "full_total_cmi": full_total_cmi,
                            "path_total_cmi": path_total_cmi,
                            "total_cmi_reduction": float(
                                1.0 - path_total_cmi / full_total_cmi
                            ),
                            "full_shuffle_cmi": int(
                                full_work["shuffle_cmi_evaluations"]
                            ),
                            "path_shuffle_cmi": int(
                                diag["shuffle_cmi_evaluations"]
                            ),
                            "shuffle_cmi_reduction": float(
                                1.0
                                - diag["shuffle_cmi_evaluations"]
                                / max(
                                    1,
                                    full_work[
                                        "shuffle_cmi_evaluations"
                                    ],
                                )
                            ),
                            "full_metrics": full_metrics,
                            "path_metrics": path_metrics,
                            "delta_tp": int(
                                path_metrics["tp"] - full_metrics["tp"]
                            ),
                            "delta_fp": int(
                                path_metrics["fp"] - full_metrics["fp"]
                            ),
                            "delta_f1": float(
                                path_metrics["f1"] - full_metrics["f1"]
                            ),
                        }
                    )

            by_retention = {}
            for retention in retentions:
                subset = [
                    row
                    for row in rows
                    if row["retention_requested"] == retention
                ]
                by_retention[f"{retention:.2f}"] = {
                    "runs": len(subset),
                    "mean_actual_retention": float(
                        np.mean([
                            row["retention_actual"] for row in subset
                        ])
                    ),
                    "mean_runtime_speedup": float(
                        np.mean([
                            row["runtime_speedup"] for row in subset
                        ])
                    ),
                    "median_runtime_speedup": float(
                        np.median([
                            row["runtime_speedup"] for row in subset
                        ])
                    ),
                    "mean_shuffle_cmi_reduction": float(
                        np.mean([
                            row["shuffle_cmi_reduction"] for row in subset
                        ])
                    ),
                    "mean_total_cmi_reduction": float(
                        np.mean([
                            row["total_cmi_reduction"] for row in subset
                        ])
                    ),
                    "mean_full_recall": float(
                        np.mean([
                            row["full_metrics"]["recall"] for row in subset
                        ])
                    ),
                    "mean_path_recall": float(
                        np.mean([
                            row["path_metrics"]["recall"] for row in subset
                        ])
                    ),
                    "mean_full_precision": float(
                        np.mean([
                            row["full_metrics"]["precision"] for row in subset
                        ])
                    ),
                    "mean_path_precision": float(
                        np.mean([
                            row["path_metrics"]["precision"] for row in subset
                        ])
                    ),
                    "mean_full_f1": float(
                        np.mean([
                            row["full_metrics"]["f1"] for row in subset
                        ])
                    ),
                    "mean_path_f1": float(
                        np.mean([
                            row["path_metrics"]["f1"] for row in subset
                        ])
                    ),
                    "mean_delta_tp": float(
                        np.mean([row["delta_tp"] for row in subset])
                    ),
                    "mean_delta_fp": float(
                        np.mean([row["delta_fp"] for row in subset])
                    ),
                    "mean_delta_f1": float(
                        np.mean([row["delta_f1"] for row in subset])
                    ),
                }

            cells.append(
                {
                    "n_nodes": n_nodes,
                    "T": T,
                    "weight_floor": floor,
                    "by_retention": by_retention,
                    "rows": rows,
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
            "alpha": alpha,
            "n_shuffles": n_shuffles,
        },
        "cells": cells,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nodes", type=int, default=50)
    parser.add_argument(
        "--sample-sizes", nargs="+", type=int, default=[300, 600]
    )
    parser.add_argument(
        "--weight-floors", nargs="+", type=float, default=[0.25, 0.5]
    )
    parser.add_argument(
        "--retentions", nargs="+", type=float, default=[0.20, 0.30, 0.40]
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--edge-probability", type=float, default=0.10)
    parser.add_argument("--rho", type=float, default=0.7)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-shuffles", type=int, default=20)
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
                alpha=args.alpha,
                n_shuffles=args.n_shuffles,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
