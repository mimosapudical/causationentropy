"""Beta-min and tuning audit for the Path-B parent-sure screening hypothesis.

The goal is not to reproduce a full-oCSE finite-sample path.  Instead this
experiment controls the minimum raw edge magnitude in a stable Gaussian VAR
and measures:

- Path-A parent recall and parent-complete target rate;
- Path-B (Path A + one conditional rescue pass) parent recall;
- the BIC/CV penalty selected by the production Path-A implementation relative
  to the classical Lasso noise-dominance threshold;
- marginal-information visibility of true parents;
- a support-Gram margin diagnostic for the weighted design.

The support-Gram quantity is only an optimistic diagnostic.  It is not the
restricted-eigenvalue constant required by the theorem in
path_b_sure_screening_theory.md.
"""

import argparse
import json
import math
import warnings

import networkx as nx
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV, LassoLarsIC

from causationentropy.core.discovery import (
    information_lasso_optimal_causation_entropy,
)
from experiments.path_b_benchmark_v2 import candidate_ids_for_target
from experiments.path_b_screen_frontier_v2 import (
    endpoint_plus_conditional_rescue,
    information_values,
    lagged_design,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def floored_linear_gaussian_var(
    n,
    T,
    p,
    rho,
    weight_floor,
    seed,
    epsilon=0.1,
    burnin=200,
):
    """Generate the repository VAR with a controlled raw edge-magnitude floor."""
    rng = np.random.default_rng(seed)
    graph = nx.erdos_renyi_graph(n, p, seed=seed, directed=True)
    mask = nx.to_numpy_array(graph).T

    raw = 2.0 * rng.random((n, n)) - 1.0
    if weight_floor > 0:
        raw = np.sign(raw) * (
            weight_floor + (1.0 - weight_floor) * np.abs(raw)
        )
    A = mask * raw

    eigvals = np.linalg.eigvals(A)
    spectral_radius = float(np.max(np.abs(eigvals)))
    if spectral_radius > 1e-12:
        A = A / spectral_radius
    A = A * rho

    total = int(T + burnin)
    series = np.zeros((total, n), dtype=float)
    series[0, :] = epsilon * rng.standard_normal(n)
    for t in range(1, total):
        series[t, :] = (
            A @ series[t - 1, :]
            + epsilon * rng.standard_normal(n)
        )
    return series[burnin:, :], A, graph


def _fit_information_lasso_with_diagnostics(X, Y):
    values = information_values(X, Y, information="gaussian")
    total = float(values.sum())
    if total <= 0:
        return {
            "selected": set(),
            "values": values,
            "weights": np.zeros_like(values),
            "alpha": None,
            "solver": "none",
        }

    weights = values / total
    X_weighted = X * weights.reshape(1, -1)

    if X.shape[0] > X.shape[1] + 1:
        model = LassoLarsIC(criterion="bic", max_iter=100).fit(
            X_weighted,
            Y.reshape(-1),
        )
        solver = "bic"
    else:
        folds = min(10, X.shape[0])
        model = LassoCV(cv=folds, max_iter=1000).fit(
            X_weighted,
            Y.reshape(-1),
        )
        solver = "cv"

    return {
        "selected": set(np.flatnonzero(model.coef_ != 0).tolist()),
        "values": values,
        "weights": weights,
        "alpha": float(model.alpha_),
        "solver": solver,
    }


def _target_row(
    X,
    Y_all,
    feature_names,
    A,
    target,
    retention,
    seed,
):
    candidate_ids = candidate_ids_for_target(feature_names, target)
    Xc = X[:, candidate_ids]
    Y = Y_all[:, [target]]

    source_ids = [int(feature_names[idx][0]) for idx in candidate_ids]
    beta = np.asarray([A[target, source] for source in source_ids])
    truth = set(np.flatnonzero(np.abs(beta) > 0).tolist())

    fit = _fit_information_lasso_with_diagnostics(Xc, Y)
    endpoint = fit["selected"]

    rescued, rescue_diag = endpoint_plus_conditional_rescue(
        Xc,
        Y,
        np.random.default_rng(seed * 10000 + target),
        retention=retention,
        information="gaussian",
    )
    rescued = set(int(idx) for idx in rescued)

    eps = Y.reshape(-1) - Xc @ beta
    weights = fit["weights"]
    Xw = Xc * weights.reshape(1, -1)
    Xwc = Xw - Xw.mean(axis=0, keepdims=True)
    epsc = eps - eps.mean()
    noise_threshold = float(
        2.0 * np.max(np.abs(Xwc.T @ epsc / Xwc.shape[0]))
    ) if Xwc.shape[1] else 0.0

    alpha = fit["alpha"]
    lambda_ratio = (
        float(alpha / noise_threshold)
        if alpha is not None and noise_threshold > 0
        else None
    )

    parent_weights = (
        np.asarray([weights[idx] for idx in sorted(truth)])
        if truth
        else np.asarray([], dtype=float)
    )
    parent_values = (
        np.asarray([fit["values"][idx] for idx in sorted(truth)])
        if truth
        else np.asarray([], dtype=float)
    )
    zero_parent_weight = bool(
        truth and np.any(parent_weights <= 0)
    )

    support_gram_min_eig = None
    theta_min = None
    optimistic_margin_ratio = None
    kkt_inverse_gram_inf = None
    kkt_error_bound = None
    kkt_certificate_ratio = None
    kkt_parent_sure = None
    if truth and not zero_parent_weight:
        S = sorted(truth)
        gram = Xwc[:, S].T @ Xwc[:, S] / Xwc.shape[0]
        support_gram_min_eig = float(
            max(0.0, np.min(np.linalg.eigvalsh(gram)))
        )
        theta_min = float(
            np.min(np.abs(beta[S]) / weights[S])
        )
        if alpha is not None and alpha > 0:
            optimistic_margin_ratio = float(
                theta_min
                * support_gram_min_eig
                / (3.0 * alpha * math.sqrt(len(S)))
            )

        full_gram = Xwc.T @ Xwc / Xwc.shape[0]
        if (
            Xwc.shape[0] > Xwc.shape[1]
            and np.linalg.matrix_rank(full_gram) == full_gram.shape[0]
        ):
            gram_inv = np.linalg.inv(full_gram)
            kkt_inverse_gram_inf = float(
                np.max(np.sum(np.abs(gram_inv), axis=1))
            )
            score_noise = noise_threshold / 2.0
            if alpha is not None:
                kkt_error_bound = float(
                    kkt_inverse_gram_inf * (score_noise + alpha)
                )
                if kkt_error_bound > 0:
                    kkt_certificate_ratio = float(
                        theta_min / kkt_error_bound
                    )
                    kkt_parent_sure = bool(
                        kkt_certificate_ratio > 1.0
                    )

    return {
        "target": int(target),
        "solver": fit["solver"],
        "parent_count": len(truth),
        "endpoint_size": len(endpoint),
        "rescue_size": len(rescued),
        "rescue_added": int(rescue_diag["rescued"]),
        "endpoint_parent_tp": len(truth & endpoint),
        "rescue_parent_tp": len(truth & rescued),
        "endpoint_parent_complete": truth.issubset(endpoint),
        "rescue_parent_complete": truth.issubset(rescued),
        "endpoint_retention": len(endpoint) / len(candidate_ids),
        "rescue_retention": len(rescued) / len(candidate_ids),
        "beta_min_target": (
            float(np.min(np.abs(beta[list(truth)])))
            if truth
            else None
        ),
        "beta_median_target": (
            float(np.median(np.abs(beta[list(truth)])))
            if truth
            else None
        ),
        "parent_information_min": (
            float(np.min(parent_values)) if truth else None
        ),
        "parent_weight_min": (
            float(np.min(parent_weights)) if truth else None
        ),
        "zero_parent_weight": zero_parent_weight,
        "chosen_alpha": alpha,
        "noise_dominance_threshold": noise_threshold,
        "lambda_ratio": lambda_ratio,
        "lambda_noise_condition": (
            bool(lambda_ratio >= 1.0)
            if lambda_ratio is not None
            else None
        ),
        "support_gram_min_eig": support_gram_min_eig,
        "theta_min": theta_min,
        "optimistic_margin_ratio": optimistic_margin_ratio,
        "kkt_inverse_gram_inf": kkt_inverse_gram_inf,
        "kkt_error_bound": kkt_error_bound,
        "kkt_certificate_ratio": kkt_certificate_ratio,
        "kkt_parent_sure": kkt_parent_sure,
    }


def _safe_mean(values):
    values = [x for x in values if x is not None]
    return float(np.mean(values)) if values else None


def _safe_median(values):
    values = [x for x in values if x is not None]
    return float(np.median(values)) if values else None


def _summarize(rows):
    parent_rows = [row for row in rows if row["parent_count"] > 0]
    parent_edges = sum(row["parent_count"] for row in parent_rows)
    endpoint_tp = sum(row["endpoint_parent_tp"] for row in parent_rows)
    rescue_tp = sum(row["rescue_parent_tp"] for row in parent_rows)
    lambda_rows = [
        row for row in rows if row["lambda_noise_condition"] is not None
    ]

    return {
        "targets": len(rows),
        "targets_with_parents": len(parent_rows),
        "true_parent_edges": int(parent_edges),
        "endpoint_parent_recall": (
            endpoint_tp / parent_edges if parent_edges else 1.0
        ),
        "rescue_parent_recall": (
            rescue_tp / parent_edges if parent_edges else 1.0
        ),
        "endpoint_parent_complete_target_rate": (
            float(np.mean([
                row["endpoint_parent_complete"] for row in parent_rows
            ]))
            if parent_rows
            else 1.0
        ),
        "rescue_parent_complete_target_rate": (
            float(np.mean([
                row["rescue_parent_complete"] for row in parent_rows
            ]))
            if parent_rows
            else 1.0
        ),
        "mean_endpoint_retention": _safe_mean(
            [row["endpoint_retention"] for row in rows]
        ),
        "mean_rescue_retention": _safe_mean(
            [row["rescue_retention"] for row in rows]
        ),
        "zero_parent_weight_target_rate": (
            float(np.mean([
                row["zero_parent_weight"] for row in parent_rows
            ]))
            if parent_rows
            else 0.0
        ),
        "median_parent_information_min": _safe_median(
            [row["parent_information_min"] for row in parent_rows]
        ),
        "median_beta_min_target": _safe_median(
            [row["beta_min_target"] for row in parent_rows]
        ),
        "lambda_noise_condition_rate": (
            float(np.mean([
                row["lambda_noise_condition"] for row in lambda_rows
            ]))
            if lambda_rows
            else None
        ),
        "median_lambda_ratio": _safe_median(
            [row["lambda_ratio"] for row in lambda_rows]
        ),
        "median_support_gram_min_eig": _safe_median(
            [row["support_gram_min_eig"] for row in parent_rows]
        ),
        "median_optimistic_margin_ratio": _safe_median(
            [row["optimistic_margin_ratio"] for row in parent_rows]
        ),
        "optimistic_margin_above_one_rate": (
            float(np.mean([
                row["optimistic_margin_ratio"] > 1.0
                for row in parent_rows
                if row["optimistic_margin_ratio"] is not None
            ]))
            if any(
                row["optimistic_margin_ratio"] is not None
                for row in parent_rows
            )
            else None
        ),
        "kkt_certificate_rate": (
            float(np.mean([
                row["kkt_parent_sure"]
                for row in parent_rows
                if row["kkt_parent_sure"] is not None
            ]))
            if any(
                row["kkt_parent_sure"] is not None
                for row in parent_rows
            )
            else None
        ),
        "median_kkt_certificate_ratio": _safe_median(
            [row["kkt_certificate_ratio"] for row in parent_rows]
        ),
        "kkt_false_certificate_count": int(
            sum(
                row["kkt_parent_sure"]
                and not row["endpoint_parent_complete"]
                for row in parent_rows
                if row["kkt_parent_sure"] is not None
            )
        ),
    }


def run_audit(
    nodes=(20, 50),
    sample_sizes=(150, 300),
    weight_floors=(0.0, 0.25, 0.5),
    seeds=1,
    edge_probability=0.10,
    rho=0.7,
    retention=0.40,
    epsilon=0.1,
    burnin=200,
):
    result = {
        "config": {
            "nodes": list(nodes),
            "sample_sizes": list(sample_sizes),
            "weight_floors": list(weight_floors),
            "seeds": seeds,
            "edge_probability": edge_probability,
            "rho": rho,
            "retention": retention,
            "epsilon": epsilon,
            "burnin": burnin,
        },
        "cells": [],
    }

    for n_nodes in nodes:
        for T in sample_sizes:
            for floor in weight_floors:
                rows = []
                realized_edge_magnitudes = []
                for seed in range(seeds):
                    data, A, _graph = floored_linear_gaussian_var(
                        n=n_nodes,
                        T=T,
                        p=edge_probability,
                        rho=rho,
                        weight_floor=floor,
                        seed=seed,
                        epsilon=epsilon,
                        burnin=burnin,
                    )
                    X, Y_all, feature_names, _series = lagged_design(
                        data, max_lag=1
                    )
                    realized_edge_magnitudes.extend(
                        np.abs(A[np.abs(A) > 0]).tolist()
                    )
                    for target in range(n_nodes):
                        row = _target_row(
                            X,
                            Y_all,
                            feature_names,
                            A,
                            target,
                            retention,
                            seed,
                        )
                        row["seed"] = seed
                        rows.append(row)

                result["cells"].append(
                    {
                        "n_nodes": n_nodes,
                        "T": T,
                        "weight_floor": floor,
                        "realized_beta_min": (
                            float(np.min(realized_edge_magnitudes))
                            if realized_edge_magnitudes
                            else None
                        ),
                        "realized_beta_median": (
                            float(np.median(realized_edge_magnitudes))
                            if realized_edge_magnitudes
                            else None
                        ),
                        "summary": _summarize(rows),
                        "rows": rows,
                    }
                )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", nargs="+", type=int, default=[20, 50])
    parser.add_argument(
        "--sample-sizes", nargs="+", type=int, default=[150, 300]
    )
    parser.add_argument(
        "--weight-floors",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5],
    )
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--edge-probability", type=float, default=0.10)
    parser.add_argument("--rho", type=float, default=0.7)
    parser.add_argument("--retention", type=float, default=0.40)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--burnin", type=int, default=200)
    args = parser.parse_args()

    print(
        json.dumps(
            run_audit(
                nodes=tuple(args.nodes),
                sample_sizes=tuple(args.sample_sizes),
                weight_floors=tuple(args.weight_floors),
                seeds=args.seeds,
                edge_probability=args.edge_probability,
                rho=args.rho,
                retention=args.retention,
                epsilon=args.epsilon,
                burnin=args.burnin,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
