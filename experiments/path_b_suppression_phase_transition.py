"""Controlled suppression benchmark for Path-B conditional rescue.

Construct a Gaussian lagged regression with two true parents X1 and X2 whose
same-time source innovations have correlation rho.  The target is

    Y_t = beta * X1_{t-1} + gamma(delta) * X2_{t-1} + noise,

with

    gamma(delta) = -(beta / rho) * (1 - delta).

For standardized sources, Cov(X1, Y) = beta * delta.  Thus:
- delta=1: X1 is strongly marginally visible;
- delta->0: X1 becomes marginally hidden;
- delta=0: X1 has exactly zero population marginal covariance with Y while its
  conditional coefficient given X2 remains beta.

This isolates the theoretical role of conditional rescue without changing the
structural parent coefficient beta.
"""

import argparse
import json
import math

import numpy as np

from causationentropy.core.discovery import (
    information_lasso_optimal_causation_entropy,
)
from causationentropy.core.information.conditional_mutual_information import (
    conditional_mutual_information,
)
from experiments.path_b_screen_frontier_v2 import (
    endpoint_plus_conditional_rescue,
    lagged_design,
)


def suppression_series(
    n_nodes=20,
    T=300,
    beta=0.2,
    rho=0.8,
    delta=0.0,
    noise_scale=0.1,
    seed=0,
):
    if n_nodes < 3:
        raise ValueError("n_nodes must be at least 3")
    if not 0 < abs(rho) < 1:
        raise ValueError("|rho| must be in (0,1)")

    rng = np.random.default_rng(seed)
    series = np.zeros((T, n_nodes), dtype=float)

    e1 = rng.standard_normal(T)
    e2 = rng.standard_normal(T)
    series[:, 1] = e1
    series[:, 2] = rho * e1 + math.sqrt(1 - rho**2) * e2

    for j in range(3, n_nodes):
        series[:, j] = rng.standard_normal(T)

    gamma = -(beta / rho) * (1.0 - delta)
    target_noise = noise_scale * rng.standard_normal(T)
    for t in range(1, T):
        series[t, 0] = (
            beta * series[t - 1, 1]
            + gamma * series[t - 1, 2]
            + target_noise[t]
        )

    return series, gamma


def _local_index(feature_names, source, lag=1):
    for idx, (src, lg) in enumerate(feature_names):
        if int(src) == int(source) and int(lg) == int(lag):
            return int(idx)
    raise KeyError((source, lag))


def _one(
    n_nodes,
    T,
    beta,
    rho,
    delta,
    noise_scale,
    retention,
    seed,
):
    series, gamma = suppression_series(
        n_nodes=n_nodes,
        T=T,
        beta=beta,
        rho=rho,
        delta=delta,
        noise_scale=noise_scale,
        seed=seed,
    )
    X, Y_all, feature_names, _ = lagged_design(series, max_lag=1)
    Y = Y_all[:, [0]]

    j_hidden = _local_index(feature_names, 1)
    j_revealer = _local_index(feature_names, 2)

    endpoint = set(
        int(j)
        for j in information_lasso_optimal_causation_entropy(
            X,
            Y,
            np.random.default_rng(seed),
            information="gaussian",
        )
    )
    rescued, diag = endpoint_plus_conditional_rescue(
        X,
        Y,
        np.random.default_rng(seed),
        retention=retention,
        information="gaussian",
    )
    rescued = set(int(j) for j in rescued)

    marginal_hidden = float(
        conditional_mutual_information(
            X[:, [j_hidden]], Y, None, method="gaussian"
        )
    )
    marginal_revealer = float(
        conditional_mutual_information(
            X[:, [j_revealer]], Y, None, method="gaussian"
        )
    )
    conditional_hidden = float(
        conditional_mutual_information(
            X[:, [j_hidden]],
            Y,
            X[:, [j_revealer]],
            method="gaussian",
        )
    )
    endpoint_conditioned_hidden = float(
        conditional_mutual_information(
            X[:, [j_hidden]],
            Y,
            X[:, sorted(endpoint)] if endpoint else None,
            method="gaussian",
        )
    )

    return {
        "seed": seed,
        "delta": delta,
        "gamma": gamma,
        "hidden_in_endpoint": j_hidden in endpoint,
        "hidden_in_rescue": j_hidden in rescued,
        "revealer_in_endpoint": j_revealer in endpoint,
        "revealer_in_rescue": j_revealer in rescued,
        "both_parents_in_endpoint": {j_hidden, j_revealer}.issubset(endpoint),
        "both_parents_in_rescue": {j_hidden, j_revealer}.issubset(rescued),
        "endpoint_size": len(endpoint),
        "rescued_count": int(diag["rescued"]),
        "screen_size": len(rescued),
        "marginal_info_hidden": marginal_hidden,
        "marginal_info_revealer": marginal_revealer,
        "conditional_info_hidden_given_revealer": conditional_hidden,
        "conditional_info_hidden_given_endpoint": endpoint_conditioned_hidden,
    }


def _summary(rows):
    def rate(key):
        return float(np.mean([bool(r[key]) for r in rows]))

    def med(key):
        return float(np.median([float(r[key]) for r in rows]))

    return {
        "seeds": len(rows),
        "hidden_endpoint_recall": rate("hidden_in_endpoint"),
        "hidden_rescue_recall": rate("hidden_in_rescue"),
        "revealer_endpoint_recall": rate("revealer_in_endpoint"),
        "revealer_rescue_recall": rate("revealer_in_rescue"),
        "both_endpoint_recall": rate("both_parents_in_endpoint"),
        "both_rescue_recall": rate("both_parents_in_rescue"),
        "median_marginal_info_hidden": med("marginal_info_hidden"),
        "median_conditional_info_hidden_given_revealer": med(
            "conditional_info_hidden_given_revealer"
        ),
        "median_conditional_info_hidden_given_endpoint": med(
            "conditional_info_hidden_given_endpoint"
        ),
        "mean_endpoint_size": float(
            np.mean([r["endpoint_size"] for r in rows])
        ),
        "mean_screen_size": float(
            np.mean([r["screen_size"] for r in rows])
        ),
    }


def run(
    n_nodes=20,
    T=300,
    beta=0.2,
    rho=0.8,
    deltas=(0.0, 0.05, 0.10, 0.25, 0.50, 1.0),
    noise_scale=0.1,
    retention=0.40,
    seeds=20,
):
    cells = []
    for delta in deltas:
        rows = [
            _one(
                n_nodes,
                T,
                beta,
                rho,
                delta,
                noise_scale,
                retention,
                seed,
            )
            for seed in range(seeds)
        ]
        cells.append(
            {
                "delta": delta,
                "population_marginal_cov_hidden": beta * delta,
                "summary": _summary(rows),
                "rows": rows,
            }
        )
    return {
        "config": {
            "n_nodes": n_nodes,
            "T": T,
            "beta": beta,
            "rho": rho,
            "deltas": list(deltas),
            "noise_scale": noise_scale,
            "retention": retention,
            "seeds": seeds,
        },
        "cells": cells,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nodes", type=int, default=20)
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--rho", type=float, default=0.8)
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.10, 0.25, 0.50, 1.0],
    )
    parser.add_argument("--noise-scale", type=float, default=0.1)
    parser.add_argument("--retention", type=float, default=0.40)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    print(
        json.dumps(
            run(
                n_nodes=args.n_nodes,
                T=args.T,
                beta=args.beta,
                rho=args.rho,
                deltas=tuple(args.deltas),
                noise_scale=args.noise_scale,
                retention=args.retention,
                seeds=args.seeds,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
