"""Controlled suppression benchmark for Path-B conditional rescue.

Two true lagged parents keep fixed nonzero coefficients throughout:

    Y_t = beta * X1_{t-1} + gamma * X2_{t-1} + noise.

The standardized source innovations have correlation rho.  Therefore

    Cov(X1, Y) = beta + gamma * rho.

With beta > 0 and gamma < 0, increasing rho toward

    rho_cancel = -beta / gamma

drives the marginal association of X1 to zero while both structural parent
coefficients remain unchanged.  Conditional on X2, X1 retains its structural
coefficient beta.

This cleanly separates marginal visibility from conditional visibility and
tests the role of Path-B's one-pass conditional rescue.
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
    n_nodes=50,
    T=200,
    beta=0.12,
    gamma=-0.15,
    rho=0.8,
    noise_scale=0.15,
    seed=0,
):
    if n_nodes < 3:
        raise ValueError("n_nodes must be at least 3")
    if not 0 <= abs(rho) < 1:
        raise ValueError("|rho| must be in [0,1)")

    rng = np.random.default_rng(seed)
    series = np.zeros((T, n_nodes), dtype=float)

    e1 = rng.standard_normal(T)
    e2 = rng.standard_normal(T)
    series[:, 1] = e1
    series[:, 2] = rho * e1 + math.sqrt(1 - rho**2) * e2

    for j in range(3, n_nodes):
        series[:, j] = rng.standard_normal(T)

    target_noise = noise_scale * rng.standard_normal(T)
    for t in range(1, T):
        series[t, 0] = (
            beta * series[t - 1, 1]
            + gamma * series[t - 1, 2]
            + target_noise[t]
        )

    return series


def _local_index(feature_names, source, lag=1):
    for idx, (src, lg) in enumerate(feature_names):
        if int(src) == int(source) and int(lg) == int(lag):
            return int(idx)
    raise KeyError((source, lag))


def _one(
    n_nodes,
    T,
    beta,
    gamma,
    rho,
    noise_scale,
    retention,
    seed,
):
    series = suppression_series(
        n_nodes=n_nodes,
        T=T,
        beta=beta,
        gamma=gamma,
        rho=rho,
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

    # The implemented rescue conditions on the whole endpoint.  This score is
    # meaningful only when the hidden parent is actually absent from endpoint.
    if j_hidden not in endpoint:
        Z = X[:, sorted(endpoint)] if endpoint else None
        endpoint_conditioned_hidden = float(
            conditional_mutual_information(
                X[:, [j_hidden]],
                Y,
                Z,
                method="gaussian",
            )
        )
    else:
        endpoint_conditioned_hidden = None

    return {
        "seed": seed,
        "rho": rho,
        "hidden_in_endpoint": j_hidden in endpoint,
        "hidden_in_rescue": j_hidden in rescued,
        "revealer_in_endpoint": j_revealer in endpoint,
        "revealer_in_rescue": j_revealer in rescued,
        "both_parents_in_endpoint": {j_hidden, j_revealer}.issubset(endpoint),
        "both_parents_in_rescue": {j_hidden, j_revealer}.issubset(rescued),
        "rescue_hidden_given_endpoint_miss": (
            (j_hidden in rescued) if j_hidden not in endpoint else None
        ),
        "revealer_present_when_hidden_missed": (
            (j_revealer in endpoint) if j_hidden not in endpoint else None
        ),
        "endpoint_size": len(endpoint),
        "rescued_count": int(diag["rescued"]),
        "screen_size": len(rescued),
        "marginal_info_hidden": marginal_hidden,
        "marginal_info_revealer": marginal_revealer,
        "conditional_info_hidden_given_revealer": conditional_hidden,
        "conditional_info_hidden_given_endpoint": endpoint_conditioned_hidden,
    }


def _summary(rows):
    def rate(key, conditional=False):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean([bool(v) for v in vals])) if vals else None

    def med(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.median(vals)) if vals else None

    misses = [r for r in rows if not r["hidden_in_endpoint"]]
    return {
        "seeds": len(rows),
        "hidden_endpoint_recall": rate("hidden_in_endpoint"),
        "hidden_rescue_recall": rate("hidden_in_rescue"),
        "revealer_endpoint_recall": rate("revealer_in_endpoint"),
        "revealer_rescue_recall": rate("revealer_in_rescue"),
        "both_endpoint_recall": rate("both_parents_in_endpoint"),
        "both_rescue_recall": rate("both_parents_in_rescue"),
        "hidden_endpoint_miss_count": len(misses),
        "rescue_success_given_hidden_endpoint_miss": rate(
            "rescue_hidden_given_endpoint_miss"
        ),
        "revealer_present_given_hidden_endpoint_miss": rate(
            "revealer_present_when_hidden_missed"
        ),
        "median_marginal_info_hidden": med("marginal_info_hidden"),
        "median_marginal_info_revealer": med("marginal_info_revealer"),
        "median_conditional_info_hidden_given_revealer": med(
            "conditional_info_hidden_given_revealer"
        ),
        "median_conditional_info_hidden_given_endpoint_on_misses": med(
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
    n_nodes=50,
    T=200,
    beta=0.12,
    gamma=-0.15,
    rhos=(0.0, 0.2, 0.4, 0.6, 0.7, 0.75, 0.8),
    noise_scale=0.15,
    retention=0.40,
    seeds=20,
):
    cells = []
    for rho in rhos:
        rows = [
            _one(
                n_nodes,
                T,
                beta,
                gamma,
                rho,
                noise_scale,
                retention,
                seed,
            )
            for seed in range(seeds)
        ]
        cells.append(
            {
                "rho": rho,
                "population_marginal_cov_hidden": beta + gamma * rho,
                "population_marginal_cov_revealer": beta * rho + gamma,
                "summary": _summary(rows),
                "rows": rows,
            }
        )
    return {
        "config": {
            "n_nodes": n_nodes,
            "T": T,
            "beta": beta,
            "gamma": gamma,
            "rho_cancel": -beta / gamma,
            "rhos": list(rhos),
            "noise_scale": noise_scale,
            "retention": retention,
            "seeds": seeds,
        },
        "cells": cells,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nodes", type=int, default=50)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--beta", type=float, default=0.12)
    parser.add_argument("--gamma", type=float, default=-0.15)
    parser.add_argument(
        "--rhos",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4, 0.6, 0.7, 0.75, 0.8],
    )
    parser.add_argument("--noise-scale", type=float, default=0.15)
    parser.add_argument("--retention", type=float, default=0.40)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    print(
        json.dumps(
            run(
                n_nodes=args.n_nodes,
                T=args.T,
                beta=args.beta,
                gamma=args.gamma,
                rhos=tuple(args.rhos),
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
