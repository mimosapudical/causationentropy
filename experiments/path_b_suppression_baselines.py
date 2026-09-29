"""Budget-matched suppression comparison for Path-B.

Compares Path-B, marginal top-k, greedy forward regression, and random screens
at exactly the same realized per-target budget on the controlled cancellation
construction.
"""

import argparse
import json
import numpy as np

from experiments.path_b_budget_matched_baselines import (
    _forward_regression_screen,
    _random_screen,
    _top_marginal_screen,
)
from experiments.path_b_screen_frontier_v2 import (
    endpoint_plus_conditional_rescue,
    lagged_design,
)
from experiments.path_b_suppression_phase_transition import (
    suppression_series,
    _local_index,
)


def one(seed, n_nodes, T, beta, gamma, rho, noise_scale, retention):
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
    hidden = _local_index(feature_names, 1)
    revealer = _local_index(feature_names, 2)

    path, _ = endpoint_plus_conditional_rescue(
        X,
        Y,
        np.random.default_rng(seed),
        retention=retention,
        information="gaussian",
    )
    budget = len(path)
    screens = {
        "path_b": [int(j) for j in path],
        "marginal_topk": _top_marginal_screen(X, Y, budget),
        "forward_regression": _forward_regression_screen(X, Y, budget),
        "random": _random_screen(X.shape[1], budget, seed + 9001),
    }
    out = {}
    for name, vals in screens.items():
        s = set(vals)
        out[name] = {
            "hidden": hidden in s,
            "revealer": revealer in s,
            "both": {hidden, revealer}.issubset(s),
            "budget": budget,
            "retention": budget / X.shape[1],
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-nodes", type=int, default=50)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--beta", type=float, default=0.12)
    p.add_argument("--gamma", type=float, default=-0.15)
    p.add_argument("--rhos", nargs="+", type=float, default=[0.7, 0.75, 0.8])
    p.add_argument("--noise-scale", type=float, default=0.15)
    p.add_argument("--retentions", nargs="+", type=float, default=[0.10, 0.20, 0.40])
    p.add_argument("--seeds", type=int, default=20)
    args = p.parse_args()

    cells = []
    for retention in args.retentions:
        for rho in args.rhos:
            runs = [
                one(
                    seed,
                    args.n_nodes,
                    args.T,
                    args.beta,
                    args.gamma,
                    rho,
                    args.noise_scale,
                    retention,
                )
                for seed in range(args.seeds)
            ]
            summary = {}
            for method in runs[0]:
                summary[method] = {
                    "hidden_recall": float(np.mean([r[method]["hidden"] for r in runs])),
                    "revealer_recall": float(np.mean([r[method]["revealer"] for r in runs])),
                    "both_recall": float(np.mean([r[method]["both"] for r in runs])),
                    "mean_retention": float(np.mean([r[method]["retention"] for r in runs])),
                }
            cells.append({
                "retention_target": retention,
                "rho": rho,
                "population_marginal_cov_hidden": args.beta + args.gamma * rho,
                "summary": summary,
            })
    print(json.dumps({"config": vars(args), "cells": cells}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
