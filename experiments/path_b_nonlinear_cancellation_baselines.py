"""Controlled nonlinear cancellation benchmark for Path-B.

Construction for lagged predictors at time t:

    X1 ~ N(0,1)
    f(X1) = X1^2 - 1
    U ~ N(0, sigma_u^2)
    X2 = -a f(X1) + U
    Y_{t+1} = X2 + a f(X1) + eps = U + eps

Hence X1 is a genuine direct parent in the structural target equation but is
population-marginally independent of Y because Y depends only on U and eps.
X2 remains marginally visible.  Given X2, X1 becomes informative about Y.

Because f is even, linear forward regression also has zero population residual
correlation with X1 after fitting X2.  A nonlinear conditional-information
rescue should detect X1.
"""

import argparse
import json
import numpy as np

from experiments.path_b_budget_matched_baselines import (
    _forward_regression_screen,
    _random_screen,
)
from experiments.path_b_screen_frontier_v2 import (
    endpoint_plus_conditional_rescue,
    information_values,
    lagged_design,
)


def generate(
    n_nodes=20,
    T=300,
    a=1.0,
    sigma_u=1.0,
    noise_scale=0.2,
    seed=0,
):
    rng = np.random.default_rng(seed)
    series = np.zeros((T, n_nodes), dtype=float)

    x1 = rng.standard_normal(T)
    u = sigma_u * rng.standard_normal(T)
    fx = x1**2 - 1.0
    x2 = -a * fx + u

    series[:, 1] = x1
    series[:, 2] = x2
    for j in range(3, n_nodes):
        series[:, j] = rng.standard_normal(T)

    eps = noise_scale * rng.standard_normal(T)
    for t in range(1, T):
        series[t, 0] = (
            series[t - 1, 2]
            + a * (series[t - 1, 1] ** 2 - 1.0)
            + eps[t]
        )
    return series


def _local(feature_names, source):
    for j, (src, lag) in enumerate(feature_names):
        if int(src) == source and int(lag) == 1:
            return int(j)
    raise KeyError(source)


def _nonlinear_marginal_topk(X, Y, budget):
    vals = np.asarray(
        information_values(X, Y, information="kde"),
        dtype=float,
    )
    vals = np.nan_to_num(vals, nan=-np.inf, neginf=-np.inf, posinf=np.inf)
    order = sorted(range(X.shape[1]), key=lambda j: (-vals[j], j))
    return order[:budget], vals


def one(seed, n_nodes, T, a, sigma_u, noise_scale, retention):
    series = generate(
        n_nodes=n_nodes,
        T=T,
        a=a,
        sigma_u=sigma_u,
        noise_scale=noise_scale,
        seed=seed,
    )
    X, Y_all, feature_names, _ = lagged_design(series, max_lag=1)
    Y = Y_all[:, [0]]
    hidden = _local(feature_names, 1)
    visible = _local(feature_names, 2)

    path, _ = endpoint_plus_conditional_rescue(
        X,
        Y,
        np.random.default_rng(seed),
        retention=retention,
        information="kde",
    )
    budget = len(path)
    marginal, vals = _nonlinear_marginal_topk(X, Y, budget)
    forward = _forward_regression_screen(X, Y, budget)
    random = _random_screen(X.shape[1], budget, seed + 1777)

    screens = {
        "path_b_kde": path,
        "marginal_kde_topk": marginal,
        "forward_regression": forward,
        "random": random,
    }
    out = {}
    for name, screen in screens.items():
        s = set(int(j) for j in screen)
        out[name] = {
            "hidden": hidden in s,
            "visible": visible in s,
            "both": {hidden, visible}.issubset(s),
            "retention": len(s) / X.shape[1],
        }

    # Diagnostics: the empirical marginal KDE score of the hidden parent and
    # the visible parent. Conditional rescue behavior is measured by inclusion.
    out["diagnostics"] = {
        "hidden_marginal_kde_info": float(vals[hidden]),
        "visible_marginal_kde_info": float(vals[visible]),
        "sample_corr_hidden_y": float(
            np.corrcoef(X[:, hidden], Y.reshape(-1))[0, 1]
        ),
        "sample_corr_visible_y": float(
            np.corrcoef(X[:, visible], Y.reshape(-1))[0, 1]
        ),
        "budget": budget,
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-nodes", type=int, default=20)
    p.add_argument("--T", type=int, default=300)
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--sigma-u", type=float, default=1.0)
    p.add_argument("--noise-scale", type=float, default=0.2)
    p.add_argument("--retentions", nargs="+", type=float, default=[0.10, 0.20, 0.40])
    p.add_argument("--seeds", type=int, default=20)
    args = p.parse_args()

    cells = []
    for retention in args.retentions:
        runs = [
            one(
                seed,
                args.n_nodes,
                args.T,
                args.a,
                args.sigma_u,
                args.noise_scale,
                retention,
            )
            for seed in range(args.seeds)
        ]
        methods = (
            "path_b_kde",
            "marginal_kde_topk",
            "forward_regression",
            "random",
        )
        summary = {}
        for m in methods:
            summary[m] = {
                "hidden_recall": float(np.mean([r[m]["hidden"] for r in runs])),
                "visible_recall": float(np.mean([r[m]["visible"] for r in runs])),
                "both_recall": float(np.mean([r[m]["both"] for r in runs])),
                "mean_retention": float(np.mean([r[m]["retention"] for r in runs])),
            }
        diagnostics = {
            "median_hidden_marginal_kde_info": float(
                np.median([r["diagnostics"]["hidden_marginal_kde_info"] for r in runs])
            ),
            "median_visible_marginal_kde_info": float(
                np.median([r["diagnostics"]["visible_marginal_kde_info"] for r in runs])
            ),
            "median_abs_corr_hidden_y": float(
                np.median([abs(r["diagnostics"]["sample_corr_hidden_y"]) for r in runs])
            ),
            "median_abs_corr_visible_y": float(
                np.median([abs(r["diagnostics"]["sample_corr_visible_y"]) for r in runs])
            ),
            "mean_budget": float(np.mean([r["diagnostics"]["budget"] for r in runs])),
        }
        cells.append({
            "retention_target": retention,
            "summary": summary,
            "diagnostics": diagnostics,
        })

    print(json.dumps({"config": vars(args), "cells": cells}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
