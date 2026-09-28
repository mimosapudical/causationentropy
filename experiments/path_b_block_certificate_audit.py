"""Population-motivated block certificate audit for screened Gaussian oCSE.

For a screen W and its excluded block R, the original oCSE theory implies

    N_I subset W  <=>  C_{R -> I | W} = 0

under the paper's faithfully Markov population assumptions.  The forward
direction is Theorem 2.2(b).  For the reverse direction, if a parent j is in R,
Theorem 2.2(c) gives C_{j -> I | W} > 0 and the CMI chain rule gives
C_{R -> I | W} >= C_{j -> I | W}.

This file audits a finite-sample Gaussian implementation of that certificate.
It is diagnostic only; it does not modify the package API.

Two worlds are measured:
1. the actual Path-B screen at the configured retention;
2. an adversarial stress world that starts from a parent-complete screen and
   removes the weakest true parent, testing whether the single block certificate
   detects the omission.

For Gaussian linear regression the block CMI and the partial F statistic encode
the same population null.  The F test provides a cheap analytical finite-sample
surrogate for the expensive per-candidate permutation checks.
"""

import argparse
import json

import networkx as nx
import numpy as np
from scipy.stats import f as f_distribution

from causationentropy.core.information.conditional_mutual_information import (
    conditional_mutual_information,
)
from causationentropy.datasets.synthetic import (
    linear_stochastic_gaussian_process,
)
from experiments.path_b_benchmark_v2 import candidate_ids_for_target
from experiments.path_b_screen_frontier_v2 import (
    endpoint_plus_conditional_rescue,
    lagged_design,
)


def _as_2d(x):
    if x is None:
        return None
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    return x


def _design(*blocks):
    kept = [_as_2d(block) for block in blocks if block is not None]
    kept = [block for block in kept if block.shape[1] > 0]
    if not kept:
        return None
    return np.hstack(kept)


def _rss_and_rank(Y, controls):
    Y = _as_2d(Y)
    n = Y.shape[0]
    if controls is None:
        A = np.ones((n, 1), dtype=float)
    else:
        A = np.column_stack([np.ones(n), _as_2d(controls)])
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    residual = Y - A @ coef
    rss = float(np.sum(residual * residual))
    rank = int(np.linalg.matrix_rank(A))
    return rss, rank


def gaussian_block_f_test(Y, controls, block):
    """Joint H0: block adds no linear-Gaussian information beyond controls."""
    block = _as_2d(block)
    if block is None or block.shape[1] == 0:
        return {
            "f": 0.0,
            "p_value": 1.0,
            "df_num": 0,
            "df_den": int(_as_2d(Y).shape[0] - 1),
            "partial_r2": 0.0,
        }

    reduced_rss, reduced_rank = _rss_and_rank(Y, controls)
    full_controls = _design(controls, block)
    full_rss, full_rank = _rss_and_rank(Y, full_controls)

    df_num = int(full_rank - reduced_rank)
    df_den = int(_as_2d(Y).shape[0] - full_rank)
    if df_num <= 0 or df_den <= 0 or full_rss <= 0:
        return {
            "f": 0.0,
            "p_value": 1.0,
            "df_num": max(df_num, 0),
            "df_den": max(df_den, 0),
            "partial_r2": 0.0,
        }

    improvement = max(0.0, reduced_rss - full_rss)
    f_stat = (improvement / df_num) / (full_rss / df_den)
    p_value = float(f_distribution.sf(f_stat, df_num, df_den))
    partial_r2 = (
        improvement / reduced_rss if reduced_rss > 0 else 0.0
    )
    return {
        "f": float(f_stat),
        "p_value": p_value,
        "df_num": df_num,
        "df_den": df_den,
        "partial_r2": float(partial_r2),
    }


def _block_cmi(X, Y, controls):
    X = _as_2d(X)
    if X is None or X.shape[1] == 0:
        return 0.0
    value = conditional_mutual_information(
        X,
        Y,
        controls,
        method="gaussian",
    )
    return float(value)


def _world(
    X,
    Y,
    Z_init,
    candidate_ids,
    screen,
    true_parent_ids,
    alpha,
):
    screen = set(int(idx) for idx in screen)
    excluded = sorted(set(candidate_ids) - screen)
    controls = _design(
        Z_init,
        X[:, sorted(screen)] if screen else None,
    )
    block = X[:, excluded] if excluded else None
    cmi = _block_cmi(block, Y, controls)
    f_test = gaussian_block_f_test(Y, controls, block)
    missing_parents = sorted(set(true_parent_ids) - screen)
    return {
        "screen_size": len(screen),
        "excluded_size": len(excluded),
        "screen_parent_complete": len(missing_parents) == 0,
        "missing_true_parents": missing_parents,
        "missing_true_parent_count": len(missing_parents),
        "block_cmi": cmi,
        "block_f": f_test["f"],
        "block_p_value": f_test["p_value"],
        "block_reject": bool(f_test["p_value"] < alpha),
        "block_df_num": f_test["df_num"],
        "block_df_den": f_test["df_den"],
        "block_partial_r2": f_test["partial_r2"],
    }


def _true_parent_candidate_ids(graph_true, feature_names, target):
    parents = set(int(source) for source in graph_true.predecessors(target))
    return {
        int(idx)
        for idx, (source, lag) in enumerate(feature_names)
        if int(source) in parents and int(source) != int(target) and int(lag) == 1
    }


def _weakest_true_parent(X, Y, Z_init, true_parent_ids):
    scored = []
    for idx in sorted(true_parent_ids):
        value = conditional_mutual_information(
            X[:, [idx]],
            Y,
            Z_init,
            method="gaussian",
        )
        scored.append((float(value), int(idx)))
    return min(scored)[1] if scored else None


def _target_row(
    X,
    Y_all,
    feature_names,
    series,
    graph_true,
    seed,
    target,
    retention,
    alpha,
):
    Y = Y_all[:, [target]]
    Z_init = series[:-1, [target]]
    candidate_ids = candidate_ids_for_target(feature_names, target)
    X_candidates = X[:, candidate_ids]

    screen_local, screen_diag = endpoint_plus_conditional_rescue(
        X_candidates,
        Y,
        np.random.default_rng(seed * 10000 + target),
        retention=retention,
        information="gaussian",
    )
    screen = {
        int(candidate_ids[int(local_idx)])
        for local_idx in screen_local
    }
    true_parent_ids = _true_parent_candidate_ids(
        graph_true, feature_names, target
    )

    base = _world(
        X,
        Y,
        Z_init,
        candidate_ids,
        screen,
        true_parent_ids,
        alpha,
    )

    stress = None
    weakest = None
    if true_parent_ids and true_parent_ids.issubset(screen):
        weakest = _weakest_true_parent(
            X, Y, Z_init, true_parent_ids
        )
        stressed_screen = set(screen) - {weakest}
        stress = _world(
            X,
            Y,
            Z_init,
            candidate_ids,
            stressed_screen,
            true_parent_ids,
            alpha,
        )
        stress["dropped_true_parent"] = int(weakest)
        singleton_controls = _design(
            Z_init,
            X[:, sorted(stressed_screen)] if stressed_screen else None,
        )
        singleton = gaussian_block_f_test(
            Y,
            singleton_controls,
            X[:, [weakest]],
        )
        stress["dropped_parent_singleton_p_value"] = singleton["p_value"]
        stress["dropped_parent_singleton_f"] = singleton["f"]

    return {
        "seed": seed,
        "target": target,
        "candidate_count": len(candidate_ids),
        "true_parent_count": len(true_parent_ids),
        "true_parent_ids": sorted(true_parent_ids),
        "screen_endpoint_size": int(screen_diag["endpoint_size"]),
        "screen_rescued": int(screen_diag["rescued"]),
        "base": base,
        "adversarial_weakest_parent_drop": stress,
    }


def _rate(values):
    return float(np.mean(values)) if values else None


def _median(values):
    return float(np.median(values)) if values else None


def _summary(rows):
    safe = [row["base"] for row in rows if row["base"]["screen_parent_complete"]]
    unsafe = [
        row["base"] for row in rows if not row["base"]["screen_parent_complete"]
    ]
    stress = [
        row["adversarial_weakest_parent_drop"]
        for row in rows
        if row["adversarial_weakest_parent_drop"] is not None
    ]
    return {
        "targets": len(rows),
        "base_parent_complete_targets": len(safe),
        "base_parent_incomplete_targets": len(unsafe),
        "base_safe_false_positive_rate": _rate(
            [row["block_reject"] for row in safe]
        ),
        "base_safe_median_p": _median(
            [row["block_p_value"] for row in safe]
        ),
        "base_safe_median_cmi": _median(
            [row["block_cmi"] for row in safe]
        ),
        "base_unsafe_detection_rate": _rate(
            [row["block_reject"] for row in unsafe]
        ),
        "base_unsafe_median_p": _median(
            [row["block_p_value"] for row in unsafe]
        ),
        "base_unsafe_median_cmi": _median(
            [row["block_cmi"] for row in unsafe]
        ),
        "adversarial_cases": len(stress),
        "adversarial_block_detection_rate": _rate(
            [row["block_reject"] for row in stress]
        ),
        "adversarial_block_median_p": _median(
            [row["block_p_value"] for row in stress]
        ),
        "adversarial_block_median_cmi": _median(
            [row["block_cmi"] for row in stress]
        ),
        "adversarial_singleton_detection_rate": _rate(
            [
                row["dropped_parent_singleton_p_value"] < 0.05
                for row in stress
            ]
        ),
        "adversarial_singleton_median_p": _median(
            [row["dropped_parent_singleton_p_value"] for row in stress]
        ),
    }


def run_audit(
    nodes=(20, 50),
    T=300,
    edge_probability=0.10,
    rho=0.7,
    seeds=1,
    retention=0.40,
    alpha=0.05,
):
    result = {
        "config": {
            "nodes": list(nodes),
            "T": T,
            "edge_probability": edge_probability,
            "rho": rho,
            "seeds": seeds,
            "retention": retention,
            "alpha": alpha,
        },
        "by_n": {},
    }
    for n_nodes in nodes:
        rows = []
        for seed in range(seeds):
            graph_true = nx.erdos_renyi_graph(
                n_nodes,
                edge_probability,
                seed=seed,
                directed=True,
            )
            data, _ = linear_stochastic_gaussian_process(
                rho=rho,
                n=n_nodes,
                T=T,
                p=edge_probability,
                seed=seed,
                G=graph_true,
            )
            X, Y_all, feature_names, series = lagged_design(
                data, max_lag=1
            )
            for target in range(n_nodes):
                rows.append(
                    _target_row(
                        X,
                        Y_all,
                        feature_names,
                        series,
                        graph_true,
                        seed,
                        target,
                        retention,
                        alpha,
                    )
                )
        result["by_n"][str(n_nodes)] = {
            "summary": _summary(rows),
            "rows": rows,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, nargs="+", default=[20, 50])
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--edge-probability", type=float, default=0.10)
    parser.add_argument("--rho", type=float, default=0.7)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--retention", type=float, default=0.40)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    print(
        json.dumps(
            run_audit(
                nodes=tuple(args.nodes),
                T=args.T,
                edge_probability=args.edge_probability,
                rho=args.rho,
                seeds=args.seeds,
                retention=args.retention,
                alpha=args.alpha,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
