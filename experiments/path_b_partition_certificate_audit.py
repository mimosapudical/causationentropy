"""Partitioned Gaussian screen-certificate audit for Path B.

This diagnostic keeps the population certificate from
path_b_screen_certificate_theory.md but studies its finite-sample power as a
function of:

1. excluded-block size, and
2. whether the certificate conditions on the full screen or only on the
   restricted forward set.

The partition is deterministic (sorted global candidate ids, fixed chunk
sizes); no data-dependent top-k completion is used.  Raw and Bonferroni
familywise decisions are both reported.
"""

import argparse
import json
import math

import networkx as nx
import numpy as np

from causationentropy.core.information.conditional_mutual_information import (
    conditional_mutual_information,
)
from causationentropy.datasets.synthetic import (
    linear_stochastic_gaussian_process,
)
from experiments.path_b_benchmark_v2 import candidate_ids_for_target
from experiments.path_b_block_certificate_audit import (
    _design,
    _true_parent_candidate_ids,
    _weakest_true_parent,
    gaussian_block_f_test,
)
from experiments.path_b_common_random_numbers import keyed_ocse
from experiments.path_b_screen_frontier_v2 import (
    endpoint_plus_conditional_rescue,
    lagged_design,
)


def _global_forward(
    X,
    Y,
    Z_init,
    working_set,
    seed,
    target,
    n_shuffles,
):
    working_set = sorted(int(idx) for idx in working_set)
    if not working_set:
        return set()
    forward_local, _ = keyed_ocse(
        X[:, working_set],
        Y,
        Z_init,
        working_set,
        seed,
        target,
        n_shuffles=n_shuffles,
    )
    return {
        int(working_set[int(local_idx)])
        for local_idx in forward_local
    }


def _chunks(values, block_size):
    values = sorted(int(x) for x in values)
    if not values:
        return []
    if block_size is None or block_size >= len(values):
        return [values]
    return [
        values[start : start + block_size]
        for start in range(0, len(values), block_size)
    ]


def _partition_certificate(
    X,
    Y,
    controls,
    excluded,
    block_size,
    alpha,
    missing_true_parents,
):
    blocks = _chunks(excluded, block_size)
    if not blocks:
        return {
            "n_blocks": 0,
            "min_p": 1.0,
            "raw_reject": False,
            "bonferroni_reject": False,
            "bonferroni_threshold": alpha,
            "missing_parent_block_min_p": None,
        }

    p_values = []
    missing_block_p = []
    missing = set(missing_true_parents)
    for block_ids in blocks:
        test = gaussian_block_f_test(
            Y,
            controls,
            X[:, block_ids],
        )
        p = float(test["p_value"])
        p_values.append(p)
        if missing.intersection(block_ids):
            missing_block_p.append(p)

    bonf = alpha / len(blocks)
    return {
        "n_blocks": len(blocks),
        "min_p": float(min(p_values)),
        "raw_reject": bool(min(p_values) < alpha),
        "bonferroni_reject": bool(min(p_values) < bonf),
        "bonferroni_threshold": float(bonf),
        "missing_parent_block_min_p": (
            float(min(missing_block_p)) if missing_block_p else None
        ),
    }


def _world(
    X,
    Y,
    Z_init,
    candidate_ids,
    screen,
    true_parent_ids,
    seed,
    target,
    alpha,
    n_shuffles,
    block_sizes,
):
    screen = set(int(idx) for idx in screen)
    excluded = sorted(set(candidate_ids) - screen)
    missing = sorted(set(true_parent_ids) - screen)

    forward = _global_forward(
        X,
        Y,
        Z_init,
        screen,
        seed,
        target,
        n_shuffles,
    )

    controls_screen = _design(
        Z_init,
        X[:, sorted(screen)] if screen else None,
    )
    controls_forward = _design(
        Z_init,
        X[:, sorted(forward)] if forward else None,
    )

    by_conditioning = {}
    for name, controls in (
        ("screen", controls_screen),
        ("forward", controls_forward),
    ):
        by_size = {}
        for block_size in block_sizes:
            key = "all" if block_size is None else str(block_size)
            by_size[key] = _partition_certificate(
                X,
                Y,
                controls,
                excluded,
                block_size,
                alpha,
                missing,
            )
        by_conditioning[name] = by_size

    return {
        "screen_parent_complete": len(missing) == 0,
        "missing_true_parents": missing,
        "screen_size": len(screen),
        "forward_size": len(forward),
        "excluded_size": len(excluded),
        "by_conditioning": by_conditioning,
    }


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
    n_shuffles,
    block_sizes,
):
    Y = Y_all[:, [target]]
    Z_init = series[:-1, [target]]
    candidate_ids = candidate_ids_for_target(feature_names, target)
    X_candidates = X[:, candidate_ids]

    screen_local, _ = endpoint_plus_conditional_rescue(
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
        seed,
        target,
        alpha,
        n_shuffles,
        block_sizes,
    )

    stress = None
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
            seed,
            target,
            alpha,
            n_shuffles,
            block_sizes,
        )
        stress["dropped_true_parent"] = int(weakest)

    return {
        "seed": seed,
        "target": target,
        "true_parent_ids": sorted(true_parent_ids),
        "base": base,
        "adversarial_weakest_parent_drop": stress,
    }


def _rate(values):
    return float(np.mean(values)) if values else None


def _median(values):
    return float(np.median(values)) if values else None


def _summarize_worlds(rows, conditioning, size_key):
    safe = [
        row["base"]["by_conditioning"][conditioning][size_key]
        for row in rows
        if row["base"]["screen_parent_complete"]
    ]
    unsafe = [
        row["base"]["by_conditioning"][conditioning][size_key]
        for row in rows
        if not row["base"]["screen_parent_complete"]
    ]
    stress = [
        row["adversarial_weakest_parent_drop"]["by_conditioning"][
            conditioning
        ][size_key]
        for row in rows
        if row["adversarial_weakest_parent_drop"] is not None
    ]

    return {
        "base_safe_n": len(safe),
        "base_unsafe_n": len(unsafe),
        "stress_n": len(stress),
        "safe_raw_false_positive_rate": _rate(
            [x["raw_reject"] for x in safe]
        ),
        "safe_bonf_false_positive_rate": _rate(
            [x["bonferroni_reject"] for x in safe]
        ),
        "unsafe_raw_detection_rate": _rate(
            [x["raw_reject"] for x in unsafe]
        ),
        "unsafe_bonf_detection_rate": _rate(
            [x["bonferroni_reject"] for x in unsafe]
        ),
        "stress_raw_detection_rate": _rate(
            [x["raw_reject"] for x in stress]
        ),
        "stress_bonf_detection_rate": _rate(
            [x["bonferroni_reject"] for x in stress]
        ),
        "safe_median_min_p": _median([x["min_p"] for x in safe]),
        "unsafe_median_min_p": _median([x["min_p"] for x in unsafe]),
        "stress_median_min_p": _median([x["min_p"] for x in stress]),
        "unsafe_missing_parent_block_median_p": _median(
            [
                x["missing_parent_block_min_p"]
                for x in unsafe
                if x["missing_parent_block_min_p"] is not None
            ]
        ),
        "stress_missing_parent_block_median_p": _median(
            [
                x["missing_parent_block_min_p"]
                for x in stress
                if x["missing_parent_block_min_p"] is not None
            ]
        ),
    }


def _summary(rows, block_sizes):
    result = {}
    for conditioning in ("screen", "forward"):
        result[conditioning] = {}
        for block_size in block_sizes:
            key = "all" if block_size is None else str(block_size)
            result[conditioning][key] = _summarize_worlds(
                rows, conditioning, key
            )
    return result


def run_audit(
    nodes=(20, 50),
    T=300,
    edge_probability=0.10,
    rho=0.7,
    seeds=1,
    retention=0.40,
    alpha=0.05,
    n_shuffles=10,
    block_sizes=(1, 2, 4, 8, 16, None),
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
            "n_shuffles": n_shuffles,
            "block_sizes": [
                "all" if size is None else size for size in block_sizes
            ],
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
                        n_shuffles,
                        block_sizes,
                    )
                )
        result["by_n"][str(n_nodes)] = {
            "summary": _summary(rows, block_sizes),
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
    parser.add_argument("--n-shuffles", type=int, default=10)
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
                n_shuffles=args.n_shuffles,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
