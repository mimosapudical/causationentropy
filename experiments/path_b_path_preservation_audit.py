"""Diagnose what a Path-B screen must preserve to reproduce full oCSE.

This is an oracle diagnostic, not a deployable screening rule.  For exactly the
same Information-LASSO + conditional-rescue screen, it separates three possible
sources of graph mismatch:

1. sequential permutation-RNG stream shifts after candidate restriction;
2. missing variables from the full final support; and
3. missing temporary variables accepted during the full forward path.

The keyed-RNG worlds derive each permutation stream from
(target, stage, candidate, conditioning set), so common statistical tests see
the same null permutations regardless of how many other candidates were
screened out.
"""

import argparse
import json

import networkx as nx
import numpy as np

from causationentropy.core.discovery import backward, standard_forward
from causationentropy.datasets.synthetic import (
    linear_stochastic_gaussian_process,
)
from experiments.path_b_benchmark_v2 import candidate_ids_for_target
from experiments.path_b_common_random_numbers import keyed_ocse
from experiments.path_b_screen_frontier_v2 import (
    endpoint_plus_conditional_rescue,
    lagged_design,
)


def _globalize(local_indices, global_ids):
    return {int(global_ids[int(idx)]) for idx in local_indices}


def _coverage(reference, selected):
    reference = set(reference)
    selected = set(selected)
    return len(reference & selected) / len(reference) if reference else 1.0


def _precision(reference, selected):
    reference = set(reference)
    selected = set(selected)
    return len(reference & selected) / len(selected) if selected else 1.0


def _sequential_ocse(
    X_full,
    Y,
    Z_init,
    candidate_ids,
    seed,
    target,
    alpha,
    n_shuffles,
):
    candidate_ids = list(candidate_ids)
    X = X_full[:, candidate_ids]
    rng = np.random.default_rng(seed * 10000 + target)
    forward_local = standard_forward(
        X,
        Y,
        Z_init,
        rng,
        alpha=alpha,
        n_shuffles=n_shuffles,
        information="gaussian",
    )
    final_local = backward(
        X,
        Y,
        forward_local,
        rng,
        alpha=alpha,
        n_shuffles=n_shuffles,
        information="gaussian",
        Z_init=Z_init,
    )
    return (
        _globalize(forward_local, candidate_ids),
        _globalize(final_local, candidate_ids),
    )


def _keyed_ocse_global(
    X_full,
    Y,
    Z_init,
    candidate_ids,
    seed,
    target,
    alpha,
    n_shuffles,
):
    candidate_ids = list(candidate_ids)
    X = X_full[:, candidate_ids]
    forward_local, final_local = keyed_ocse(
        X,
        Y,
        Z_init,
        candidate_ids,
        seed,
        target,
        alpha=alpha,
        n_shuffles=n_shuffles,
    )
    return (
        _globalize(forward_local, candidate_ids),
        _globalize(final_local, candidate_ids),
    )


def _target_row(
    X,
    Y_all,
    feature_names,
    series,
    seed,
    target,
    retention,
    max_lag,
    alpha,
    n_shuffles,
):
    Y = Y_all[:, [target]]
    Z_init = np.column_stack(
        [
            series[max_lag - lag : series.shape[0] - lag, target]
            for lag in range(1, max_lag + 1)
        ]
    )
    candidate_ids = candidate_ids_for_target(feature_names, target)
    X_candidates = X[:, candidate_ids]

    screen_local, screen_diag = endpoint_plus_conditional_rescue(
        X_candidates,
        Y,
        np.random.default_rng(seed * 10000 + target),
        retention=retention,
        information="gaussian",
    )
    base_screen = _globalize(screen_local, candidate_ids)

    full_seq_forward, full_seq_final = _sequential_ocse(
        X,
        Y,
        Z_init,
        candidate_ids,
        seed,
        target,
        alpha,
        n_shuffles,
    )
    _, seq_base_final = _sequential_ocse(
        X,
        Y,
        Z_init,
        sorted(base_screen),
        seed,
        target,
        alpha,
        n_shuffles,
    )

    full_keyed_forward, full_keyed_final = _keyed_ocse_global(
        X,
        Y,
        Z_init,
        candidate_ids,
        seed,
        target,
        alpha,
        n_shuffles,
    )
    _, keyed_base_final = _keyed_ocse_global(
        X,
        Y,
        Z_init,
        sorted(base_screen),
        seed,
        target,
        alpha,
        n_shuffles,
    )

    plus_final = sorted(base_screen | full_keyed_final)
    _, keyed_plus_final_final = _keyed_ocse_global(
        X,
        Y,
        Z_init,
        plus_final,
        seed,
        target,
        alpha,
        n_shuffles,
    )

    plus_forward = sorted(base_screen | full_keyed_forward)
    _, keyed_plus_forward_final = _keyed_ocse_global(
        X,
        Y,
        Z_init,
        plus_forward,
        seed,
        target,
        alpha,
        n_shuffles,
    )

    denom = len(candidate_ids)
    return {
        "seed": seed,
        "target": target,
        "candidate_count": denom,
        "screen_endpoint_size": int(screen_diag["endpoint_size"]),
        "screen_rescued": int(screen_diag["rescued"]),
        "base_retention": len(base_screen) / denom,
        "plus_final_retention": len(plus_final) / denom,
        "plus_forward_retention": len(plus_forward) / denom,
        "full_seq_forward": sorted(full_seq_forward),
        "full_seq_final": sorted(full_seq_final),
        "full_keyed_forward": sorted(full_keyed_forward),
        "full_keyed_final": sorted(full_keyed_final),
        "base_screen": sorted(base_screen),
        "seq_base_final": sorted(seq_base_final),
        "keyed_base_final": sorted(keyed_base_final),
        "keyed_plus_final_final": sorted(keyed_plus_final_final),
        "keyed_plus_forward_final": sorted(keyed_plus_forward_final),
        "base_forward_closure_recall": _coverage(
            full_keyed_forward, base_screen
        ),
        "base_final_support_recall": _coverage(
            full_keyed_final, base_screen
        ),
        "full_seq_vs_keyed_final_recall": _coverage(
            full_keyed_final, full_seq_final
        ),
        "full_seq_vs_keyed_final_precision": _precision(
            full_keyed_final, full_seq_final
        ),
        "full_seq_vs_keyed_final_exact": (
            full_seq_final == full_keyed_final
        ),
        "seq_base_final_recall": _coverage(
            full_seq_final, seq_base_final
        ),
        "seq_base_final_precision": _precision(
            full_seq_final, seq_base_final
        ),
        "seq_base_final_exact": seq_base_final == full_seq_final,
        "keyed_base_final_recall": _coverage(
            full_keyed_final, keyed_base_final
        ),
        "keyed_base_final_precision": _precision(
            full_keyed_final, keyed_base_final
        ),
        "keyed_base_final_exact": keyed_base_final == full_keyed_final,
        "keyed_plus_final_recall": _coverage(
            full_keyed_final, keyed_plus_final_final
        ),
        "keyed_plus_final_precision": _precision(
            full_keyed_final, keyed_plus_final_final
        ),
        "keyed_plus_final_exact": (
            keyed_plus_final_final == full_keyed_final
        ),
        "keyed_plus_forward_recall": _coverage(
            full_keyed_final, keyed_plus_forward_final
        ),
        "keyed_plus_forward_precision": _precision(
            full_keyed_final, keyed_plus_forward_final
        ),
        "keyed_plus_forward_exact": (
            keyed_plus_forward_final == full_keyed_final
        ),
    }


def _micro_set_metric(rows, reference_key, selected_key, metric):
    numer = 0
    denom = 0
    for row in rows:
        reference = set(row[reference_key])
        selected = set(row[selected_key])
        numer += len(reference & selected)
        denom += len(reference if metric == "recall" else selected)
    return numer / denom if denom else 1.0


def _exact_rate(rows, key):
    return float(np.mean([bool(row[key]) for row in rows])) if rows else 1.0


def _summary(rows):
    ordinary_recall = _micro_set_metric(
        rows, "full_seq_final", "seq_base_final", "recall"
    )
    keyed_recall = _micro_set_metric(
        rows, "full_keyed_final", "keyed_base_final", "recall"
    )
    plus_final_recall = _micro_set_metric(
        rows, "full_keyed_final", "keyed_plus_final_final", "recall"
    )
    plus_forward_recall = _micro_set_metric(
        rows, "full_keyed_final", "keyed_plus_forward_final", "recall"
    )

    return {
        "targets": len(rows),
        "mean_base_retention": float(
            np.mean([row["base_retention"] for row in rows])
        ),
        "mean_plus_final_retention": float(
            np.mean([row["plus_final_retention"] for row in rows])
        ),
        "mean_plus_forward_retention": float(
            np.mean([row["plus_forward_retention"] for row in rows])
        ),
        "base_forward_closure_recall": _micro_set_metric(
            rows, "full_keyed_forward", "base_screen", "recall"
        ),
        "base_final_support_screen_recall": _micro_set_metric(
            rows, "full_keyed_final", "base_screen", "recall"
        ),
        "full_seq_vs_keyed_final_recall": _micro_set_metric(
            rows, "full_keyed_final", "full_seq_final", "recall"
        ),
        "full_seq_vs_keyed_final_precision": _micro_set_metric(
            rows, "full_keyed_final", "full_seq_final", "precision"
        ),
        "full_seq_vs_keyed_exact_target_rate": float(
            np.mean(
                [
                    set(row["full_seq_final"])
                    == set(row["full_keyed_final"])
                    for row in rows
                ]
            )
        ),
        "ordinary_restricted_final_recall": ordinary_recall,
        "ordinary_restricted_final_precision": _micro_set_metric(
            rows, "full_seq_final", "seq_base_final", "precision"
        ),
        "ordinary_restricted_exact_target_rate": _exact_rate(
            rows, "seq_base_final_exact"
        ),
        "keyed_restricted_final_recall": keyed_recall,
        "keyed_restricted_final_precision": _micro_set_metric(
            rows, "full_keyed_final", "keyed_base_final", "precision"
        ),
        "keyed_restricted_exact_target_rate": _exact_rate(
            rows, "keyed_base_final_exact"
        ),
        "keyed_plus_final_recall": plus_final_recall,
        "keyed_plus_final_precision": _micro_set_metric(
            rows, "full_keyed_final", "keyed_plus_final_final", "precision"
        ),
        "keyed_plus_final_exact_target_rate": _exact_rate(
            rows, "keyed_plus_final_exact"
        ),
        "keyed_plus_forward_recall": plus_forward_recall,
        "keyed_plus_forward_precision": _micro_set_metric(
            rows,
            "full_keyed_final",
            "keyed_plus_forward_final",
            "precision",
        ),
        "keyed_plus_forward_exact_target_rate": _exact_rate(
            rows, "keyed_plus_forward_exact"
        ),
        "keyed_gain_over_ordinary_recall": keyed_recall - ordinary_recall,
        "final_oracle_gain_over_keyed_recall": (
            plus_final_recall - keyed_recall
        ),
        "forward_oracle_gain_over_final_oracle_recall": (
            plus_forward_recall - plus_final_recall
        ),
    }


def run_audit(
    nodes=(20, 50),
    T=300,
    edge_probability=0.10,
    rho=0.7,
    seeds=1,
    retention=0.40,
    max_lag=1,
    alpha=0.05,
    n_shuffles=10,
):
    output = {
        "config": {
            "nodes": list(nodes),
            "T": T,
            "edge_probability": edge_probability,
            "rho": rho,
            "seeds": seeds,
            "retention": retention,
            "max_lag": max_lag,
            "alpha": alpha,
            "n_shuffles": n_shuffles,
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
                data, max_lag=max_lag
            )
            for target in range(n_nodes):
                rows.append(
                    _target_row(
                        X,
                        Y_all,
                        feature_names,
                        series,
                        seed,
                        target,
                        retention,
                        max_lag,
                        alpha,
                        n_shuffles,
                    )
                )

        output["by_n"][str(n_nodes)] = {
            "summary": _summary(rows),
            "rows": rows,
        }

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, nargs="+", default=[20, 50])
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--edge-probability", type=float, default=0.10)
    parser.add_argument("--rho", type=float, default=0.7)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--retention", type=float, default=0.40)
    parser.add_argument("--max-lag", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-shuffles", type=int, default=10)
    args = parser.parse_args()

    result = run_audit(
        nodes=tuple(args.nodes),
        T=args.T,
        edge_probability=args.edge_probability,
        rho=args.rho,
        seeds=args.seeds,
        retention=args.retention,
        max_lag=args.max_lag,
        alpha=args.alpha,
        n_shuffles=args.n_shuffles,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
