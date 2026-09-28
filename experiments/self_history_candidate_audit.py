"""Audit duplicate target-history candidates in standard oCSE.

Standard oCSE conditions on the target's own lagged history through Z_init, but
the same lagged target columns also remain in X_lagged.  This audit measures:

1. how often I(X_target_lag; Y | Z_init) is non-finite;
2. how many futile shuffle tests the duplicate candidates can induce;
3. whether removing those duplicate candidates changes the keyed-RNG final
   support for the non-duplicate candidate universe.

The audit is diagnostic only and does not change discover_network.
"""

import argparse
import json

import networkx as nx
import numpy as np

from causationentropy.core.information.conditional_mutual_information import (
    conditional_mutual_information,
)
from causationentropy.datasets.synthetic import linear_stochastic_gaussian_process
from experiments.path_b_common_random_numbers import keyed_ocse
from experiments.path_b_screen_frontier_v2 import lagged_design


def _globalize(local_support, global_ids):
    return {int(global_ids[int(local_idx)]) for local_idx in local_support}


def run_audit(
    n_nodes=12,
    T=300,
    edge_probability=0.12,
    rho=0.7,
    max_lag=1,
    seeds=5,
    n_shuffles=20,
):
    rows = []

    for seed in range(seeds):
        graph = nx.erdos_renyi_graph(
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
            G=graph,
        )
        X, Y_all, feature_names, series = lagged_design(
            data,
            max_lag=max_lag,
        )
        all_ids = list(range(X.shape[1]))

        for target in range(n_nodes):
            Y = Y_all[:, [target]]
            Z_init = np.column_stack(
                [
                    series[max_lag - lag : series.shape[0] - lag, target]
                    for lag in range(1, max_lag + 1)
                ]
            )
            duplicate_ids = [
                idx
                for idx, (source, _lag) in enumerate(feature_names)
                if source == target
            ]

            duplicate_cmi = []
            for idx in duplicate_ids:
                with np.errstate(all="ignore"):
                    value = conditional_mutual_information(
                        X[:, [idx]],
                        Y,
                        Z_init,
                        method="gaussian",
                    )
                duplicate_cmi.append(float(value))

            keep_ids = [idx for idx in all_ids if idx not in set(duplicate_ids)]

            full_forward, full_final = keyed_ocse(
                X,
                Y,
                Z_init,
                all_ids,
                seed,
                target,
                n_shuffles=n_shuffles,
            )
            dedup_forward, dedup_final = keyed_ocse(
                X[:, keep_ids],
                Y,
                Z_init,
                keep_ids,
                seed,
                target,
                n_shuffles=n_shuffles,
            )

            full_forward_global = _globalize(full_forward, all_ids)
            full_final_global = _globalize(full_final, all_ids)
            dedup_forward_global = _globalize(dedup_forward, keep_ids)
            dedup_final_global = _globalize(dedup_final, keep_ids)

            rows.append(
                {
                    "seed": seed,
                    "target": target,
                    "duplicate_candidates": len(duplicate_ids),
                    "duplicate_nonfinite_cmi": int(
                        sum(not np.isfinite(value) for value in duplicate_cmi)
                    ),
                    "duplicate_cmi": duplicate_cmi,
                    "full_forward_duplicate_accepts": len(
                        full_forward_global & set(duplicate_ids)
                    ),
                    "full_final_duplicate_accepts": len(
                        full_final_global & set(duplicate_ids)
                    ),
                    "nonduplicate_forward_equal": (
                        full_forward_global - set(duplicate_ids)
                    )
                    == dedup_forward_global,
                    "nonduplicate_final_equal": (
                        full_final_global - set(duplicate_ids)
                    )
                    == dedup_final_global,
                }
            )

    duplicate_total = sum(row["duplicate_candidates"] for row in rows)
    nonfinite_total = sum(row["duplicate_nonfinite_cmi"] for row in rows)

    return {
        "config": {
            "n_nodes": n_nodes,
            "T": T,
            "edge_probability": edge_probability,
            "rho": rho,
            "max_lag": max_lag,
            "seeds": seeds,
            "n_shuffles": n_shuffles,
        },
        "summary": {
            "targets": len(rows),
            "duplicate_candidates": duplicate_total,
            "duplicate_nonfinite_cmi": nonfinite_total,
            "nonfinite_fraction": (
                nonfinite_total / duplicate_total if duplicate_total else 0.0
            ),
            "targets_with_duplicate_forward_accept": int(
                sum(row["full_forward_duplicate_accepts"] > 0 for row in rows)
            ),
            "targets_with_duplicate_final_accept": int(
                sum(row["full_final_duplicate_accepts"] > 0 for row in rows)
            ),
            "nonduplicate_forward_equality_rate": float(
                np.mean([row["nonduplicate_forward_equal"] for row in rows])
            ),
            "nonduplicate_final_equality_rate": float(
                np.mean([row["nonduplicate_final_equal"] for row in rows])
            ),
            "estimated_futile_shuffle_evaluations_if_tested_once": (
                nonfinite_total * n_shuffles
            ),
        },
        "rows": rows,
        "implementation_changed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nodes", type=int, default=12)
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--max-lag", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-shuffles", type=int, default=20)
    args = parser.parse_args()

    print(
        json.dumps(
            run_audit(
                n_nodes=args.n_nodes,
                T=args.T,
                max_lag=args.max_lag,
                seeds=args.seeds,
                n_shuffles=args.n_shuffles,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
