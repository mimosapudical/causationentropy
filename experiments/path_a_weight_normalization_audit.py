"""Numerical audit for Path-A Information-LASSO weight normalization.

Path A currently uses

    w_j = I_j / sum_k I_k.

Replacing the denominator by max_j I_j multiplies every feature weight by one
common positive constant. The relative weighted-L1 penalties are therefore
unchanged; this audit measures whether the alternative normalization improves
solver scale without changing selected support in ordinary finite-precision
runs.

This file is diagnostic only and does not modify the Path-A implementation.
"""

import argparse
import json
import warnings

import networkx as nx
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoLarsIC

from causationentropy.core.information.conditional_mutual_information import (
    conditional_mutual_information,
)
from causationentropy.datasets.synthetic import linear_stochastic_gaussian_process
from experiments.path_b_screen_frontier_v2 import lagged_design


def information_values(X, Y):
    values = np.empty(X.shape[1], dtype=float)
    for j in range(X.shape[1]):
        value = conditional_mutual_information(
            X[:, [j]],
            Y,
            None,
            method="gaussian",
        )
        if not np.isfinite(value):
            raise ValueError(f"non-finite information score for candidate {j}")
        values[j] = max(0.0, value)
    return values


def fit_support(X, Y, values, normalization):
    if normalization == "sum":
        denominator = float(values.sum())
    elif normalization == "max":
        denominator = float(values.max()) if values.size else 0.0
    else:
        raise ValueError(normalization)

    if denominator <= 0:
        return [], {
            "warning_count": 0,
            "weighted_max_abs": 0.0,
            "weighted_min_nonzero_abs": 0.0,
        }

    weights = values / denominator
    X_weighted = X * weights.reshape(1, -1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = LassoLarsIC(criterion="bic", max_iter=100).fit(
            X_weighted,
            Y.reshape(-1),
        )

    nonzero = np.abs(X_weighted[np.nonzero(X_weighted)])
    return np.flatnonzero(model.coef_ != 0).tolist(), {
        "warning_count": sum(
            issubclass(item.category, (ConvergenceWarning, RuntimeWarning))
            for item in caught
        ),
        "weighted_max_abs": float(np.max(np.abs(X_weighted))),
        "weighted_min_nonzero_abs": (
            float(np.min(nonzero)) if nonzero.size else 0.0
        ),
    }


def run_audit(
    n_nodes=50,
    T=300,
    edge_probability=0.08,
    rho=0.7,
    seeds=10,
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
        X, Y_all, _, _ = lagged_design(data, max_lag=1)

        for target in range(n_nodes):
            Y = Y_all[:, [target]]
            values = information_values(X, Y)
            sum_support, sum_diag = fit_support(
                X,
                Y,
                values,
                normalization="sum",
            )
            max_support, max_diag = fit_support(
                X,
                Y,
                values,
                normalization="max",
            )
            rows.append(
                {
                    "seed": seed,
                    "target": target,
                    "support_equal": sum_support == max_support,
                    "sum_support_size": len(sum_support),
                    "max_support_size": len(max_support),
                    "sum_warning_count": sum_diag["warning_count"],
                    "max_warning_count": max_diag["warning_count"],
                    "sum_weighted_max_abs": sum_diag["weighted_max_abs"],
                    "max_weighted_max_abs": max_diag["weighted_max_abs"],
                }
            )

    return {
        "config": {
            "n_nodes": n_nodes,
            "T": T,
            "edge_probability": edge_probability,
            "rho": rho,
            "seeds": seeds,
        },
        "summary": {
            "targets": len(rows),
            "support_equality_rate": float(
                np.mean([row["support_equal"] for row in rows])
            ),
            "sum_warning_total": int(
                sum(row["sum_warning_count"] for row in rows)
            ),
            "max_warning_total": int(
                sum(row["max_warning_count"] for row in rows)
            ),
            "mean_sum_weighted_max_abs": float(
                np.mean([row["sum_weighted_max_abs"] for row in rows])
            ),
            "mean_max_weighted_max_abs": float(
                np.mean([row["max_weighted_max_abs"] for row in rows])
            ),
        },
        "rows": rows,
        "implementation_changed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nodes", type=int, default=50)
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    print(
        json.dumps(
            run_audit(
                n_nodes=args.n_nodes,
                T=args.T,
                seeds=args.seeds,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
