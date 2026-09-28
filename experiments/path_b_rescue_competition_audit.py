"""Audit the deterministic one-shot rescue competition condition.

For a Path-A endpoint A, let M be the true parents omitted by A.  Score every
excluded variable by c_j = I(X_j; Y | X_A).  Let

    c_min = min_{j in M} c_j

and let h(A) be the number of excluded nonparents whose score is at least
c_min.  If q rescue slots are available and

    q >= |M| + h(A),

then descending-score one-shot rescue must contain every missed parent.

This audit measures how often that sufficient condition explains success and
failure in the current Gaussian regimes.
"""

import argparse
import json
import math

import networkx as nx
import numpy as np

from causationentropy.core.discovery import (
    information_lasso_optimal_causation_entropy,
)
from causationentropy.core.information.conditional_mutual_information import (
    conditional_mutual_information,
)
from causationentropy.datasets.synthetic import (
    linear_stochastic_gaussian_process,
)
from experiments.path_b_benchmark_v2 import candidate_ids_for_target
from experiments.path_b_beta_min_audit import floored_linear_gaussian_var
from experiments.path_b_screen_frontier_v2 import (
    hidden_parent_stress,
    lagged_design,
)


def _truth_local(A, target, candidate_ids, feature_names):
    return {
        int(local_idx)
        for local_idx, global_idx in enumerate(candidate_ids)
        if abs(
            float(
                A[
                    target,
                    int(feature_names[global_idx][0]),
                ]
            )
        )
        > 0
    }


def _audit_target(X, Y, truth, seed, target, retention):
    endpoint = set(
        int(j)
        for j in information_lasso_optimal_causation_entropy(
            X,
            Y,
            np.random.default_rng(seed * 10000 + target),
            information="gaussian",
        )
    )
    missed = set(truth) - endpoint

    target_size = max(
        int(math.ceil(retention * X.shape[1])),
        len(endpoint) + (1 if len(endpoint) < X.shape[1] else 0),
    )
    target_size = min(X.shape[1], target_size)
    q = max(0, target_size - len(endpoint))

    Z = X[:, sorted(endpoint)] if endpoint else None
    scores = {}
    for j in range(X.shape[1]):
        if j in endpoint:
            continue
        value = conditional_mutual_information(
            X[:, [j]],
            Y,
            Z,
            method="gaussian",
        )
        scores[int(j)] = (
            float(value) if np.isfinite(value) else -np.inf
        )

    ranking = sorted(
        scores,
        key=lambda j: (-scores[j], int(j)),
    )
    rescued = set(endpoint)
    rescued.update(ranking[:q])

    if missed:
        c_min = min(scores[j] for j in missed)
        h = sum(
            1
            for j, value in scores.items()
            if j not in truth and value >= c_min
        )
        condition = q >= len(missed) + h
    else:
        c_min = None
        h = 0
        condition = True

    success = truth.issubset(rescued)

    return {
        "target": int(target),
        "parent_count": len(truth),
        "endpoint_parent_complete": len(missed) == 0,
        "missed_parent_count": len(missed),
        "rescue_slots": int(q),
        "competition_nonparents": int(h),
        "weakest_missed_parent_score": c_min,
        "sufficient_condition": bool(condition),
        "rescue_parent_complete": bool(success),
        "condition_true_but_failure": bool(condition and not success),
        "condition_false_but_success": bool((not condition) and success),
    }


def _summary(rows):
    parent_rows = [r for r in rows if r["parent_count"] > 0]
    incomplete = [
        r for r in parent_rows if not r["endpoint_parent_complete"]
    ]
    return {
        "targets": len(rows),
        "targets_with_parents": len(parent_rows),
        "path_a_incomplete_targets": len(incomplete),
        "rescue_success_rate_on_incomplete": (
            float(np.mean([
                r["rescue_parent_complete"] for r in incomplete
            ]))
            if incomplete else 1.0
        ),
        "condition_rate_on_incomplete": (
            float(np.mean([
                r["sufficient_condition"] for r in incomplete
            ]))
            if incomplete else 1.0
        ),
        "condition_true_but_failure_count": int(
            sum(r["condition_true_but_failure"] for r in incomplete)
        ),
        "condition_false_but_success_count": int(
            sum(r["condition_false_but_success"] for r in incomplete)
        ),
        "median_missed_parent_count": (
            float(np.median([
                r["missed_parent_count"] for r in incomplete
            ]))
            if incomplete else 0.0
        ),
        "median_rescue_slots": (
            float(np.median([
                r["rescue_slots"] for r in incomplete
            ]))
            if incomplete else 0.0
        ),
        "median_competition_nonparents": (
            float(np.median([
                r["competition_nonparents"] for r in incomplete
            ]))
            if incomplete else 0.0
        ),
    }


def _random_case(
    n_nodes,
    T,
    p,
    rho,
    seed,
    retention,
    weight_floor=None,
):
    if weight_floor is None:
        graph = nx.erdos_renyi_graph(
            n_nodes, p, seed=seed, directed=True
        )
        data, A = linear_stochastic_gaussian_process(
            rho=rho,
            n=n_nodes,
            T=T,
            p=p,
            seed=seed,
            G=graph,
        )
    else:
        data, A, _graph = floored_linear_gaussian_var(
            n=n_nodes,
            T=T,
            p=p,
            rho=rho,
            weight_floor=weight_floor,
            seed=seed,
            burnin=200,
        )

    X, Y_all, feature_names, _series = lagged_design(data, max_lag=1)
    rows = []
    for target in range(n_nodes):
        candidate_ids = candidate_ids_for_target(
            feature_names, target
        )
        Xc = X[:, candidate_ids]
        truth = _truth_local(
            A, target, candidate_ids, feature_names
        )
        rows.append(
            _audit_target(
                Xc,
                Y_all[:, [target]],
                truth,
                seed,
                target,
                retention,
            )
        )
    return rows


def run_audit(retention=0.40, hidden_seeds=20):
    cases = {}

    for n_nodes in (20, 50):
        rows = _random_case(
            n_nodes,
            300,
            0.10,
            0.7,
            0,
            retention,
            weight_floor=None,
        )
        cases[f"repo_random_n{n_nodes}"] = {
            "summary": _summary(rows),
            "rows": rows,
        }

    for floor in (0.0, 0.25, 0.5):
        rows = _random_case(
            50,
            300,
            0.10,
            0.7,
            0,
            retention,
            weight_floor=floor,
        )
        cases[f"floored_n50_f{floor:.2f}"] = {
            "summary": _summary(rows),
            "rows": rows,
        }

    hidden_rows = []
    for seed in range(hidden_seeds):
        X, Y, truth = hidden_parent_stress(seed=seed, T=1000)
        hidden_rows.append(
            _audit_target(
                X,
                Y,
                truth,
                seed,
                0,
                retention,
            )
        )
    cases["hidden_parent"] = {
        "summary": _summary(hidden_rows),
        "rows": hidden_rows,
    }

    return {
        "config": {
            "retention": retention,
            "hidden_seeds": hidden_seeds,
        },
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retention", type=float, default=0.40)
    parser.add_argument("--hidden-seeds", type=int, default=20)
    args = parser.parse_args()
    print(
        json.dumps(
            run_audit(
                retention=args.retention,
                hidden_seeds=args.hidden_seeds,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
