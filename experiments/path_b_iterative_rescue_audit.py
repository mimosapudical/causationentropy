"""Compare one-shot Path-B rescue with iterative conditional screening.

This is a theory-motivated diagnostic inspired by ISIS and by the original
oCSE aggregative-discovery principle.  It does not change production code.

Both screens start from the same production Information-LASSO endpoint and use
the same target budget.  The only difference is:

- one-shot rescue: score every excluded variable once given the endpoint and
  fill all remaining budget slots from that ranking;
- iterative rescue: repeatedly add the currently largest conditional-MI
  candidate and recompute scores after each addition.

The audit reports parent recall / parent-complete targets and the number of
no-shuffle CMI score evaluations required by each rescue.
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
    endpoint_plus_conditional_rescue,
    hidden_parent_stress,
    lagged_design,
)


def _target_size(p, endpoint_size, retention, min_rescue=1):
    size = max(
        int(math.ceil(retention * p)),
        endpoint_size + (min_rescue if endpoint_size < p else 0),
    )
    return min(p, size)


def iterative_conditional_rescue(
    X,
    Y,
    endpoint,
    retention=0.40,
    information="gaussian",
):
    selected = set(int(j) for j in endpoint)
    target_size = _target_size(X.shape[1], len(selected), retention)
    score_evals = 0

    while len(selected) < target_size:
        Z = X[:, sorted(selected)] if selected else None
        best_j = None
        best_score = -np.inf
        for j in range(X.shape[1]):
            if j in selected:
                continue
            value = conditional_mutual_information(
                X[:, [j]],
                Y,
                Z,
                method=information,
            )
            score_evals += 1
            score = float(value) if np.isfinite(value) else -np.inf
            if (
                score > best_score
                or (
                    score == best_score
                    and (best_j is None or int(j) < best_j)
                )
            ):
                best_score = score
                best_j = int(j)

        if best_j is None:
            break
        selected.add(best_j)

    return sorted(selected), score_evals


def _truth_local(A, target, candidate_ids, feature_names):
    truth = set()
    for local_idx, global_idx in enumerate(candidate_ids):
        source = int(feature_names[global_idx][0])
        if abs(float(A[target, source])) > 0:
            truth.add(int(local_idx))
    return truth


def _target_row(
    X,
    Y,
    truth,
    seed,
    target,
    retention,
):
    endpoint = set(
        int(j)
        for j in information_lasso_optimal_causation_entropy(
            X,
            Y,
            np.random.default_rng(seed * 10000 + target),
            information="gaussian",
        )
    )

    one_shot, diag = endpoint_plus_conditional_rescue(
        X,
        Y,
        np.random.default_rng(seed * 10000 + target),
        retention=retention,
        information="gaussian",
    )
    one_shot = set(int(j) for j in one_shot)
    one_shot_scores = X.shape[1] - len(endpoint) if len(endpoint) < X.shape[1] else 0

    iterative, iterative_scores = iterative_conditional_rescue(
        X,
        Y,
        endpoint,
        retention=retention,
        information="gaussian",
    )
    iterative = set(iterative)

    return {
        "target": int(target),
        "parent_count": len(truth),
        "endpoint_size": len(endpoint),
        "screen_size": len(one_shot),
        "one_shot_parent_tp": len(truth & one_shot),
        "iterative_parent_tp": len(truth & iterative),
        "one_shot_parent_complete": truth.issubset(one_shot),
        "iterative_parent_complete": truth.issubset(iterative),
        "one_shot_score_evals": int(one_shot_scores),
        "iterative_score_evals": int(iterative_scores),
        "rescue_added": int(diag["rescued"]),
        "screen_equal": one_shot == iterative,
    }


def _summary(rows):
    parent_rows = [row for row in rows if row["parent_count"] > 0]
    edges = sum(row["parent_count"] for row in parent_rows)
    return {
        "targets": len(rows),
        "targets_with_parents": len(parent_rows),
        "true_parent_edges": int(edges),
        "one_shot_parent_recall": (
            sum(row["one_shot_parent_tp"] for row in parent_rows) / edges
            if edges else 1.0
        ),
        "iterative_parent_recall": (
            sum(row["iterative_parent_tp"] for row in parent_rows) / edges
            if edges else 1.0
        ),
        "one_shot_parent_complete_target_rate": (
            float(np.mean([
                row["one_shot_parent_complete"] for row in parent_rows
            ]))
            if parent_rows else 1.0
        ),
        "iterative_parent_complete_target_rate": (
            float(np.mean([
                row["iterative_parent_complete"] for row in parent_rows
            ]))
            if parent_rows else 1.0
        ),
        "mean_one_shot_score_evals": float(
            np.mean([row["one_shot_score_evals"] for row in rows])
        ),
        "mean_iterative_score_evals": float(
            np.mean([row["iterative_score_evals"] for row in rows])
        ),
        "score_eval_ratio": (
            sum(row["iterative_score_evals"] for row in rows)
            / max(1, sum(row["one_shot_score_evals"] for row in rows))
        ),
        "screen_equality_rate": float(
            np.mean([row["screen_equal"] for row in rows])
        ),
    }


def _random_var_case(
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
        data, A, graph = floored_linear_gaussian_var(
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
        candidate_ids = candidate_ids_for_target(feature_names, target)
        Xc = X[:, candidate_ids]
        truth = _truth_local(
            A, target, candidate_ids, feature_names
        )
        rows.append(
            _target_row(
                Xc,
                Y_all[:, [target]],
                truth,
                seed,
                target,
                retention,
            )
        )
    return rows


def _hidden_parent_case(seeds, retention):
    rows = []
    for seed in range(seeds):
        X, Y, truth = hidden_parent_stress(seed=seed, T=1000)
        rows.append(
            _target_row(
                X,
                Y,
                truth,
                seed,
                0,
                retention,
            )
        )
    return rows


def run_audit(
    retention=0.40,
    hidden_seeds=20,
):
    cases = {}

    for n_nodes in (20, 50):
        rows = _random_var_case(
            n_nodes=n_nodes,
            T=300,
            p=0.10,
            rho=0.7,
            seed=0,
            retention=retention,
            weight_floor=None,
        )
        cases[f"repo_random_n{n_nodes}"] = {
            "summary": _summary(rows),
            "rows": rows,
        }

    for floor in (0.0, 0.25, 0.5):
        rows = _random_var_case(
            n_nodes=50,
            T=300,
            p=0.10,
            rho=0.7,
            seed=0,
            retention=retention,
            weight_floor=floor,
        )
        cases[f"floored_n50_f{floor:.2f}"] = {
            "summary": _summary(rows),
            "rows": rows,
        }

    hidden_rows = _hidden_parent_case(hidden_seeds, retention)
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
