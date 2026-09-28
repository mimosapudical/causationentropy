"""Exact working-set certificate for Path-B Gaussian oCSE.

The initial Information-LASSO + conditional-rescue screen is treated only as a
working set.  A restricted keyed-oCSE path is accepted only after a global
violation check certifies that no excluded candidate could have changed any
forward decision.

This audit is deliberately non-heuristic:
- every accepted step checks excluded candidates that outrank the accepted one;
- the terminal step checks all still-live excluded candidates;
- the first excluded candidate that would pass the same keyed permutation test
  is added to the working set and the restricted solve is repeated;
- if no violator exists, induction over the forward steps certifies equality
  with the full keyed forward path.

The procedure terminates in finitely many iterations because each failed
certificate adds at least one candidate and the full candidate set is finite.
"""

import argparse
import json

import networkx as nx
import numpy as np

from causationentropy.core.information.conditional_mutual_information import (
    conditional_mutual_information,
    gaussian_conditional_mutual_information,
    prepare_gaussian_cmi_context,
)
from causationentropy.datasets.synthetic import (
    linear_stochastic_gaussian_process,
)
from experiments.path_b_benchmark_v2 import candidate_ids_for_target
from experiments.path_b_common_random_numbers import (
    _keyed_shuffle_pass,
    _stable_seed,
    keyed_backward,
)
from experiments.path_b_screen_frontier_v2 import (
    endpoint_plus_conditional_rescue,
    lagged_design,
)


def _score_state(X, Y, Z, candidate_local_ids):
    context = prepare_gaussian_cmi_context(Y, Z) if Z is not None else None
    values = {}
    for local_idx in candidate_local_ids:
        Xj = X[:, [local_idx]]
        if context is None:
            value = conditional_mutual_information(
                Xj, Y, None, method="gaussian"
            )
        else:
            value = gaussian_conditional_mutual_information(
                Xj, Y, Z, context=context
            )
        values[int(local_idx)] = float(value)
    return values


def _keyed_forward_trace(
    X,
    Y,
    Z_init,
    global_ids,
    base_seed,
    target,
    alpha=0.05,
    n_shuffles=10,
):
    candidates = list(range(X.shape[1]))
    selected = []
    Z = Z_init.copy() if Z_init is not None else None
    trace = []
    counts = {"observed_scores": 0, "shuffle_tests": 0}

    while candidates:
        values_map = _score_state(X, Y, Z, candidates)
        counts["observed_scores"] += len(candidates)

        round_candidates = list(candidates)
        round_values = [values_map[idx] for idx in round_candidates]
        failed = []
        accepted = None
        accepted_score = None
        selected_before = list(selected)

        while round_candidates:
            k_best = int(np.asarray(round_values).argmax())
            local_idx = int(round_candidates[k_best])
            global_idx = int(global_ids[local_idx])
            observed = float(round_values[k_best])
            conditioning_global = [
                int(global_ids[idx]) for idx in selected_before
            ]
            counts["shuffle_tests"] += 1
            passed = _keyed_shuffle_pass(
                X[:, [local_idx]],
                Y,
                Z,
                observed,
                alpha,
                n_shuffles,
                base_seed,
                target,
                "forward",
                global_idx,
                conditioning_global,
            )
            if passed:
                accepted = local_idx
                accepted_score = observed
                break

            failed.append(local_idx)
            round_candidates.pop(k_best)
            round_values.pop(k_best)

        if failed:
            failed_set = set(failed)
            candidates = [
                idx for idx in candidates if idx not in failed_set
            ]

        if accepted is None:
            trace.append(
                {
                    "selected_before": [
                        int(global_ids[idx]) for idx in selected_before
                    ],
                    "accepted": None,
                    "accepted_score": None,
                }
            )
            break

        trace.append(
            {
                "selected_before": [
                    int(global_ids[idx]) for idx in selected_before
                ],
                "accepted": int(global_ids[accepted]),
                "accepted_score": float(accepted_score),
            }
        )
        selected.append(accepted)
        X_best = X[:, [accepted]]
        Z = np.hstack([Z, X_best]) if Z is not None else X_best
        candidates.remove(accepted)

    return [int(global_ids[idx]) for idx in selected], trace, counts


def _state_matrix(X_full, Z_init, selected_global):
    if not selected_global:
        return Z_init
    Z_selected = X_full[:, list(selected_global)]
    if Z_init is None:
        return Z_selected
    return np.hstack((Z_init, Z_selected))


def _candidate_score_global(X_full, Y, Z, global_idx):
    Xj = X_full[:, [global_idx]]
    if Z is None:
        return float(
            conditional_mutual_information(Xj, Y, None, method="gaussian")
        )
    context = prepare_gaussian_cmi_context(Y, Z)
    return float(
        gaussian_conditional_mutual_information(
            Xj, Y, Z, context=context
        )
    )


def _find_first_violator(
    X_full,
    Y,
    Z_init,
    all_candidate_ids,
    working_set,
    trace,
    base_seed,
    target,
    alpha,
    n_shuffles,
):
    excluded = sorted(set(all_candidate_ids) - set(working_set))
    permanently_failed = set()
    counts = {"observed_scores": 0, "shuffle_tests": 0}

    for state in trace:
        selected_before = list(state["selected_before"])
        Z = _state_matrix(X_full, Z_init, selected_before)
        live_excluded = [
            idx for idx in excluded if idx not in permanently_failed
        ]

        scored = []
        for global_idx in live_excluded:
            score = _candidate_score_global(
                X_full, Y, Z, global_idx
            )
            counts["observed_scores"] += 1
            scored.append((score, int(global_idx)))
        scored.sort(key=lambda item: (-item[0], item[1]))

        accepted = state["accepted"]
        if accepted is None:
            candidates_to_test = scored
        else:
            accepted_score = float(state["accepted_score"])
            candidates_to_test = [
                (score, global_idx)
                for score, global_idx in scored
                if (
                    score > accepted_score
                    or (
                        score == accepted_score
                        and global_idx < int(accepted)
                    )
                )
            ]

        for score, global_idx in candidates_to_test:
            counts["shuffle_tests"] += 1
            passed = _keyed_shuffle_pass(
                X_full[:, [global_idx]],
                Y,
                Z,
                score,
                alpha,
                n_shuffles,
                base_seed,
                target,
                "forward",
                global_idx,
                selected_before,
            )
            if passed:
                return int(global_idx), counts
            permanently_failed.add(int(global_idx))

        if accepted is None:
            return None, counts

    return None, counts


def certified_working_set(
    X_full,
    Y,
    Z_init,
    all_candidate_ids,
    initial_working_set,
    base_seed,
    target,
    alpha=0.05,
    n_shuffles=10,
):
    working_set = sorted(set(int(x) for x in initial_working_set))
    total_counts = {
        "restricted_observed_scores": 0,
        "restricted_shuffle_tests": 0,
        "certificate_observed_scores": 0,
        "certificate_shuffle_tests": 0,
    }
    added = []
    iterations = 0

    while True:
        iterations += 1
        local_X = X_full[:, working_set]
        forward_global, trace, restricted_counts = _keyed_forward_trace(
            local_X,
            Y,
            Z_init,
            working_set,
            base_seed,
            target,
            alpha=alpha,
            n_shuffles=n_shuffles,
        )
        total_counts["restricted_observed_scores"] += restricted_counts[
            "observed_scores"
        ]
        total_counts["restricted_shuffle_tests"] += restricted_counts[
            "shuffle_tests"
        ]

        violator, certificate_counts = _find_first_violator(
            X_full,
            Y,
            Z_init,
            all_candidate_ids,
            working_set,
            trace,
            base_seed,
            target,
            alpha,
            n_shuffles,
        )
        total_counts["certificate_observed_scores"] += certificate_counts[
            "observed_scores"
        ]
        total_counts["certificate_shuffle_tests"] += certificate_counts[
            "shuffle_tests"
        ]

        if violator is None:
            local_index = {
                global_idx: local_idx
                for local_idx, global_idx in enumerate(working_set)
            }
            forward_local = [local_index[idx] for idx in forward_global]
            final_local = keyed_backward(
                local_X,
                Y,
                forward_local,
                working_set,
                base_seed,
                target,
                alpha=alpha,
                n_shuffles=n_shuffles,
                Z_init=Z_init,
            )
            final_global = {
                int(working_set[int(local_idx)])
                for local_idx in final_local
            }
            return {
                "working_set": set(working_set),
                "forward": set(forward_global),
                "final": final_global,
                "added": list(added),
                "iterations": iterations,
                **total_counts,
            }

        if violator in working_set:
            raise RuntimeError("certificate returned an existing working-set member")
        working_set = sorted(set(working_set) | {violator})
        added.append(int(violator))


def _full_keyed(
    X_full,
    Y,
    Z_init,
    all_candidate_ids,
    base_seed,
    target,
    alpha,
    n_shuffles,
):
    local_X = X_full[:, all_candidate_ids]
    forward_global, _trace, counts = _keyed_forward_trace(
        local_X,
        Y,
        Z_init,
        all_candidate_ids,
        base_seed,
        target,
        alpha=alpha,
        n_shuffles=n_shuffles,
    )
    local_index = {
        global_idx: local_idx
        for local_idx, global_idx in enumerate(all_candidate_ids)
    }
    forward_local = [local_index[idx] for idx in forward_global]
    final_local = keyed_backward(
        local_X,
        Y,
        forward_local,
        all_candidate_ids,
        base_seed,
        target,
        alpha=alpha,
        n_shuffles=n_shuffles,
        Z_init=Z_init,
    )
    final_global = {
        int(all_candidate_ids[int(local_idx)])
        for local_idx in final_local
    }
    return set(forward_global), final_global, counts


def _target_run(
    X,
    Y_all,
    feature_names,
    series,
    seed,
    target,
    retention,
    alpha,
    n_shuffles,
):
    Y = Y_all[:, [target]]
    Z_init = series[:-1, [target]]
    all_candidate_ids = candidate_ids_for_target(feature_names, target)
    X_candidates = X[:, all_candidate_ids]

    screen_local, _diag = endpoint_plus_conditional_rescue(
        X_candidates,
        Y,
        np.random.default_rng(seed * 10000 + target),
        retention=retention,
        information="gaussian",
    )
    screen = {
        int(all_candidate_ids[int(local_idx)])
        for local_idx in screen_local
    }

    full_forward, full_final, full_counts = _full_keyed(
        X,
        Y,
        Z_init,
        all_candidate_ids,
        seed,
        target,
        alpha,
        n_shuffles,
    )
    certified = certified_working_set(
        X,
        Y,
        Z_init,
        all_candidate_ids,
        screen,
        seed,
        target,
        alpha=alpha,
        n_shuffles=n_shuffles,
    )

    total_certificate_shuffles = (
        certified["restricted_shuffle_tests"]
        + certified["certificate_shuffle_tests"]
    )
    total_certificate_scores = (
        certified["restricted_observed_scores"]
        + certified["certificate_observed_scores"]
    )

    return {
        "seed": seed,
        "target": target,
        "candidate_count": len(all_candidate_ids),
        "initial_retention": len(screen) / len(all_candidate_ids),
        "certified_retention": (
            len(certified["working_set"]) / len(all_candidate_ids)
        ),
        "witnesses_added": len(certified["added"]),
        "added": certified["added"],
        "added_from_full_forward": len(
            set(certified["added"]) & full_forward
        ),
        "added_from_full_final": len(
            set(certified["added"]) & full_final
        ),
        "iterations": certified["iterations"],
        "forward_exact": certified["forward"] == full_forward,
        "final_exact": certified["final"] == full_final,
        "full_forward_size": len(full_forward),
        "full_final_size": len(full_final),
        "full_observed_scores": full_counts["observed_scores"],
        "full_shuffle_tests": full_counts["shuffle_tests"],
        "certified_observed_scores": total_certificate_scores,
        "certified_shuffle_tests": total_certificate_shuffles,
        "shuffle_test_ratio": (
            total_certificate_shuffles / full_counts["shuffle_tests"]
            if full_counts["shuffle_tests"]
            else 1.0
        ),
        "observed_score_ratio": (
            total_certificate_scores / full_counts["observed_scores"]
            if full_counts["observed_scores"]
            else 1.0
        ),
    }


def _summary(rows):
    return {
        "targets": len(rows),
        "mean_initial_retention": float(
            np.mean([r["initial_retention"] for r in rows])
        ),
        "mean_certified_retention": float(
            np.mean([r["certified_retention"] for r in rows])
        ),
        "total_witnesses_added": int(
            sum(r["witnesses_added"] for r in rows)
        ),
        "witnesses_from_full_forward": int(
            sum(r["added_from_full_forward"] for r in rows)
        ),
        "witnesses_from_full_final": int(
            sum(r["added_from_full_final"] for r in rows)
        ),
        "mean_iterations": float(np.mean([r["iterations"] for r in rows])),
        "forward_exact_rate": float(
            np.mean([r["forward_exact"] for r in rows])
        ),
        "final_exact_rate": float(
            np.mean([r["final_exact"] for r in rows])
        ),
        "total_full_shuffle_tests": int(
            sum(r["full_shuffle_tests"] for r in rows)
        ),
        "total_certified_shuffle_tests": int(
            sum(r["certified_shuffle_tests"] for r in rows)
        ),
        "aggregate_shuffle_test_ratio": (
            sum(r["certified_shuffle_tests"] for r in rows)
            / sum(r["full_shuffle_tests"] for r in rows)
            if sum(r["full_shuffle_tests"] for r in rows)
            else 1.0
        ),
        "aggregate_observed_score_ratio": (
            sum(r["certified_observed_scores"] for r in rows)
            / sum(r["full_observed_scores"] for r in rows)
            if sum(r["full_observed_scores"] for r in rows)
            else 1.0
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
    n_shuffles=10,
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
                    _target_run(
                        X,
                        Y_all,
                        feature_names,
                        series,
                        seed,
                        target,
                        retention,
                        alpha,
                        n_shuffles,
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
