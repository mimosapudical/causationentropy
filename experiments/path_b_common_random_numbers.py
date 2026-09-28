"""Common-random-number diagnostic for Path-B path preservation.

This file does not change the package API.  It exists to separate two effects:

1. restricting the candidate universe, and
2. shifting the shared permutation RNG stream because fewer tests are executed.

For a fixed candidate and conditioning set, the keyed shuffle test below derives
its RNG seed from (base_seed, target, stage, candidate, conditioning_set).  Full
and restricted oCSE therefore see the same null permutations whenever they
perform the same statistical test.
"""

import argparse
import hashlib
import json

import networkx as nx
import numpy as np

from causationentropy.core.information.conditional_mutual_information import (
    conditional_mutual_information,
    gaussian_conditional_mutual_information,
    prepare_gaussian_cmi_context,
)
from causationentropy.datasets.synthetic import linear_stochastic_gaussian_process
from experiments.path_b_screen_frontier_v2 import (
    endpoint_plus_conditional_rescue,
    lagged_design,
)


def _stable_seed(base_seed, target, stage, candidate, conditioning):
    payload = (
        f"{base_seed}|{target}|{stage}|{candidate}|"
        + ",".join(str(int(x)) for x in sorted(conditioning))
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**63 - 1)


def _observed_cmi(X, Y, Z):
    if Z is None:
        return conditional_mutual_information(X, Y, None, method="gaussian")
    context = prepare_gaussian_cmi_context(Y, Z)
    return gaussian_conditional_mutual_information(X, Y, Z, context=context)


def _keyed_shuffle_pass(
    X,
    Y,
    Z,
    observed,
    alpha,
    n_shuffles,
    base_seed,
    target,
    stage,
    candidate,
    conditioning,
):
    rng = np.random.default_rng(
        _stable_seed(base_seed, target, stage, candidate, conditioning)
    )
    context = prepare_gaussian_cmi_context(Y, Z) if Z is not None else None
    null = np.empty(n_shuffles, dtype=float)
    for i in range(n_shuffles):
        X_perm = X[rng.permutation(len(X)), :]
        if context is None:
            null[i] = conditional_mutual_information(
                X_perm, Y, None, method="gaussian"
            )
        else:
            null[i] = gaussian_conditional_mutual_information(
                X_perm, Y, Z, context=context
            )
    threshold = np.percentile(null, 100 * (1 - alpha))
    return bool(observed >= threshold)


def keyed_forward(
    X,
    Y,
    Z_init,
    global_ids,
    base_seed,
    target,
    alpha=0.05,
    n_shuffles=100,
):
    candidates = list(range(X.shape[1]))
    selected = []
    Z = Z_init.copy() if Z_init is not None else None

    while candidates:
        context = prepare_gaussian_cmi_context(Y, Z) if Z is not None else None
        values = []
        for local_idx in candidates:
            Xj = X[:, [local_idx]]
            if context is None:
                value = conditional_mutual_information(
                    Xj, Y, None, method="gaussian"
                )
            else:
                value = gaussian_conditional_mutual_information(
                    Xj, Y, Z, context=context
                )
            values.append(value)

        round_candidates = list(candidates)
        round_values = list(values)
        accepted = None
        failed = []
        conditioning_global = [global_ids[idx] for idx in selected]

        while round_candidates:
            k_best = int(np.asarray(round_values).argmax())
            local_idx = round_candidates[k_best]
            global_idx = global_ids[local_idx]
            observed = round_values[k_best]

            if _keyed_shuffle_pass(
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
            ):
                accepted = local_idx
                break

            failed.append(local_idx)
            round_candidates.pop(k_best)
            round_values.pop(k_best)

        if failed:
            failed_set = set(failed)
            candidates = [idx for idx in candidates if idx not in failed_set]

        if accepted is None:
            break

        selected.append(accepted)
        X_best = X[:, [accepted]]
        Z = np.hstack([Z, X_best]) if Z is not None else X_best
        candidates.remove(accepted)

    return selected


def keyed_backward(
    X,
    Y,
    forward,
    global_ids,
    base_seed,
    target,
    alpha=0.05,
    n_shuffles=100,
    Z_init=None,
):
    selected = list(forward)

    # The order itself is keyed to the global forward set, so a full and
    # restricted run with the same accepted forward closure use the same order.
    order_seed = _stable_seed(
        base_seed,
        target,
        "backward_order",
        -1,
        [global_ids[idx] for idx in forward],
    )
    order_rng = np.random.default_rng(order_seed)

    for local_idx in order_rng.permutation(forward):
        conditioning = [idx for idx in selected if idx != local_idx]
        Z_selected = X[:, conditioning] if conditioning else None
        Z = (
            Z_init
            if Z_selected is None
            else (
                Z_selected
                if Z_init is None
                else np.hstack((Z_init, Z_selected))
            )
        )
        observed = _observed_cmi(X[:, [local_idx]], Y, Z)
        conditioning_global = [global_ids[idx] for idx in conditioning]

        if not _keyed_shuffle_pass(
            X[:, [local_idx]],
            Y,
            Z,
            observed,
            alpha,
            n_shuffles,
            base_seed,
            target,
            "backward",
            global_ids[local_idx],
            conditioning_global,
        ):
            selected.remove(local_idx)

    return selected


def keyed_ocse(
    X,
    Y,
    Z_init,
    global_ids,
    base_seed,
    target,
    alpha=0.05,
    n_shuffles=100,
):
    forward = keyed_forward(
        X,
        Y,
        Z_init,
        global_ids,
        base_seed,
        target,
        alpha=alpha,
        n_shuffles=n_shuffles,
    )
    final = keyed_backward(
        X,
        Y,
        forward,
        global_ids,
        base_seed,
        target,
        alpha=alpha,
        n_shuffles=n_shuffles,
        Z_init=Z_init,
    )
    return forward, final


def _recall(reference, selected):
    reference = set(reference)
    selected = set(selected)
    return len(reference & selected) / len(reference) if reference else 1.0


def run_diagnostic(
    n_nodes=20,
    T=350,
    edge_probability=0.10,
    rho=0.7,
    seeds=3,
    n_shuffles=20,
    retentions=(0.20, 0.30, 0.40, 0.50),
):
    rows = []

    for seed in range(seeds):
        graph_true = nx.erdos_renyi_graph(
            n_nodes, edge_probability, seed=seed, directed=True
        )
        data, _ = linear_stochastic_gaussian_process(
            rho=rho,
            n=n_nodes,
            T=T,
            p=edge_probability,
            seed=seed,
            G=graph_true,
        )
        X, Y_all, _, series = lagged_design(data, max_lag=1)
        all_ids = list(range(X.shape[1]))

        full_by_target = {}
        for target in range(n_nodes):
            Y = Y_all[:, [target]]
            Z_init = series[:-1, [target]]
            forward, final = keyed_ocse(
                X,
                Y,
                Z_init,
                all_ids,
                seed,
                target,
                n_shuffles=n_shuffles,
            )
            full_by_target[target] = (set(forward), set(final))

        for retention in retentions:
            forward_total = 0
            final_total = 0
            forward_screened = 0
            final_screened = 0
            final_reproduced = 0
            actual_retentions = []

            for target in range(n_nodes):
                Y = Y_all[:, [target]]
                Z_init = series[:-1, [target]]
                screened, _ = endpoint_plus_conditional_rescue(
                    X,
                    Y,
                    np.random.default_rng(seed * 10000 + target),
                    retention=retention,
                )
                screened = sorted(int(idx) for idx in screened)
                screened_set = set(screened)
                actual_retentions.append(len(screened) / X.shape[1])

                full_forward, full_final = full_by_target[target]
                forward_total += len(full_forward)
                final_total += len(full_final)
                forward_screened += len(full_forward & screened_set)
                final_screened += len(full_final & screened_set)

                _, restricted_final_local = keyed_ocse(
                    X[:, screened],
                    Y,
                    Z_init,
                    screened,
                    seed,
                    target,
                    n_shuffles=n_shuffles,
                )
                restricted_final = {
                    screened[local_idx] for local_idx in restricted_final_local
                }
                final_reproduced += len(full_final & restricted_final)

            rows.append(
                {
                    "seed": seed,
                    "retention_target": retention,
                    "actual_retention": float(np.mean(actual_retentions)),
                    "forward_closure_recall": (
                        forward_screened / forward_total
                        if forward_total
                        else 1.0
                    ),
                    "final_support_screen_recall": (
                        final_screened / final_total if final_total else 1.0
                    ),
                    "restricted_final_recall": (
                        final_reproduced / final_total if final_total else 1.0
                    ),
                }
            )

    summary = {}
    for retention in retentions:
        subset = [row for row in rows if row["retention_target"] == retention]
        summary[f"{retention:.2f}"] = {
            "mean_actual_retention": float(
                np.mean([row["actual_retention"] for row in subset])
            ),
            "mean_forward_closure_recall": float(
                np.mean([row["forward_closure_recall"] for row in subset])
            ),
            "mean_final_support_screen_recall": float(
                np.mean([row["final_support_screen_recall"] for row in subset])
            ),
            "mean_restricted_final_recall": float(
                np.mean([row["restricted_final_recall"] for row in subset])
            ),
            "seeds": len(subset),
        }

    return {"rows": rows, "summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n-shuffles", type=int, default=20)
    args = parser.parse_args()
    result = run_diagnostic(
        seeds=args.seeds,
        n_shuffles=args.n_shuffles,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
