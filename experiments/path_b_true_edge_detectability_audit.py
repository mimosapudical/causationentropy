"""Audit Path-B screening against true-edge detectability, not baseline identity.

Consumes a path-preservation JSON and regenerates the same Gaussian synthetic
graphs from its recorded configuration.  This separates three questions:

1. Does the screen omit true parents that full keyed oCSE can actually detect?
2. Are full-oCSE final edges omitted by the screen true edges or false positives?
3. How do screen/full/restricted recovery rates vary with true edge magnitude?

This is a post-processing audit; it does not rerun discovery.
"""

import argparse
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np

from causationentropy.datasets.synthetic import (
    linear_stochastic_gaussian_process,
)


DEFAULT_BINS = (0.0, 0.025, 0.05, 0.10, 0.20, 0.40, float("inf"))


def _metrics(truth, predicted):
    truth = set(truth)
    predicted = set(predicted)
    tp = len(truth & predicted)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _bin_label(lo, hi):
    if math.isinf(hi):
        return f"[{lo:.3f}, inf)"
    return f"[{lo:.3f}, {hi:.3f})"


def _rate(rows, key):
    return float(np.mean([bool(row[key]) for row in rows])) if rows else None


def _summarize_bins(edge_rows, bins):
    out = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        subset = [
            row
            for row in edge_rows
            if row["abs_weight"] >= lo and row["abs_weight"] < hi
        ]
        out[_bin_label(lo, hi)] = {
            "true_edges": len(subset),
            "screen_recall": _rate(subset, "in_screen"),
            "full_keyed_forward_recall": _rate(
                subset, "in_full_keyed_forward"
            ),
            "full_keyed_final_recall": _rate(
                subset, "in_full_keyed_final"
            ),
            "restricted_keyed_final_recall": _rate(
                subset, "in_restricted_keyed_final"
            ),
        }
    return out


def _regenerate_graph_and_weights(n_nodes, edge_probability, rho, seed):
    graph = nx.erdos_renyi_graph(
        n_nodes,
        edge_probability,
        seed=seed,
        directed=True,
    )
    # Use the repository generator itself so graph/weight construction remains
    # exactly aligned with the benchmark.
    _data, A = linear_stochastic_gaussian_process(
        rho=rho,
        n=n_nodes,
        T=2,
        p=edge_probability,
        seed=seed,
        G=graph,
    )
    return graph, A


def run_audit(path_preservation, bins=DEFAULT_BINS):
    payload = json.loads(Path(path_preservation).read_text())
    config = payload["config"]
    by_n = {}

    for n_key, block in payload["by_n"].items():
        n_nodes = int(n_key)
        rows_by_seed_target = {
            (int(row["seed"]), int(row["target"])): row
            for row in block["rows"]
        }

        true_edge_rows = []
        all_truth = set()
        all_full = set()
        all_restricted = set()
        all_screen = set()
        full_edges_missed_by_screen = []

        for seed in range(int(config["seeds"])):
            graph, A = _regenerate_graph_and_weights(
                n_nodes,
                float(config["edge_probability"]),
                float(config["rho"]),
                seed,
            )

            for target in range(n_nodes):
                row = rows_by_seed_target[(seed, target)]
                screen = set(int(x) for x in row["base_screen"])
                full_forward = set(
                    int(x) for x in row["full_keyed_forward"]
                )
                full_final = set(int(x) for x in row["full_keyed_final"])
                restricted_final = set(
                    int(x) for x in row["keyed_base_final"]
                )

                truth_target = {
                    int(source) for source in graph.predecessors(target)
                }
                all_truth.update(
                    (seed, source, target) for source in truth_target
                )
                all_screen.update(
                    (seed, source, target) for source in screen
                )
                all_full.update(
                    (seed, source, target) for source in full_final
                )
                all_restricted.update(
                    (seed, source, target)
                    for source in restricted_final
                )

                for source in sorted(truth_target):
                    true_edge_rows.append(
                        {
                            "seed": seed,
                            "source": int(source),
                            "target": target,
                            "weight": float(A[target, source]),
                            "abs_weight": float(abs(A[target, source])),
                            "in_screen": source in screen,
                            "in_full_keyed_forward": source in full_forward,
                            "in_full_keyed_final": source in full_final,
                            "in_restricted_keyed_final": (
                                source in restricted_final
                            ),
                        }
                    )

                for source in sorted(full_final - screen):
                    full_edges_missed_by_screen.append(
                        {
                            "seed": seed,
                            "source": int(source),
                            "target": target,
                            "is_true_edge": source in truth_target,
                            "weight": (
                                float(A[target, source])
                                if source in truth_target
                                else 0.0
                            ),
                        }
                    )

        full_detectable_true = [
            row for row in true_edge_rows if row["in_full_keyed_final"]
        ]
        screen_missed_true = [
            row for row in true_edge_rows if not row["in_screen"]
        ]

        by_n[n_key] = {
            "summary": {
                "true_edges": len(true_edge_rows),
                "screen_true_parent_recall": _rate(
                    true_edge_rows, "in_screen"
                ),
                "full_keyed_true_parent_recall": _rate(
                    true_edge_rows, "in_full_keyed_final"
                ),
                "restricted_keyed_true_parent_recall": _rate(
                    true_edge_rows, "in_restricted_keyed_final"
                ),
                "screen_recall_of_full_detectable_true_parents": _rate(
                    full_detectable_true, "in_screen"
                ),
                "screen_missed_true_edges": len(screen_missed_true),
                "screen_missed_true_also_missed_by_full": int(
                    sum(
                        not row["in_full_keyed_final"]
                        for row in screen_missed_true
                    )
                ),
                "full_final_edges_missed_by_screen": len(
                    full_edges_missed_by_screen
                ),
                "full_final_true_edges_missed_by_screen": int(
                    sum(
                        row["is_true_edge"]
                        for row in full_edges_missed_by_screen
                    )
                ),
                "full_keyed_metrics": _metrics(all_truth, all_full),
                "restricted_keyed_metrics": _metrics(
                    all_truth, all_restricted
                ),
            },
            "by_abs_weight": _summarize_bins(true_edge_rows, bins),
            "screen_missed_true_edges": screen_missed_true,
            "full_final_edges_missed_by_screen": (
                full_edges_missed_by_screen
            ),
        }

    return {
        "config": {
            **config,
            "weight_bins": [
                "inf" if math.isinf(x) else x for x in bins
            ],
        },
        "by_n": by_n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_preservation_json")
    args = parser.parse_args()
    print(
        json.dumps(
            run_audit(args.path_preservation_json),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
