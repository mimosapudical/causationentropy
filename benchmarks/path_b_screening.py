"""Parent-containment and rescue ablation audit of the production Path-B screen.

Run ``python -m benchmarks.path_b_screening --help``. This extends the earlier
true-parent-retention and nonlinear-cancellation audits, using the production
screen (including target history) rather than copying its ranking algorithm.
Results are descriptive pilot evidence, not a sure-screening guarantee.
"""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import sklearn

import causationentropy.core.discovery as discovery
from benchmarks.adapters import generate_case


def generate_cancellation(n: int, T: int, seed: int) -> np.ndarray:
    """Reuse the earlier engineered nonlinear-cancellation stress mechanism.

    X1 is marginally independent of Y, but is a structural parent given X2.
    This engineered cancellation is not representative of generic faithful
    systems; only target zero and its two external parents are evaluated.
    """
    if n < 3:
        raise ValueError("Cancellation requires at least three variables.")
    rng = np.random.default_rng(seed)
    data = np.zeros((T, n))
    x1 = rng.standard_normal(T)
    u = rng.standard_normal(T)
    data[:, 1] = x1
    data[:, 2] = -(x1**2 - 1.0) + u
    for j in range(3, n):
        data[:, j] = rng.standard_normal(T)
    eps = 0.2 * rng.standard_normal(T)
    data[1:, 0] = data[:-1, 2] + (data[:-1, 1] ** 2 - 1.0) + eps[1:]
    return data


def column_indices(full: np.ndarray, subset: np.ndarray) -> list:
    """Map instrumented production columns to unique original feature indices."""
    selected = []
    for column in subset.T:
        matches = np.flatnonzero(np.all(full == column[:, None], axis=0))
        if len(matches) != 1:
            raise ValueError("Audit requires unique candidate columns.")
        selected.append(int(matches[0]))
    return selected


def containment(truth: set, screen: set, full_support) -> dict:
    """Keep generating parents distinct from finite-sample oCSE support."""
    return {
        "parent_count": len(truth),
        "parent_hits": len(truth & screen),
        "parent_complete": bool(truth <= screen),
        "full_support_count": len(full_support) if full_support is not None else None,
        "full_support_hits": (
            len(full_support & screen) if full_support is not None else None
        ),
    }


def refine(X, Y, Z, screen: list, seed: int, information: str, n_shuffles: int) -> list:
    """Refine a fixed screen with the same corrected standard-oCSE baseline."""
    if not screen:
        return []
    local = discovery.standard_optimal_causation_entropy(
        X[:, screen],
        Y,
        Z,
        np.random.default_rng(seed),
        n_shuffles=n_shuffles,
        information=information,
    )
    return [screen[int(j)] for j in local]


def measured_refinement(X, Y, Z, screen, seed, information, n_shuffles):
    """Return selections, elapsed seconds, and all observed CMI invocations."""
    cmi = Mock(wraps=discovery.conditional_mutual_information)
    start = time.perf_counter()
    with patch.object(discovery, "conditional_mutual_information", cmi):
        result = refine(X, Y, Z, screen, seed, information, n_shuffles)
    return result, time.perf_counter() - start, cmi.call_count


def audit_target(X, Y, Z, truth, seed, retentions, information, n_shuffles, final):
    """Audit actual production screens and two explicitly defined ablations.

    The proposal-only ablation changes size; marginal-fill keeps the proposal
    and fills to the *realized* production size by I(Xj;Y | H), without S0.
    Nonlinear screen-only runs leave full-support and final metrics absent.
    """
    p = X.shape[1]
    full_support = None
    full_time = full_calls = None
    if final:
        support, full_time, full_calls = measured_refinement(
            X,
            Y,
            Z,
            list(range(p)),
            seed,
            information,
            n_shuffles,
        )
        full_support = set(support)
    proposal = discovery.information_lasso_optimal_causation_entropy(
        X,
        Y,
        np.random.default_rng(seed),
        information=information,
    )
    proposal = sorted(int(j) for j in proposal)
    marginal_scores = [
        discovery.conditional_mutual_information(
            X[:, [j]],
            Y,
            Z,
            method=information,
        )
        for j in range(p)
    ]
    marginal_order = sorted(
        set(range(p)) - set(proposal),
        key=lambda j: (
            -(marginal_scores[j] if np.isfinite(marginal_scores[j]) else -np.inf),
            j,
        ),
    )
    rows = []
    original_refine = discovery.standard_optimal_causation_entropy
    for retention in retentions:
        # Observe the production boundary between screening and refinement.
        # No copied rescue formula or new public diagnostics API is required.
        observer = Mock(wraps=original_refine) if final else Mock(return_value=[])
        cmi = Mock(wraps=discovery.conditional_mutual_information)
        start = time.perf_counter()
        with patch.object(discovery, "standard_optimal_causation_entropy", observer):
            with patch.object(discovery, "conditional_mutual_information", cmi):
                predicted = discovery.information_screened_optimal_causation_entropy(
                    X,
                    Y,
                    Z,
                    np.random.default_rng(seed),
                    retention=retention,
                    information=information,
                    n_shuffles=n_shuffles,
                )
        elapsed = time.perf_counter() - start
        if observer.call_count != 1:
            raise AssertionError("Expected exactly one restricted refinement.")
        screen = column_indices(X, observer.call_args.args[0])
        marginal = sorted(
            set(proposal) | set(marginal_order[: len(screen) - len(proposal)])
        )
        for method, selected in (
            ("conditional_rescue", screen),
            ("proposal_only", proposal),
            ("marginal_fill_matched", marginal),
        ):
            result = {
                "method": method,
                "retention_target": retention,
                "screen": selected,
                "truth": sorted(truth),
                "full_support": (
                    sorted(full_support) if full_support is not None else None
                ),
                "retained_fraction": len(selected) / p,
                **containment(set(truth), set(selected), full_support),
            }
            if final:
                if method == "conditional_rescue":
                    final_support, seconds, calls = predicted, elapsed, cmi.call_count
                else:
                    final_support, seconds, calls = measured_refinement(
                        X,
                        Y,
                        Z,
                        selected,
                        seed,
                        information,
                        n_shuffles,
                    )
                result.update(
                    {
                        "final": final_support,
                        "TP": len(set(final_support) & truth),
                        "FP": len(set(final_support) - truth),
                        "FN": len(truth - set(final_support)),
                        "measured_seconds": seconds,
                        "cmi_calls": calls,
                        "full_seconds": full_time,
                        "full_cmi_calls": full_calls,
                        "full_TP": len(full_support & truth),
                        "full_FP": len(full_support - truth),
                        "full_FN": len(truth - full_support),
                    }
                )
            rows.append(result)
    return rows


def summarize(rows: list) -> list:
    """Micro-average edges; report target completeness separately, by cell."""
    groups = {}
    for row in rows:
        key = (row["family"], row["T"], row["retention_target"], row["method"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (family, T, retention, method), group in sorted(groups.items()):
        parents = sum(row["parent_count"] for row in group)
        nonempty = [row for row in group if row["parent_count"]]
        support_rows = [row for row in group if row["full_support_count"] is not None]
        support_count = sum(row["full_support_count"] for row in support_rows)
        summary = {
            "family": family,
            "T": T,
            "retention_target": retention,
            "method": method,
            "targets": len(group),
            "seeds": len({row["seed"] for row in group}),
            "parent_edges": parents,
            "parent_recall": (
                sum(row["parent_hits"] for row in group) / parents if parents else None
            ),
            "parent_complete_target_rate": (
                float(np.mean([row["parent_complete"] for row in nonempty]))
                if nonempty
                else None
            ),
            "mean_retention": float(
                np.mean([row["retained_fraction"] for row in group])
            ),
            "full_support_coverage": (
                sum(row["full_support_hits"] for row in support_rows) / support_count
                if support_count
                else None
            ),
        }
        if support_rows:
            # Counts already use exact parent-column identity within each target.
            tp, fp, fn = (
                sum(row[name] for row in group) for name in ("TP", "FP", "FN")
            )
            summary["final_F1"] = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 0.0
            full_tp, full_fp, full_fn = (
                sum(row[name] for row in group)
                for name in ("full_TP", "full_FP", "full_FN")
            )
            summary["full_F1"] = (
                2 * full_tp / (2 * full_tp + full_fp + full_fn)
                if (full_tp + full_fp + full_fn)
                else 0.0
            )
            summary["cmi_calls"] = sum(row["cmi_calls"] for row in group)
            summary["full_cmi_calls"] = sum(row["full_cmi_calls"] for row in group)
            # Only production includes complete screening overhead in timing/calls.
            if method == "conditional_rescue":
                summary["cmi_reduction"] = (
                    1 - summary["cmi_calls"] / summary["full_cmi_calls"]
                )
                summary["aggregate_speedup"] = sum(
                    row["full_seconds"] for row in group
                ) / sum(row["measured_seconds"] for row in group)
        summaries.append(summary)
    return summaries


def run_audit(family, n, sample_sizes, seeds, retentions, n_shuffles, output):
    """Write target-level JSONL as each seed completes, then a summary and CSV."""
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    caught = []
    with (output / "targets.jsonl").open("w") as raw:
        for T in sample_sizes:
            for seed in seeds:
                if family == "gaussian":
                    data, edges, _ = generate_case(n, T, seed)
                    targets, information = range(n), "gaussian"
                else:
                    data = generate_cancellation(n, T, seed)
                    edges, targets, information = {(1, 0, 1), (2, 0, 1)}, [0], "kde"
                for target in targets:
                    sources = [source for source in range(n) if source != target]
                    X, Y, Z = (
                        data[:-1, sources],
                        data[1:, [target]],
                        data[:-1, [target]],
                    )
                    truth = {
                        j
                        for j, source in enumerate(sources)
                        if (source, target, 1) in edges
                    }
                    with warnings.catch_warnings(record=True) as caught_target:
                        warnings.simplefilter("always")
                        target_rows = audit_target(
                            X,
                            Y,
                            Z,
                            truth,
                            seed * 10000 + target,
                            retentions,
                            information,
                            n_shuffles,
                            family == "gaussian",
                        )
                    caught.extend(type(w.message).__name__ for w in caught_target)
                    for row in target_rows:
                        row.update(
                            {
                                "family": family,
                                "n": n,
                                "T": T,
                                "seed": seed,
                                "target": target,
                            }
                        )
                        raw.write(json.dumps(row) + "\n")
                    raw.flush()
                    rows.extend(target_rows)
                print(f"{family}: T={T} seed={seed} complete", flush=True)
    summary = summarize(rows)
    metadata = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"])
        ).hexdigest(),
        "source_sha256": {
            "screening_audit": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "discovery": hashlib.sha256(
                Path(discovery.__file__).read_bytes()
            ).hexdigest(),
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "family": family,
        "n": n,
        "sample_sizes": sample_sizes,
        "seeds": seeds,
        "retentions": retentions,
        "n_shuffles": n_shuffles,
        "warning_counts": {name: caught.count(name) for name in sorted(set(caught))},
        "notes": "Nonlinear runs evaluate the screen only; support/final/timing comparisons are Gaussian only. Ablation times omit screening overhead. No inferential CI/FWER guarantee is claimed.",
    }
    (output / "summary.json").write_text(
        json.dumps({"config": metadata, "cells": summary}, indent=2) + "\n"
    )
    fields = sorted({key for row in summary for key in row})
    with (output / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    return summary


def main() -> None:
    """Run a fixed sample-size/retention grid without overwriting previous runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family", choices=["gaussian", "cancellation"], default="gaussian"
    )
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[150, 300, 600])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--retentions", type=float, nargs="+", default=[0.1, 0.2, 0.4])
    parser.add_argument("--n-shuffles", type=int, default=49)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.n < 3 or min(args.sample_sizes) < 20 or args.n_shuffles < 1:
        parser.error("Require n >= 3, T >= 20 and positive shuffle count.")
    if any(not 0 < r <= 1 for r in args.retentions):
        parser.error("Retention must be in (0, 1].")
    run_audit(
        args.family,
        args.n,
        args.sample_sizes,
        args.seeds,
        args.retentions,
        args.n_shuffles,
        args.output,
    )


if __name__ == "__main__":
    main()
