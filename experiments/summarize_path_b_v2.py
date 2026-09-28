"""Summarize local Path-B v2 validation artifacts as Markdown."""

import json
import sys
from pathlib import Path

import numpy as np


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean(values):
    return float(np.mean(values)) if values else float("nan")


def fmt(value, digits=3):
    if value != value:
        return "N/A"
    return f"{value:.{digits}f}"


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m experiments.summarize_path_b_v2 OUTDIR")

    outdir = Path(sys.argv[1])
    frontier = load_json(outdir / "03_screen_frontier.json")

    gaussian_files = sorted(outdir.glob("05_gaussian_seed_*.json"))
    gaussian = [load_json(path)["gaussian"] for path in gaussian_files]

    print("# Path B v2 local validation summary")
    print()
    print(f"- Gaussian seeds completed: {len(gaussian)}")
    frontier_summary = frontier["gaussian_frontier"]["summary"]
    frontier_targets = (
        next(iter(frontier_summary.values()))["targets"] if frontier_summary else 0
    )
    print(f"- Frontier targets per screen: {frontier_targets}")
    print()

    summary = frontier["gaussian_frontier"]["summary"]
    print("## Screening frontier")
    print()
    print(
        "| Screen | Mean retention | Full forward-closure recall | "
        "Full final-support recall |"
    )
    print("|---|---:|---:|---:|")
    for key in sorted(summary):
        row = summary[key]
        print(
            f"| {key} | {fmt(row['mean_retention'])} | "
            f"{fmt(row['mean_full_forward_recall'])} | "
            f"{fmt(row['mean_full_support_recall'])} |"
        )

    print()
    print("## Common-random-number path preservation")
    print()
    crn_path = outdir / "04_common_random_numbers.json"
    if crn_path.exists():
        crn = load_json(crn_path)["summary"]
        print(
            "| Target retention | Actual retention | Forward closure recall | "
            "Restricted final recall |"
        )
        print("|---:|---:|---:|---:|")
        for key in sorted(crn, key=float):
            row = crn[key]
            print(
                f"| {key} | {fmt(row['mean_actual_retention'])} | "
                f"{fmt(row['mean_forward_closure_recall'])} | "
                f"{fmt(row['mean_restricted_final_recall'])} |"
            )
    else:
        print("CRN diagnostic: NOT_RUN")

    print()
    print("## Gaussian end-to-end")
    print()
    if gaussian:
        full_runtime = [row["standard"]["runtime_seconds"] for row in gaussian]
        path_runtime = [row["path_b_v2"]["runtime_seconds"] for row in gaussian]
        speedups = [
            full / path
            for full, path in zip(full_runtime, path_runtime)
            if path > 0
        ]
        retention = [row["path_b_v2"]["candidate_retention"] for row in gaussian]
        screen_recall = [
            row["path_b_v2"]["screen_full_support_recall"] for row in gaussian
        ]
        refined_recall = [
            row["path_b_v2"]["refined_full_support_recall"] for row in gaussian
        ]
        full_f1 = [row["standard"]["f1"] for row in gaussian]
        path_f1 = [row["path_b_v2"]["f1"] for row in gaussian]

        print(f"- Mean candidate retention: {fmt(mean(retention))}")
        print(f"- Mean screen recall vs full oCSE support: {fmt(mean(screen_recall))}")
        print(
            f"- Mean refined recall vs full oCSE support: "
            f"{fmt(mean(refined_recall))}"
        )
        print(f"- Mean full-oCSE F1: {fmt(mean(full_f1))}")
        print(f"- Mean Path-B-v2 F1: {fmt(mean(path_f1))}")
        print(f"- Mean runtime speedup: {fmt(mean(speedups), 2)}x")

        full_shuffle_evals = [
            row["standard"].get("shuffle_cmi_evaluations", float("nan"))
            for row in gaussian
        ]
        path_shuffle_evals = [
            row["path_b_v2"].get("shuffle_cmi_evaluations", float("nan"))
            for row in gaussian
        ]
        shuffle_reductions = [
            1.0 - path / full
            for full, path in zip(full_shuffle_evals, path_shuffle_evals)
            if full == full and path == path and full > 0
        ]
        screen_scores = [
            row["path_b_v2"].get("screen_marginal_cmi_scores", 0)
            + row["path_b_v2"].get("screen_rescue_cmi_scores", 0)
            for row in gaussian
        ]
        print(
            f"- Mean shuffle-CMI evaluation reduction: "
            f"{fmt(mean(shuffle_reductions))}"
        )
        print(
            f"- Mean no-shuffle screen CMI scores: "
            f"{fmt(mean(screen_scores), 1)}"
        )
        total_cmi_reductions = [
            1.0 - row["path_b_v2"]["total_cmi_evaluations"]
            / row["standard"]["total_cmi_evaluations"]
            for row in gaussian
            if row["standard"].get("total_cmi_evaluations", 0) > 0
        ]
        print(
            f"- Mean total-CMI evaluation reduction: "
            f"{fmt(mean(total_cmi_reductions))}"
        )
    else:
        print("No Gaussian result files found.")

    print()
    print("## Gaussian scaling")
    print()
    scale_files = sorted(outdir.glob("06_scale_n*_seed_*.json"))
    if scale_files:
        grouped = {}
        for path in scale_files:
            payload = load_json(path)["gaussian"]
            n_nodes = int(payload["config"]["n_nodes"])
            grouped.setdefault(n_nodes, []).append(payload)

        print(
            "| N | Runs | Candidate retention | Screen recall | Refined recall | "
            "Full runtime (s) | Path-B runtime (s) | Speedup | "
            "Shuffle-CMI reduction | Total-CMI reduction |"
        )
        print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for n_nodes in sorted(grouped):
            rows = grouped[n_nodes]
            full_runtime = [row["standard"]["runtime_seconds"] for row in rows]
            path_runtime = [row["path_b_v2"]["runtime_seconds"] for row in rows]
            speedups = [
                full / path
                for full, path in zip(full_runtime, path_runtime)
                if path > 0
            ]
            retention = [
                row["path_b_v2"]["candidate_retention"] for row in rows
            ]
            screen_recall = [
                row["path_b_v2"]["screen_full_support_recall"] for row in rows
            ]
            refined_recall = [
                row["path_b_v2"]["refined_full_support_recall"] for row in rows
            ]
            full_shuffle = [
                row["standard"].get("shuffle_cmi_evaluations", float("nan"))
                for row in rows
            ]
            path_shuffle = [
                row["path_b_v2"].get("shuffle_cmi_evaluations", float("nan"))
                for row in rows
            ]
            shuffle_reduction = [
                1.0 - path / full
                for full, path in zip(full_shuffle, path_shuffle)
                if full == full and path == path and full > 0
            ]
            total_cmi_reduction = [
                1.0 - row["path_b_v2"]["total_cmi_evaluations"]
                / row["standard"]["total_cmi_evaluations"]
                for row in rows
                if row["standard"].get("total_cmi_evaluations", 0) > 0
            ]
            print(
                f"| {n_nodes} | {len(rows)} | {fmt(mean(retention))} | "
                f"{fmt(mean(screen_recall))} | {fmt(mean(refined_recall))} | "
                f"{fmt(mean(full_runtime), 2)} | {fmt(mean(path_runtime), 2)} | "
                f"{fmt(mean(speedups), 2)}x | {fmt(mean(shuffle_reduction))} | "
                f"{fmt(mean(total_cmi_reduction))} |"
            )
    else:
        print("Scaling benchmark: NOT_RUN")

    print()
    print("## Stress diagnostics")
    print()
    for name, row in frontier["stress"].items():
        print(
            f"- {name}: endpoint parent recall "
            f"{fmt(row['endpoint_parent_recall'])}; rescue parent recall "
            f"{fmt(row['rescue_parent_recall'])}"
        )

    print()
    print("## Independent Path-A / Poisson audits")
    print()

    path_a_audit = outdir / "08_path_a_weight_normalization.json"
    if path_a_audit.exists():
        payload = load_json(path_a_audit)
        row = payload["summary"]
        print(
            f"- Path A normalization support equality: "
            f"{fmt(row['support_equality_rate'])}"
        )
        print(
            f"- Path A warnings, sum-normalized vs max-normalized: "
            f"{row['sum_warning_total']} vs {row['max_warning_total']}"
        )
        print(
            f"- Mean weighted max |X|, sum vs max normalization: "
            f"{fmt(row['mean_sum_weighted_max_abs'], 6)} vs "
            f"{fmt(row['mean_max_weighted_max_abs'], 6)}"
        )
    else:
        print("- Path A normalization audit: NOT_RUN")

    poisson_audit = outdir / "08_poisson_rate_structure.json"
    if poisson_audit.exists():
        payload = load_json(poisson_audit)
        low = payload["low_rate_system"]
        high = payload["high_rate_system"]
        print(
            "- Poisson low-rate empirical means vs corrcoef-implied rates: "
            f"{low['empirical_means']} vs "
            f"{low['current_corrcoef_implied_rates']}"
        )
        print(
            "- Poisson high-rate empirical means vs corrcoef-implied rates: "
            f"{high['empirical_means']} vs "
            f"{high['current_corrcoef_implied_rates']}"
        )
    else:
        print("- Poisson rate-structure audit: NOT_RUN")

    gaussian_audit = outdir / "08_gaussian_constant_feature.json"
    if gaussian_audit.exists():
        payload = load_json(gaussian_audit)["cases"]
        constant = payload["constant_vs_random"]
        near_constant = payload["near_constant_vs_random"]
        independent = payload["independent_random"]
        print(
            "- Gaussian MI constant / near-constant / independent: "
            f"{fmt(constant['mi'], 6)} / {fmt(near_constant['mi'], 6)} / "
            f"{fmt(independent['mi'], 6)}"
        )
    else:
        print("- Gaussian constant-feature audit: NOT_RUN")

    self_history_audit = outdir / "08_self_history_candidate.json"
    if self_history_audit.exists():
        row = load_json(self_history_audit)["summary"]
        print(
            "- Self-history duplicate nonfinite fraction: "
            f"{fmt(row['nonfinite_fraction'])}"
        )
        print(
            "- Self-history nonduplicate final-support equality under keyed RNG: "
            f"{fmt(row['nonduplicate_final_equality_rate'])}"
        )
        print(
            "- Estimated futile shuffle-CMI evaluations from duplicate self-history: "
            f"{row['estimated_futile_shuffle_evaluations_if_tested_once']}"
        )
    else:
        print("- Self-history candidate audit: NOT_RUN")

    significant_audit = outdir / "08_only_return_significant.json"
    if significant_audit.exists():
        payload = load_json(significant_audit)
        print(
            "- only_return_significant=True with forced final Pass=False "
            f"returned edges: {payload['returned_edge_count']}"
        )
    else:
        print("- only_return_significant audit: NOT_RUN")

    print()
    print("## Nonlinear / estimator audit")
    print()
    for case in ("logistic", "poisson"):
        path = outdir / f"07_{case}_seed_0.json"
        if not path.exists():
            path = outdir / f"06_{case}_seed_0.json"
        if not path.exists():
            print(f"- {case}: NOT_RUN")
            continue
        payload = load_json(path)[case]
        role = payload["config"].get("benchmark_role", "unknown")
        print(
            f"- {case} ({role}): full F1={fmt(payload['standard']['f1'])}, "
            f"Path-B-v2 F1={fmt(payload['path_b_v2']['f1'])}, "
            f"screen recall={fmt(payload['path_b_v2']['screen_full_support_recall'])}"
        )


if __name__ == "__main__":
    main()
