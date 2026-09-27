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

    gaussian_files = sorted(outdir.glob("04_gaussian_seed_*.json"))
    gaussian = [load_json(path)["gaussian"] for path in gaussian_files]

    print("# Path B v2 local validation summary")
    print()
    print(f"- Gaussian seeds completed: {len(gaussian)}")
    print()

    summary = frontier["gaussian_frontier"]["summary"]
    print("## Screening frontier")
    print()
    print("| Screen | Mean retention | Full-oCSE support recall |")
    print("|---|---:|---:|")
    for key in sorted(summary):
        row = summary[key]
        print(
            f"| {key} | {fmt(row['mean_retention'])} | "
            f"{fmt(row['mean_full_forward_recall'])} | "
            f"{fmt(row['mean_full_support_recall'])} |"
        )

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
    else:
        print("No Gaussian result files found.")

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
    print("## Nonlinear / estimator audit")
    print()
    for case in ("logistic", "poisson"):
        path = outdir / f"05_{case}_seed_0.json"
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
