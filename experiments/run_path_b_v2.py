"""Cross-platform one-command validation for the Path-B v2 experiment.

Examples
--------
Quick correctness pass:
    python -m experiments.run_path_b_v2 --mode quick

Full paper-oriented local run:
    python -m experiments.run_path_b_v2 --mode full

The runner writes every machine-readable result under one output directory and
finishes by generating a Markdown summary.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(args, outfile=None, env=None, allow_failure=False):
    print("+", " ".join(str(arg) for arg in args), flush=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    if outfile is None:
        completed = subprocess.run(args, check=False, env=merged_env)
        if completed.returncode != 0 and not allow_failure:
            raise subprocess.CalledProcessError(completed.returncode, args)
        return completed.returncode

    outfile.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        args,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=merged_env,
    )
    outfile.write_text(completed.stdout, encoding="utf-8")
    if completed.stderr:
        outfile.with_suffix(outfile.suffix + ".stderr.txt").write_text(
            completed.stderr,
            encoding="utf-8",
        )
    if completed.returncode != 0 and not allow_failure:
        print(completed.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(completed.returncode, args)
    return completed.returncode


def python_module(module, *args):
    return [sys.executable, "-m", module, *map(str, args)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument(
        "--outdir",
        default="experiments/results/path_b_v2_local",
    )
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--shuffles", type=int, default=None)
    parser.add_argument("--retention", type=float, default=0.40)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "quick":
        seeds = 1 if args.seeds is None else args.seeds
        shuffles = 20 if args.shuffles is None else args.shuffles
        crn_seeds = 1
        crn_shuffles = 10
        scale_nodes = (20, 50)
        scale_seeds = 1
        scale_shuffles = min(shuffles, 20)
        nonlinear_seeds = (0,)
    else:
        seeds = 5 if args.seeds is None else args.seeds
        shuffles = 100 if args.shuffles is None else args.shuffles
        crn_seeds = 3
        crn_shuffles = 20
        scale_nodes = (20, 50, 100, 200)
        scale_seeds = 3
        scale_shuffles = 20
        nonlinear_seeds = (0, 1, 2)

    print("== [1/10] Style / syntax gates ==")
    style_targets = [
        "causationentropy/core/discovery.py",
        "causationentropy/core/information/conditional_mutual_information.py",
        "causationentropy/tests/test_path_b_v2_engineering.py",
        "causationentropy/tests/test_path_b_benchmark_smoke.py",
        "experiments/path_b_benchmark_v1.py",
        "experiments/path_b_benchmark_v2.py",
        "experiments/path_b_common_random_numbers.py",
        "experiments/path_b_screen_frontier_v2.py",
        "experiments/path_a_weight_normalization_audit.py",
        "experiments/poisson_rate_structure_audit.py",
        "experiments/gaussian_constant_feature_audit.py",
        "experiments/run_path_b_v2.py",
        "experiments/summarize_path_b_v2.py",
    ]
    style_commands = [
        [sys.executable, "-m", "black", "--check", *style_targets],
        [sys.executable, "-m", "isort", "--check-only", *style_targets],
        [
            sys.executable,
            "-m",
            "flake8",
            "--select=E9,F63,F7,F82",
            *style_targets,
        ],
    ]
    style_log = outdir / "00_style_checks.txt"
    with style_log.open("w", encoding="utf-8") as handle:
        for command in style_commands:
            print("+", " ".join(command), file=handle, flush=True)
            subprocess.run(
                command,
                check=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )

    print("== [2/10] Focused engineering tests ==")
    focused = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "causationentropy/tests/test_path_b_v2_engineering.py",
        "causationentropy/tests/test_path_b_benchmark_smoke.py",
    ]
    with (outdir / "01_focused_tests.txt").open("w", encoding="utf-8") as handle:
        subprocess.run(
            focused,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    print("== [3/10] Discovery + information regression ==")
    regression = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "causationentropy/tests/test_discovery.py",
        (
            "causationentropy/tests/core/information/"
            "test_conditional_mutual_information.py"
        ),
    ]
    with (outdir / "02_regression_tests.txt").open("w", encoding="utf-8") as handle:
        subprocess.run(
            regression,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if args.mode == "full":
        print("   running full default pytest suite")
        with (outdir / "02_full_pytest.txt").open("w", encoding="utf-8") as handle:
            subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                check=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )

    print("== [4/10] Screening frontier + stress diagnostics ==")
    run_command(
        python_module(
            "experiments.path_b_screen_frontier_v2",
            "--seeds",
            seeds,
            "--n-shuffles",
            shuffles,
        ),
        outdir / "03_screen_frontier.json",
    )

    print("== [5/10] Common-random-number path diagnostic ==")
    run_command(
        python_module(
            "experiments.path_b_common_random_numbers",
            "--seeds",
            crn_seeds,
            "--n-shuffles",
            crn_shuffles,
        ),
        outdir / "04_common_random_numbers.json",
    )

    print("== [6/10] End-to-end Gaussian accuracy benchmark ==")
    for seed in range(seeds):
        run_command(
            python_module(
                "experiments.path_b_benchmark_v2",
                "--case",
                "gaussian",
                "--seed",
                seed,
                "--retention",
                args.retention,
                "--n-shuffles",
                shuffles,
            ),
            outdir / f"05_gaussian_seed_{seed}.json",
        )

    print("== [7/10] Gaussian scaling benchmark ==")
    for n_nodes in scale_nodes:
        T = max(300, 5 * n_nodes)
        for seed in range(scale_seeds):
            run_command(
                python_module(
                    "experiments.path_b_benchmark_v2",
                    "--case",
                    "gaussian",
                    "--seed",
                    seed,
                    "--retention",
                    args.retention,
                    "--n-shuffles",
                    scale_shuffles,
                    "--n-nodes",
                    n_nodes,
                    "--T",
                    T,
                ),
                outdir / f"06_scale_n{n_nodes}_seed_{seed}.json",
            )

    print("== [8/10] Nonlinear logistic + Poisson estimator audit ==")
    for seed in nonlinear_seeds:
        run_command(
            python_module(
                "experiments.path_b_benchmark_v2",
                "--case",
                "logistic",
                "--seed",
                seed,
                "--retention",
                args.retention,
                "--n-shuffles",
                shuffles,
            ),
            outdir / f"07_logistic_seed_{seed}.json",
        )

    # Poisson remains an estimator audit rather than a primary benchmark until
    # the repository's existing Poisson integration behavior is resolved.
    poisson_code = run_command(
        python_module(
            "experiments.path_b_benchmark_v2",
            "--case",
            "poisson",
            "--seed",
            0,
            "--retention",
            args.retention,
            "--n-shuffles",
            shuffles,
        ),
        outdir / "07_poisson_seed_0.json",
        allow_failure=True,
    )
    (outdir / "07_poisson_status.txt").write_text(
        "PASS\n" if poisson_code == 0 else f"AUDIT_FAILED_RETURN_CODE={poisson_code}\n",
        encoding="utf-8",
    )

    print("== [9/10] Independent Path-A / Poisson audits ==")
    path_a_nodes = 20 if args.mode == "quick" else 50
    path_a_seeds = 1 if args.mode == "quick" else 10
    run_command(
        python_module(
            "experiments.path_a_weight_normalization_audit",
            "--n-nodes",
            path_a_nodes,
            "--T",
            300,
            "--seeds",
            path_a_seeds,
        ),
        outdir / "08_path_a_weight_normalization.json",
    )
    run_command(
        python_module(
            "experiments.poisson_rate_structure_audit",
            "--seed",
            0,
            "--n-samples",
            5000,
        ),
        outdir / "08_poisson_rate_structure.json",
    )
    run_command(
        python_module(
            "experiments.gaussian_constant_feature_audit",
            "--seed",
            0,
            "--n-samples",
            500,
        ),
        outdir / "08_gaussian_constant_feature.json",
    )

    print("== [10/10] Markdown summary ==")
    summary_path = outdir / "08_summary.md"
    with summary_path.open("w", encoding="utf-8") as handle:
        subprocess.run(
            python_module("experiments.summarize_path_b_v2", outdir),
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    print(f"PATH_B_V2_VALIDATION_COMPLETE={outdir}")
    print(f"SUMMARY={summary_path}")


if __name__ == "__main__":
    main()
