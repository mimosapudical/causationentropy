"""Run all fork-only causationentropy validation branches.

Start from a clean tracked worktree on experiment/path-b-benchmark-v2:

    python -m experiments.run_all_fork_validations

Optional dependency installation:

    python -m experiments.run_all_fork_validations --install

Run the full Path-B scaling suite instead of quick:

    python -m experiments.run_all_fork_validations --path-b-full
"""

import argparse
import subprocess
import sys


VALIDATIONS = [
    (
        "Poisson conditional marginalization",
        "experiment/poisson-marginalization-fix-v1",
        ["-m", "experiments.run_poisson_marginalization_fix_v1"],
    ),
    (
        "Standard backward Z_init",
        "experiment/standard-backward-zinit-fix-v1",
        ["-m", "experiments.run_standard_backward_zinit_fix_v1"],
    ),
    (
        "Gaussian constant feature",
        "experiment/gaussian-constant-feature-fix-v1",
        ["-m", "experiments.run_gaussian_constant_feature_fix_v1"],
    ),
    (
        "Standard self-history dedup",
        "experiment/standard-self-history-dedup-v1",
        ["-m", "experiments.run_standard_self_history_dedup_v1"],
    ),
    (
        "Sparse post-hoc significance",
        "experiment/lasso-posthoc-significance-v1",
        ["-m", "experiments.run_lasso_posthoc_significance_v1"],
    ),
]


def run(command, check=True):
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=check)


def current_branch():
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def require_clean_tracked_worktree():
    for command in (
        ["git", "diff", "--quiet"],
        ["git", "diff", "--cached", "--quiet"],
    ):
        result = subprocess.run(command)
        if result.returncode != 0:
            raise SystemExit(
                "Tracked worktree changes detected. Commit or stash them "
                "before cross-branch validation."
            )


def switch_branch(branch):
    local = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"]
    )
    if local.returncode == 0:
        run(["git", "switch", branch])
    else:
        run(["git", "switch", "--track", f"origin/{branch}"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install the repository dev dependencies before validation.",
    )
    parser.add_argument(
        "--path-b-full",
        action="store_true",
        help="Run Path B in full scaling mode instead of quick mode.",
    )
    args = parser.parse_args()

    require_clean_tracked_worktree()
    start_branch = current_branch()
    if not start_branch:
        raise SystemExit("Detached HEAD is not supported by this orchestrator.")

    if args.install:
        run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])

    run(["git", "fetch", "origin"])

    results = []
    try:
        for label, branch, command in VALIDATIONS:
            print()
            print("=" * 72)
            print(label)
            print("=" * 72)
            try:
                switch_branch(branch)
                result = run([sys.executable, *command], check=False)
                passed = result.returncode == 0
            except subprocess.CalledProcessError:
                passed = False
            results.append((label, branch, passed))

        print()
        print("=" * 72)
        print("Path B validation")
        print("=" * 72)
        switch_branch("experiment/path-b-benchmark-v2")
        mode = "full" if args.path_b_full else "quick"
        result = run(
            [
                sys.executable,
                "-m",
                "experiments.run_path_b_v2",
                "--mode",
                mode,
            ],
            check=False,
        )
        results.append(
            (
                f"Path B {mode}",
                "experiment/path-b-benchmark-v2",
                result.returncode == 0,
            )
        )
    finally:
        print()
        print(f"Restoring starting branch: {start_branch}")
        switch_branch(start_branch)

    print()
    print("Validation summary")
    print("-" * 72)
    for label, branch, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{status:4}  {label}  [{branch}]")

    if not all(passed for _, _, passed in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
