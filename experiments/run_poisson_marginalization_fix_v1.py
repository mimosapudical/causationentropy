"""Validate the Poisson conditional-marginalization correction.

Run from the repository root:

    python -m experiments.run_poisson_marginalization_fix_v1

The validation deliberately uses the repository's existing Poisson integration
tests instead of inventing a replacement benchmark.
"""

import subprocess
import sys
from pathlib import Path


def run(command, log_path):
    print("+", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )


def main():
    root = Path(__file__).resolve().parents[1]
    outdir = root / "experiments" / "results" / "poisson_marginalization_fix_v1"
    outdir.mkdir(parents=True, exist_ok=True)

    targets = [
        (
            "causationentropy/core/information/"
            "conditional_mutual_information.py"
        ),
        (
            "causationentropy/tests/core/information/"
            "test_conditional_mutual_information.py"
        ),
        "experiments/run_poisson_marginalization_fix_v1.py",
    ]
    run(
        [sys.executable, "-m", "compileall", "-q", *targets],
        outdir / "01_compileall.txt",
    )
    run(
        [
            sys.executable,
            "-m",
            "flake8",
            "--select=E9,F63,F7,F82",
            *targets,
        ],
        outdir / "02_flake8.txt",
    )

    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            (
                "causationentropy/tests/core/information/"
                "test_conditional_mutual_information.py"
            ),
            "-k",
            "poisson",
        ],
        outdir / "04_poisson_unit_tests.txt",
    )

    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "causationentropy/tests/test_data_integration.py",
            "-m",
            "integration",
            "-k",
            "standard_poisson or alternative_poisson",
        ],
        outdir / "05_poisson_integration_tests.txt",
    )

    print(f"POISSON_FIX_VALIDATION_COMPLETE={outdir}")


if __name__ == "__main__":
    main()
