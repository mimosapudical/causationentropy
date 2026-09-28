"""Validate standard-oCSE backward preservation of Z_init.

Run from the repository root:

    python -m experiments.run_standard_backward_zinit_fix_v1
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
    outdir = root / "experiments" / "results" / "standard_backward_zinit_fix_v1"
    outdir.mkdir(parents=True, exist_ok=True)

    targets = [
        "causationentropy/core/discovery.py",
        "causationentropy/tests/test_integration.py",
        "experiments/run_standard_backward_zinit_fix_v1.py",
    ]

    run(
        [sys.executable, "-m", "black", "--check", *targets],
        outdir / "01_black.txt",
    )
    run(
        [sys.executable, "-m", "isort", "--check-only", *targets],
        outdir / "02_isort.txt",
    )
    run(
        [
            sys.executable,
            "-m",
            "flake8",
            "--select=E9,F63,F7,F82",
            *targets,
        ],
        outdir / "03_flake8.txt",
    )
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "causationentropy/tests/test_integration.py",
        ],
        outdir / "04_ocse_integration_tests.txt",
    )
    run(
        [sys.executable, "-m", "pytest", "-q"],
        outdir / "05_full_default_pytest.txt",
    )

    print(f"BACKWARD_ZINIT_VALIDATION_COMPLETE={outdir}")


if __name__ == "__main__":
    main()
