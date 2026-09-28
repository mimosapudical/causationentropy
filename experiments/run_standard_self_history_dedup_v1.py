"""Validate standard-oCSE self-history candidate deduplication.

Run from the repository root:

    python -m experiments.run_standard_self_history_dedup_v1
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
    outdir = root / "experiments" / "results" / "standard_self_history_dedup_v1"
    outdir.mkdir(parents=True, exist_ok=True)

    targets = [
        "causationentropy/core/discovery.py",
        "causationentropy/tests/test_discovery.py",
        "experiments/run_standard_self_history_dedup_v1.py",
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
            "causationentropy/tests/test_discovery.py",
        ],
        outdir / "04_discovery_tests.txt",
    )
    run(
        [sys.executable, "-m", "pytest", "-q"],
        outdir / "05_full_default_pytest.txt",
    )

    print(f"SELF_HISTORY_DEDUP_VALIDATION_COMPLETE={outdir}")


if __name__ == "__main__":
    main()
