"""Validate standard-oCSE conditioning consistency.

Run from the repository root:

    python -m experiments.run_standard_conditioning_consistency_v1
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
    outdir = (
        root
        / "experiments"
        / "results"
        / "standard_conditioning_consistency_v1"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    targets = [
        "causationentropy/core/discovery.py",
        "causationentropy/tests/test_discovery.py",
        "experiments/standard_backward_z_init_audit.py",
        "experiments/run_standard_conditioning_consistency_v1.py",
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
            "-k",
            (
                "standard_excludes_target_history_from_candidates or "
                "standard_optimal_passes_z_init_to_backward or "
                "backward_conditions_on_fixed_z_init or "
                "standard_final_reporting_keeps_target_history"
            ),
        ],
        outdir / "04_focused_tests.txt",
    )
    run(
        [
            sys.executable,
            "-m",
            "experiments.standard_backward_z_init_audit",
            "--seed",
            "123",
            "--n",
            "500",
            "--n-shuffles",
            "500",
        ],
        outdir / "05_backward_z_init_audit.json",
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
            "standard_gaussian",
        ],
        outdir / "06_standard_gaussian_integration.txt",
    )
    run(
        [sys.executable, "-m", "pytest", "-q"],
        outdir / "07_full_default_pytest.txt",
    )

    print(f"STANDARD_CONDITIONING_VALIDATION_COMPLETE={outdir}")


if __name__ == "__main__":
    main()
