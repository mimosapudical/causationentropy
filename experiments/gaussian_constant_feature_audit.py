"""Audit Gaussian MI behavior for constant and near-constant features.

The current Gaussian MI path uses correlation_log_determinant.  Singular or
non-finite joint correlation matrices are mapped to a fixed log-determinant of
-1000, while a scalar correlation matrix returns 0.  For a constant scalar X
and random scalar Y, this can therefore yield an artificial MI near 500 nats.

This diagnostic does not change the estimator.
"""

import argparse
import json

import numpy as np

from causationentropy.core.information.mutual_information import (
    gaussian_mutual_information,
)


def run_audit(seed=0, n_samples=500):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=(n_samples, 1))
    x_constant = np.ones((n_samples, 1))
    x_near_constant = 1.0 + 1e-12 * rng.normal(size=(n_samples, 1))
    x_independent = rng.normal(size=(n_samples, 1))
    x_identical = y.copy()

    cases = {
        "constant_vs_random": (x_constant, y),
        "near_constant_vs_random": (x_near_constant, y),
        "independent_random": (x_independent, y),
        "identical": (x_identical, y),
    }

    results = {}
    for name, (x, target) in cases.items():
        with np.errstate(all="ignore"):
            value = gaussian_mutual_information(x, target)
        results[name] = {
            "mi": float(value),
            "x_std": float(np.std(x)),
            "y_std": float(np.std(target)),
            "finite": bool(np.isfinite(value)),
        }

    return {
        "cases": results,
        "diagnostic": {
            "constant_feature_should_not_rank_as_informative": True,
            "implementation_changed": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=500)
    args = parser.parse_args()
    print(
        json.dumps(
            run_audit(seed=args.seed, n_samples=args.n_samples),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
