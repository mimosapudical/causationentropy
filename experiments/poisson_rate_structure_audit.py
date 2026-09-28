"""Audit the rate structure used by the current Poisson information estimator.

The current Poisson MI/CMI code starts from np.corrcoef and then feeds the
result into poisson_joint_entropy, whose diagonal is interpreted as Poisson
rate/variance parameters. Correlation normalization forces those diagonals to
one, so this experiment makes the loss of marginal-rate information explicit.

This is a diagnostic only. It does not propose a replacement estimator.

Reference
---------
Fish, Sun & Bollt, Applied Network Science (2022),
doi:10.1007/s41109-022-00510-x.
"""

import argparse
import json

import numpy as np


def current_implied_rates(data):
    """Reproduce the marginal-rate vector implied by the current MI code."""
    corr = np.corrcoef(np.asarray(data).T)
    off_diag = corr - np.diag(np.diag(corr))
    adjusted = corr.copy()
    np.fill_diagonal(
        adjusted,
        np.diagonal(adjusted) - np.sum(off_diag, axis=0),
    )
    implied = np.diag(adjusted) + np.sum(off_diag, axis=0)
    return implied


def shared_poisson_sample(
    seed,
    n_samples=5000,
    x_private=2.0,
    y_private=7.0,
    shared=3.0,
):
    rng = np.random.default_rng(seed)
    common = rng.poisson(shared, size=n_samples)
    x = rng.poisson(x_private, size=n_samples) + common
    y = rng.poisson(y_private, size=n_samples) + common
    return np.column_stack([x, y])


def summarize(data):
    data = np.asarray(data)
    covariance = np.cov(data.T, ddof=1)
    correlation = np.corrcoef(data.T)
    return {
        "empirical_means": data.mean(axis=0).tolist(),
        "empirical_variances": data.var(axis=0, ddof=1).tolist(),
        "empirical_covariance": covariance.tolist(),
        "empirical_correlation": correlation.tolist(),
        "current_corrcoef_implied_rates": current_implied_rates(data).tolist(),
    }


def run_audit(seed=0, n_samples=5000):
    low = shared_poisson_sample(
        seed,
        n_samples=n_samples,
        x_private=2.0,
        y_private=7.0,
        shared=3.0,
    )
    high = shared_poisson_sample(
        seed + 1,
        n_samples=n_samples,
        x_private=20.0,
        y_private=70.0,
        shared=30.0,
    )

    return {
        "low_rate_system": summarize(low),
        "high_rate_system": summarize(high),
        "diagnostic": {
            "expected_behavior": (
                "Poisson rate/variance structure should retain marginal count scale."
            ),
            "current_behavior": (
                "The corrcoef-based construction yields unit marginal rates "
                "independent of the observed count scale."
            ),
            "estimator_changed": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=5000)
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
