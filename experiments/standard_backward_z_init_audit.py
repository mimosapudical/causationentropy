"""Minimal audit for dropping Z_init during standard backward elimination.

Construct:

    Z ~ N(0, 1)
    X = Z + 0.2 eps_x
    Y = Z + 0.2 eps_y

X and Y are strongly associated marginally but conditionally independent given
Z. A backward phase that drops the fixed target-history baseline therefore
keeps X, while the same test with Z_init should remove it.

This script is diagnostic and does not modify discovery code.
"""

import argparse
import json

import numpy as np

from causationentropy.core.information.conditional_mutual_information import (
    gaussian_conditional_mutual_information,
)
from causationentropy.core.information.mutual_information import (
    gaussian_mutual_information,
)


def permutation_threshold(X, Y, Z, seed, n_shuffles, alpha=0.05):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_shuffles):
        X_perm = X[rng.permutation(len(X)), :]
        if Z is None:
            value = gaussian_mutual_information(X_perm, Y)
        else:
            value = gaussian_conditional_mutual_information(
                X_perm,
                Y,
                Z,
            )
        values.append(value)
    return float(np.percentile(values, 100 * (1 - alpha)))


def run_audit(seed=123, n=500, n_shuffles=500):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 1))
    X = Z + 0.2 * rng.normal(size=(n, 1))
    Y = Z + 0.2 * rng.normal(size=(n, 1))

    marginal = gaussian_mutual_information(X, Y)
    conditional = gaussian_conditional_mutual_information(X, Y, Z)

    marginal_threshold = permutation_threshold(
        X,
        Y,
        None,
        seed + 1000,
        n_shuffles,
    )
    conditional_threshold = permutation_threshold(
        X,
        Y,
        Z,
        seed + 2000,
        n_shuffles,
    )

    return {
        "marginal": {
            "observed": float(marginal),
            "threshold": marginal_threshold,
            "pass": bool(marginal >= marginal_threshold),
        },
        "conditional_on_z_init": {
            "observed": float(conditional),
            "threshold": conditional_threshold,
            "pass": bool(conditional >= conditional_threshold),
        },
        "diagnostic": {
            "expected": (
                "Marginal test passes but conditional-on-Z_init test fails."
            ),
            "implementation_changed": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--n-shuffles", type=int, default=500)
    args = parser.parse_args()

    print(
        json.dumps(
            run_audit(
                seed=args.seed,
                n=args.n,
                n_shuffles=args.n_shuffles,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
