"""Audit the conditional Poisson marginalization used in the current estimator.

The Fish-Sun-Bollt estimator intentionally uses pairwise correlations as scaled
surrogates for shared Poisson rates. The defect audited here is different:

1. Eq. (46) requires each private rate to be the correlation diagonal minus the
   SUM of all off-diagonal shared rates for that variable.
2. Eq. (38) requires proper Poisson marginals for H(X,Z), H(Y,Z), and H(Z).

The current conditional implementation does not perform the diagonal reduction
correctly and has incomplete marginalization bookkeeping. This diagnostic
compares that implementation algebra with the direct entropy decomposition
required by Eq. (38).

No core estimator is modified by this file.
"""

import argparse
import json

import numpy as np

from causationentropy.core.information.entropy import poisson_joint_entropy


def rate_matrix_from_correlation(correlation):
    """Paper Eq. (46): shared off-diagonals plus recovered private diagonal."""
    correlation = np.atleast_2d(np.asarray(correlation, dtype=float))
    shared = correlation - np.diag(np.diag(correlation))
    rates = shared.copy()
    private = np.diag(correlation) - np.sum(shared, axis=1)
    np.fill_diagonal(rates, private)
    return rates


def joint_entropy_from_samples(samples):
    correlation = np.corrcoef(np.asarray(samples), rowvar=False)
    return poisson_joint_entropy(rate_matrix_from_correlation(correlation))


def paper_consistent_cmi(X, Y, Z=None):
    """Eq. (38) using proper marginals estimated from each retained set."""
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    if Z is None:
        return (
            joint_entropy_from_samples(X)
            + joint_entropy_from_samples(Y)
            - joint_entropy_from_samples(np.hstack((X, Y)))
        )

    Z = np.atleast_2d(Z)
    return (
        joint_entropy_from_samples(np.hstack((X, Z)))
        + joint_entropy_from_samples(np.hstack((Y, Z)))
        - joint_entropy_from_samples(np.hstack((X, Y, Z)))
        - joint_entropy_from_samples(Z)
    )


def current_conditional_algebra(X, Y, Z):
    """Reproduce the current repository conditional-Poisson implementation."""
    SzX = X.shape[1]
    SzY = Y.shape[1]
    SzZ = Z.shape[1]
    indX = np.arange(SzX)
    indY = np.arange(SzY) + SzX
    indZ = np.arange(SzZ) + SzX + SzY
    XYZ = np.concatenate((X, Y, Z), axis=1)
    SXYZ = np.corrcoef(XYZ.T)
    SS = SXYZ
    Sa = SXYZ - np.diag(np.diag(SXYZ))
    np.fill_diagonal(SS, np.diagonal(SS) - Sa)
    SS[0:SzX, 0:SzX] = (
        SS[0:SzX, 0:SzX] + SXYZ[0:SzX, SzX : SzX + SzY]
    )
    SS[SzX : SzX + SzY, SzX : SzX + SzY] = (
        SS[SzX : SzX + SzY, SzX : SzX + SzY]
        + SXYZ[SzX : SzX + SzY, 0:SzX]
    )
    yz_idx = np.concatenate((indY, indZ))
    xz_idx = np.concatenate((indX, indZ))
    S_est1 = SS[np.ix_(yz_idx, yz_idx)]
    S_est2 = SS[np.ix_(xz_idx, xz_idx)]
    HYZ = poisson_joint_entropy(S_est1)
    HZ = poisson_joint_entropy(SS[np.ix_(indZ, indZ)])
    HXYZ = poisson_joint_entropy(SXYZ - np.diag(Sa))
    HXZ = poisson_joint_entropy(S_est2)
    return (HYZ - HZ) - (HXYZ - HXZ)


def shared_poisson_example(seed=123, n=2000):
    rng = np.random.default_rng(seed)
    shared_xy = rng.poisson(0.3, size=n)
    shared_xz = rng.poisson(0.2, size=n)
    shared_yz = rng.poisson(0.1, size=n)
    X = (
        rng.poisson(0.8, size=n) + shared_xy + shared_xz
    ).reshape(-1, 1)
    Y = (
        rng.poisson(0.9, size=n) + shared_xy + shared_yz
    ).reshape(-1, 1)
    Z = (
        rng.poisson(1.0, size=n) + shared_xz + shared_yz
    ).reshape(-1, 1)
    return X, Y, Z


def run_audit(seed=123, n=2000):
    X, Y, Z = shared_poisson_example(seed=seed, n=n)
    current = current_conditional_algebra(X, Y, Z)
    corrected = paper_consistent_cmi(X, Y, Z)

    rng = np.random.default_rng(seed + 1)
    X_multi = rng.poisson(1.0, size=(1000, 2)).astype(float)
    Y_multi = rng.poisson(1.2, size=(1000, 3)).astype(float)
    Z_multi = rng.poisson(0.8, size=(1000, 2)).astype(float)

    try:
        current_multi = current_conditional_algebra(
            X_multi, Y_multi, Z_multi
        )
        current_multi_status = {
            "status": "ok",
            "value": float(current_multi),
        }
    except Exception as exc:
        current_multi_status = {
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }

    corrected_multi_xy = paper_consistent_cmi(
        X_multi, Y_multi, Z_multi
    )
    corrected_multi_yx = paper_consistent_cmi(
        Y_multi, X_multi, Z_multi
    )

    known_correlation = np.array(
        [
            [1.0, 0.10, 0.05],
            [0.10, 1.0, 0.08],
            [0.05, 0.08, 1.0],
        ]
    )

    return {
        "scalar_shared_poisson": {
            "current_raw_cmi": float(current),
            "current_dispatcher_value": float(max(0.0, current)),
            "paper_consistent_cmi": float(corrected),
        },
        "multivariate_partition": {
            "current": current_multi_status,
            "paper_consistent_xy": float(corrected_multi_xy),
            "paper_consistent_yx": float(corrected_multi_yx),
            "symmetry_abs_error": float(
                abs(corrected_multi_xy - corrected_multi_yx)
            ),
        },
        "eq46_rate_reconstruction": {
            "input_correlation": known_correlation.tolist(),
            "rate_matrix": rate_matrix_from_correlation(
                known_correlation
            ).tolist(),
        },
        "diagnostic": {
            "correlation_scaling_is_intentional": True,
            "conditional_marginalization_is_target": True,
            "implementation_changed": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n", type=int, default=2000)
    args = parser.parse_args()
    print(
        json.dumps(
            run_audit(seed=args.seed, n=args.n),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
