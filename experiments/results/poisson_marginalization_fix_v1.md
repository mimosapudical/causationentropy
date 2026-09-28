# Poisson conditional-marginalization fix v1

Branch:

    experiment/poisson-marginalization-fix-v1

## What the paper actually says

Fish, Sun & Bollt (Applied Network Science, 2022) use two related ideas:

1. the multivariate-Poisson covariance structure identifies shared component
   rates in the underlying model;
2. for their practical network experiments, they intentionally estimate those
   rates from pairwise correlations instead of raw covariances, so the surrogate
   rates remain in the small-rate regime where the entropy approximation is
   accurate.

Therefore this branch does **not** replace correlation with covariance.

Their Eq. (46) reconstructs the private rate as the diagonal entry minus the
sum of all off-diagonal shared rates. Their Eq. (38) computes conditional
mutual information using proper Poisson marginals:

    I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z).

## Repository defect targeted here

The current conditional implementation:

- performs a diagonal update without summing all shared rates;
- manually adjusts only selected X/Y blocks when marginalizing;
- does not consistently absorb dropped-variable shared components into every
  retained variable's private rate;
- can broadcast-fail for multivariate X/Y partitions.

## Formula-level smoke

On a shared-Poisson construction with X-Y, X-Z, and Y-Z latent shared
components:

- legacy raw CMI: about -0.2407;
- legacy dispatcher result after non-negativity clamp: 0;
- Eq. (38)-consistent implementation: about +0.1169.

For X with 2 features, Y with 3 features, and Z with 2 features:

- legacy conditional code: broadcasting ValueError;
- corrected implementation: finite;
- corrected X/Y swap absolute difference: approximately machine precision.

For the fixed correlation matrix

    [[1.00, 0.10, 0.05],
     [0.10, 1.00, 0.08],
     [0.05, 0.08, 1.00]]

Eq. (46) gives

    [[0.85, 0.10, 0.05],
     [0.10, 0.82, 0.08],
     [0.05, 0.08, 0.87]].

## Repository acceptance gate

The existing integration suite requires both Poisson discovery variants on its
fixed 5-node, T=200, 1000-shuffle benchmark to satisfy:

- TPR >= 0.95;
- FPR <= 0.10.

Run:

    pip install -e ".[dev]"
    python -m experiments.run_poisson_marginalization_fix_v1

Do not propose this upstream until those existing integration tests pass.
