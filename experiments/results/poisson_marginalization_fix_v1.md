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


## Source-equivalent 1000-shuffle integration proxy

Because this environment cannot execute a GitHub checkout, the repository's
Poisson generator, forward/backward oCSE logic, sequential RNG behavior, and
corrected Poisson CMI were reconstructed directly from the branch source.

The only acceleration used in the permutation loop was algebraically
equivalent batching:

- Pearson correlations for 100 shuffled candidates were compared against
  individual np.corrcoef calls;
- maximum absolute difference in CMI was 1.24e-14;
- Poisson PMFs were evaluated by the exact recurrence
  p_k = p_(k-1) * lambda / k rather than repeated scipy.stats object calls;
- the recurrence matched the repository poisson_entropy implementation to
  4.44e-16 on representative rates.

The repository's own integration configuration was then reproduced:

    n = 5
    T = 200
    p = 0.2
    seed = 42
    max_lag = 1
    random_state = 42
    n_shuffles = 1000

Results:

| Method | TP | FP | TPR | FPR | Repository gate |
|---|---:|---:|---:|---:|---|
| standard Poisson | 6 / 6 | 1 | 1.000 | 0.07143 | PASS |
| alternative Poisson | 6 / 6 | 1 | 1.000 | 0.07143 | PASS |

Both satisfy the existing test_data_integration.py requirements:

    TPR >= 0.95
    FPR <= 0.10

The selected directed lag-1 edges were identical for standard and alternative
in this proxy:

    3 -> 0
    3 -> 1
    2 -> 1
    0 -> 2
    4 -> 3
    1 -> 4
    0 -> 4

Six are true edges and one is a false positive.

Status remains:

    PACKAGE_PYTEST = NOT_VERIFIED

The actual branch must still pass:

    python -m experiments.run_poisson_marginalization_fix_v1

before an upstream PR is opened.


## Finite-sample non-negativity

The direct Poisson MI/CMI estimator now applies the same finite-value
non-negativity clamp as the public conditional_mutual_information dispatcher.
This keeps direct and dispatcher calls consistent when sampling noise produces
a small negative estimate (for example about -0.0041 in the repository unit
test) while preserving NaN/inf for error handling.
