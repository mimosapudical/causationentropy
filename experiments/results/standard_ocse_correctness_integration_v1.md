# Standard-oCSE correctness integration v1

Branch:

    experiment/standard-ocse-correctness-integration-v1

This fork-only branch combines two independently reviewable fixes:

1. exclude target-history columns from the standard candidate universe when the
   same columns are already present in Z_init;
2. retain Z_init throughout standard backward elimination and final edge
   CMI/p-value reporting.

It exists to test interaction between the fixes. It is not intended to replace
the two smaller review branches.

## Source-equivalent 1000-shuffle Gaussian integration

Repository configuration:

    n = 5
    T = 200
    p = 0.2
    rho = 0.7
    seed = 42
    max_lag = 1
    alpha_forward = 0.05
    alpha_backward = 0.05
    n_shuffles = 1000
    random_state = 42

The Gaussian CMI permutation loop was evaluated with the mathematically
equivalent scalar partial-correlation identity. Before the benchmark, that
identity was compared with the repository's correlation-logdet Gaussian CMI
on random conditioning dimensions 0/1/2/4:

    maximum absolute CMI difference = 2.29e-16

The batched permutation calculation was also compared with scalar evaluation:

    maximum absolute shuffle-CMI difference = 6.25e-17

Results:

    true edges      = 6
    predicted edges = 6
    TP              = 6
    FP              = 0
    TPR             = 1.000
    FPR             = 0.000

Predicted directed lag-1 edges exactly matched the ground truth:

    3 -> 0
    3 -> 1
    2 -> 1
    0 -> 2
    4 -> 3
    1 -> 4

## Compute implication of self-history dedup

For standard oCSE, every target contributes max_lag columns that are already
present identically in Z_init.

The earlier source-equivalent audit found all 10/10 duplicate columns non-finite
for n=5 and max_lag=2.

If each duplicate reaches one permutation test, the avoidable null-CMI work is

    n * max_lag * n_shuffles.

At n=5, max_lag=2, and 1000 shuffles, that is 10,000 futile shuffle-CMI
evaluations before counting the redundant observed CMI evaluations.

## Status

    COMBINED_GAUSSIAN_PROXY = PASS
    PACKAGE_PYTEST = NOT_VERIFIED

The independent review branches remain:

    experiment/standard-self-history-dedup-v1
    experiment/standard-backward-zinit-fix-v1
