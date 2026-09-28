# Standard oCSE self-history candidate dedup v1

Branch:

    experiment/standard-self-history-dedup-v1

## Defect

For target i, standard discover_network already puts

    X_i(t-1), ..., X_i(t-max_lag)

into Z_init.

The same columns also remain in X_lagged and are passed to standard forward
selection as candidate predictors. This creates redundant tests of the form

    I(X_i(t-lag); Y_i(t) | ..., X_i(t-lag), ...).

For correlation-based estimators the duplicated conditioning column can make
the correlation matrix singular.

The repository's own discovery test already states that a one-variable
standard run should produce no self-loops.

## Minimal correction

Only standard mode changes:

1. build the full lagged design exactly as before;
2. for target i, remove all candidate columns whose source variable is i;
3. run standard oCSE on the reduced candidate matrix;
4. map selected local indices back to the global lagged-feature indices.

Alternative oCSE, LASSO, and Information-LASSO retain the original candidate
universe.

Report-all mode also iterates only over candidates that standard oCSE actually
tested.

## Source-equivalent diagnostic

For independent Gaussian data with

    n = 5
    T = 200
    max_lag = 2

all 10 duplicated target-history candidates produced non-finite Gaussian CMI:

    duplicate nonfinite CMI = 10 / 10

Those candidates are already represented identically in Z_init and therefore
cannot contribute a new conditioning direction.

## RNG note

Removing candidates also removes their observed-CMI and permutation-test work.
With a shared sequential RNG, later finite-shuffle draws therefore change.

This branch does NOT claim bit-identical legacy graphs. The intended invariant
is the statistical candidate universe: columns already conditioned on through
Z_init are no longer tested as new causes.

## Validation

Run:

    pip install -e ".[dev]"
    python -m experiments.run_standard_self_history_dedup_v1

Status:

    SOURCE_EQUIVALENT_DUPLICATE_AUDIT = PASS
    PACKAGE_PYTEST = NOT_VERIFIED

Do not open an upstream PR until discovery and full default tests pass.
