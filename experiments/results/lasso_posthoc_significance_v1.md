# Sparse-method post-hoc significance v1

Branch:

    experiment/lasso-posthoc-significance-v1

## Problem

discover_network computes a final CMI + shuffle test for every selected edge.

For standard and alternative oCSE, the selected set S already passed the
algorithm's forward/backward permutation tests.

For LASSO and Information-LASSO, S is only coefficient support. It has not
passed a significance test before the final reporting loop.

PR #36 introduced only_return_significant=False while explicitly preserving the
default standard/alternative selected-set behavior. Therefore applying a second
random gate to standard/alternative would be incorrect, but treating every
LASSO support coefficient as significant is also inconsistent.

## Minimal correction

Define edge significance as:

    standard / alternative:
        significant because S already passed oCSE testing

    lasso / information_lasso:
        significant iff final shuffle_test["Pass"] is True

With only_return_significant=True, failed sparse supports are omitted.

With only_return_significant=False, failed sparse supports remain in the graph
with significant=False, which is useful for delay analysis.

## Regression coverage

Focused tests lock four cases:

1. sparse support + final Pass=False -> omitted by default;
2. sparse support + final Pass=True -> retained;
3. standard selected edge + diagnostic Pass=False -> retained, avoiding a
   second stochastic gate;
4. report-all sparse support + Pass=False -> present with significant=False.

## Validation

Run:

    pip install -e ".[dev]"
    python -m experiments.run_lasso_posthoc_significance_v1

Status:

    SEMANTIC_AUDIT = PASS
    PACKAGE_PYTEST = NOT_VERIFIED

This branch is independent of the Path-A implementation PR itself.
