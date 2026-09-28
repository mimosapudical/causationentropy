# Standard backward Z_init fix v1

Branch:

    experiment/standard-backward-zinit-fix-v1

## Defect

The repository's standard-oCSE documentation defines backward elimination with

    Z_-j = Z_init,i union (S_i \ {X_j}).

The implementation currently calls backward without Z_init, so the backward
phase tests only against the other selected predictors.

That changes the question being asked:

- forward: does X add information beyond the target's autoregressive history?
- legacy backward: does X remain informative when the target history is removed?

Those are not equivalent.

## Minimal correction

backward gains one optional parameter appended at the end:

    Z_init=None

For each selected j it now constructs:

    Z = Z_init + all other currently retained predictors

when Z_init is provided.

standard_optimal_causation_entropy passes its Z_init.
alternative_optimal_causation_entropy keeps the default None and is unchanged.

The appended optional parameter preserves existing positional callers.

## Repository documentation evidence

docs/source/theory/methods/standard_oce.rst explicitly defines backward
conditioning as:

    Z_-j = Z_init,i union (S_i \ {X_j}).

The same document states that standard oCSE focuses on information supplied
beyond the target's autoregressive history.

## Source-equivalent suppression example

A Gaussian construction was used where:

    Z ~ N(0, 10^2)
    X = Z + e
    Y = e + 0.2 * noise

with 500 samples.

X is strongly informative about Y once Z is controlled, but their marginal
association is deliberately weak.

At alpha=0.01 with 1000 permutations:

| Backward conditioning | MI / CMI | Threshold | Decision |
|---|---:|---:|---|
| legacy, without Z_init | 0.00555 | 0.00642 | FAIL |
| standard definition, with Z_init | 1.62106 | 0.00642 | PASS |

So the legacy backward phase can delete a predictor that the standard forward
phase correctly selected specifically because it adds information beyond the
initial target history.

## Validation

Run:

    pip install -e ".[dev]"
    python -m experiments.run_standard_backward_zinit_fix_v1

Status:

    DOCUMENTATION_CONSISTENCY = PASS
    SOURCE_EQUIVALENT_SUPPRESSION_CASE = PASS
    PACKAGE_PYTEST = NOT_VERIFIED

Do not open an upstream PR until the oCSE integration and full default suites
pass.
