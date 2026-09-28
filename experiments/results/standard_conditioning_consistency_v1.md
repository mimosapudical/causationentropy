# Standard oCSE conditioning consistency v1

Branch:

    experiment/standard-conditioning-consistency-v1

This branch is intentionally layered on top of:

    experiment/standard-self-history-dedup-v1

## Contract being tested

The public documentation defines standard oCSE as using lagged target history
as an initial conditioning set. Once that fixed baseline is introduced, the
same baseline should remain present whenever standard oCSE evaluates a
candidate edge.

Before this branch:

- forward selection conditions on Z_init;
- backward elimination drops Z_init;
- final edge CMI/p-value reporting drops Z_init;
- the target-history columns are also redundantly present as candidates.

This makes the statistical question change between stages.

## Changes

1. Keep the self-history candidate de-duplication from the base branch.
2. Extend backward(..., Z_init=None).
3. Standard oCSE passes Z_init to backward.
4. Alternative oCSE leaves Z_init=None and preserves legacy behavior.
5. Final selected-edge reporting in standard mode conditions on:
   Z_init + all other selected predictors.
6. Report-all mode uses:
   Z_init + selected predictors.

## Why this is separate from the small de-dup fix

Removing a candidate already present in Z_init is a narrow candidate-universe
cleanup. Preserving Z_init through backward/reporting changes the conditioning
contract of standard oCSE and can therefore alter graph accuracy.

It must be benchmarked independently.

## Theory reference

The oCSE discovery/removal framework evaluates each candidate conditional on
the currently retained conditioning set. In this repository, standard mode
explicitly augments that set with target history during forward selection.
This branch tests the natural invariant that the fixed baseline remains part of
the conditioning set rather than disappearing between phases.

## Validation

Run:

    pip install -e ".[dev]"
    python -m experiments.run_standard_conditioning_consistency_v1

The runner executes:

- Black / isort / flake8 hard gates;
- focused conditioning-contract regressions;
- the repository's existing standard-Gaussian integration test;
- full default pytest.

Do not propose upstream until all four layers pass.
