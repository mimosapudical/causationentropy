# Path B v2 diagnostic record — 2026-09-28

This file preserves the local diagnostic evidence that motivated the v2 design.
It is **not** an upstream benchmark result and **not** a CI artifact.

## Current v2 design

Path A Information-LASSO endpoint
→ one conditional-CMI rescue pass (no shuffle tests)
→ exact standard oCSE refinement.

The rescue stage always admits at least one excluded candidate when any remain,
while otherwise targeting a fixed candidate-retention budget.

## Gaussian screening diagnostic

Configuration used for the support-coverage diagnostic:

- 12 variables
- T = 300
- sparse directed linear-Gaussian systems
- 10 graph seeds × 12 targets
- 100 shuffles for the full-oCSE reference
- max_lag = 1

Reference set: the final support returned by full standard oCSE for each target.

| Screen | Mean candidate retention | Mean full-oCSE support recall |
|---|---:|---:|
| Path-A endpoint | ~10.6% | ~85.6% |
| Relaxed path, ~25% retained | ~25.0% | ~94.2% |
| Relaxed path, ~33% retained | ~33.3% | ~97.4% |
| v2 rescue, target 20% | ~26.8% | ~98.9% |
| v2 rescue, target 30% | ~34.1% | ~99.63% |
| v2 rescue, target 40% | ~41.9% | 100% |

Interpretation: the main failure of Path-B v1 was the Path-A endpoint cutoff,
not the exact oCSE refinement. A single conditional rescue pass recovered nearly
all of the full-oCSE support while still discarding roughly two thirds of the
candidate space at the 30% target.

## Stress diagnostics

### Hidden-parent construction

A parent was constructed to have weak/near-zero marginal information but strong
conditional relevance. Across the diagnostic seeds:

- Earlier diagnostic Path-A endpoint parent recall: ~87%
- Independent final smoke rerun of the same failure mode: ~92%
- endpoint + minimum conditional rescue: 100% in both checks

The exact endpoint percentage is therefore not treated as a paper result yet;
the robust observation is that the endpoint can miss hidden parents while the
single rescue pass recovered them in these diagnostics.

This directly tests the failure mode motivating conditional screening.

### Correlated-lag construction

A highly autocorrelated source influenced the target at two true lags.

- Path-A endpoint retained both true lags reliably in the diagnostic.
- A raw relaxed LARS-entry path was not consistently better.

This is why v2 does **not** add a separate lag-group mechanism yet.

## Exact baseline optimizations retained in this branch

1. **Observed-CMI reuse inside standard forward selection.**
   When the best candidate fails its shuffle test, the conditioning set is
   unchanged. Remaining observed CMI values are therefore reused instead of
   recomputed. The original repeated-argmax/removal order is replayed exactly.

2. **Gaussian conditioning-context reuse.**
   For fixed (Y, Z), the Gaussian terms involving only Z and (Y, Z) are cached
   across candidates and shuffle surrogates. The X-dependent terms are still
   recomputed.

A standalone local numerical check over 200 shuffled candidates showed identical
Gaussian CMI values (max absolute difference 0) and roughly 1.5–1.8× speedup for that
inner calculation across local smoke runs. This is not a full-pipeline speedup claim.

## Explicitly excluded as duplicate/out of scope

- Target-level parallel discovery already has a separate implementation in the
  project, so the candidate-level n_jobs experiment was removed from v2.
- KD-tree neighbor search, shuffle early stopping, conditional permutation,
  and multiple-testing corrections already have separate ongoing work.
- The known kNN conditional-MI path is not used as the main nonlinear benchmark
  in v2; the nonlinear logistic benchmark uses KDE.

## Newly observed repository issue not fixed here

Standard oCSE initializes Z with the target's own lagged values while the same
target-lag columns also remain in the candidate matrix. In Gaussian CMI this can
produce singular correlation matrices / NaNs and performs redundant tests.
This is recorded as a separate follow-up because changing candidate semantics is
broader than the v2 screening experiment.

## Validation status

- Mathematical equivalence of Gaussian cached vs uncached inner CMI: checked locally.
- Screening frontier / stress diagnostics: checked locally.
- Focused regression tests were added to the fork for score reuse, Gaussian
  context equivalence, shuffle equivalence, and conditional rescue.
- Full repository pytest / formatting / multi-platform CI has **not** been
  verified in this environment because the fork produced no Actions workflow
  run and the local container cannot clone the connected repository.

Do not use the timing numbers in a paper until full repository CI and the
output-equivalent end-to-end benchmark have been run on a fixed machine.
