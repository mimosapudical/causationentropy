# Path B v2 diagnostic record — 2026-09-28

This file preserves the diagnostic evidence that motivated the v2 design.
It is **not** an upstream benchmark result and **not** a CI artifact.

## Current v2 design

Path A Information-LASSO endpoint
→ one conditional-CMI rescue pass (no shuffle tests)
→ exact standard oCSE refinement.

The rescue stage admits at least one excluded candidate when any remain and
otherwise targets a fixed candidate-retention budget.

## Reproduced Gaussian screening frontier

Configuration:

- 12 variables
- T = 300
- sparse directed linear-Gaussian systems
- 10 graph seeds × 12 targets
- 100 shuffles for the full-oCSE support reference
- max_lag = 1
- reference = final support returned by full standard oCSE for each target

The later exact-equivalent local reproduction supersedes the earlier provisional
frontier values.

| Screen | Mean candidate retention | Mean full-oCSE support recall |
|---|---:|---:|
| Path-A endpoint | 11.46% | 78.13% |
| v2 rescue, target 10% | 22.15% | 97.65% |
| v2 rescue, target 20% | 27.29% | 99.04% |
| v2 rescue, target 30% | 33.96% | 99.63% |
| v2 rescue, target 40% | 41.81% | 99.79% |
| v2 rescue, target 50% | 50.00% | 99.79% |

The robust conclusion is that the Path-A endpoint is too aggressive for use as
a high-recall screen. A single conditional-CMI rescue pass recovers almost all
of the full-oCSE support while retaining roughly one third of the candidate
space at the 30% target.

## Stress diagnostics

### Hidden-parent construction

A true parent was constructed to have weak/near-zero marginal information but
strong conditional relevance.

Across 30 seeds:

- Path-A endpoint parent recall: ~88.3%
- endpoint + one conditional rescue pass: 100%

This directly exercises the failure mode that motivates conditional screening.

### Correlated-lag construction

A highly autocorrelated source influenced the target at two true lags.

Across the diagnostic seeds:

- Path-A endpoint parent recall: 100%
- endpoint + rescue parent recall: 100%

A separate lag-group mechanism is therefore not justified by this diagnostic
yet.

## Rescue-conditioning ablation

A follow-up checked whether the rescue score should also condition on the
standard-oCSE target-history set Z_init, i.e.

I(X_j; Y | Z_init, X_endpoint)

instead of the simpler

I(X_j; Y | X_endpoint).

On the tested Gaussian seeds this did not improve support coverage: the 30% and
40% retention settings were effectively tied, while the 20% setting was slightly
worse with the extra Z_init conditioning. v2 therefore keeps the simpler
endpoint-conditioned rescue rather than adding another conditioning layer.

## Screen coverage is not final-graph identity

A candidate screen containing a full-oCSE support does **not** guarantee that a
finite-shuffle rerun on the screened set returns exactly the same graph.

The reason is stochastic: full and screened refinement execute different
numbers/orders of permutation tests, so a shared RNG stream advances
differently. This can change borderline finite-permutation decisions even when
the screen contains every variable selected by the full run.

The v2 benchmark therefore reports three quantities separately:

1. `screen_full_support_recall` — whether the screen contains the full-oCSE
   support;
2. `refined_full_support_recall` — whether screened exact refinement recovers
   those support variables under its own permutation sequence;
3. `full_ocse_edge_recall` — final graph overlap.

Do not substitute one of these quantities for another in a paper.

## Output-equivalent Gaussian end-to-end diagnostic

Configuration:

- 12 variables
- T = 300
- p = 0.12
- max_lag = 1
- Path-B target retention = 30%
- 100 shuffles
- 5 graph seeds
- full and Path-B runs use the same per-target RNG initialization rule
- both sides include final edge CMI + shuffle reporting work

Mean results over the five seeds:

| Metric | Result |
|---|---:|
| Actual Path-B candidate retention | ~33.61% |
| Runtime speedup vs optimized full oCSE | ~1.98× |
| Final full-oCSE edge recall | ~97.49% |
| Full-oCSE recall | ~87.23% |
| Path-B recall | ~87.23% |
| Full-oCSE precision | ~70.26% |
| Path-B precision | ~71.39% |
| Full-oCSE F1 | ~0.772 |
| Path-B F1 | ~0.780 |

At 200 and 500 shuffles on the smaller follow-up seed set, the approximately
2× speed advantage persisted and final graph agreement was generally high.
Those runs are diagnostics, not the final scaling table.

The old v1 ~6.45× number must **not** be used as the paper result: that comparison
used a less optimized baseline and did not yet make the output-stage work fully
equivalent.

## Exact baseline optimizations retained in this branch

1. **Observed-CMI reuse inside standard forward selection.**
   When the best candidate fails its shuffle test, the conditioning set is
   unchanged. Remaining observed CMI values are therefore reused instead of
   recomputed. The original repeated-argmax/removal behavior is replayed.

2. **Gaussian conditioning-context reuse.**
   For fixed (Y, Z), Gaussian terms involving only Z and (Y, Z) are cached
   across candidates and shuffle surrogates. X-dependent terms are recomputed.

Exact-equivalent local checks found zero numerical difference between cached and
uncached Gaussian CMI values. The fixed-context inner calculation was roughly
1.5–1.9× faster across smoke runs. This is **not** a full-pipeline speedup claim.

## Nonlinear logistic benchmark audit

The repository default coupled-logistic parameters can become numerically
unstable on the longer benchmark trajectory. In a 20-seed local stability scan
at n=8, p=0.15, T=250, the default-like r=3.99, sigma=0.1 was frequently
non-finite/outside the intended [0,1] interval.

The v2 primary nonlinear benchmark therefore uses:

- r = 3.9
- sigma = 0.01
- n = 8
- T = 250
- p = 0.25

This exact benchmark configuration was rechecked over 20 seeds: all trajectories
were finite and remained in [0,1], with an overall observed range of approximately
0.0026 to 0.9900 and a mean of 14.6 directed truth edges. The benchmark code now
refuses to score a non-finite trajectory.

## Poisson benchmark status

The Poisson generator is retained, but v2 labels it an
`estimator_audit` rather than a primary paper benchmark.

A local formula-level reproduction did not recover useful standard-oCSE edges at
the modest shuffle budgets tested, while the repository integration test expects
high TPR with 1000 shuffles. Because the exact repository integration behavior
has not yet been reproduced in this environment, this discrepancy is treated as
an estimator-validation question rather than evidence for or against Path B.

Do not put Poisson results in the primary benchmark table until that integration
behavior is reproduced exactly.

## Explicitly excluded as duplicate/out of scope

- Target-level parallel discovery already has a separate project implementation,
  so the candidate-level n_jobs experiment was removed from v2.
- KD-tree neighbor search, shuffle early stopping, conditional permutation, and
  multiple-testing corrections already have separate ongoing work.
- The known kNN conditional-MI path is not used as the primary nonlinear
  benchmark.

## Newly observed repository issue not fixed here

Standard oCSE initializes Z with the target's own lagged values while the same
target-lag columns also remain in the candidate matrix. In Gaussian CMI this can
create a singular correlation matrix / NaN and a futile self-lag permutation
test.

This is recorded as a separate follow-up because removing that candidate changes
the exact test/RNG sequence and is broader than the v2 screening experiment.

## Validation status

Completed locally with exact-equivalent reproductions:

- Gaussian cached vs uncached CMI equivalence;
- shuffle decision/p-value equivalence for fixed RNG;
- Gaussian screening frontier;
- hidden-parent and correlated-lag stress diagnostics;
- output-equivalent Gaussian end-to-end diagnostics;
- nonlinear generator stability scan.

Focused regression tests are present in the fork for score reuse, Gaussian
context equivalence, shuffle equivalence, and conditional rescue.

Still not verified:

- full repository pytest;
- Black/isort/flake8 on the actual branch checkout;
- multi-platform GitHub Actions;
- final high-dimensional scaling benchmark with a fixed compute environment.

Do not describe Path B as upstream-ready or paper-final until those checks pass.


## Update: forward-closure preservation is the stronger target

A later diagnostic separated final-parent coverage from preservation of the
full oCSE selection path.

Full oCSE can accept predictors during forward selection that are later removed
by backward elimination. These temporary predictors can act as conditioning
witnesses. Therefore, preserving only the final parent set is not in general
the right screening target for a restricted oCSE run.

The stronger diagnostic target is the **full forward closure**:

```
C_forward = predictors accepted by full oCSE forward selection
```

The Path-B screen should aim for:

```
C_forward ⊆ C_screen
```

rather than only:

```
C_final ⊆ C_screen
```

### Common-random-number (CRN) diagnostic

Candidate restriction changes how many shuffle tests are executed. With one
shared sequential RNG, this also changes which random permutations later tests
receive, even when the same candidate and conditioning set are tested. That
confounds algorithmic path differences with RNG-stream differences.

The new `experiments/path_b_common_random_numbers.py` diagnostic keys the
permutation RNG to:

```
(base seed, target, stage, candidate, conditioning set)
```

so full and restricted oCSE receive identical null permutations for identical
tests.

In the independent local reference runner at N=20, T=350, 3 seeds, 20 shuffles:

| Target retention | Mean actual retention | Mean forward-closure recall | Final-support screen recall | Restricted final recall (CRN) |
|---:|---:|---:|---:|---:|
| 20% | ~22.7% | ~87.8% | ~95.6% | ~93.8% |
| 30% | ~31.4% | ~94.9% | ~98.8% | ~97.5% |
| **40%** | **~40.8%** | **~98.5%** | **100%** | **100%** |
| 50% | ~50.5% | ~99.0% | 100% | 100% |

A larger N=30, T=350, one-seed smoke at 40% retention gave:

- actual retention: ~40.0%
- forward-closure recall: ~99.30%
- final-support screen recall: 100%
- restricted final recall under CRN: 100%

These numbers are still diagnostic, not paper-ready benchmark claims.

### Revised working hypothesis

The paper-level sufficient-condition story should now be framed around
**selection-path preservation**:

1. the screen retains the full forward closure with high probability;
2. identical shared tests use coupled/common random numbers (or enough
   permutations that Monte Carlo noise is negligible);
3. restricted forward selection therefore reproduces the accepted full-oCSE
   path;
4. backward elimination then receives the same accepted set and reproduces the
   same final graph.

This is stronger and more oCSE-specific than generic final-parent sure
screening.

### Revised default operating point

The v2 benchmark default is now 40% candidate retention, not 30%.

The 30% point is still useful on the accuracy/compute frontier, but 40% is the
current conservative point because the independent diagnostic showed exact
restricted-final reproduction under CRN across the tested N=20 seeds and the
N=30 smoke.
