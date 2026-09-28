# Repository-wide audit — 2026-09-28

This note records concrete optimization/correctness findings for
Center-For-Complex-Systems-Science/causationentropy and separates them from
work that already exists upstream.  It is a fork-only planning artifact.

## 1. Existing upstream work: do not duplicate

The following substantial directions already have dedicated upstream work.

| Area | Existing work | What it covers | Our action |
|---|---|---|---|
| kNN neighbor search | PR #12 | Optional KD-tree path instead of full pairwise cdist | Do not reimplement |
| Conditional permutation | PR #33 | Z-local shuffle surrogates and finite-permutation p-value handling | Do not fold into Path B |
| Target-level parallelism | PR #28 / issue #17 | Makes n_jobs control parallel target discovery | Our duplicate candidate-parallel experiment was removed |
| Multiple testing | PR #37 / issue #10 | Bonferroni/BH/BY/adaptive reporting | Keep separate |
| Shuffle futility stopping | PR #41 / issue #7 | Exact early rejection of candidates that can no longer pass | Keep separate |
| KDE / Poisson sign cleanup | PR #42 | KDE log-space fix, ndarray cleanup, Poisson conditional-MI sign | Do not duplicate |
| kNN estimator correctness | issue #14 | Complete-graph failure and p/k conflation | Avoid kNN as primary Path-B nonlinear baseline until resolved |

This means generic “parallelize it,” “use KD-tree,” “stop shuffles early,”
“add multiple testing,” and “fix the Poisson sign” are not new contribution
opportunities for us.

## 2. New high-value findings from this audit

### A. Poisson conditional marginalization

The first Poisson audit initially suspected that using np.corrcoef instead of
covariance was a bug. Reading the full Fish-Sun-Bollt paper corrected that
interpretation.

The paper explicitly states that its network experiments intentionally estimate
the shared rates from pairwise correlations rather than covariances, because
correlations keep the surrogate rates small enough for the joint-entropy
approximation. Their Eq. (46) then recovers the private rate with

    lambda_ii = e_ii - sum_{j != i} lambda_ij.

Therefore, the unit diagonal produced by correlation scaling is intentional and
is NOT itself a correctness defect.

The actual defect is in the conditional branch:

1. the current diagonal update uses

       np.fill_diagonal(SS, np.diagonal(SS) - Sa)

   rather than subtracting the sum of the shared off-diagonal rates;
2. the H(X,Z), H(Y,Z), and H(Z) marginals do not consistently absorb shared
   components involving variables that were marginalized out;
3. multivariate X/Y partitions can fail because rectangular cross-blocks are
   added directly to square blocks.

Formula-level reproduction on a shared-Poisson construction found:

- current raw conditional CMI: about -0.2407;
- dispatcher value after non-negativity clamp: 0;
- direct Eq. (38)-consistent entropy decomposition: about +0.1169.

For a multivariate partition with X in R^2 and Y in R^3, the current
implementation raised a broadcasting ValueError, while the Eq. (38)
decomposition returned finite, X/Y-symmetric values.

Fork diagnostics:

    experiments/poisson_rate_structure_audit.py
    experiments/poisson_conditional_marginalization_audit.py

The first file is retained only to document the paper's intentional correlation
scaling. The second file targets the actual bug.

A separate experimental correction lives on:

    experiment/poisson-marginalization-fix-v1

That branch keeps correlation scaling and rewrites Poisson CMI directly as

    H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z),

with each retained variable set receiving its own Eq. (46) rate reconstruction.

This is independent of Path B and should become a separate correctness PR only
after the package-level Poisson tests and integration benchmark pass.


### B. Gaussian constant-feature singularity

core/linalg.py::correlation_log_determinant maps singular/non-finite
correlation determinants to -1000.

For scalar constant X and random scalar Y, the current Gaussian MI path can
therefore return about 500 nats even though a deterministic predictor carries
zero mutual information.

Formula-level reproduction:

- constant X vs random Y: 500.0 nats;
- independent random X vs Y: about 0.000322 nats.

A dedicated experimental correction now exists:

    experiment/gaussian-constant-feature-fix-v1

The correction is deliberately narrow:

- remove only dimensions that are exactly constant before constructing the
  correlation matrix;
- preserve near-constant, highly correlated, and other nonconstant singular
  cases;
- leave the existing singular sentinel behavior untouched for those cases.

This fixes both Gaussian MI and CMI because they share
correlation_log_determinant.

Regression coverage now includes:

- constant predictor MI = 0;
- adding a constant nuisance dimension leaves MI unchanged;
- constant predictor CMI = 0;
- adding a constant conditioning dimension leaves CMI unchanged.

Validation entry point:

    python -m experiments.run_gaussian_constant_feature_fix_v1

This is a correctness/numerical-stability change and is a strong candidate for
a small upstream PR once focused + full pytest pass.

### C. Standard oCSE conditioning is internally inconsistent

There are two layers.

#### C1. Target-history candidates are duplicated

For target i, standard discover_network already puts

    X_i(t-1), ..., X_i(t-max_lag)

into Z_init, while the same columns remain in the candidate universe.

This creates redundant tests of

    I(X_i(t-lag); Y_i(t) | ..., X_i(t-lag), ...)

and can make correlation-based estimators singular.

A narrow experimental correction exists:

    experiment/standard-self-history-dedup-v1

It removes only target-history columns from the standard candidate universe and
maps selected local indices back to the full lagged-feature indices.
Alternative/LASSO/Information-LASSO are unchanged.

#### C2. Z_init disappears after forward selection

The deeper problem is that the current standard pipeline changes its
conditioning question between stages:

- forward selection uses Z_init;
- backward elimination drops Z_init;
- final selected-edge CMI/p-value reporting drops Z_init;
- report-all edge statistics also drop Z_init.

A direct Gaussian counterexample shows why this matters. With

    Z ~ N(0,1)
    X = Z + 0.2 eps_x
    Y = Z + 0.2 eps_y

a formula-level reproduction gave approximately:

- I(X;Y) = 1.243;
- marginal 95% shuffle threshold = 0.00368;
- I(X;Y|Z) = 0.00152;
- conditional 95% shuffle threshold = 0.00392.

So the candidate passes when the fixed baseline is dropped but fails when it is
retained.

A layered experimental correction exists:

    experiment/standard-conditioning-consistency-v1

That branch is based on the self-history de-dup branch and additionally:

- extends backward(..., Z_init=None);
- passes Z_init from standard oCSE into backward;
- preserves Z_init in final selected-edge reporting;
- preserves Z_init in report-all statistics;
- leaves alternative oCSE with Z_init=None.

Validation entry point:

    python -m experiments.run_standard_conditioning_consistency_v1

This branch changes the statistical semantics of standard oCSE and therefore
must pass the repository's existing standard-Gaussian integration test plus the
full default suite before any upstream proposal.

### D. only_return_significant violates its documented contract

discover_network documents only_return_significant=True as returning only
statistically significant links.

The implementation already computes a final shuffle test for every selected
edge but ignores test_result["Pass"]:

- True mode adds the selected edge unconditionally;
- report-all mode marks every selected edge significant=True unconditionally;
- non-selected candidates are marked significant=False unconditionally even
  though a final shuffle decision was just computed.

This is especially visible for LASSO and Information-LASSO because coefficient
support is not itself a shuffle significance decision.

A minimal experimental correction exists:

    experiment/only-return-significant-fix-v1

Selection is unchanged. Only graph reporting changes:

- True mode emits a selected edge only if its final shuffle Pass is true;
- False mode sets significant=bool(test_result["Pass"]) for every reported
  candidate.

Focused regression forces a LASSO-selected edge to receive final Pass=False and
checks both public modes.

Validation entry point:

    python -m experiments.run_only_return_significant_fix_v1

This can change standalone Path-A/LASSO graph metrics, so it should remain
separate from the existing Path-A PR and be discussed as an API/reporting
correction after full pytest passes.

### E. Path-A information-weight normalization has avoidable absolute scaling

Current Path A uses

    w_j = I_j / sum_k I_k.

At high feature count this can make every design column numerically tiny.  A
normalization such as

    w_j = I_j / max_k I_k

preserves all relative weighted-L1 penalties and differs only by one global
positive factor, while keeping the largest weighted feature at the original
scale.

This does not automatically guarantee bit-identical finite-precision
LassoLarsIC model selection, so it is being audited rather than changed.

Fork diagnostic:

    experiments/path_a_weight_normalization_audit.py

Decision rule:

- if support equality is essentially 100% while warnings/failures drop, prepare
  a small Path-A numerical follow-up;
- if supports change materially, do not modify the submitted Path-A method on
  numerical grounds alone.

### F. Version metadata is inconsistent

At the audited revision:

- pyproject.toml reports 1.1.0;
- causationentropy/core/__init__.py reports 1.1.0;
- causationentropy/__init__.py reports 0.1.0.

This is a real release/API hygiene bug but not a research contribution.  It is
low priority unless preparing a release.

## 3. Exact optimizations already retained in our Path-B v2 branch

These do not intentionally change the statistical decision rule.

### Forward observed-CMI reuse

When the best candidate fails its shuffle test and Z is unchanged, all
remaining observed I(X_j;Y|Z) values are unchanged.  v2 computes them once for
that conditioning context and replays the original repeated-argmax/removal
order.

### Gaussian conditioning-context reuse

For fixed Y and Z, Gaussian CMI terms involving only Z and (Y,Z) are invariant
across candidate X and across X permutations.  v2 reuses those terms while
recomputing X-dependent terms.

These are appropriate parts of an optimized full-oCSE baseline because they
remove redundant arithmetic rather than reducing the candidate universe.

## 4. Path-B research contribution after the audit

The strongest current Path-B story is not generic “screen then refine.”

Working method:

    Information-LASSO endpoint
      -> one conditional-CMI rescue pass
      -> exact optimized oCSE refinement

The stronger preservation target identified by the CRN diagnostic is the full
forward closure, not only the final parent set:

    C_forward(full oCSE) subset C_screen

At roughly 40% retention in the current diagnostic, forward-closure coverage is
near complete and restricted-final reproduction under common random numbers was
complete on the tested configurations.

The benchmark now records both wall-clock time and algorithmic work:

- screen candidate retention;
- forward-closure/final-support coverage;
- final edge precision/recall/F1;
- forward observed-CMI scores;
- backward observed-CMI scores;
- output observed-CMI scores;
- shuffle-test count;
- shuffle-CMI evaluation count;
- total CMI evaluation count.

This is the part most plausibly supporting a paper: high-recall,
information-guided preservation of the oCSE selection path while reducing the
number of expensive conditional-information tests.

## 5. Priority order from here

### Priority 1 — finish Path B validation

Run:

    python -m experiments.run_path_b_v2 --mode quick

and, only after quick passes:

    python -m experiments.run_path_b_v2 --mode full

The paper question remains whether the roughly 40% screen preserves the full
forward closure/final graph while total CMI work falls materially as N scales
through 20/50/100/200.

### Priority 2 — validate the two smallest correctness fixes

Run independently:

    git checkout experiment/gaussian-constant-feature-fix-v1
    python -m experiments.run_gaussian_constant_feature_fix_v1

    git checkout experiment/only-return-significant-fix-v1
    python -m experiments.run_only_return_significant_fix_v1

These have the narrowest diffs and the clearest local contracts.

### Priority 3 — validate standard-oCSE conditioning in two layers

First the narrow de-dup:

    git checkout experiment/standard-self-history-dedup-v1
    python -m experiments.run_standard_self_history_dedup_v1

Then the full conditioning-consistency experiment:

    git checkout experiment/standard-conditioning-consistency-v1
    python -m experiments.run_standard_conditioning_consistency_v1

Do not collapse these into one upstream PR until the second layer's effect on
the repository integration benchmark is known.

### Priority 4 — validate Poisson conditional marginalization

Run:

    git checkout experiment/poisson-marginalization-fix-v1
    python -m experiments.run_poisson_marginalization_fix_v1

The mathematical defect is stronger than the other estimator audits, but the
code change is larger and must satisfy the repository's existing Poisson
TPR/FPR gates before upstreaming.

### Priority 5 — Path-A weight normalization remains diagnostic only

Read:

- 08_path_a_weight_normalization.json.

Do not change Path A's normalization unless support equality remains essentially
complete while numerical warnings or failures improve.

### Priority 6 — deeper Gaussian exact fast path only if profiling justifies it

After Path B scaling is measured, consider shared QR/Cholesky/residualization
for Gaussian partial correlations only if Gaussian CMI arithmetic remains a
dominant cost after permutation-count reduction.

## 6. What not to do now

Do not add:

- another group-lag heuristic;
- iterative rescue rounds;
- generic stability selection;
- another parallel implementation;
- another KD-tree implementation;
- another shuffle early-stop implementation;
- another multiple-testing implementation.

The current diagnostics do not justify those additions, and several already
exist upstream.
