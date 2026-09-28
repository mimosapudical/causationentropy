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
correlation determinants to -1000.  The epsilon argument is currently unused.

For scalar constant X and random scalar Y:

- the scalar X correlation path returns 0;
- the joint (X,Y) correlation matrix is non-finite and maps to -1000;
- Gaussian MI becomes approximately 0.5 * (0 + 0 - (-1000)) = 500 nats.

A formula-level local reproduction with 500 samples returned:

- constant X vs random Y: 500.0 nats;
- independent random X vs Y: about 0.000322 nats;
- identical continuous X=Y: 500.0 nats because it hits the same sentinel.

The identical-variable case is genuinely singular/infinite in the ideal
continuous model; the problematic diagnostic is that a deterministic constant
feature receives the same sentinel-driven score.

A deterministic constant predictor should not become an extremely informative
Path-A feature.

This is directly relevant to Information-LASSO because Path A obtains its
weights from marginal Gaussian information scores.

Fork diagnostic:

    experiments/gaussian_constant_feature_audit.py

Recommended next action:

- first measure how often zero/near-zero variance columns occur in realistic
  data and how they affect Path-A support;
- likely fixes are either zero-variance filtering at the Path-A boundary or a
  principled regularized Gaussian log-determinant;
- do not replace -1000 with another arbitrary sentinel.

This is a correctness/numerical-stability change and may alter selected support.

### C. Standard oCSE retests the target-history columns already in Z_init

discover_network constructs X_lagged from every variable and lag.  For standard
oCSE it also places the target's own lagged columns in Z_init, while those same
columns remain candidates in X_lagged.

This can create tests of the form

    I(X_target(t-lag); Y | ..., X_target(t-lag), ...)

which are redundant and can create singular Gaussian correlation matrices /
NaNs.

No existing upstream issue/PR was found in the audit for this exact point.

Potential fix:

- remove candidate columns that are already represented identically in Z_init,
  or represent the conditioning/candidate universe explicitly so duplicates
  cannot occur.

Caution: removing candidates changes the number/order of permutation tests and
therefore the shared RNG stream under finite shuffles.  It is statistically
cleaner but is not bit-identical legacy behavior.

### D. only_return_significant is ambiguous for LASSO / Information-LASSO

discover_network computes a final shuffle test for selected edges, but when
only_return_significant=True it still adds each edge in S without checking the
new test_result["Pass"].

For standard/alternative oCSE, S already went through shuffle-based selection.
For ordinary LASSO and Information-LASSO, S is coefficient support rather than
a significance-filtered set.

Therefore the public option name does not describe the same semantics across
methods.

No existing upstream issue/PR was found for this behavior.

Recommended next action:

- decide API semantics before changing code;
- if only_return_significant should mean final post-hoc filtering for every
  method, add a method-agnostic regression test;
- otherwise rename/document it so Path-A benchmark interpretation is explicit.

A fix may change standalone Path-A graph metrics and should not be mixed into
the current Path-A PR without maintainer agreement.

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

Run the cross-platform validation:

    python -m experiments.run_path_b_v2 --mode quick

then, after all gates pass:

    python -m experiments.run_path_b_v2 --mode full

The primary question is whether the approximately 40% screen continues to
preserve the forward closure and final graph as N grows to 20/50/100/200 while
total CMI work falls materially.

### Priority 2 — decide the two Path-A follow-ups from evidence

Read:

- 08_path_a_weight_normalization.json
- 08_gaussian_constant_feature.json

Only create a Path-A follow-up PR if the audits show a clear numerical or
correctness gain without introducing a new modeling choice.

### Priority 3 — Poisson conditional-marginalization correction

Read:

- 08_poisson_rate_structure.json;
- 08_poisson_conditional_marginalization.json.

The paper mathematics has now been checked: correlation scaling is intentional,
while the conditional marginalization is the actual defect. Validate the
separate branch experiment/poisson-marginalization-fix-v1 against the full
Poisson unit/integration tests before preparing any upstream PR.

### Priority 4 — deeper Gaussian exact fast path

After the simple context cache is validated, the next mathematically grounded
performance step is shared QR/Cholesky/residualization for partial correlations.
Classic work by Delosme, Ipsen & Paige shows that partial correlations can be
computed efficiently from shared factorizations.

This should only be pursued if the scaling table shows Gaussian CMI arithmetic,
rather than permutation count, remains the dominant bottleneck.

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
