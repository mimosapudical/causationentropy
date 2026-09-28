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

### A. Poisson rate-structure collapse

Current Poisson MI/CMI code begins from np.corrcoef and then passes a derived
matrix into poisson_joint_entropy, whose diagonal is treated as marginal
Poisson rate/variance information.

Correlation normalization forces the original diagonal to one. Replaying the
current construction therefore produces an implied marginal rate vector of
approximately [1, 1, ...] regardless of the observed count scale.

A formula-level local reproduction using shared-Poisson samples gave:

- low-rate empirical means about [5.0086, 10.0528] -> implied [1.0, 1.0];
- high-rate empirical means about [49.9618, 99.9160] -> implied [1.0, 1.0].

This reproduction uses the same corrcoef-to-implied-rate algebra as the current
estimator; it is not yet a full package-level regression run.

Why this matters:

- The Fish–Sun–Bollt Poisson estimator is specifically intended to use
  multivariate Poisson intensity/covariance structure.
- PR #42 fixes the conditional-MI sign but does not address the loss of marginal
  count scale.
- This is a plausible explanation for the current Poisson benchmark mismatch.

Fork diagnostic:

    experiments/poisson_rate_structure_audit.py

Recommended next action:

1. reproduce the exact estimator equations from Fish, Sun & Bollt,
   Applied Network Science 2022, DOI 10.1007/s41109-022-00510-x;
2. verify whether the estimator expects empirical covariance, a reconstructed
   latent-component intensity matrix, or another moment parameter;
3. only then implement a separate Poisson-correctness PR.

This would change estimator values and potentially graph accuracy, so it must
not be bundled with exact Path-B acceleration.

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

### Priority 3 — Poisson estimator correction as a separate project

Read:

- 08_poisson_rate_structure.json

Then reproduce the 2022 Poisson estimator mathematics before touching core
code.  This could be a substantial correctness contribution independent of
Path B.

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
