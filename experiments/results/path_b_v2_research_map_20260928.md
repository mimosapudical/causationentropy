# Path B research map — 2026-09-28

This note connects the current Path-B diagnostics to the literature and to the
remaining experiments. It is a research planning artifact, not an upstream
claim.

## Current empirical diagnosis

The current Path-B v2 design is intentionally small:

Information-LASSO endpoint
→ one conditional-CMI rescue pass
→ exact standard oCSE refinement.

Why this shape:

- The standalone Path-A endpoint is too sparse for screening: in the reproduced
  12-node Gaussian frontier it retained about 11.5% of candidates but covered
  only about 78.1% of full-oCSE support.
- A single no-shuffle conditional rescue pass raised support coverage to about
  99.6% while retaining about 34% of candidates at the 30% target.
- A hidden-parent stress case showed the intended mechanism: marginal screening
  missed some conditionally active parents, while the one-pass rescue recovered
  them.
- A correlated-lag stress case did not justify adding a separate lag-group rule.
- Adding the standard-oCSE target-history set Z_init to the rescue conditioning
  set did not improve the tested frontier, so v2 keeps the simpler endpoint-only
  conditioning rule.

## Exact-computation evidence

Two implementation optimizations are separate from the screening method.

### Forward observed-CMI reuse

When a candidate fails its shuffle test, Z does not change. Therefore all
remaining observed I(X_j;Y|Z) values are unchanged.

An exact-equivalent local regression compared the old recompute loop with the
new reuse loop across 20 Gaussian graph seeds × 8 targets:

- 160 / 160 forward supports were identical.

This optimization changes only redundant computation.

### Gaussian conditioning-context reuse

For Gaussian CMI,

I(X;Y|Z) = 1/2 [log|R_XZ| + log|R_YZ| - log|R_Z| - log|R_XYZ|].

For fixed (Y,Z), the R_Z and R_YZ terms are invariant across candidate X and
across X permutations.

Local exact-equivalent checks found:

- zero numerical difference between cached and uncached CMI values in the
  tested cases;
- 80 / 80 full standard-oCSE supports identical across a 10-seed × 8-target
  regression;
- roughly 1.5–1.9× speedup for the fixed-context inner Gaussian-CMI calculation
  in smoke timing.

The full-pipeline paper benchmark must still measure speed on a fixed machine.

## Literature that directly informs the method

### Conditional / sure screening

**Barut, Fan & Verhasselt (JASA, 2016), Conditional Sure Independence
Screening.**
Shows why a conditional screen is useful when marginal signals are weak or
covariates are highly correlated, and establishes sure-screening results under
its model assumptions.

Direct relevance: our rescue stage addresses the same failure class
(marginally weak but conditionally relevant predictors), but our score and
time-series setting are different. This paper is motivation, not a theorem for
our method.

**Yousuf (Electronic Journal of Statistics, 2018), Variable Screening for High
Dimensional Time Series, DOI 10.1214/18-EJS1402.**
Develops sure screening for dependent/heavy-tailed time series and a GLSS screen
that exploits serial dependence.

Direct relevance: any Path-B theorem should explicitly account for dependent
observations rather than importing iid screening arguments.

**Yousuf & Feng, Partial Distance Correlation Screening for High-Dimensional
Time Series / Targeting Predictors via Partial Distance Correlation.**
Develops model-free screening for NARX/VAR-type time series and proves
sure-screening properties using functional-dependence / beta-mixing control.

Direct relevance: strongest template for a time-series-specific screening
guarantee.

**Fang, Yuan & Yin (Statistica Sinica, 2024), Variable Screening via
Conditional Martingale Difference Divergence, DOI 10.5705/ss.202022.0346.**
Targets both marginally and jointly active variables and emphasizes the
instability/computational cost of choosing a conditioning set. Proves a
sure-screening result for CMDH.

Direct relevance: warns us not to make the rescue conditioning set iterative or
large without evidence. The current one-pass rescue is deliberately cheaper.

**Xiong, Pan & Shen (Biometrics, 2025), PDC-MAKES,
DOI 10.1093/biomtc/ujaf042.**
Uses partial distance correlation to recover marginally unrelated but
conditionally related predictors, with sure-screening and FDR-control results.

Direct relevance: recent evidence that conditional recovery under strong
predictor correlation is still an active screening problem. It also raises FDR
as a possible later extension, but FDR is not needed for the current v2.

### Scalable causal discovery / fewer CI tests

**Runge et al. (Science Advances, 2019), PCMCI.**
Parent/condition selection followed by targeted conditional-independence tests.

Direct relevance: Path B cannot claim novelty for “screen then test.” PCMCI and
PCMCI+ should be external baselines.

**CUTS+ (AAAI 2024), High-Dimensional Causal Discovery from Irregular
Time-Series, DOI 10.1609/aaai.v38i10.29034.**
Uses coarse-to-fine discovery for scalability.

Direct relevance: “coarse-to-fine” itself is not the contribution. Our claim
must be specifically about information-guided high-recall screening that
preserves exact-oCSE refinement.

**Shiragur, Zhang & Uhler (2024), Causal Discovery with Fewer Conditional
Independence Tests, arXiv:2406.01823.**
Studies what causal structure can be learned with fewer CI queries and gives
polynomial-query results in identifiable special cases.

Direct relevance: supports reporting number of CI/CMI tests as a first-class
complexity metric, not only wall-clock time.

**E-CIT (2025), Efficient Ensemble Conditional Independence Test Framework for
Causal Discovery, arXiv:2509.21021.**
Reduces the cost of individual CI tests by divide-and-aggregate sampling and
gives consistency guarantees under its assumptions.

Direct relevance: complementary axis. Path B reduces how many candidates/tests
are needed; E-CIT-like work reduces the cost of each test. The paper should make
this distinction explicit.

### Exact Gaussian acceleration

**Delosme, Ipsen & Paige (1987/1988), partial correlations from QR /
Cholesky / Schur complements.**
Shows that partial covariances/correlations can be extracted efficiently and
reliably from shared factorizations.

Direct relevance: the current SZ/SYZ cache is only the first exact reuse step.
A future Gaussian fast path could residualize or reuse QR/Cholesky structure
rather than rebuilding correlation log-determinants for every candidate.

**Zhang, Zhou & Guan (AAAI 2018 / TIST 2019), independent-residual CI tests.**
For linear Gaussian SEMs, relates conditional independence to independence of
residuals after conditioning.

Direct relevance: provides another exact-Gaussian perspective for replacing
repeated high-dimensional determinant work with shared residualization.

## What is and is not the paper contribution

Not sufficient as novelty:

- “two-stage causal discovery”;
- “coarse-to-fine”;
- “LASSO before a CI method”;
- generic parallelism or KD-tree acceleration.

Candidate contribution supported by current diagnostics:

> Construct a cheap information-guided candidate superset for oCSE that
> explicitly targets high parent/support recall, repairs marginal-screening
> failures with one conditional rescue pass, and then preserves the original
> exact oCSE refinement semantics.

The strongest theory target is a composition argument:

1. screening guarantee: P(P_true subset C_hat) → 1 under explicit
   time-dependence and signal assumptions;
2. conditional on P_true subset C_hat, invoke the relevant oCSE consistency
   conditions;
3. combine the two failure probabilities.

This is only a proof strategy until the exact assumptions are checked against
the oCSE theory.

## Primary benchmark plan

### Primary synthetic tracks

1. linear Gaussian;
2. stable coupled logistic nonlinear dynamics with r=3.9, sigma=0.01;
3. hidden-parent stress;
4. correlated-lag stress.

Poisson stays an estimator-audit track until the repository's own 1000-shuffle
integration behavior is reproduced exactly.

### Scaling axes

- variables N: 20, 50, 100, 200 where feasible;
- T: at least 200, 500, 1000;
- max_lag: 1, 3, 5;
- graph sparsity / expected degree;
- multiple held-out graph seeds.

Do not run the full Cartesian product. Freeze a development subset, choose the
screen rule, then evaluate on held-out seeds/configurations.

### Required metrics

Screening layer:

- true-parent recall when ground truth is known;
- full-oCSE-support recall as a diagnostic;
- candidate retention/reduction;
- marginal-MI rank of missed parents;
- rescue contribution.

Final discovery:

- edge precision, recall, F1, FPR;
- final graph overlap with optimized full oCSE;
- wall-clock runtime;
- number of observed CMI evaluations;
- number of shuffle/permutation CMI evaluations;
- peak memory if practical.

### External baselines

At minimum:

- optimized full standard oCSE;
- ordinary LASSO;
- standalone Path-A Information-LASSO;
- PCMCI / PCMCI+;
- Path-B v2.

CUTS+ is a useful high-dimensional comparison when its assumptions/data
interface are compatible.

## Important finite-shuffle caveat

Screen coverage and final graph identity must remain separate metrics.

Even if the screen contains every variable in a reference full-oCSE support,
screened refinement and full refinement can consume different permutation
sequences. With a finite shuffle budget, borderline significance decisions may
therefore differ.

The final paper should either:

- report this stochastic variation explicitly over multiple permutation seeds,
  or
- use a deterministic/per-test keyed RNG scheme as a separate controlled
  experiment without changing the default repository semantics.

Do not hide this effect by comparing only a single shared global RNG stream.
