# Path B paper-readiness ledger

This file tracks the current scientific state of the Path-B project.  It is
deliberately stricter than a pull-request checklist: a green engineering test
does not imply that a paper claim is ready.

## A. Core scientific claim

Target claim:

> A parent-sure information-guided screen can reduce the candidate universe of
> oCSE substantially while preserving population causal-parent recovery, and
> can reduce finite-sample computation without requiring reproduction of every
> noisy edge in the full-oCSE path.

Current status: **strong hypothesis, partially proved, partially validated**.

### A1. Restricted-oCSE population consistency

Status: **proof skeleton complete**.

If the screen contains all true parents, the original oCSE aggregative
discovery argument still cannot terminate before including the parents, and
progressive removal deletes nonparents.  The argument is a restricted-candidate
corollary of the original oCSE theorem.

Remaining:
- write the result in publication notation;
- state baseline/self-history conditioning explicitly;
- check every assumption against the exact theorem numbering and wording in
  Sun, Taylor & Bollt.

### A2. Parent-sure screening theorem

Status: **deterministic Lasso lemma complete; probability theorem incomplete**.

Completed:
- weighted-Lasso reparameterization;
- deterministic no-false-negative lemma under weighted RE + beta-min;
- relation between weighted and unweighted RE under a parent-weight lower
  bound;
- Gaussian marginal-visibility -> information-weight lower-bound route;
- stable-VAR dependence identified as requiring time-series concentration
  rather than iid concentration.

Remaining:
- instantiate the Basu-Michailidis / mixing-time deviation and sample-RE
  constants for the exact stable VAR setup;
- turn the schematic beta-min scaling into a citation-level corollary;
- decide the final treatment of marginally invisible parents.

### A3. Marginally invisible parents

Status: **mechanism demonstrated; universal theorem not yet attached to the
production one-shot rescue**.

Facts:
- marginal-information weighting cannot by itself be universally parent-sure;
- the hidden-parent stress demonstrates this boundary;
- one-shot conditional rescue recovers the hidden-parent stress in existing
  diagnostics;
- ISIS/iterative screening literature provides a principled route to remove
  the marginal-correlation assumption.

Pending experiment:
- one-shot vs iterative conditional rescue at identical budget and explicit
  no-shuffle CMI cost.

Decision rule:
- if iterative rescue materially improves parent-sure recovery, develop a
  vectorized/low-cost iterative screen and adapt an ISIS-style theorem;
- otherwise keep the simpler one-shot rescue and state a conditional
  competition / visibility assumption rather than adding complexity.

## B. Empirical evidence

### B1. Baseline identity vs causal accuracy

Status: **important negative result established**.

The current quick audit showed that true edges missed by the screen were also
missed by full keyed oCSE, while full-oCSE final edges missing from the N=50
screen were false positives in that audit.  Therefore exact full-path identity
is a diagnostic, not the primary correctness target.

### B2. Beta-min phase transition

Status: **one-seed signal established; multiseed audit running**.

One-seed N=50, T=300:
- raw random weights: Path-B parent recall about 0.908;
- raw weight floor 0.25: about 0.992;
- raw weight floor 0.50: 1.000.

This is qualitatively consistent with a beta-min screening theorem.

Pending:
- N=50, T in {150,300,600}, weight floor in {0,0.1,0.25,0.5}, 3 seeds.

### B3. Retention phase transition

Status: **running**.

Goal:
identify the smallest candidate fraction that preserves the generating parents,
rather than the smallest fraction that reproduces the noisy full-oCSE support.

Grid:
- N=50;
- T in {150,300,600};
- weight floor in {0,0.1,0.25,0.5};
- retention in {0.1,0.2,0.3,0.4,0.5};
- 3 seeds.

This result will determine the theoretically defensible speed/accuracy
operating point.

### B4. End-to-end quick benchmark

Status: **works in Gaussian primary regime, not yet paper-scale**.

Current quick summary:
- Gaussian mean runtime speedup about 1.38x;
- shuffle-CMI evaluation reduction about 40.6%;
- total-CMI evaluation reduction about 35.5%;
- N=20 quick scaling about 1.53x;
- N=50 quick scaling about 1.28x.

The current nonlinear logistic audit is poor and Poisson remains an estimator
audit rather than a successful generalization result.

Interpretation:
- a Gaussian/stable-VAR paper scope is already coherent;
- a general estimator-agnostic paper is not yet supported.

## C. Routes explicitly ruled out

These are useful negative results and should not be silently revisited.

1. **Top-k witness completion as the main theory**
   - not safe because conditional MI is not monotone in the conditioning set.

2. **Exact finite-sample working-set certificate as the accelerator**
   - exact graph identity is recoverable, but certificate cost is >= full in
     the tested regimes.

3. **One giant excluded-block F test**
   - population certificate is valid, but finite-sample power collapses with
     block dimension, especially at N=50.

4. **Full-Gram KKT/OLS inverse certificate**
   - mathematically valid but empirically far too conservative because tiny
     information-weight columns make the weighted Gram inverse ill-conditioned.

## D. Novelty position

Current search has not found an existing method that combines all of:

- information-weighted sparse screening;
- a parent-sure restricted-candidate oCSE theorem;
- conditional rescue for marginally weak parents;
- explicit separation of causal-support accuracy from full noisy-path identity;
- measured oCSE computation reduction.

Related areas that must be compared carefully:
- original oCSE and later CausationEntropy package work;
- SIS / ISIS and iterative sure screening;
- adaptive / weighted Lasso;
- high-dimensional VAR Lasso theory;
- local / restricted-conditioning causal discovery;
- safe screening / working-set methods.

## E. Paper-level gaps

A credible methods paper still needs all of the following.

### Theory
- full stable-VAR probability corollary;
- final rescue theorem/assumption;
- computational complexity statement.

### Experiments
- multiseed beta-min and retention frontiers;
- larger-N end-to-end scaling;
- confidence intervals / variability across seeds;
- ablations: unweighted Lasso, Path A only, one-shot rescue, iterative rescue
  if retained;
- at least one realistic or established time-series causal benchmark if the
  venue expects more than synthetic Gaussian VARs.

### Engineering
- clean implementation branch separate from diagnostics;
- production tests and API design;
- remove benchmark-only oracle hooks from the final method.

### Writing
- theorem statements/proofs in paper form;
- related-work table;
- runtime/accuracy phase-transition figures;
- limitations section defining the Gaussian/stable-VAR scope if nonlinear
  performance is not solved.
