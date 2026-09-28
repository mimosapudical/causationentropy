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

Experiment result:
- iterative conditional rescue did **not** improve the main regimes enough to
  justify its cost;
- floored N=50, no floor: one-shot parent recall 0.908 vs iterative 0.877;
- floor 0.25: 0.992 vs 0.988;
- floor 0.50: both 1.000;
- hidden-parent stress: both 1.000;
- iterative rescue required roughly 12--13x as many no-shuffle CMI score
  evaluations in the N=50 regimes.

Decision:
- keep the simpler one-shot rescue in the primary method;
- use ISIS / iterative screening as related theory showing why conditional
  screening is principled, but do not import its iterative algorithm merely for
  theorem convenience;
- state and analyze a conditional-competition / rescue-budget condition for the
  one-shot stage.

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


## F. September 28 theory/phase-transition update

### F1. One-shot rescue condition is empirically exact in the audited regimes

For the deterministic condition

[
q\ge |M|+h(A),
]

where (M) is the Path-A missed-parent set and (h(A)) counts excluded
nonparents that outrank the weakest missed parent under
(I(X_j;Y\mid X_A)), the audit found:

- zero cases where the condition held but rescue failed;
- zero cases where the condition failed but rescue nevertheless succeeded.

Examples at N=50, T=300:
- no floor: condition/rescue success on 52.5% of Path-A-incomplete targets;
- raw weight floor 0.25: 92%;
- raw weight floor 0.50: 100%;
- hidden-parent stress: 100%.

The median conditional-competition count fell from 15 nonparents at no floor
to 4 at floor 0.25 and 0 at floor 0.50.  This gives a direct finite-sample
explanation for the beta-min phase transition of the one-shot rescue.

### F2. Multiseed beta-min phase transition is stable

N=50, three graph seeds, 40% retention:

| T | raw-weight floor | realized beta-min | Path-A parent recall | Path-B parent recall | Path-B parent-complete targets |
|---:|---:|---:|---:|---:|---:|
| 150 | 0.00 | 0.0019 | 0.658 | 0.864 | 0.486 |
| 150 | 0.10 | 0.0453 | 0.692 | 0.901 | 0.628 |
| 150 | 0.25 | 0.0978 | 0.760 | 0.958 | 0.838 |
| 150 | 0.50 | 0.1620 | 0.847 | 0.985 | 0.932 |
| 300 | 0.00 | 0.0019 | 0.758 | 0.917 | 0.669 |
| 300 | 0.10 | 0.0453 | 0.811 | 0.958 | 0.831 |
| 300 | 0.25 | 0.0978 | 0.892 | 0.994 | 0.973 |
| 300 | 0.50 | 0.1620 | 0.969 | 1.000 | 1.000 |
| 600 | 0.00 | 0.0019 | 0.810 | 0.936 | 0.723 |
| 600 | 0.10 | 0.0453 | 0.871 | 0.986 | 0.939 |
| 600 | 0.25 | 0.0978 | 0.960 | 0.999 | 0.993 |
| 600 | 0.50 | 0.1620 | 0.999 | 1.000 | 1.000 |

The monotone improvement with both sample size and minimum signal is the
qualitative behavior predicted by the parent-sure screening theorem.

### F3. True-parent retention frontier

The minimum useful candidate budget is signal-dependent rather than a universal
40% constant.

Examples, N=50 and three seeds:

- T=600, floor 0.50: 10% retention already gives 100% parent recall and 100%
  parent-complete targets.
- T=600, floor 0.25: 20% gives 99.72% recall / 98.65% complete targets; 30%
  gives 99.86% / 99.32%; 50% reaches 100%.
- T=300, floor 0.50: 30% gives 99.86% / 99.32%; 40% reaches 100%.
- T=300, floor 0.25: 30% gives 98.75% / 95.27%; 40% gives 99.44% / 97.30%.
- no-floor T=600: even 50% reaches only 95.28% recall, consistent with the
  generator containing near-zero structural edges.

Interpretation:
candidate retention should be treated as a statistical/computational operating
point governed by signal strength and sample size, not as a magic constant.
