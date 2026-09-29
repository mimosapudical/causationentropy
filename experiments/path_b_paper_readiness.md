# Path B paper-readiness ledger

This document tracks scientific readiness, not repository merge readiness.
A row is DONE only when the current branch has either a theorem/proof argument
or a replicated experiment directly supporting it.

| Component | Status | Current evidence | Remaining blocker |
|---|---|---|---|
| Core problem formulation: expensive oCSE candidate search | DONE | original oCSE theory + repository benchmark work counters | none |
| Distinguish causal correctness from finite-sample baseline identity | DONE | true-edge detectability audit: baseline-only omitted screen edges are false positives in the current N=50 diagnostic | replicate in broader settings |
| Restricted-candidate population oCSE corollary | DONE THEORY | proof pinned to the original oCSE faithful-Markov assumptions; screened-universe contradiction argument recorded in the theory notes | polish notation for manuscript |
| Impossibility of generic monotone score pruning | DONE | Gaussian suppression/collider counterexample | write as lemma/example, not overclaim general impossibility beyond stated class |
| Weighted Information-LASSO support-containment route | DONE MODULAR / CV OPEN | deterministic RE+beta-min theorem, stable-Gaussian-VAR probability bridge, Gaussian weight-visibility bound, and n>p BIC-compatible KKT/OLS certificates are recorded | high-dimensional ordinary K-fold CV branch is a practical variant without a matching support-containment theorem |
| Conditional rescue mechanism | DONE CONDITIONAL | fixed-parent suppression sweep + 10-40% budget replication; one-shot ranking-margin theorem recorded for uniform CMI error event | manuscript should state the uniform-concentration event as an assumption unless a full data-dependent-set concentration lemma is added |
| Exact finite-sample path certificate | DONE NEGATIVE | 100% keyed path identity but >1x test/evaluation cost | keep as diagnostic/negative result, not main algorithm |
| Giant residual-block certificate | DONE NEGATIVE | population identity is clean; finite-sample power collapses with dimension | not a primary method |
| Partitioned certificate | DONE DIAGNOSTIC | confirms block-size power dilution; actual misses remain ultra-weak | optional appendix |
| Beta-min phase transition | DONE INITIAL | controlled DAG VAR and 5-seed floored-VAR replication | add confidence intervals / broader N,T,density sweep |
| Suppression phase transition | DONE INITIAL | fixed coefficients, correlation-controlled cancellation, 20 seeds; budget sweep | replicate at additional T/noise levels if space permits |
| End-to-end Gaussian accuracy | DONE PRIMARY | five-seed N=12 replication: full F1 0.787±0.136 vs Path-B 0.781±0.124; aggregate 1.57x speedup; 39.2% total-CMI and 41.4% shuffle-CMI reduction | broader N/T/density sweep still useful |
| Scaling advantage | DONE PRIMARY | 3-seed N20/N50/N100: ~40% candidate retention, mean screen support recall >=0.991, CMI reduction 39.8%/44.4%/48.7%, aggregate speedup 1.54x/1.46x/1.76x | broader density/noise sweeps are optional robustness, not a gate |
| Comparison baselines | DONE PRIMARY | budget-matched marginal information, forward regression, and random screens at N=20/50; controlled linear and nonlinear cancellation comparisons | broader external baselines are optional rather than a primary gate |
| Nonlinear generality | OPEN / OUT OF SCOPE | logistic/Poisson quick results are poor | either explicitly scope paper to Gaussian linear regime or design separate extension |
| Clean production implementation | IN PROGRESS / CLEAN BRANCH | `feature/information-screened-ocse-path-b` contains only the new API/core path and tests, stacked on Path A; standard-oCSE prerequisite branch is separately validated | final stacked CI + upstream dependency handling |
| Full paper write-up | DRAFTED | `PATH_B_PAPER_DRAFT.md` contains abstract, method, theorem statements, main tables, negative results, limitations, and figure plan | convert results to figures and polish proof/related-work prose |
| Upstream review | OPEN | maintainer expressed interest in Path B | only after clean, verified result |

## Current scientific claim that is supported

The strongest defensible current claim is:

> In sparse Gaussian linear dynamical systems, information-guided screening plus
> conditional rescue can substantially reduce the candidate space before oCSE.
> Population oCSE only requires retention of the causal-parent set, not
> reproduction of the full finite-sample greedy path. Controlled beta-min and
> suppression experiments support the two complementary screening mechanisms:
> marginally visible parents are retained as signal strength/sample size grow,
> while conditionally visible but marginally cancelled parents are recovered by
> the rescue stage.

Do not yet claim a complete consistency theorem for the production BIC/CV
implementation.  The weighted-Lasso theorem route currently applies to a
theorem-backed tuning regime, not automatically to scikit-learn's selected
penalty.

## Critical path to a submission-quality result

1. Close the tuning/proof alignment:
   - analyze the selected LARS/BIC path directly, OR
   - introduce a theorem-backed screening/tuning variant and clearly separate
     it from the practical BIC/CV variant.
2. Formalize the rescue ranking-margin theorem with a valid treatment of the
   data-dependent conditioning set (uniform concentration or cross-fitting).
3. Finish replicated Gaussian end-to-end/scaling experiments and strongest
   baseline comparisons.
4. Freeze the algorithm, produce a clean implementation, and write the paper.


## Five-seed primary Gaussian replication

Using the primary N=12, T=300 Gaussian benchmark with 50 shuffles and seeds
0--4:

- candidate retention: 0.4561 ± 0.0034;
- screen recall of the full-oCSE per-target support: 1.000 for all five seeds;
- refined/full-oCSE edge recall: mean 0.9662 (range 0.90--1.00);
- full-oCSE truth F1: 0.7869 ± 0.1364;
- Path-B truth F1: 0.7810 ± 0.1243;
- mean F1 difference Path-B minus full: -0.0059;
- mean precision difference: -0.00095;
- mean recall difference: -0.0105;
- aggregate end-to-end speedup: 1.566x;
- aggregate total-CMI reduction: 39.18%;
- aggregate shuffle-CMI reduction: 41.40%.

The result is not exact graph reproduction on every seed.  In particular, seed
3 retains only 90% of full-oCSE edges yet improves truth F1 from 0.595 to 0.629,
which reinforces the distinction between baseline identity and causal accuracy.


## Replicated Gaussian scaling (3 seeds each)

The completed scaling workflow uses N=20,50 with T=300 and N=100 with T=500.

| N | Full F1 | Path-B F1 | Delta F1 | candidate retention | screen full-support recall | CMI reduction | shuffle-CMI reduction | aggregate speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.7932 | 0.8071 | +0.0139 | 0.4211 | 0.9952 | 39.80% | 43.39% | 1.54x |
| 50 | 0.7234 | 0.7168 | -0.0066 | 0.4082 | 0.9932 | 44.35% | 45.17% | 1.46x |
| 100 | 0.7100 | 0.7145 | +0.0045 | 0.4040 | 0.9909 | 48.74% | 45.41% | 1.76x |

Across all three sizes the truth-level F1 difference remains small while the
fraction of expensive CMI work removed grows with the candidate universe.


## Nonlinear conditional-cancellation baseline

A controlled nonlinear construction separates the conditional-information
rescue from both marginal information screening and linear forward regression:

[
X_2=-a(X_1^2-1)+U,
qquad
Y=X_2+a(X_1^2-1)+epsilon=U+epsilon.
]

Thus (X_1) is a direct structural parent but is population-marginally
independent of (Y), while its linear residual correlation after fitting
(X_2) also vanishes in the population.

Across 20 seeds:

| realized retention | Path-B KDE hidden recall | forward regression | marginal KDE top-k | random |
|---:|---:|---:|---:|---:|
| ~12% | 1.00 | 0.25 | 0.15 | 0.10 |
| ~21% | 1.00 | 0.30 | 0.15 | 0.15 |
| 40% | 1.00 | 0.55 | 0.40 | 0.35 |

The visible parent is recovered in every run by Path-B, forward regression,
and marginal KDE.  The difference is therefore localized to the marginally
invisible nonlinear parent.
