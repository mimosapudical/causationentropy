# Path B paper-readiness ledger

This document tracks scientific readiness, not repository merge readiness.
A row is DONE only when the current branch has either a theorem/proof argument
or a replicated experiment directly supporting it.

| Component | Status | Current evidence | Remaining blocker |
|---|---|---|---|
| Core problem formulation: expensive oCSE candidate search | DONE | original oCSE theory + repository benchmark work counters | none |
| Distinguish causal correctness from finite-sample baseline identity | DONE | true-edge detectability audit: baseline-only omitted screen edges are false positives in the current N=50 diagnostic | replicate in broader settings |
| Restricted-candidate population oCSE corollary | PARTIAL | proof sketch adapted from aggregative discovery + progressive removal | formalize assumptions/notation and write full proof |
| Impossibility of generic monotone score pruning | DONE | Gaussian suppression/collider counterexample | write as lemma/example, not overclaim general impossibility beyond stated class |
| Weighted Information-LASSO support-containment route | PARTIAL | transformed-design reduction + beta-min/RE proof route; 5-seed beta-min experiments | production BIC/CV tuning not covered by fixed-lambda sufficient condition |
| Conditional rescue mechanism | STRONG PARTIAL | fixed-parent suppression sweep: endpoint collapses near cancellation while rescue restores 100%; 10-40% budget replication | formal uniform-concentration/ranking-margin proof for data-dependent endpoint |
| Exact finite-sample path certificate | DONE NEGATIVE | 100% keyed path identity but >1x test/evaluation cost | keep as diagnostic/negative result, not main algorithm |
| Giant residual-block certificate | DONE NEGATIVE | population identity is clean; finite-sample power collapses with dimension | not a primary method |
| Partitioned certificate | DONE DIAGNOSTIC | confirms block-size power dilution; actual misses remain ultra-weak | optional appendix |
| Beta-min phase transition | DONE INITIAL | controlled DAG VAR and 5-seed floored-VAR replication | add confidence intervals / broader N,T,density sweep |
| Suppression phase transition | DONE INITIAL | fixed coefficients, correlation-controlled cancellation, 20 seeds; budget sweep | replicate at additional T/noise levels if space permits |
| End-to-end Gaussian accuracy | DONE PRIMARY | five-seed N=12 replication: full F1 0.787±0.136 vs Path-B 0.781±0.124; aggregate 1.57x speedup; 39.2% total-CMI and 41.4% shuffle-CMI reduction | broader N/T/density sweep still useful |
| Scaling advantage | PARTIAL | quick N20/N50 shows ~1.3-1.5x wall-clock and ~40% CMI-work reduction | replicated N20/N50/N100 run is the remaining scaling gate |
| Comparison baselines | PARTIAL | full oCSE, plain Lasso, Information-Lasso already emitted by benchmark | add/justify strongest relevant screening baselines (e.g. SIS/forward/HOLP where appropriate) |
| Nonlinear generality | OPEN / OUT OF SCOPE | logistic/Poisson quick results are poor | either explicitly scope paper to Gaussian linear regime or design separate extension |
| Clean production implementation | OPEN | current branch is experiment-heavy | make clean core branch after method freezes |
| Full paper write-up | OPEN | theory note + experiment artifacts exist | theorem proof, figures, related work, manuscript |
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
