# Path-B review evidence — 2026-10-22, run 02

This audit addresses parent containment, sample-size/retention sensitivity, and the
contribution of conditional rescue. Production corrections remain on the feature
branch; this research branch holds the executable audit and raw evidence.

## Scope and provenance

- Production revision: `f19d099221d05b9c75105db42331d82dca1b8fb2`.
- Recorded local revision `49dde737abb97a1749010db1ae1618188ba128ff`
  has exactly the same Git tree (`6be27d5f31e081107bbe3b09266210394e26a6af`).
  Publishing through the authenticated API changed commit metadata, not content.
- Both JSON summaries record Python/library versions and exact SHA256 hashes of
  the discovery implementation and the audit. Audit code was untracked during
  execution; source hashes identify it independently of the recorded revision.
- Gaussian: n=12, T=150/300/600, five paired seeds, 49 shuffles, retention targets
  0.1/0.2/0.4. Reuses `benchmarks.adapters.generate_case`; all 12 targets measured.
  These are five generating graphs reused across T, not 180 independent graphs.
- Nonlinear: n=20, T=150/300/600, 20 seeds, same retention targets; evaluates only
  target zero using KDE. Reuses the earlier cancellation mechanism, with target
  history now included through the **production** screen.
- The audit intercepts the production screen/refinement boundary; it does not
  implement a second rescue ranking. This fixes the old audit's missing-history
  mismatch without exporting a new production diagnostics API.

## Definitions and controls

Ground-truth parent recall and parent-complete target rate use generating
external parent-lag pairs. Targets without parents are excluded from the latter.
Full-oCSE-support coverage is a separate finite-sample agreement diagnostic;
empty support denominators are null, not 100%. Final F1 micro-aggregates exact
parent identities within each target. Both searches share the corrected history
conditioning semantics. No comparison with the legacy inconsistent baseline is
used to claim acceleration.

`proposal_only` disables rescue and retains the Information-LASSO proposal.
`marginal_fill_matched` keeps that same proposal and fills to the production
screen's **realized** size using I(Xj;Y | target history), without conditioning on
the proposal. The second control isolates conditional ranking from candidate
count. Retention is a soft target and can exceed its nominal value.

## Gaussian results at nominal retention 0.4

| T | Actual retention | Truth parent recall | All-parent target rate | Full-support coverage | Full F1 | Screened F1 | CMI reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 150 | 46.8% | 87.2% | 70.9% | 100.0% | 0.761 | 0.751 | 43.9% |
| 300 | 46.4% | 92.8% | 83.6% | 99.2% | 0.822 | 0.798 | 44.0% |
| 600 | 46.1% | 93.6% | 87.3% | 97.7% | 0.844 | 0.837 | 44.9% |

CMI counts include proposal, rescue and restricted selection. Target-discovery
runtime includes screening and refinement but excludes public graph creation and
final edge reporting; measured aggregate speedups are about 1.77–1.92x, not a
claim about public-API end-to-end latency. Ablation timing/counts omit their
screening overhead and must not be used to claim comparative speedups.

## Ablations at nominal retention 0.4

| Family | T | Method | Truth parent recall | All-parent target rate | Actual retention | Final F1 |
|---|---:|---|---:|---:|---:|---:|
| gaussian | 150 | conditional_rescue | 87.2% | 70.9% | 46.8% | 0.751 |
| gaussian | 150 | marginal_fill_matched | 86.4% | 72.7% | 46.8% | 0.767 |
| gaussian | 150 | proposal_only | 73.6% | 54.5% | 18.8% | 0.788 |
| gaussian | 300 | conditional_rescue | 92.8% | 83.6% | 46.4% | 0.798 |
| gaussian | 300 | marginal_fill_matched | 90.4% | 78.2% | 46.4% | 0.817 |
| gaussian | 300 | proposal_only | 79.2% | 58.2% | 17.6% | 0.856 |
| gaussian | 600 | conditional_rescue | 93.6% | 87.3% | 46.1% | 0.837 |
| gaussian | 600 | marginal_fill_matched | 92.0% | 81.8% | 46.1% | 0.850 |
| gaussian | 600 | proposal_only | 83.2% | 65.5% | 18.0% | 0.888 |
| cancellation | 150 | conditional_rescue | 100.0% | 100.0% | 42.1% | not evaluated |
| cancellation | 150 | marginal_fill_matched | 72.5% | 45.0% | 42.1% | not evaluated |
| cancellation | 150 | proposal_only | 55.0% | 10.0% | 6.6% | not evaluated |
| cancellation | 300 | conditional_rescue | 100.0% | 100.0% | 42.1% | not evaluated |
| cancellation | 300 | marginal_fill_matched | 67.5% | 35.0% | 42.1% | not evaluated |
| cancellation | 300 | proposal_only | 55.0% | 10.0% | 7.4% | not evaluated |
| cancellation | 600 | conditional_rescue | 100.0% | 100.0% | 42.1% | not evaluated |
| cancellation | 600 | marginal_fill_matched | 62.5% | 25.0% | 42.1% | not evaluated |
| cancellation | 600 | proposal_only | 50.0% | 0.0% | 5.3% | not evaluated |

The complete 0.1/0.2/0.4 frontier is in both `summary.csv` files, with target-level
supports and truth sets in `targets.jsonl.gz` (gzip-compressed JSONL). Nonlinear runs are **screen-only**;
no nonlinear final graph quality, full-oCSE agreement, or speedup is inferred.

## Interpretation for a methods paper

There is a reproducible mechanism signal, not yet a completed paper claim.
At T=150, 100% full-support coverage coexists with only 87.2% true-parent recall.
Support agreement cannot substitute for truth containment. In Gaussian runs,
conditional rescue modestly improves parent recall over matched marginal fill,
but does not uniformly improve final F1; proposal-only refinement has higher F1
while missing more true parents. This trade-off must be shown, not hidden.

In the engineered nonlinear mechanism, production rescue retains both parents
across all seeds and sample sizes at retention 0.4, with a clear advantage over
the size-matched control. This is evidence that conditioning can rescue a hidden
signal. It is not evidence of universal graph recovery or novelty of conditional
screening itself. Exact cancellation is a designed stress case, not a typical
faithful observational network. Broad real-data, high-dimensional and additional
mechanism-family validation are still missing; these are limits, not added tasks
in this repair.

Related primary literature: Barut, Fan & Verhasselt, *Conditional Sure Independence
Screening*, JASA 111(515), 1266–1277 (2016), https://arxiv.org/abs/1206.1024;
Zhao et al., *Fast Flow Matching based Conditional Independence Tests for Causal
Discovery* (2026), https://arxiv.org/abs/2602.08315. Conditional screening and
screen-then-refine precede this work. CSIS assumptions/results for GLMs do not
establish a nonlinear time-series sure-screening theorem for this implementation.
A credible contribution must establish the specific rescue/accuracy/cost gain and
its assumptions and failure boundaries relative to these directions.

## Reproduction

Run from this research checkout with the project and development dependencies:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m benchmarks.path_b_screening \
  --n 12 --sample-sizes 150 300 600 --seeds 0 1 2 3 4 \
  --retentions .1 .2 .4 --n-shuffles 49 \
  --output benchmark-results/pathb_gaussian_reproduction_01

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m benchmarks.path_b_screening \
  --family cancellation --n 20 --sample-sizes 150 300 600 \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --retentions .1 .2 .4 \
  --output benchmark-results/pathb_cancellation_reproduction_01
```

Output paths must be new; the runner refuses to overwrite an existing directory.
The smoke tests distinguish support agreement from truth recall, check production
full-retention equivalence, and ensure screen-only runs never invent final metrics.

## Engineering validation

- Production + three research-audit checks: 496 passed, 17 integration tests
  deselected by the repository's standard test configuration; coverage 94%.
- Formatting/import ordering and the CI syntax/undefined-name lint gate pass.
- Documentation build completes. Its five docutils indentation diagnostics in
  `information_theory.rst` and `lasso_methods.rst` also occur on unchanged upstream;
  none is introduced in the modified method pages. These unrelated pages were
  left outside the repair scope.
- No estimator, permutation scheme, or new production API was added for the audit.

## Completion checklist

- [x] Append retention parameter after legacy positional parameters.
- [x] Execute legacy positional-call regression.
- [x] Document the independent default-method correction and observable changes.
- [x] Preserve per-target background and masks during deferred edge reporting.
- [x] Test standard/screened reporting and selected-edge RNG consistency.
- [x] Measure truth containment separately from full-support agreement.
- [x] Run three sample sizes and three nominal retention levels.
- [x] Run proposal-only and size-matched no-conditional-rescue controls.
- [x] Preserve exact raw results, versions, source hashes, and reproduction commands.
- [x] Record both favorable mechanism evidence and Gaussian recall/F1 limits.
