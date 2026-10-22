> Prepared replacement PR description. It has not been posted: the authenticated integration returned HTTP 403 for updating upstream PR metadata. Production code and its successful CI run are already live.

## Summary

Implement Path B: an Information-LASSO proposal, conditional-CMI rescue given target history plus the proposal, and restricted oCSE refinement. The objective is to retain the generating external parents while reducing the candidate universe, not to reproduce the exact finite-sample full-oCSE selection trajectory.

Standalone Information-LASSO (Path A) remains independently reviewable. This branch includes its implementation and current conditioning-scope documentation pending integration.

## Review corrections

- Append `screen_retention` after all existing `discover_network` parameters. Legacy positional calls preserve the binding of `n_jobs`, `random_state`, and `only_return_significant`; an executed regression checks both the signature and graph output.
- Treat the standard-path changes as an **independent conditioning-consistency correction**, documented in `standard_oce.rst`. Target-history lags are fixed background, not selectable candidates; forward selection, backward elimination, and reported edge CMI/p-values retain that background. This can change selected edges, self-lag output, scores, and seeded results. It is an intentional observable correction, not numerical backward compatibility.
- Integrate current upstream deferred reporting without reusing the final target's history or candidate mask for earlier targets. Regression coverage checks both standard and screened methods, significant/all-candidate reporting, and selected-edge RNG consistency.
- Keep the existing rescue-conditioning fix: target history is included alongside the Information-LASSO proposal.

The added retention parameter preserves API positional compatibility. The standard-path behavior correction is separate and explicitly documented.

## New methodology evaluation

[Executable audit, complete frontier, raw target-level results, source hashes, and reproduction commands](https://github.com/mimosapudical/causationentropy/blob/f74fd1b87383bfd4b2aee5f3b64cb2492f887f85/benchmarks/results/pathb_review_20261022_02/README.md) remain on a separate research branch.

The audit observes the **production** screening/refinement boundary, rather than duplicating the screening implementation. Both full and restricted searches use the same corrected conditioning semantics. It distinguishes:

1. generating-parent recall;
2. parent-complete target rate (excluding targets with no parents);
3. coverage of finite-sample full-oCSE support;
4. final graph recovery and computational cost.

Gaussian pilot: N=12, five paired seeds, T=150/300/600, retention targets 0.1/0.2/0.4, 49 shuffles. The same five generating graphs are reused across sample sizes.

At nominal retention 0.4:

| T | Actual retention | True-parent recall | Parent-complete targets | Full-support coverage | Full F1 | Screened F1 | CMI reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 150 | 46.8% | 87.2% | 70.9% | 100.0% | 0.761 | 0.751 | 43.9% |
| 300 | 46.4% | 92.8% | 83.6% | 99.2% | 0.822 | 0.798 | 44.0% |
| 600 | 46.1% | 93.6% | 87.3% | 97.7% | 0.844 | 0.837 | 44.9% |

Full-support agreement is not evidence of complete truth containment: at T=150 the former is 100%, but true-parent recall is 87.2%.

Two no-conditional-rescue controls are reported: proposal-only refinement, and retention-matched marginal fill that keeps the proposal and ranks excluded candidates given target history **without the proposal**. The first changes candidate count; the second isolates conditional ranking at the same realized size.

The 20-seed nonlinear-cancellation screen-only audit uses N=20 and the same sample-size/retention grid. At retention 0.4, rescue retains both structural parents on all 20 seeds at every sample size; matched marginal fill retains both in 45%, 35%, and 25% of runs, respectively. This is an engineered cancellation stress test, not a representative faithful-network benchmark. Nonlinear final graph recovery, full-support coverage, and runtime gains were not evaluated in that audit.

CMI reductions include proposal, rescue, and refinement. Gaussian target-discovery speedup is about 1.77–1.92x; it excludes public graph creation and final edge reporting. Ablation timings omit screening overhead and are not used for speedup comparisons.

## Interpretation and limits

The results support a specific conditional-rescue mechanism and a useful computational trade-off. They do **not** establish universal parent containment, uniform final-F1 superiority, a finite-sample sure-screening theorem, or a completed methods-paper claim. In the Gaussian pilot, proposal-only refinement has higher final F1 while missing more true parents. Both sides of that trade-off are included in the evidence.

Conditional screening and screen-then-refine are established directions. The potential contribution is the oCSE-specific proposal/rescue/restricted-refinement mechanism, its accuracy/cost frontier, and its assumptions and failure boundaries. Theory development and broader benchmarks remain separate research work.

## Validation

Production head: `f19d099221d05b9c75105db42331d82dca1b8fb2`.

- Upstream Actions: **all 14 jobs successful**, including Windows/macOS/Linux and documentation.
- Local production tests: 493 passed; with the three separate research-audit smoke checks: 496 passed. The repository's default configuration deselects 17 integration tests.
- Overall local coverage: 94%.
- Black, isort, and the CI syntax/undefined-name lint gate pass.
- Documentation build completes. Five docutils indentation diagnostics in unrelated pages also occur on unchanged upstream; no such diagnostic was introduced in the modified method pages.

No new estimator, permutation scheme, or production diagnostics API is introduced. The production diff contains implementation corrections, regression tests, and method documentation; the extended audit and raw results remain on the research branch.
