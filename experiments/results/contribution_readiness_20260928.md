# Contribution readiness ledger — 2026-09-28

This is a fork-only coordination document. No item below should be proposed
upstream until its branch-level validation runner passes on a real checkout.

## Current branches

| Branch | Purpose | Strongest evidence already obtained | Remaining gate | Current decision |
|---|---|---|---|---|
| feature/information-lasso-path-a | Standalone Information-LASSO | Existing implementation and focused tests from Path A work | Do not perturb current PR without new evidence | KEEP AS-IS |
| experiment/path-b-benchmark-v2 | High-recall Information-LASSO screen + one CMI rescue + exact oCSE | ~40% retention gave near-complete forward-closure coverage in diagnostics; output-equivalent 30% benchmark ~1.98x speedup; complete CMI-work counters | quick, then full N=20/50/100/200 run on real checkout | PRIMARY RESEARCH TRACK |
| experiment/poisson-marginalization-fix-v1 | Correct conditional Poisson marginalization | Eq.38/Eq.46 analysis; old raw CMI ~-0.2407 vs corrected ~+0.1169; multivariate crash fixed; source-equivalent 1000-shuffle standard+alternative both TPR=1.0, FPR=0.07143 | branch runner / real pytest | HIGH-VALUE CORRECTNESS FIX |
| experiment/gaussian-constant-feature-fix-v1 | Stop deterministic constant predictors receiving ~500 nats Gaussian MI | constant MI 500 -> 0; informative MI unchanged after adding constant nuisance coordinate; near-constant variables preserved | branch runner / full default pytest | SMALL CLEAN CORRECTNESS FIX |
| experiment/standard-self-history-dedup-v1 | Remove standard-oCSE candidate columns already present identically in Z_init | 10/10 duplicated self-history candidates were non-finite in source-equivalent Gaussian audit; repository test already expects no self-loop in one-variable standard case | branch runner / discovery tests | CLEAN LOGIC + COMPUTE FIX |
| experiment/standard-backward-zinit-fix-v1 | Preserve Z_init during standard backward elimination | Repository theory docs explicitly require Z_init union S_-j; suppression case: legacy MI 0.00555 < 0.00642 threshold, corrected CMI 1.621 > threshold | branch runner / oCSE integration tests | HIGH-VALUE ALGORITHM CORRECTNESS FIX |
| experiment/lasso-posthoc-significance-v1 | Make only_return_significant meaningful for sparse baselines without double-gating oCSE | PR #36 semantics audited; regression tests cover failed/passed sparse supports and preserve standard selected-set semantics | branch runner / discovery tests | API/SEMANTIC FIX |
| Path-A normalization audit only | Compare I/sum(I) vs I/max(I) | N=20/50/100/200 audited; 100% support equality, 0 warnings both; only design scale changed | None needed now | DO NOT CHANGE PATH A |

## Path-A normalization decision

Source-equivalent audit:

| N | Targets checked | Support equality | Warnings sum/max | Mean design scale max/sum |
|---:|---:|---:|---:|---:|
| 20 | all, 3 seeds | 1.000 | 0 / 0 | 2.166x |
| 50 | all, 3 seeds | 1.000 | 0 / 0 | 3.369x |
| 100 | 20 sampled | 1.000 | 0 / 0 | 5.739x |
| 200 | 20 sampled | 1.000 | 0 / 0 | 12.069x |

The larger absolute design scale did not fix any observed warning or alter any
support. Changing Path A only for this normalization is therefore not justified.

## Validation order

The best order is:

1. Poisson conditional marginalization.
2. Standard backward Z_init.
3. Gaussian constant feature.
4. Standard self-history dedup.
5. LASSO/Information-LASSO post-hoc significance.
6. Path B quick.
7. Path B full only after all quick/focused gates are green.

The first two have the strongest correctness evidence and the clearest
methodological impact.

## Single-command local validation

From a clean checkout of:

    experiment/path-b-benchmark-v2

run:

    python -m experiments.run_all_fork_validations

The orchestrator fetches the fork branches, runs each branch-specific validation
runner, restores the starting branch, and reports a final pass/fail table.

Use:

    python -m experiments.run_all_fork_validations --install

if the dev dependencies have not yet been installed.

Use:

    python -m experiments.run_all_fork_validations --path-b-full

only after the default quick pass succeeds.

## Public GitHub rule

These remain fork-only experiments until their real checkout tests pass.
Do not cite an upstream issue from an incomplete or failing experimental branch.
