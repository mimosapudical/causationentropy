# Path B implementation and contribution map

Last updated: 2026-09-29

This file is the compact implementation map for the Information-LASSO /
screened-oCSE work originating from upstream issue #44.

## 1. Public contribution stack

### Upstream issue #44

https://github.com/Center-For-Complex-Systems-Science/causationentropy/issues/44

Canonical proposal and ownership thread for:

- Path A: standalone Information-LASSO;
- Path B: information-guided screening + oCSE refinement.

Kevin Slote explicitly approved both directions and noted that a strong Path-B
benchmark could support a paper.

### Upstream PR #45 — Path A

https://github.com/Center-For-Complex-Systems-Science/causationentropy/pull/45

State at this update:

- OPEN;
- non-draft;
- mergeable/clean;
- base: upstream `main`;
- head: `mimosapudical:feature/information-lasso-path-a`;
- one commit, 3 files.

Path A implements the information-weighted sparse proposal.

### Fork PR #8 — standard-oCSE correctness prerequisite

https://github.com/mimosapudical/causationentropy/pull/8

State:

- OPEN;
- non-draft;
- mergeable/clean;
- head: `fix/standard-ocse-conditioning-correctness`;
- exact base mirror: upstream-main SHA
  `116073eef33f8036b0a6f090a9c3a260fb767b8a`;
- 3 commits, 3 files;
- full validation passed.

The exact cross-fork upstream PR was attempted through the connected GitHub App
on 2026-09-29 and rejected with HTTP 403
`Resource not accessible by integration`.

### Fork PR #7 — clean Path B implementation

https://github.com/mimosapudical/causationentropy/pull/7

State:

- OPEN;
- non-draft;
- mergeable/clean;
- base: `feature/information-lasso-path-a`;
- head: `feature/information-screened-ocse-path-b`;
- 6 commits;
- exactly 3 changed files;
- 357 additions / 10 deletions.

The three changed files are:

1. `causationentropy/core/discovery.py`;
2. `causationentropy/tests/test_discovery.py`;
3. `causationentropy/tests/test_information_screened_ocse.py`.

The clean Path-B PR is intentionally stacked on Path A and should not be opened
against upstream main until #45 and the standard-oCSE prerequisite are resolved.

## 2. Frozen Path-B algorithm

For one target, let the external lagged candidate matrix be (X), the target be
(Y), and target-history baseline be (Z_0).

The production architecture is:

```text
all external lagged candidates
        |
        v
Information-LASSO proposal
        |
        v
one-shot conditional-CMI rescue
        |
        v
retained working set (~40%)
        |
        v
restricted standard oCSE
        |
        v
final causal parents
```

### Stage A — Information-LASSO proposal

Implementation:

`information_lasso_optimal_causation_entropy(...)`

Each candidate receives an information-derived weight. The weighted design is
passed to the existing sparse-selection machinery.

Scientific role:

- cheaply retain marginally visible causal parents;
- reduce the candidate universe before expensive permutation-based oCSE.

### Stage B — conditional rescue

Implementation:

`information_screened_optimal_causation_entropy(...)`

The Information-LASSO endpoint is used as the conditioning set. Every excluded
candidate is scored by

[
\widehat I(X_j;Y\mid X_{W_0}).
]

Candidates are added in descending conditional-information order until the
retention budget is reached.

Scientific role:

- recover marginally weak / cancelled parents;
- specifically address cases where marginal information or linear residual
  screening can fail.

This is one-shot rescue, not an iterative witness-completion procedure.

### Stage C — restricted standard oCSE

The retained columns are sent to
`standard_optimal_causation_entropy(...)` with the target-history
`Z_init` preserved.

Scientific role:

- use screening only to reduce the expensive search universe;
- let standard oCSE perform conditional forward discovery and backward removal.

The clean public method is:

```python
discover_network(
    data,
    method="information_screened",
    screen_retention=0.40,
    ...
)
```

## 3. Why standard-oCSE correctness is a prerequisite

Standard oCSE uses target history as the initial conditioning set.

The audit found two inconsistent behaviors in the existing baseline:

1. lagged target-history columns already present in `Z_init` could be tested
   again as external candidates;
2. backward / final edge reporting could lose the same `Z_init` conditioning.

The isolated prerequisite branch fixes those semantics without including any
Information-LASSO or Path-B code.

This separation is deliberate: Path B should not receive credit for a baseline
correctness fix, and reviewers should be able to evaluate each change
independently.

## 4. Scientific correctness target

The final method does **not** attempt to reproduce the finite-sample full-oCSE
greedy path exactly.

The central population target is parent containment:

[
N_I\subseteq W.
]

If the retained screen (W) contains the causal parents, the original oCSE
aggregative-discovery / progressive-removal argument can be restricted to (W).

Therefore:

[
\text{parent-sure screen}
+
\text{restricted oCSE}
\Rightarrow
\text{causal-parent recovery}
]

under the stated oCSE assumptions.

This is why exact-path witness completion, giant-block certificates, and
full-graph identity were retained only as diagnostics / negative results.

## 5. Main empirical evidence

### Five-seed primary Gaussian

- candidate retention: 45.6%;
- full-oCSE support covered by screen: 100% on all five seeds;
- full truth F1: 0.787 +/- 0.136;
- screened truth F1: 0.781 +/- 0.124;
- aggregate speedup: 1.57x;
- total CMI evaluations: -39.2%;
- shuffle-CMI evaluations: -41.4%.

### Three-seed scaling

| N | retention | CMI reduction | speedup | full F1 | Path-B F1 |
|---:|---:|---:|---:|---:|---:|
| 20 | 42.1% | 39.8% | 1.54x | 0.793 | 0.807 |
| 50 | 40.8% | 44.4% | 1.46x | 0.723 | 0.717 |
| 100 | 40.4% | 48.7% | 1.76x | 0.710 | 0.715 |

### Nonlinear cancellation

At a matched ~12% budget:

- Path-B nonlinear CMI rescue: 100% hidden-parent recall;
- forward regression: 25%;
- marginal KDE top-k: 15%;
- random: 10%.

This is the clearest mechanism experiment for why the conditional-information
rescue is not interchangeable with purely marginal or linear screening.

## 6. Research / manuscript files

Canonical research branch:

`experiment/path-b-benchmark-v2`

Main records:

- `experiments/PATH_B_RESEARCH_LEDGER.md` — complete research/contribution
  ledger;
- `experiments/path_b_paper_readiness.md` — paper-readiness gate table;
- `experiments/PATH_B_PAPER_DRAFT.md` — manuscript draft;
- `experiments/path_b_sure_screening_theory.md` — parent-sure / weighted
  screen theorem route;
- `experiments/path_b_screen_certificate_theory.md` — restricted-oCSE and
  diagnostic certificate theory;
- `experiments/UPSTREAM_ACTIONS_READY.md` — exact remaining browser actions;
- `experiments/results/path_b_paper_summary_20260929.json` — frozen main
  numbers;
- `experiments/make_path_b_paper_figures.py` — figure generation.

## 7. Historical / diagnostic branches

Main contribution branches:

- `feature/information-lasso-path-a`;
- `feature/information-screened-ocse-path-b`;
- `fix/standard-ocse-conditioning-correctness`;
- `experiment/path-b-benchmark-v2`.

Older experiment / correctness branches are preserved in the research ledger
and are not separate Path-B algorithm components.

## 8. Final upstream dependency order

1. upstream PR #45 (Path A);
2. upstream standard-oCSE conditioning correctness PR;
3. rebase `feature/information-screened-ocse-path-b` on the resulting
   upstream main;
4. open final minimal Path-B upstream PR;
5. continue paper / release review from the canonical issue #44 thread.

The connected GitHub App cannot perform steps 2 or post the prepared #44
progress comment because upstream writes return HTTP 403.  The exact title,
body, compare link, and comment are stored in `UPSTREAM_ACTIONS_READY.md`.
