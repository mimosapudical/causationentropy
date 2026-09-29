# Information-LASSO / Screened-oCSE research ledger

Last updated: 2026-09-29

This is the canonical map for the work originating from upstream issue #44.
It separates upstream contribution state, fork-only CI machinery, research
evidence, correctness fixes discovered during benchmarking, and the remaining
paper / merge gates.

## 1. Canonical repositories and branches

- Upstream: https://github.com/Center-For-Complex-Systems-Science/causationentropy
- Working fork: https://github.com/mimosapudical/causationentropy
- Path A implementation branch: `feature/information-lasso-path-a`
- Canonical Path B research branch: `experiment/path-b-benchmark-v2`
- Previous Path B prototype: `experiment/path-b-benchmark-v1`
- Fork default branch: `main`

The v2 branch is the scientific audit branch.  It intentionally contains many
experiments and workflows and is **not** the branch that should be submitted
upstream as a final Path-B implementation.

## 2. Upstream issue / PR state

### Upstream issue #44 — canonical proposal / ownership thread

https://github.com/Center-For-Complex-Systems-Science/causationentropy/issues/44

Title: **Proposal: complete the Information-LASSO pathway for scalable
high-dimensional discovery**

Status: **OPEN**

Important maintainer response from Kevin Slote:

> "I believe we would like both. Please do take ownership of this and we can
> add it into the next release. If we can show good improvements in benchmarks
> with code path B, then there might be a nice paper in there."

The issue therefore authorizes both:

- Path A: standalone Information-LASSO;
- Path B: information-guided screening + oCSE refinement.

Vaisakhi Mishra also offered to collaborate in the same issue thread.

### Upstream PR #45 — Path A

https://github.com/Center-For-Complex-Systems-Science/causationentropy/pull/45

Title: **feat: implement standalone Information-LASSO pathway**

Status at this ledger update:

- OPEN;
- non-draft;
- mergeable = true;
- mergeable_state = clean;
- one commit;
- 3 files changed;
- 286 additions / 18 deletions;
- assignee: `jefish003`;
- no review submissions;
- no PR conversation comments;
- upstream Tests workflow completed successfully;
- head commit: `26888e294e30bdfde47de8825d7246f1f524b0b0`.

This is the canonical upstream Path-A contribution.  Path B should not be
opened as an upstream PR that redundantly includes Path A while #45 remains
unmerged.

## 3. Fork PR history

These PRs are historical/CI records, not additional upstream contributions.

| Fork PR | Purpose | Final state |
|---|---|---|
| #1 | Path A implementation in the fork | CLOSED / MERGED into fork main |
| #2 | temporary Path-A CI validation | CLOSED, not merged |
| #3 | Path-B benchmark v1 experiment | CLOSED, not merged |
| #4 | Path-B v2 pull-request CI trigger | CLOSED, not merged |
| #5 | Path-B v2 validation against Path-A branch | CLOSED, not merged |
| #6 | isolated Path-B CI trigger using CI base/run branches | CLOSED, not merged |

Links:

- https://github.com/mimosapudical/causationentropy/pull/1
- https://github.com/mimosapudical/causationentropy/pull/2
- https://github.com/mimosapudical/causationentropy/pull/3
- https://github.com/mimosapudical/causationentropy/pull/4
- https://github.com/mimosapudical/causationentropy/pull/5
- https://github.com/mimosapudical/causationentropy/pull/6

No fork CI-only PR needs to remain open.

## 4. Branch map

### Main contribution branches

- `feature/information-lasso-path-a` — exact Path-A head used by upstream #45.
- `experiment/path-b-benchmark-v2` — canonical current research branch.
- `experiment/path-b-benchmark-v1` — superseded first benchmark.
- `experiment/path-b-ci-base`, `experiment/path-b-ci-run` — CI-only support
  branches; superseded by push-triggered workflows.
- `ci/information-lasso-path-a-check` — historical Path-A CI branch.

### Isolated correctness / audit branches discovered while validating Path B

These are related because Path-B benchmarking exposed repository semantics that
could otherwise make the baseline comparison misleading.

- `experiment/standard-backward-zinit-fix-v1`
  - keeps target-history conditioning in standard-oCSE backward refinement.
- `experiment/standard-self-history-dedup-v1`
  - isolates duplicated target-history candidates from the external candidate
    universe.
- `experiment/standard-conditioning-consistency-v1`
  - integrates conditioning semantics consistently.
- `experiment/standard-ocse-correctness-integration-v1`
  - combined correctness validation branch.
- `experiment/only-return-significant-fix-v1`
  - audits/fixes output semantics for insignificant links.
- `experiment/lasso-posthoc-significance-v1`
  - isolates post-hoc significance semantics for Lasso-family outputs.
- `experiment/gaussian-constant-feature-fix-v1`
  - isolates Gaussian information singularity handling for constant features.
- `experiment/poisson-marginalization-fix-v1`
  - isolates the Poisson conditional-marginalization issue.

These branches should **not** silently be bundled into a clean Path-B upstream
PR.  Each correctness change should either already exist upstream, be proposed
separately, or be explicitly listed as a prerequisite.

## 5. Path-B method that survived the audit

The frozen research architecture is:

```text
all external lagged candidates
        |
        v
Information-LASSO proposal
        |
        v
one-shot conditional rescue
        |
        v
parent-superset working set
        |
        v
restricted standard oCSE
        |
        v
progressive removal / final graph
```

The research target is **parent-sure screening**, not exact reproduction of the
finite-sample full-oCSE forward path.

Old routes that are now diagnostics / negative results rather than primary
methods:

- exact finite-sample path certificate;
- top-k witness completion;
- giant excluded-block certificate;
- partitioned residual certificate as a primary method;
- iterative rescue as a default replacement for the one-shot rescue;
- forcing exact full-oCSE graph identity.

## 6. Main theory files

- `experiments/path_b_sure_screening_theory.md`
  - transformed weighted-Lasso theorem;
  - RE + beta-min parent-containment proof;
  - Gaussian marginal-visibility bound;
  - stable-Gaussian-VAR RE/deviation bridge;
  - n>p BIC-compatible KKT certificate;
  - observable OLS-to-Lasso certificate;
  - one-shot rescue ranking-margin theorem;
  - CMI-complexity proposition.

- `experiments/path_b_screen_certificate_theory.md`
  - restricted-candidate oCSE corollary from Sun--Taylor--Bollt;
  - population excluded-block certificate;
  - non-monotonicity / suppression counterexample;
  - negative finite-sample certificate results;
  - parent-sure vs baseline-identity distinction.

Current theory boundary:

- low-dimensional BIC production branch has deterministic finite-sample
  certificates;
- the theoretically calibrated stable-VAR weighted-Lasso branch has a clean
  asymptotic route;
- ordinary K-fold `LassoCV` in the high-dimensional production branch is
  still a practical tuning policy without a matched support-containment theorem.
  This is an explicit scope boundary, not a hidden claim.

## 7. Main replicated empirical results

### Five-seed primary Gaussian

N=12, T=300, 50 shuffles, seeds 0--4:

- mean candidate retention: 45.6%;
- full-oCSE support covered by the screen: 100% on all five seeds;
- mean refined/full-oCSE edge recall: 96.6%;
- full-oCSE truth F1: 0.787 ± 0.136;
- Path-B truth F1: 0.781 ± 0.124;
- mean delta F1: -0.006;
- aggregate end-to-end speedup: 1.57x;
- total CMI evaluations: -39.2%;
- shuffle-CMI evaluations: -41.4%.

### Replicated scaling

Three seeds each:

| N | T | Full F1 | Path-B F1 | Delta F1 | retention | screen support recall | CMI reduction | shuffle reduction | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 300 | 0.7932 | 0.8071 | +0.0139 | 0.4211 | 0.9952 | 39.80% | 43.39% | 1.54x |
| 50 | 300 | 0.7234 | 0.7168 | -0.0066 | 0.4082 | 0.9932 | 44.35% | 45.17% | 1.46x |
| 100 | 500 | 0.7100 | 0.7145 | +0.0045 | 0.4040 | 0.9909 | 48.74% | 45.41% | 1.76x |

### Beta-min / parent-sure replication

At T=300, five seeds:

- N=20, random weak edges: Path A parent recall 87.4%, Path B 95.2%;
- N=20, weight floor .25: Path B 100%;
- N=20, weight floor .50: Path B 100%;
- N=50, random weak edges: Path A 75.2%, Path B 91.4%;
- N=50, weight floor .25: Path B 99.6%;
- N=50, weight floor .50: Path B 100%.

This empirically supports the expected beta-min / sample-size transition.

### Controlled suppression / cancellation

Fixed structural coefficients beta=0.12, gamma=-0.15; only predictor
correlation changes.

At rho=0.8 the hidden parent's population marginal covariance is zero.

Endpoint vs Path-B hidden-parent recall:

- rho=.60: 95% -> 100%;
- rho=.70: 60% -> 100%;
- rho=.75: 50% -> 100%;
- rho=.80: 35% -> 100%.

The 100% rescue result persists in the tested 10%, 20%, 30%, and 40% candidate
budgets.

### Budget-matched generic Gaussian screening baselines

Five seeds, exact same realized Path-B budget:

N=20:

- Path-B: parent recall 95.5%, parent-complete targets 88.0%, F1 0.826;
- forward regression: 94.3%, 86.0%, F1 0.825;
- marginal top-k: 94.8%, 86.0%, F1 0.840;
- random: 41.9%, 20.0%, F1 0.432.

N=50:

- Path-B: parent recall 91.0%, parent-complete targets 58.8%, F1 0.716;
- forward regression: 90.6%, 58.0%, F1 0.717;
- marginal top-k: 87.3%, 45.2%, F1 0.733;
- random: 43.0%, 2.0%, F1 0.427.

Interpretation: on ordinary random Gaussian systems, marginal top-k can achieve
slightly higher final F1 even while retaining fewer true parents.  The Path-B
claim should therefore focus on **parent-preserving robustness**, especially
under cancellation/suppression, rather than universal F1 dominance.

## 8. Important negative / diagnostic results

- Exact keyed full-path reproduction can be achieved, but costs more tests than
  full oCSE and therefore is not the acceleration mechanism.
- Giant excluded-block CMI is a correct population certificate but loses
  finite-sample power badly as block dimension grows.
- Partitioning the excluded block explains the power loss but does not remove
  the beta-min problem for ultra-weak parents.
- Full-oCSE identity is not a correctness oracle:
  some edges omitted by Path B are baseline false positives, and in at least
  one primary seed Path B has higher truth F1 while reproducing only 90% of the
  full-oCSE edges.
- The repository's default Gaussian generator has no beta-min condition; some
  labeled true edges are arbitrarily weak after spectral-radius scaling.

## 9. Key workflow / artifact records

Research runs whose artifacts were explicitly inspected:

- Theory audit:
  https://github.com/mimosapudical/causationentropy/actions/runs/36497112250
  - artifact ID 11003821810.
- Controlled suppression:
  https://github.com/mimosapudical/causationentropy/actions/runs/36497471219
  - artifact ID 11004410277.
- Theory replication:
  https://github.com/mimosapudical/causationentropy/actions/runs/36497703466
  - artifact ID 11003992532.
- Gaussian primary + scaling replication:
  https://github.com/mimosapudical/causationentropy/actions/runs/36497861222
  - artifact ID 11004098531.
- Focused five-seed primary Gaussian:
  https://github.com/mimosapudical/causationentropy/actions/runs/36498263236
  - artifact ID 11004800521.
- Budget-matched Gaussian baselines:
  https://github.com/mimosapudical/causationentropy/actions/runs/36503290914
  - artifact ID 11006410588.

The active branch also contains focused workflows for beta-min, rescue
competition, iterative rescue, true-parent retention, suppression, and the full
Path-B v2 validation suite.

## 10. Key commits / milestones

This is a milestone list, not every experimental commit.  The complete history
is preserved on `experiment/path-b-benchmark-v2`.

- `1783a28` — correct standard-oCSE conditioning in the benchmark baseline.
- `fa707db` — preserve target history during keyed backward refinement.
- `fcda2ea` — separate true-edge detectability from baseline identity.
- `edfb2a0` — formalize baseline-identity vs causal-accuracy distinction.
- `83a173e` — formulate parent-sure weighted-Lasso theorem target.
- `ea0e25c` — BIC-compatible KKT screening certificate.
- `08114c0` — observable OLS-to-Lasso screening certificate.
- `2737507` — connect weighted RE to stable-VAR theory.
- `72ed02d` — one-shot rescue ranking-margin theorem.
- `8579543` — stable-VAR parent-sure corollary.
- `08d2c15` — Gaussian information-visibility concentration step.
- `04e2b67` — controlled beta-min phase transition.
- `ec3f5d7` — fixed-parent suppression design.
- `4ed9753` — replicated theory benchmarks.
- `c1212fa` — multi-seed Gaussian scaling replication.
- `88b2b98` — record five-seed primary result.
- `155347a` — budget-matched screening baselines.
- `d212caf` — close the replicated Gaussian scaling gate.

## 11. Upstream / paper gates that remain

### Upstream

1. Path A PR #45 needs maintainer review/merge.
2. A clean Path-B implementation branch must be based on the post-#45 upstream
   main so the Path-B PR does not duplicate Path A.
3. The isolated standard-oCSE correctness changes must be handled explicitly
   rather than smuggled into the Path-B PR.

### Paper

The main scientific core is complete enough to freeze:

- parent-sure formulation;
- restricted-oCSE theorem;
- weighted-screen theorem route;
- BIC finite-sample certificate in n>p;
- stable-VAR calibrated theorem route;
- rescue ranking-margin theorem;
- replicated beta-min and suppression mechanisms;
- replicated end-to-end Gaussian and scaling results.

Remaining manuscript boundaries:

- high-dimensional ordinary K-fold CV is a practical variant, not covered by
  the strongest support-containment theorem;
- nonlinear logistic/Poisson behavior is not ready for a general nonlinear
  claim and should be scoped out of the primary paper unless a separate method
  extension is developed;
- strongest screening-baseline interpretation should include both generic
  Gaussian and suppression/cancellation regimes.

## 12. Canonical next-state rule

Do **not** create another detector family.

Research changes from here should be accepted only if they close one of:

- clean Path-B production implementation;
- fair baseline comparison;
- theorem exposition / citation precision;
- robustness replication;
- manuscript figures / writing;
- upstream review requirements.

Everything else belongs in an appendix/diagnostic branch rather than the
primary method.


## 13. New baseline and clean-branch closure

### Linear suppression against forward regression

The budget-matched suppression comparison confirms that greedy forward
regression is a strong baseline in the purely linear cancellation regime.
At exact cancellation and the smallest tested realized budget (~12%), Path B
recovers the hidden parent in 100% of 20 seeds versus 95% for forward
regression; at 20% and 40% budgets both reach 100%.

This prevents claiming that conditional rescue is uniquely necessary for
linear-Gaussian suppression.

Workflow:
https://github.com/mimosapudical/causationentropy/actions/runs/36503565105

Artifact:
11005984921

### Nonlinear conditional cancellation

A harder construction makes the hidden parent marginally independent of the
target and removes its population linear residual correlation, while leaving
positive nonlinear conditional dependence.

Across 20 seeds:

- ~12% realized budget: Path-B KDE 100%, forward regression 25%, marginal KDE
  top-k 15%, random 10%;
- ~21%: 100%, 30%, 15%, 15%;
- 40%: 100%, 55%, 40%, 35%.

This is the strongest current mechanism result distinguishing information-
theoretic conditional rescue from linear forward screening.

Workflow:
https://github.com/mimosapudical/causationentropy/actions/runs/36504148473

Artifact:
11007120574

### Clean implementation branch

Canonical clean stacked Path-B branch:

`feature/information-screened-ocse-path-b`

It is based on `feature/information-lasso-path-a` rather than the
experiment-heavy research branch.

Public API addition:

`discover_network(..., method="information_screened", screen_retention=0.40)`

Core architecture:

1. remove target-history columns from the external candidate universe;
2. Information-LASSO endpoint;
3. one-shot conditional-CMI rescue;
4. restricted standard-oCSE refinement with target-history conditioning.

Key clean commits:

- `ac4af8e` — initial clean information-screened pathway;
- `195e804` — source-newline repair after CI caught a generated-code syntax
  issue; full test workflow then passed;
- `78e06ed` — align screened refinement with corrected standard-oCSE
  conditioning semantics;
- `f0cb894` — regression coverage for the conditioning prerequisite.

### Standalone standard-oCSE correctness prerequisite

Clean fork branch:

`fix/standard-ocse-conditioning-correctness`

Base: upstream main commit
`116073eef33f8036b0a6f090a9c3a260fb767b8a`.

It contains only:

- target-history candidate de-duplication;
- preservation of `Z_init` in backward refinement;
- preservation of `Z_init` in final edge reporting;
- regression tests.

Final fork test workflow passed on the validated branch.

An attempt to open the upstream PR through the GitHub connector returned
HTTP 403 `Resource not accessible by integration`.  This is a connector
permission limitation, not a code or CI failure.  The branch is ready for a
browser-side upstream PR.

Suggested upstream PR title:

`fix: preserve standard-oCSE conditioning semantics`

Do not combine this fix with Path A or Path B in the upstream review.

## 14. Submission artifact state

A research manuscript draft now exists at:

`experiments/PATH_B_PAPER_DRAFT.md`

It includes the abstract, method, theorem statements, replicated main tables,
controlled cancellation results, negative results, scope/limitations, related
work structure, and figure/table plan.

The project has therefore moved out of detector-search mode.  Remaining work is
review/integration and manuscript polishing rather than creation of another
algorithm family.


## 15. Current PR review surfaces and permission boundary

### Fork PR #7 — clean Path B

https://github.com/mimosapudical/causationentropy/pull/7

Title: `feat: add information-screened oCSE pathway`

Base: `feature/information-lasso-path-a`

Head: `feature/information-screened-ocse-path-b`

At creation it is a clean stacked diff containing only core/test changes, not
the research workflows.  The latest push workflow on the clean head passes.

This is the canonical review surface for Path B until upstream Path A #45 and
the standard-oCSE prerequisite are resolved.

### Fork PR #8 — standalone prerequisite review mirror

https://github.com/mimosapudical/causationentropy/pull/8

Title: `review: standard-oCSE conditioning correctness fix`

Base: `review/upstream-main-116073`, pinned exactly to upstream main commit
`116073eef33f8036b0a6f090a9c3a260fb767b8a`.

Head: `fix/standard-ocse-conditioning-correctness`.

This fork PR exists only because the connected GitHub App cannot open a
cross-repository upstream PR.  The standalone branch itself has full CI
success.

### GitHub integration permission boundary

Two attempted upstream writes returned HTTP 403
`Resource not accessible by integration`:

1. creating the standalone standard-oCSE correctness PR in
   `Center-For-Complex-Systems-Science/causationentropy`;
2. posting the Path-B research update comment to upstream issue #44.

Nothing is missing from the branches because of this permission failure.  The
remaining browser-side actions are administrative only.

Suggested cross-fork correctness compare page:

`https://github.com/Center-For-Complex-Systems-Science/causationentropy/compare/main...mimosapudical:fix/standard-ocse-conditioning-correctness?expand=1`

Suggested upstream fix PR title:

`fix: preserve standard-oCSE conditioning semantics`

The intended issue-#44 progress update is already represented by the body of
fork PR #7 plus the empirical/theory sections in this ledger, so it can be
copied without reconstructing results from chat history.


## 16. Paper figure generation

The frozen paper summary is committed at:

`experiments/results/path_b_paper_summary_20260929.json`

Figure generator:

`experiments/make_path_b_paper_figures.py`

It generates:

- `scaling.svg`;
- `linear_suppression.svg`;
- `nonlinear_cancellation.svg`.

The figure workflow completed successfully:

https://github.com/mimosapudical/causationentropy/actions/runs/36504903923

Artifact ID: `11006168783` (`path-b-paper-figures`).

This means the current manuscript numbers and first three core figures are
generated from one frozen machine-readable summary rather than copied manually
from separate CI artifacts.


## Final fork-side completion status — 2026-09-29

The fork-side implementation/research work is now frozen and organized.

### Clean review objects

- Path A upstream PR #45: OPEN, non-draft, mergeable/clean.
- Path B fork PR #7: OPEN, non-draft, mergeable/clean; 6 commits / 3 files.
- Standard-oCSE prerequisite fork PR #8: OPEN, non-draft, mergeable/clean;
  3 commits / 3 files.

### Upstream write-permission blocker

On 2026-09-29 the connected GitHub App was used to attempt both remaining
upstream writes:

1. create the cross-fork standard-oCSE correctness PR;
2. post the prepared Path-B progress comment to upstream issue #44.

GitHub rejected both with HTTP 403 `Resource not accessible by integration`.
No further repository-side action can remove this permission boundary.

Exact browser-ready actions are stored in
`experiments/UPSTREAM_ACTIONS_READY.md`.

### Canonical compact implementation map

See `experiments/PATH_B_IMPLEMENTATION_MAP.md`.

No additional Path-B algorithm branch should be created unless new evidence
invalidates the frozen Proposal -> conditional rescue -> restricted-oCSE
architecture.
