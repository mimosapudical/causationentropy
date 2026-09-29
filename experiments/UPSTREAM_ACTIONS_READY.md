# Upstream actions ready to paste

These are the only remaining actions blocked by the connected GitHub App's
cross-repository write permission.  All source branches and CI results already
exist.

## Action 1 — open the standalone standard-oCSE correctness PR

Compare page:

https://github.com/Center-For-Complex-Systems-Science/causationentropy/compare/main...mimosapudical:fix/standard-ocse-conditioning-correctness?expand=1

Title:

`fix: preserve standard-oCSE conditioning semantics`

Body:

---

## Summary

Fix two conditioning inconsistencies in the standard oCSE pathway:

1. lagged target-history columns are already present in `Z_init`, so standard
   oCSE should not test the same columns again as external candidates;
2. backward refinement and final edge reporting should retain `Z_init`,
   matching the conditioning semantics used by the standard forward phase.

These issues surfaced while validating the candidate-screened oCSE work
related to #44, but this PR is intentionally isolated from the
Information-LASSO / screening changes.

## What changes

- standard `discover_network` searches only lagged predictors from other
  variables and maps local candidate indices back to global lagged-feature IDs;
- `backward(..., Z_init=None)` optionally preserves the initial conditioning
  set;
- `standard_optimal_causation_entropy` passes its `Z_init` into backward
  refinement;
- standard edge CMI / permutation reporting uses the same target-history
  conditioning;
- `only_return_significant=False` reports only the external candidates that
  were actually tested.

The optional `Z_init=None` default preserves alternative-oCSE behavior.

## Tests

Regression coverage checks:

- target-history columns are excluded from the external candidate universe;
- reduced candidate indices map back correctly without self edges;
- report-all mode uses the actual tested candidate universe;
- every standard backward test retains `Z_init`;
- final standard edge reporting retains `Z_init`.

The full fork test workflow passes on commit
`bff82797c4faac9197de0e370d2d460e265e97cc`.

## Scope

No Information-LASSO or Path-B screening code is included.  This is a
standalone standard-oCSE correctness fix so the screening work can be reviewed
against a clean baseline.

---

Fork mirror for reviewing the exact diff first:

https://github.com/mimosapudical/causationentropy/pull/8

## Action 2 — post the Path-B progress update on upstream issue #44

Issue:

https://github.com/Center-For-Complex-Systems-Science/causationentropy/issues/44

Comment:

---

Path B update: I finished the research audit and have now frozen a clean
implementation for review.

The key change from the initial benchmark idea is that I no longer target exact
finite-sample full-oCSE path identity. Under the oCSE population argument, the
screen only needs to retain the causal-parent set; restricted aggregative
discovery + progressive removal can then do the refinement.

The current clean Path B is:

**Information-LASSO proposal -> one conditional-CMI rescue pass -> restricted
standard oCSE**

Clean stacked review PR in my fork:
https://github.com/mimosapudical/causationentropy/pull/7

It is based on the Path-A branch used by upstream #45, is mergeable/clean, and
the latest full test workflow passes.

Replicated Gaussian results:

- primary 5 seeds (N=12, T=300): 45.6% candidate retention, 100% screen
  coverage of the full-oCSE per-target support, truth F1 0.787 full vs 0.781
  screened, 1.57x aggregate speedup, 39.2% fewer total CMI evaluations;
- 3-seed scaling:
  - N=20: 39.8% fewer CMI evaluations, 1.54x speedup, delta F1 +0.014;
  - N=50: 44.4% fewer, 1.46x, delta F1 -0.007;
  - N=100: 48.7% fewer, 1.76x, delta F1 +0.005.

I also tested the reason for the conditional rescue rather than treating it as
a heuristic. In a controlled nonlinear cancellation construction where a true
parent is marginally invisible and has zero population linear residual
correlation, at the same ~12% candidate budget:

- Path B (KDE CMI rescue): 100% hidden-parent recall;
- forward regression: 25%;
- marginal KDE top-k: 15%.

One separate correctness issue surfaced while making the baseline comparison
fair: standard oCSE currently retests target-history columns already contained
in `Z_init` and drops `Z_init` during backward/output refinement. I
isolated that into `fix/standard-ocse-conditioning-correctness`, with full
tests passing, rather than hiding it inside Path B.

The experiment-heavy branch remains separate from the clean implementation.
I also wrote a research ledger and manuscript draft in the fork so the
negative results and theorem assumptions do not get lost.

I will keep Path B stacked in the fork until #45 and the standard-oCSE
prerequisite are resolved, then rebase it into a minimal upstream PR rather
than duplicating Path A.

---

## Action 3 — after upstream dependencies merge

Current upstream dependency order:

1. Path A upstream PR #45:
   https://github.com/Center-For-Complex-Systems-Science/causationentropy/pull/45
2. standalone standard-oCSE correctness PR from Action 1;
3. rebase `feature/information-screened-ocse-path-b` onto the new upstream
   main;
4. open the final Path-B upstream PR using fork PR #7 as the review template.

Do not open the current Path-B branch against upstream main before these
dependencies are resolved, because it would duplicate Path A and the
prerequisite fix in the upstream diff.
