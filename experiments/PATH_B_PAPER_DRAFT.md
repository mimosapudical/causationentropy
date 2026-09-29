# Screened Optimal Causation Entropy: Parent-Sure Candidate Reduction for Scalable Causal Network Inference

> Research manuscript draft. Claims are intentionally limited to results
> already supported by the current theory notes and replicated experiments.

## Abstract

Optimal Causation Entropy (oCSE) provides an information-theoretic procedure
for recovering causal parents in networked dynamical systems, but its forward
search repeatedly evaluates conditional mutual information (CMI) over the full
candidate universe and can become expensive as network dimension grows.  We
study whether the candidate universe can be reduced before oCSE without
requiring exact reproduction of the finite-sample greedy path.

Our key observation is that population oCSE does not require path identity:
under the original faithful-Markov assumptions, any candidate screen that
contains all causal parents is sufficient for restricted aggregative discovery
followed by progressive removal to recover the parent set.  This motivates a
parent-sure screening objective.

We propose a three-stage screened-oCSE procedure: an information-weighted Lasso
proposal, a one-shot conditional-information rescue pass, and standard oCSE
restricted to the retained working set.  The proposal is designed for parents
that are marginally visible, while the rescue handles variables that are
marginally weak or cancelled but become informative conditionally.  We give a
deterministic weighted-Lasso no-false-negative result under restricted
eigenvalue and beta-min conditions, a stable-Gaussian-VAR probability route,
finite-sample BIC-compatible certificates in the full-rank regime, and a
ranking-margin guarantee for the rescue stage.

Across five primary Gaussian seeds, the method retains 45.6% of candidates,
reduces total CMI evaluations by 39.2% and shuffle-CMI evaluations by 41.4%,
achieves a 1.57x aggregate end-to-end speedup, and changes mean truth-level F1
by only -0.006 relative to full oCSE.  In three-seed scaling experiments from
20 to 100 nodes, candidate retention remains near 40%, CMI reduction increases
from 39.8% to 48.7%, and aggregate speedup reaches 1.76x at 100 nodes while F1
remains within 0.014 of full oCSE.  Controlled beta-min experiments show the
predicted transition in parent containment.  Under exact linear marginal
cancellation, the rescue restores hidden-parent recall from 35% to 100%.
Under a nonlinear cancellation construction in which both marginal
information screening and linear forward regression are weak, screened-oCSE
recovers the hidden parent in 100% of 20 seeds at a roughly 12% candidate
budget, compared with 15% for marginal KDE screening and 25% for forward
regression.

These results support screening for parent containment rather than finite-sample
path identity as a principled route to scalable information-theoretic causal
network inference.

## 1. Motivation

oCSE alternates between:

1. aggregative discovery, which repeatedly searches remaining variables for
   positive conditional causation entropy; and
2. progressive removal, which prunes redundant variables from the discovered
   superset.

The expensive component is not merely the final number of edges.  It is the
number of candidate CMI scores and permutation-test evaluations performed while
searching a large candidate universe.

A natural acceleration is to pre-screen variables.  The central difficulty is
that generic marginal screening is not safe for causal discovery: conditional
mutual information is not monotone in the conditioning set, and a causal
parent may be marginally weak or exactly cancelled.

The paper therefore asks:

> What is the weakest screening property required by oCSE, and can that
> property be achieved cheaply enough to reduce CMI work without sacrificing
> causal-parent recovery?

## 2. Main conceptual shift: parent containment, not path identity

Let (N_I) denote the causal-parent set of target (I), (B) a fixed baseline
conditioning set such as target history, and (W) the retained candidate
screen.

Under the original oCSE faithful-Markov assumptions, the relevant condition is

[
N_I subseteq B cup W.
]

If restricted aggregative discovery stopped while a parent
(jin N_Isetminus(Bcup K)) remained, then (jin Wsetminus K), and the
oCSE parent-positivity result implies

[
C_{j	o Imid Bcup K}>0,
]

contradicting the stopping condition.  Progressive removal then removes
nonparents once all parents are present.

Therefore the scientific correctness target is a **parent-superset screen**.
Exact equality with the full finite-sample oCSE path is neither necessary nor,
empirically, always desirable.

## 3. Method

For each target:

### 3.1 Information-weighted proposal

Compute a marginal information score for each external lagged candidate:

[
q_j propto I(X_j;Y).
]

Solve the weighted Lasso

[
widehateta
in
argmin_eta
rac{1}{2n}|Y-Xeta|_2^2
+
lambdasum_j rac{|eta_j|}{q_j}.
]

The nonzero endpoint defines (W_0).

### 3.2 One-shot conditional rescue

For each excluded candidate, compute

[
widehat c_j
=
widehat I(X_j;Ymid X_{W_0}).
]

Fill the retained working-set budget by descending (widehat c_j).  This step
targets parents that are hidden from a purely marginal proposal.

### 3.3 Restricted standard oCSE

Run standard oCSE only on the retained external candidates, while preserving
the target-history baseline throughout forward selection, backward refinement,
and output significance evaluation.

## 4. Theory

### Proposition 1: restricted-candidate oCSE

Under the Sun--Taylor--Bollt population assumptions, if (N_Isubseteq W),
standard oCSE restricted to (W) returns (N_I).  This is a direct
restricted-universe corollary of aggregative discovery and progressive
removal.

### Proposition 2: weighted-Lasso parent containment

Write the weighted design as

[
widetilde X = X D_q,
qquad
	heta^star=D_q^{-1}eta^star.
]

On the usual Lasso cone, suppose the realized weighted design has restricted
eigenvalue (kappa_q>0) and

[
lambda
ge
2left|
rac{widetilde X^	oparepsilon}{n}
ight|_infty.
]

Then the standard basic inequality yields an (ell_2) error bound of order

[
|widehat	heta_S-	heta^star_S|_2
lesssim
rac{lambdasqrt{s}}{kappa_q^2}.
]

Thus the beta-min condition

[
	heta_{min}
>
Crac{lambdasqrt{s}}{kappa_q^2}
]

implies

[
Ssubseteqoperatorname{supp}(widehat	heta).
]

The theorem is deliberately a no-false-negative result; it does not require
exact support selection.

### Proposition 3: stable Gaussian VAR route

For a stable Gaussian VAR, time-series concentration and sample-RE results give
high-probability control of the score and Gram matrix with a
stability-dependent factor.  Combined with a nonvanishing parent marginal
correlation condition, uniform covariance concentration gives a lower bound on
the true-parent information weights.

The resulting sufficient signal scale has the schematic form

[
eta_{min}
gtrsim
rac{
mathcal M(A,Sigma_arepsilon)sqrt{slog p/n}
}{
a_{min,S}^{2}kappa_X^2
}.
]

The detailed derivation and constants are recorded in
`path_b_sure_screening_theory.md`.

### Proposition 4: BIC-compatible finite-sample certificate for (n>p)

When the weighted design Gram matrix (G) is invertible, the Lasso KKT system
implies

[
|widehat	heta-	heta^star|_infty
le
|G^{-1}|_infty
left(
left|
rac{widetilde X^	oparepsilon}{n}
ight|_infty
+lambda
ight).
]

This implication holds for the realized BIC-selected penalty and therefore
provides a finite-sample screening certificate without claiming BIC exact-model
selection consistency.

A sharper observable OLS-to-Lasso bound is also available in the full-rank
regime.

### Proposition 5: one-shot rescue ranking margin

Let (M=Ssetminus W_0) be the true parents missed by the proposal and let the
rescue have (q) slots.  Define population conditional scores

[
c_j^star=I(X_j;Ymid X_{W_0}).
]

If the weakest missed parent exceeds the relevant nonparent competition
boundary by margin (Delta_{m rescue}>0), and

[
max_{j
otin W_0}
|widehat c_j-c_j^star|
<
Delta_{m rescue}/2,
]

then the top-(q) rescue contains every missed parent.

The current manuscript treats the required uniform CMI estimation event as an
explicit condition.  A full data-dependent-set concentration lemma would
strengthen the theorem but is not needed to state the deterministic ranking
implication.

## 5. Why pure marginal screening is insufficient

For jointly Gaussian variables, suppression can make a true structural parent
marginally uninformative while leaving positive conditional information.  This
is not a numerical artifact but a structural limitation of any purely
marginal screening rule.

The controlled linear cancellation experiment holds both parent coefficients
fixed and varies only predictor correlation.  At exact cancellation, the
Information-Lasso endpoint finds the hidden parent in 35% of seeds, while
conditional rescue restores it in 100%.

## 6. Nonlinear conditional cancellation

To distinguish the proposed conditional-information rescue from linear forward
screening, use

[
X_1sim N(0,1),qquad
f(X_1)=X_1^2-1,
]

[
X_2=-a f(X_1)+U,
]

and

[
Y=X_2+a f(X_1)+arepsilon=U+arepsilon.
]

(X_1) is a direct structural parent of (Y), but is marginally independent
of (Y) in the population.  Because (f) is even, linear residual
correlation is also zero in the population.  Conditioning on (X_2), however,
reveals the nonlinear dependence.

Across 20 seeds:

| realized budget | Path-B KDE hidden recall | forward regression | marginal KDE top-k | random |
|---:|---:|---:|---:|---:|
| ~12% | **100%** | 25% | 15% | 10% |
| ~21% | **100%** | 30% | 15% | 15% |
| 40% | **100%** | 55% | 40% | 35% |

The visible parent is recovered in 100% of runs by the three nonrandom
screening procedures.

This experiment isolates the role of nonlinear conditional information rather
than generic signal strength.

## 7. Primary Gaussian results

### 7.1 Five-seed primary benchmark

N=12, T=300, 50 shuffles:

- candidate retention: 45.6%;
- screen recall of the full-oCSE support: 100% on all five seeds;
- refined/full-oCSE edge recall: 96.6% mean;
- full truth F1: 0.787 ± 0.136;
- screened-oCSE truth F1: 0.781 ± 0.124;
- mean F1 difference: -0.006;
- end-to-end aggregate speedup: 1.57x;
- total CMI reduction: 39.2%;
- shuffle-CMI reduction: 41.4%.

One seed improves truth F1 even while reproducing only 90% of the full-oCSE
edges, illustrating why finite-sample baseline identity is not the primary
correctness target.

### 7.2 Scaling

| N | T | Full F1 | Screened F1 | Delta | Retention | Screen support recall | CMI reduction | Shuffle reduction | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 300 | 0.793 | 0.807 | +0.014 | 42.1% | 99.5% | 39.8% | 43.4% | 1.54x |
| 50 | 300 | 0.723 | 0.717 | -0.007 | 40.8% | 99.3% | 44.4% | 45.2% | 1.46x |
| 100 | 500 | 0.710 | 0.715 | +0.005 | 40.4% | 99.1% | 48.7% | 45.4% | 1.76x |

## 8. Beta-min transition

The repository's default Gaussian generator has no beta-min condition because
edge weights may be arbitrarily close to zero before global spectral-radius
rescaling.  Controlled floor experiments therefore separate finite-sample
detectability from graph labels.

At T=300 over five seeds:

| setting | Proposal parent recall | + rescue parent recall |
|---|---:|---:|
| N=20, raw weak-edge generator | 87.4% | 95.2% |
| N=20, floor .25 | 98.6% | 100% |
| N=20, floor .50 | 100% | 100% |
| N=50, raw weak-edge generator | 75.2% | 91.4% |
| N=50, floor .25 | 90.0% | 99.6% |
| N=50, floor .50 | 97.2% | 100% |

This supports the expected transition with increasing
(eta_{min}sqrt{n/log p}).

## 9. Budget-matched screening baselines

All screens receive exactly the same realized candidate budget before the same
restricted-oCSE refinement.

On ordinary random Gaussian systems, marginal top-k can have slightly better
final F1 than Path-B, while Path-B has the strongest parent-containment
statistics.

N=50, five seeds:

- Path-B parent recall 91.0%, parent-complete targets 58.8%, final F1 0.716;
- forward regression 90.6%, 58.0%, F1 0.717;
- marginal top-k 87.3%, 45.2%, F1 0.733;
- random 43.0%, 2.0%, F1 0.427.

This prevents an overclaim of universal F1 dominance.  The proposed method is
best motivated by preserving the parent-superset property, especially in
conditional-cancellation regimes.

## 10. Negative results that shaped the method

### Exact finite-sample path certification

A keyed certificate can reproduce the full forward and final supports exactly,
but it requires at least as many expensive tests as the full procedure.
Therefore exact path certification is a correctness diagnostic, not the
acceleration mechanism.

### Giant excluded-block certificate

The population identity is exact, but a high-dimensional excluded-block CMI
test loses finite-sample power where weak parents are most difficult.

### Partitioned certificate

Partitioning improves the dimensionality problem but does not remove the
fundamental beta-min limitation.

### Iterative rescue

The iterative variant was investigated but did not justify replacing the
simpler one-shot conditional rescue as the default architecture.

These negative results are useful because they remove unnecessary algorithmic
branches from the final method.

## 11. Computational interpretation

Let (p) be the full candidate count, (d=ho p) the screen size, (k) the
number of accepted forward variables, and (R) the number of permutation
shuffles.

The screen adds (O(p)) information-estimation work and no permutation tests.
Restricted oCSE then moves the expensive permutation component from (p)
candidates to (d) candidates.  In sparse regimes the observed-score work
improves approximately linearly in (ho); dense worst-case forward scoring
can improve quadratically in (ho).

The replicated experiments show that the empirical CMI-work reduction grows as
the candidate universe grows.

## 12. Related work

The paper should position itself relative to four lines of work:

1. **Optimal Causation Entropy**
   - Sun, Taylor & Bollt: population optimal-causation-entropy principle,
     aggregative discovery, progressive removal.

2. **Sure screening**
   - Fan & Lv: sure independence screening.
   - iterative/conditional screening work showing why marginally weak but
     jointly important variables require conditional information.
   - forward-regression screening and strong-screening consistency results.

3. **High-dimensional time-series Lasso**
   - Basu & Michailidis: nonasymptotic regularized estimation for sparse stable
     Gaussian time series.
   - related mixing/sub-Gaussian time-series Lasso theory.

4. **Reducing conditional-independence computation**
   - recent causal-discovery work reducing the number or dimensionality of
     conditional independence tests.
   - the distinction here is candidate-space screening specifically tied to the
     parent-superset property of oCSE.

## 13. Scope and limitations

The primary paper should make the following boundaries explicit:

- The strongest theorem and replicated end-to-end results concern sparse
  Gaussian linear dynamical systems.
- The rescue mechanism itself can use nonlinear information estimators and the
  nonlinear cancellation experiment demonstrates that capability, but a full
  nonlinear end-to-end consistency theorem is not claimed.
- The production low-dimensional BIC branch has deterministic finite-sample
  screening certificates; the high-dimensional ordinary K-fold LassoCV branch
  remains a practical tuning variant without a matched support-containment
  theorem.
- The fixed retention budget is currently a practical operating point, not an
  oracle-optimal adaptive budget.
- Full-oCSE output is a baseline, not ground truth.

## 14. Claimed contributions

A defensible contribution list is:

1. A restricted-candidate oCSE result showing that parent containment, rather
   than greedy path identity, is sufficient for population correctness.
2. A parent-sure screened-oCSE architecture combining information-weighted
   proposal, conditional rescue, and restricted oCSE.
3. Theory connecting weighted-Lasso beta-min / RE conditions and stable-VAR
   concentration to no-false-negative screening, together with finite-sample
   certificates in the full-rank BIC regime.
4. A ranking-margin analysis of one-shot conditional rescue.
5. Controlled linear and nonlinear cancellation experiments showing why pure
   marginal or linear screens can miss causal parents.
6. Replicated end-to-end evidence of roughly 40--49% CMI-work reduction and
   1.46--1.76x speedup across 20--100 nodes with small truth-level F1 changes.
7. Negative results clarifying why exact finite-sample path certification and
   giant-block certificates are not effective acceleration mechanisms.

## 15. Figure and table plan

**Figure 1:** method schematic: proposal -> conditional rescue -> restricted
oCSE.

**Figure 2:** beta-min phase transition versus
(eta_{min}sqrt{n/log p}).

**Figure 3:** linear suppression: marginal MI collapse and rescue recovery.

**Figure 4:** nonlinear cancellation: hidden-parent recall versus screen budget
for Path-B, marginal KDE, forward regression, and random.

**Figure 5:** scaling: candidate retention, CMI reduction, runtime speedup, and
truth F1 versus network size.

**Table 1:** five-seed primary Gaussian.

**Table 2:** budget-matched screening baselines.

**Appendix:** exact-path certificate, block certificate, partitioned
certificate, iterative-rescue negative results, tuning diagnostics.

## 16. Remaining work before submission

The remaining work is no longer method search.

Required:

1. polish theorem notation and verify citation-level constants;
2. finish the clean production branch and upstream dependency handling;
3. turn the existing JSON results into publication figures with confidence
   intervals;
4. add a small number of robustness sweeps over density / noise if compute
   permits;
5. write the final related-work and proof appendices.

Optional extensions:

- a theorem-calibrated high-dimensional tuning mode;
- nonlinear end-to-end generalization beyond the current mechanism experiment;
- adaptive data-driven retention budgets.
