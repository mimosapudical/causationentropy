# Path B: theory-backed screening and certification notes

This note records the current theoretical target for Path B.  It intentionally
separates population causal correctness from finite-sample reproduction of a
particular noisy oCSE path.

## 1. Parent-preserving screening is sufficient in the population

Let (I) be a target, (N_I) its causal parents, (B) a fixed baseline
conditioning set (for example, target self-history), and (W) the candidate
screen retained by Path B.  Assume the faithfully Markov conditions used by
Sun, Taylor & Bollt (2015).

If

[
N_I \subseteq B \cup W,
]

then aggregative discovery restricted to candidates in (W), while always
conditioning on (B), still terminates at a set (K) satisfying

[
N_I \subseteq B \cup K.
]

Proof sketch: if the restricted forward search stopped while a parent
(j \in N_I \setminus (B \cup K)) remained, then (j \in W \setminus K).
Theorem 2.2(c) of Sun et al. gives

[
C_{j\to I\mid B\cup K} > 0,
]

contradicting the stopping condition that every remaining candidate has zero
causation entropy.  Progressive removal then applies the same argument as
Lemma 2.5: true parents remain positive when removed from the conditioning set,
whereas nonparents have zero causation entropy once all parents are conditioned
upon.  The final population support is therefore (N_I) (apart from any parent
already fixed in (B)).

This is a restricted-candidate corollary of the original aggregative-discovery
and progressive-removal arguments; it does **not** require the restricted run
to reproduce the same intermediate forward path as the full run.

Reference: Jie Sun, Dane Taylor, Erik M. Bollt, "Causal Network Inference by
Optimal Causation Entropy", SIAM J. Applied Dynamical Systems 14(1), 2015,
DOI: 10.1137/140956166.

## 2. A population certificate for screen completeness

Let

[
R = V \setminus (B \cup W)
]

be the variables excluded by the screen.  Under the same assumptions,

[
N_I \subseteq B \cup W
\quad\Longleftrightarrow\quad
C_{R\to I\mid B\cup W}=0.
]

The forward direction follows directly from Theorem 2.2(b): once all parents
are conditioned upon, every remaining set contributes zero causation entropy.

For the reverse direction, suppose some parent
(j \in N_I \cap R) is missing.  Theorem 2.2(c) gives

[
C_{j\to I\mid B\cup W}>0.
]

By the conditional-mutual-information chain rule,

[
C_{R\to I\mid B\cup W}
=
C_{j\to I\mid B\cup W}
+
C_{R\setminus\{j\}\to I\mid B\cup W\cup\{j\}}
>0.
]

Thus zero residual block causation entropy is a population certificate that the
screen did not omit a causal parent.

## 3. Why a score-threshold witness rule cannot be generally safe

Conditional mutual information is not monotone in the conditioning set.
A simple Gaussian suppression/collider construction has (I(X;Y)=0), while
for (Z=X+Y+\epsilon), (epsilon\sim N(0,\sigma^2)),

[
\rho_{XY\mid Z}=-\frac{1}{1+\sigma^2},
\qquad
I(X;Y\mid Z)
=
-\frac12\log\left(1-(1+\sigma^2)^{-2}\right),
]

which diverges as (sigma\to0^+).  Therefore a generic rule of the form
"the current CMI is small, so this candidate can never matter later" cannot be
safe without additional structural assumptions.

## 4. Finite-sample exact-path certification is correct but not an accelerator

The certified working-set audit on the current Gaussian quick benchmark uses
keyed permutation tests and global violation checks.  It reproduces the full
keyed forward and final supports exactly:

| N | forward exact | final exact | witnesses added | all added in full forward closure | shuffle-test ratio vs full | observed-score ratio vs full |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 1.000 | 1.000 | 5 | 5/5 | 1.095 | 1.257 |
| 50 | 1.000 | 1.000 | 18 | 18/18 | 1.167 | 1.447 |

At N=20, none of the five required additions belongs to the full final support.
At N=50, only 6/18 do.  This supports the empirical distinction between final
support and transient forward witnesses.

However, a black-box exact certificate eventually has to rule out excluded
candidates that the full permutation procedure would test.  Without an
additional safe bound or cheaper test, exact finite-sample path certification
cannot be the source of test-count speedup.

## 5. A single giant block test has poor finite-sample power

The population block certificate above is theoretically exact, but the current
Gaussian finite-sample audit shows strong dimensionality/power loss when the
whole excluded set is tested as one regression block.

At 40% screening, one seed, T=300:

| N | parent-complete base screens | parent-incomplete base screens | base unsafe detected by giant block | weakest-parent-drop block detection | weakest-parent-drop singleton detection |
|---:|---:|---:|---:|---:|---:|
| 20 | 17 | 3 | 0.000 | 0.8125 | 1.000 |
| 50 | 34 | 16 | 0.000 | 0.000 | 0.588 |

Safe base screens produced zero block-test false positives in this diagnostic,
but the high-dimensional block test lacked power exactly where the original
screen missed weak parents.

Therefore the next finite-sample design should preserve the population
certificate while replacing a single high-dimensional block with
low-dimensional analytical checks / a partitioned certificate.  This follows
the same statistical motivation as recent work that limits conditioning-set
complexity in causal discovery.

## 6. Current design target

The current principled architecture is:

1. **Propose** a small working set (W) using Information-LASSO plus
   conditional rescue.
2. **Certify** residual information outside (W) using tests whose population
   null is implied by the oCSE Markov property.
3. If residual dependence is detected, **expand** (W) and re-certify.
4. Once certified, run restricted oCSE and progressive removal.

The proposal stage may be aggressive; correctness is not delegated to a
top-k/threshold witness heuristic.  The next experiment should compare
low-dimensional Gaussian partial-correlation/F checks (including partitioned
excluded sets) against the giant-block certificate in terms of omitted-parent
power, false expansions, and total test/evaluation cost.


## 7. Baseline identity is not the same as causal accuracy

The current Gaussian path-preservation audit reveals that several apparent
"screening misses" are not causally meaningful misses.

For the one-seed quick regimes used in the audit:

- N=20: the 40% screen omits three true graph edges; full keyed oCSE also omits
  all three.
- N=50: the screen omits 21 true graph edges across 16 targets; full keyed oCSE
  also omits all 21.
- N=50: six edges in the full keyed final support are absent from the screen;
  all six are false positives relative to the generating graph.

The synthetic generator assigns every graph edge an independent random weight
in (-1, 1) and then globally rescales the matrix to the requested spectral
radius.  It therefore has no beta-min condition: arbitrarily weak "true" edges
are present by construction.

Consequently, exact reproduction of the finite-sample full-oCSE graph is not an
appropriate scientific target by itself.  A screened method can differ from the
baseline by removing baseline false positives while preserving every
finite-sample-detectable true parent.

The theory target should therefore be stated in terms of parent-sure screening
and causal-support consistency.  Finite-sample full-graph identity remains a
diagnostic for path instability, not the primary correctness criterion.


## 8. Exact restricted-candidate corollary and original assumptions

The restricted-candidate argument uses exactly the population assumptions in
Sun, Taylor & Bollt (2015), Eq. (2.8):

1. stationarity with a continuous distribution;
2. temporally Markov dynamics;
3. spatially Markov dynamics with respect to the causal-parent set;
4. faithful Markov dependence, so changing which true parents are conditioned
   upon changes the relevant conditional distribution.

Theorem 2.2(b) states that once all causal parents are included in the
conditioning set, any remaining set contributes zero causation entropy.
Theorem 2.2(c) states that an unconditioned true parent contributes strictly
positive causation entropy.  Lemma 2.4 uses precisely this fact to prove that
aggregative discovery cannot stop while a true parent is still absent.  Lemma
2.5 then requires only a starting superset of the true parents and removes
nonparents.

Therefore the original proof extends immediately from candidate universe
(V) to any screened universe (W) satisfying

[
N_Isubseteq W.
]

More explicitly, run Algorithm 2.1 with every occurrence of
(V-K) replaced by (W-K).  If the restricted algorithm stops at (K_q)
while some (jin N_Isetminus K_q) remains, then (jin Wsetminus K_q)
and Theorem 2.2(c) gives

[
C_{j	o Imid K_q}>0,
]

contradicting the zero stopping value.  Hence (N_Isubseteq K_q).  Lemma 2.5
then applies unchanged to (K_q) and returns (N_I).

For higher-order lag models or explicit target-history conditioning, the clean
paper notation should use the standard expanded first-order state
representation noted by Sun et al.; the screen is then a subset of lagged
state coordinates and the same argument applies coordinate-wise.


## 8. Theorem target: weighted-screen containment + restricted-oCSE consistency

The theorem should be stated for **support containment**, not exact weighted-LASSO
model selection.

Consider the Gaussian linearized target equation

[
Y = X\beta^\star + \varepsilon,
\qquad
S = \operatorname{supp}(\beta^\star),
\qquad |S|=s.
]

Let the information weights be represented only up to a common positive scale:

[
q_j \propto I(X_j;Y), \qquad q_j \ge 0,
]

and define

[
D_q=\operatorname{diag}(q_1,\ldots,q_p),
\qquad
Z=XD_q,
\qquad
\gamma^\star=D_q^{-1}\beta^\star.
]

On coordinates with positive weights, Information-LASSO is ordinary Lasso in
the transformed design:

[
\widehat\gamma
\in
\arg\min_\gamma
\frac{1}{2n}\|Y-Z\gamma\|_2^2
+\lambda\|\gamma\|_1.
]

A sufficient support-containment route is:

1. **True-support positive weighting.**
   There is a constant (q_{\min}>0) such that

   [
   \min_{j\in S} q_j \ge q_{\min}.
   ]

   This is a marginal-faithfulness condition for the weighted first-stage
   screen.  It is intentionally not assumed for the rescue theorem below.

2. **Noise event.**

   [
   \|Z^T\varepsilon/n\|_\infty \le \lambda/2.
   ]

   For sub-Gaussian noise and normalized transformed columns this holds with
   high probability for
   (lambda\asymp\sigma\sqrt{\log(p)/n}).

3. **Restricted-eigenvalue / compatibility condition.**
   The transformed design (Z) has compatibility constant
   (kappa(S)>0) on the usual Lasso cone.

4. **Beta-min on the transformed coefficients.**
   A conservative sufficient condition is

   [
   \gamma_{\min}
   =
   \min_{j\in S}|\gamma_j^\star|
   >
   C\frac{\sqrt{s}\lambda}{\kappa(S)^2},
   ]

   where (C) is the constant inherited from the chosen Lasso
   (ell_2)-error bound.

Under these conditions, the Lasso estimation error is too small to set any
true transformed coefficient to zero, and therefore

[
S\subseteq\widehat S_{\mathrm{InfoLasso}}
]

on the stated high-probability event.

This proposition deliberately avoids an irrepresentable condition because
false positives are allowed at the screening stage.  It only needs a
no-false-negative guarantee.

Combining it with the restricted-candidate oCSE corollary gives the main
screened-oCSE statement:

[
P(S\subseteq W_n)\to1
\quad\text{and population-oCSE consistency}
\quad\Longrightarrow\quad
P(\widehat N_I^{\mathrm{restricted\ oCSE}}=N_I)\to1.
]

### Why rescue is theoretically necessary

The condition (q_j>0) for every true parent cannot hold universally because
conditional dependence is not monotone in the conditioning set.  A true parent
can have zero marginal mutual information but positive conditional mutual
information after a suitable revealing set is conditioned upon.

The rescue theorem therefore needs a separate assumption:

For every marginally hidden parent (j\in S\setminus W_0), there exists a
revealing set (R_j\subseteq W_0) of bounded size such that

[
I(X_j;Y\mid R_j)\ge \eta_n
]

for a detectable signal level (eta_n), and the rescue stage evaluates a
conditioning set containing such an (R_j).

Under uniform concentration of the conditional-information estimator and a
threshold below (eta_n), the rescue stage recovers all such hidden parents
with high probability.

This separates two assumptions cleanly:

- weighted Lasso handles parents that are **marginally visible**;
- conditional rescue handles parents that are **jointly visible but marginally
  hidden**.

The next theorem-facing experiment should therefore vary both:

[
\beta_{\min}\sqrt{n/\log p}
\quad\text{and}\quad
\text{marginal-visibility / suppression strength}.
]

The current beta-min phase-transition benchmark controls the first axis.


## 9. Rescue proposition aligned with the actual implementation

The current rescue stage does not search over arbitrary revealing sets.  If the
Information-LASSO endpoint is (W_0), it scores every excluded variable using

[
\widehat I_j(W_0)
=
\widehat I(X_j;Y\mid X_{W_0})
]

and retains the top (r=d-|W_0|) excluded variables required by the screen
budget (d).

A theorem for the implemented rescue therefore needs a **conditional ranking
margin**, not merely existence of some unspecified revealing set.

Let (H=S\setminus W_0) be the true parents missed by the endpoint.  Let
(a_r(W_0)) denote the (r)-th largest population conditional information
among excluded nonparents (with the convention (a_r=-\infty) if fewer than
(r) nonparents exist).  If

[
\min_{j\in H} I(X_j;Y\mid X_{W_0})
>
a_{r-|H|+1}(W_0) + 2\epsilon_n,
]

and the conditional-information estimates satisfy the uniform error bound

[
\max_{j\notin W_0}
\left|
\widehat I_j(W_0)-I_j(W_0)
\right|
\le \epsilon_n,
]

then every hidden parent ranks inside the rescue budget, hence

[
S\subseteq W_0\cup W_{\rm rescue}.
]

The indexing above can equivalently be written as: at most
(r-|H|) excluded nonparents may have population conditional information
larger than the weakest hidden parent by less than the estimation margin.

### Data-dependent endpoint issue

Because (W_0) is estimated from the same sample, a rigorous proof cannot use a
pointwise concentration statement that treats (W_0) as fixed.  Two clean
routes are available:

1. prove a uniform CMI concentration bound over all admissible conditioning
   sets with size at most the endpoint budget; or
2. introduce sample splitting / cross-fitting between proposal and rescue.

The current implementation uses the first-data-twice design.  Therefore
uniform concentration is the theorem-compatible route unless an explicit
cross-fit variant is introduced.

### Tuning-parameter issue

The standard weighted-Lasso support-containment argument assumes a deterministic
or high-probability tuning level satisfying approximately

[
\lambda \gtrsim
\|Z^T\varepsilon/n\|_\infty
\asymp
\sigma\sqrt{\log p/n}.
]

The production implementation selects (lambda) with BIC (or CV when
(p\ge n)).  BIC/CV are not automatically covered by the above theorem.
The beta-min audit therefore records the ratio between the selected penalty and
the empirical noise-dominance threshold.  If the production selector
systematically violates the required event, the final theorem must either:

- analyze the data-driven tuning rule directly; or
- expose a theorem-backed tuning mode while retaining BIC/CV as the practical
  default.

This is a genuine remaining theory/implementation alignment question.
