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
