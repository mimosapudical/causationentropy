# Path B sure-screening theory

This note states the current theorem target for the Information-LASSO screen and
separates deterministic implications from claims that still require a tuning
parameter or probability argument.

## 1. Setup

For one target in a stable linear-Gaussian VAR(1), after forming the lagged
design, write

[
y = Xeta^* + arepsilon,
qquad
S = operatorname{supp}(eta^*),
qquad
s = |S|.
]

Path A computes nonnegative marginal-information weights and solves a weighted
Lasso.  A common scaling of all weights changes only the Lasso penalty scale,
so for theory it is convenient to write relative weights

[
0 le a_j le 1,
qquad
a_j = I(X_j;Y) / max_k I(X_k;Y),
]

and the transformed design

[
widetilde X = X D_a,
qquad
D_a = operatorname{diag}(a_1,ldots,a_p).
]

Whenever (a_j>0), setting
(	heta_j=eta_j/a_j) turns

[
rac{1}{2n}|y-Xeta|_2^2
+lambdasum_jrac{|eta_j|}{a_j}
]

into the ordinary Lasso

[
widehat	heta
=
argmin_	heta
rac{1}{2n}|y-widetilde X	heta|_2^2
+lambda|	heta|_1.
]

The production implementation normalizes by the sum rather than the maximum.
That is a common positive rescaling of all columns and therefore represents the
same weighted-L1 path after rescaling (lambda).

## 2. Deterministic no-false-negative proposition

Define the cone

[
mathcal C(S,3)
=
{Delta:|Delta_{S^c}|_1le3|Delta_S|_1}.
]

Assume the realized weighted design obeys the restricted-eigenvalue condition

[
rac{|widetilde XDelta|_2}{sqrt n}
ge
kappa_a|Delta_S|_2
quad
	ext{for every }Deltainmathcal C(S,3),
]

and choose a penalty satisfying

[
lambda
ge
2left|
rac{widetilde X^	oparepsilon}{n}
ight|_infty.
]

If every true coordinate has nonzero information weight and

[
	heta_{min}
:=
min_{jin S}rac{|eta_j^*|}{a_j}
>
rac{3lambdasqrt{s}}{kappa_a^2},
]

then

[
Ssubseteqoperatorname{supp}(widehat	heta).
]

### Proof

Let (Delta=widehat	heta-	heta^*).  The Lasso basic inequality and the
noise-dominance condition give

[
rac{1}{2n}|widetilde XDelta|_2^2
+
rac{lambda}{2}|Delta_{S^c}|_1
le
rac{3lambda}{2}|Delta_S|_1.
]

Hence (Deltainmathcal C(S,3)), and

[
rac{1}{n}|widetilde XDelta|_2^2
le
3lambdasqrt{s}|Delta_S|_2.
]

Applying the restricted-eigenvalue inequality yields

[
|Delta_S|_2
le
rac{3lambdasqrt{s}}{kappa_a^2}.
]

If a true coordinate (jin S) were absent from the fitted support, then
(|Delta_j|=|	heta_j^*|), contradicting the beta-min inequality above.
Therefore no true coordinate can be zero in the fitted model.

This is a deterministic implication for the realized data and realized
weights.  The weights may be data-dependent; independence between the weight
estimator and the regression noise is not needed for this implication.  A
probability theorem follows by proving that the three required events
(nonzero parent weights, weighted RE, and noise-dominating (lambda)) hold
with probability tending to one.

## 3. Sure-screening corollary

Let (mathcal E_n) be the event that all assumptions in Section 2 hold.  If

[
P(mathcal E_n)	o1,
]

then Path A has the parent-sure property

[
P{Ssubseteqwidehat S_A}	o1.
]

Combining this with the restricted-candidate oCSE corollary recorded in
`path_b_screen_certificate_theory.md` gives

[
P{
widehat N_I^{mathrm{restricted oCSE}}=N_I
}
	o1
]

under the original oCSE faithful-Markov assumptions.

The scientific decomposition is therefore

[
	ext{parent-sure screen}
+
	ext{oCSE population consistency}
Longrightarrow
	ext{screened-oCSE consistency}.
]

## 4. Why marginal-information visibility is a real assumption

Production Path A sets (a_j=0) when the estimated marginal mutual information
is zero.  Such a column is multiplied by zero and receives an effectively
infinite weighted-L1 penalty.  Therefore Path A alone cannot be universally
parent-sure.

The Gaussian suppression construction already used in the project makes this
boundary explicit: it is possible to have a true parent with

[
I(X_j;Y)=0
]

but

[
I(X_j;Ymid Z)>0.
]

Thus a theorem for Path A must include an information-visibility condition such
as

[
min_{jin S} a_j ge a_{min}>0,
]

or an equivalent probabilistic lower bound.

This is not merely a proof artifact.  It is exactly the regime in which the
conditional-rescue stage is needed.

## 5. Deterministic rescue condition

Let (A) be the Path-A endpoint, let (M=Ssetminus A) be the missed parents,
and let the rescue score for an excluded variable be

[
c_j=I(X_j;Ymid X_A).
]

Suppose the rescue stage has (q) available slots.  Define

[
c_{min,M}=min_{jin M}c_j
]

and the conditional competition number

[
h(A)
=
left|
{k
otin Scup A:c_kge c_{min,M}}
ight|.
]

If

[
qge |M|+h(A),
]

then a descending-CMI rescue necessarily includes every missed parent.

Under the oCSE faithful-Markov assumptions, every omitted true parent has
positive conditional causation entropy for a conditioning set that does not
yet contain all parents.  The unresolved finite-sample question is therefore
not whether missed parents can become conditionally visible; it is how many
nonparents outrank the weakest missed parent and how that competition scales
with graph sparsity, sample size, and signal strength.

This condition is intentionally stated as a diagnostic identity, not yet as a
new asymptotic theorem.

## 6. Tuning-parameter gap in the current implementation

The deterministic proposition requires

[
lambda
ge
2|widetilde X^	oparepsilon/n|_infty.
]

The current implementation does **not** choose (lambda) from this formula:

- when (n>p+1), it uses `LassoLarsIC(criterion="bic")`;
- when (pge n-1), it uses ordinary K-fold `LassoCV`.

There is literature establishing BIC-type selection consistency for penalized
estimators and adaptive Lasso under additional assumptions, including
stationary autoregressive settings.  There is also theory showing strong
prediction/risk guarantees for cross-validated Lasso.  However, those results
are not automatically identical to the weighted-information implementation
here, and ordinary K-fold CV is not generally a model-selection-consistent
device.

Therefore the current paper must not claim that the proposition above already
proves the exact production solver.

The immediate empirical audit is to record, for the actual BIC/CV solution,

[
r_lambda
=
rac{lambda_{mathrm{chosen}}}
{2|widetilde X^	oparepsilon/n|_infty}
]

together with parent recall.  Values (r_lambdage1) satisfy the classical
noise-dominance side of the proposition; values below one identify precisely
where the fixed-(lambda) proof does not apply.

## 7. Literature anchors

The proof structure above follows the standard restricted-eigenvalue/oracle-
inequality line for Lasso, especially Bickel-Ritov-Tsybakov and the comparison
of Lasso design assumptions by van de Geer and Bühlmann.

Related screening and tuning work that motivates the next theorem step:

- Fan & Lv: sure independence screening for ultrahigh-dimensional linear
  models.
- Fan, Samworth & Wu: iterative sure independence screening for jointly
  important but marginally weak variables.
- Zou: adaptive Lasso and oracle properties.
- Chetverikov, Liao & Chernozhukov: nonasymptotic rates for cross-validated
  Lasso.
- Hui, Warton & Foster: ERIC, a tuning rule designed for selection consistency
  of adaptive Lasso, with a wider consistency regime than ordinary BIC.
- Kock: BIC-tuned adaptive Lasso can be selection consistent in stationary and
  nonstationary autoregressions, while also illustrating the loss of power
  against sufficiently shrinking alternatives.

The next paper-level step is not to import these results verbatim, but to prove
a probability bound for the information-weight event and connect it to a
weighted-design RE/beta-min condition in the stable Gaussian VAR setting.
