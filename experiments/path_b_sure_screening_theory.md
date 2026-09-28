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


## 8. Making the weighted-design condition interpretable

The weighted restricted-eigenvalue assumption can be related to an
unweighted-design condition when the true-parent information weights are
bounded away from zero.

Let

[
a_{min,S}=min_{jin S}a_j>0,
qquad
0le a_jle1.
]

For (gamma=D_aDelta),

[
|gamma_{S^c}|_1
le
|Delta_{S^c}|_1
le
3|Delta_S|_1
le
rac{3}{a_{min,S}}|gamma_S|_1.
]

Thus if the original design (X) satisfies an RE condition with constant
(kappa_X) on the enlarged cone

[
|gamma_{S^c}|_1
le
rac{3}{a_{min,S}}|gamma_S|_1,
]

then

[
rac{|widetilde XDelta|_2}{sqrt n}
=
rac{|Xgamma|_2}{sqrt n}
ge
kappa_X|gamma_S|_2
ge
a_{min,S}kappa_X|Delta_S|_2.
]

Therefore one may take

[
kappa_age a_{min,S}kappa_X.
]

A simple sufficient beta-min condition is consequently

[
eta_{min}
>
rac{3lambdasqrt{s}}
{a_{min,S}^2kappa_X^2},
]

where this expression is conservative because it replaces the actual weighted
RE constant by a lower bound and uses (a_jle1).

This formula exposes the intended role of the information weights: a true
parent whose marginal information weight collapses toward zero can make the
screening requirement dramatically harder, even when its structural
coefficient is not tiny.  That is precisely the regime targeted by conditional
rescue / iterative screening rather than by the one-shot weighted Lasso.

## 9. Dependence matters: the VAR probability step

The deterministic no-false-negative result does not require iid rows.  The
probability step does.

For a stable Gaussian VAR, the lagged design is serially dependent.  The right
theoretical bridge is therefore high-dimensional time-series Lasso theory,
not an iid concentration argument.

Basu and Michailidis (Annals of Statistics, 2015) derive nonasymptotic
deviation and estimation bounds for sparse stable Gaussian time series and VAR
transition-matrix estimation.  Their bounds use spectral properties of the
stationary process to quantify how temporal dependence degrades effective
concentration.  Wong, Li and Tewari extend this style of Lasso guarantee to
mixing sub-Gaussian time series under more general data-generating processes.

For Path B, the probability theorem should therefore have the schematic form

[
lambda_n
asymp
mathcal M(A,Sigma_arepsilon)
sqrt{rac{log p}{n}},
]

where (mathcal M) is a stability/dependence factor inherited from the
time-series deviation bound, together with a sample-RE event for the weighted
lagged design.  The resulting sufficient signal condition becomes

[
eta_{min}
gtrsim
rac{
mathcal M(A,Sigma_arepsilon)
sqrt{slog p/n}
}{
a_{min,S}^2kappa_X^2
}.
]

The exact constants and the correct Basu-Michailidis stability functional
still need to be instantiated carefully from their theorem; the expression
above is the target scaling, not yet a completed citation-level corollary.

This is a remaining proof task for a paper submission, but it is now sharply
localized: the algorithmic part of the no-false-negative argument is already
deterministic, and only the stable-VAR concentration/RE probability bound has
to be imported and adapted.


## 10. From Gaussian marginal visibility to a weight lower bound

For the Gaussian information estimator used by the primary benchmark,

[
I_j
=
-rac12log(1-widehatho_j^2),
]

where (widehatho_j) is the sample correlation between candidate (j) and
the target.  Let

[
g(r)=-rac12log(1-r^2),
qquad 0le r<1.
]

Assume a population marginal-visibility condition for the true parents,

[
min_{jin S}|ho_j|ge ho_{min}>0,
]

and suppose a uniform correlation-concentration event holds,

[
max_j|widehatho_j-ho_j|ledelta,
qquad
0<delta<ho_{min}.
]

If (ho_{max}=max_j|ho_j|) and
(ho_{max}+delta<1), then max-normalized information weights obey

[
a_{min,S}
ge
rac{
g(ho_{min}-delta)
}{
g(ho_{max}+delta)
}.
]

The proof is immediate from monotonicity of (g) on ([0,1)).

This gives a concrete route for closing the probability theorem:

1. use stable-Gaussian-VAR concentration to control all sample correlations;
2. convert that event into an explicit parent-weight lower bound;
3. combine the weight bound with a sample RE event for the lagged design;
4. use a noise-dominating penalty and the deterministic no-false-negative
   proposition.

The visibility condition is genuinely structural.  It fails under exact
suppression/cancellation, in which case no marginal-information weighting
scheme can have a nonvanishing parent-weight lower bound.  That case should be
handled by the conditional/iterative screening component rather than hidden
inside the one-shot Path-A theorem.


## 11. A BIC-compatible finite-sample screening certificate when n > p

The production Path-A BIC branch is used when the number of observations
exceeds the number of candidate predictors plus one.  In this regime there is
a simpler deterministic no-false-negative result that does **not** require the
penalty to dominate the score noise.

Center the weighted design and response so the intercept is removed, and let

[
G
=
rac{widetilde X^	opwidetilde X}{n}.
]

Assume (G) is invertible.  The Lasso KKT equations give

[
rac{widetilde X^	op
(y-widetilde Xwidehat	heta)}{n}
=
lambda z,
qquad
|z|_inftyle1.
]

Using
(y=widetilde X	heta^*+arepsilon),

[
G(widehat	heta-	heta^*)
=
rac{widetilde X^	oparepsilon}{n}
-
lambda z.
]

Therefore

[
|widehat	heta-	heta^*|_infty
le
|G^{-1}|_infty
left(
left|
rac{widetilde X^	oparepsilon}{n}
ight|_infty
+
lambda
ight).
]

Consequently, if

[
	heta_{min}
=
min_{jin S}|	heta_j^*|
>
|G^{-1}|_infty
left(
left|
rac{widetilde X^	oparepsilon}{n}
ight|_infty
+
lambda
ight),
]

then every true parent must have a nonzero fitted coefficient:

[
Ssubseteqoperatorname{supp}(widehat	heta).
]

This implication holds for **any realized nonnegative penalty**, including a
penalty chosen by BIC, because it is simply the KKT system evaluated at the
chosen solution.  No claim of BIC model-selection consistency is needed for
this finite-sample screening certificate.

This result is particularly well matched to the present implementation:

- the (n>p+1) branch uses `LassoLarsIC(criterion="bic")`;
- screening only needs no false negatives, not exact support recovery;
- a smaller BIC penalty can be favorable for screening even when it would be
  unsuitable for exact model selection.

The high-dimensional (pge n-1) CV branch still requires a different
argument, such as the RE-based result above or a dedicated high-dimensional
screening theorem.


## 12. Sharper observable OLS-to-Lasso screening certificate for n > p

The previous KKT bound compares the fitted Lasso coefficient to the unknown
population coefficient.  In the full-column-rank regime there is an even
sharper finite-sample statement.

Let

[
widehat	heta_{mathrm{OLS}}
=
G^{-1}rac{widetilde X^	op y}{n}.
]

The OLS normal equations give zero score, while the Lasso KKT equations give

[
G(
widehat	heta_{mathrm{Lasso}}
-
widehat	heta_{mathrm{OLS}}
)
=
-lambda z,
qquad
|z|_inftyle1.
]

Hence

[
|
widehat	heta_{mathrm{Lasso}}
-
widehat	heta_{mathrm{OLS}}
|_infty
le
lambda|G^{-1}|_infty.
]

Therefore, for any coordinate (j),

[
|widehat	heta_{mathrm{OLS},j}|
>
lambda|G^{-1}|_infty
quadLongrightarrowquad
widehat	heta_{mathrm{Lasso},j}
e0.
]

For an audit in which the true parent set (S) is known, the condition

[
min_{jin S}
|widehat	heta_{mathrm{OLS},j}|
>
lambda|G^{-1}|_infty
]

is a deterministic finite-sample certificate that the BIC-tuned Path-A Lasso
cannot drop any true parent.

This certificate uses the actual chosen BIC penalty and observed design.  It
does not require iid samples, a noise model, an RE condition, or asymptotic BIC
consistency.  Its limitation is the full-rank requirement (n>p), so it
complements rather than replaces the high-dimensional RE/time-series theorem.
