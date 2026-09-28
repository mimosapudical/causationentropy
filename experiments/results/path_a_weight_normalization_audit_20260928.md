# Path-A weight-normalization audit — 2026-09-28

This is a fork-only numerical audit. It does not modify the Path-A method.

## Question

Path A currently uses

    w_j = I_j / sum_k I_k.

A possible numerical alternative is

    w_j = I_j / max_k I_k.

The two choices preserve every relative weight ratio and differ only by one
global positive feature-scale factor. The audit asks whether the current
sum-normalization is actually causing finite-precision model-selection or
convergence problems.

## Source-equivalent Gaussian audit

The audit used the repository's linear stochastic Gaussian generator,
marginal Gaussian information scores, and LassoLarsIC with BIC.

| N | Seeds | Targets checked | Support equality | Sum warnings | Max warnings | Mean max/sum design-scale ratio | Mean support size |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 3 | all | 1.000 | 0 | 0 | 2.166x | 1.917 |
| 50 | 3 | all | 1.000 | 0 | 0 | 3.369x | 3.467 |
| 100 | 2 | 10 per seed | 1.000 | 0 | 0 | 5.739x | 6.300 |
| 200 | 2 | 10 per seed | 1.000 | 0 | 0 | 12.069x | 15.550 |

For N=100 the sampled-target scale-ratio range was approximately
3.242x to 11.418x. For N=200 it was approximately 7.967x to 17.125x.

## Decision

The max-normalized design is numerically larger, as expected, but this audit
found:

- no convergence/runtime warning advantage;
- no selected-support change in any audited target;
- no evidence that current sum-normalization is a practical bottleneck.

Therefore:

    PATH_A_NORMALIZATION_CHANGE = NOT_JUSTIFIED

Do not modify PR #45 only to change the normalization constant. Keep the
sampled audit available for larger future settings if a real solver-scale
failure appears.

Reproducible audit command example:

    python -m experiments.path_a_weight_normalization_audit \
        --n-nodes 200 \
        --T 300 \
        --seeds 2 \
        --targets-per-seed 10
