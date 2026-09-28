# Gaussian constant-feature fix v1

Branch:

    experiment/gaussian-constant-feature-fix-v1

## Defect

The Gaussian mutual-information path computes

    I(X;Y) = 0.5 * (logdet R_X + logdet R_Y - logdet R_XY).

correlation_log_determinant maps a singular or non-finite correlation matrix to
the finite sentinel -1000.

For an exactly constant scalar X and a random scalar Y:

- the one-dimensional X term returns 0;
- the one-dimensional Y term returns 0;
- R_XY contains NaNs and maps to -1000;
- the resulting MI is approximately 500 nats.

A deterministic constant predictor should carry zero mutual information.

## Minimal correction

Before constructing the correlation matrix,
correlation_log_determinant removes only columns that are exactly constant
across all samples.

This is intentionally narrower than a variance threshold:

- exactly constant dimensions contribute no information;
- near-constant nonconstant dimensions are preserved;
- highly correlated and duplicate nonconstant dimensions remain singular and
  retain the existing very-negative log-determinant behavior.

## Formula-level smoke

- constant X vs random Y: 500 -> 0 nats;
- adding a constant nuisance coordinate to an informative X leaves MI unchanged
  exactly in the smoke test;
- a near-constant nonconstant coordinate, 1 + 1e-12 * t, was preserved and the
  two-column perfectly correlated example remained strongly singular
  (logdet about -20.66).

## Validation

Run:

    pip install -e ".[dev]"
    python -m experiments.run_gaussian_constant_feature_fix_v1

Status:

    FORMULA_SMOKE = PASS
    PACKAGE_PYTEST = NOT_VERIFIED

Do not open an upstream PR until the focused and full default suites pass.


## CMI consistency

Gaussian conditional mutual information previously used a private _detcorr
implementation instead of correlation_log_determinant, so the exact-constant
handling applied to MI did not propagate to CMI. The branch now routes both MI
and CMI through the same constant-safe log-determinant implementation.
