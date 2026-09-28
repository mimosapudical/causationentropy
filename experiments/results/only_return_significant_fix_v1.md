# only_return_significant contract fix v1

Branch:

    experiment/only-return-significant-fix-v1

## Defect

discover_network documents:

    only_return_significant=True

as returning only statistically significant links.

The implementation already computes a final shuffle_test for every selected
edge, but then ignores test_result["Pass"]:

- True mode adds the edge unconditionally;
- report-all mode marks every selected edge significant=True unconditionally;
- non-selected candidates are marked significant=False unconditionally even
  though a final shuffle decision was just computed for them.

This is especially visible for LASSO and Information-LASSO, whose selected
support is not itself a shuffle-test significance decision.

## Minimal correction

Selection is unchanged.

Only graph reporting changes:

- True mode adds a selected edge only when its final shuffle Pass is true;
- False mode stores significant=bool(test_result["Pass"]) for every reported
  candidate.

This cleanly separates two concepts:

    method support selection
    final reported statistical significance

## Validation

Run:

    pip install -e ".[dev]"
    python -m experiments.run_only_return_significant_fix_v1

The focused tests force a LASSO-selected edge to have final Pass=False and
verify both public modes.

Do not propose upstream until full default pytest passes.
