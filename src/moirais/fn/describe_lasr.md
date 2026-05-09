# lasr - Lasso regression (L1 regularised OLS)

## WHAT IT DOES

Fits a linear regression with an L1 (absolute-value) penalty. The L1
penalty has the property that it drives some coefficients exactly to
zero - performing automatic variable selection.

## WHEN TO USE

- High-dimensional regression where you suspect a sparse solution
  (only a few predictors actually matter).
- You want a single procedure that does selection AND shrinkage.
- Predictors numerous and possibly collinear.

## WHEN NOT TO USE

- All predictors are believed important - use ridge (`rdgr`).
- You want prediction stability across slightly-different datasets -
  ridge or elastic net are more stable than pure lasso when predictors
  are highly collinear.
- You need exact post-selection inference - lasso's selected predictors
  have biased standard errors; use lasso + a debiased / desparsified
  procedure.

## ASSUMPTIONS

- Linear relationship.
- Predictors standardized.
- Errors uncorrelated.

## FORMULA

```
beta_lasso = argmin || y - X*beta ||^2 + alpha * || beta ||_1
```

No closed form; solved by coordinate descent or LARS.

## INPUTS / OUTPUTS

```
lasr(X, y, alpha=1.0, fit_intercept=True, max_iter=10000) -> RichResult
  alpha        L1 strength (> 0; for alpha=0 use plain OLS)
  returns      coefficient table marked selected/zeroed, R^2, nonzero count.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import lasr
>>> r = lasr([[1,0],[2,1],[3,0],[4,1],[5,0]], [1,2,3,4,5], alpha=0.05)
>>> r["nonzero"]   # how many predictors survived
```

## COMMON MISTAKES

- Picking alpha by hand without cross-validation - LASSO is more
  alpha-sensitive than Ridge.
- Forgetting to standardize X - same fairness issue as Ridge.
- Reporting unbiased SEs from a fitted lasso - they're biased; use
  bootstrap or a debiased estimator.

## REFERENCES

- Tibshirani (1996). Regression shrinkage and selection via the lasso.
- Hastie et al. (2009) ch.3.
