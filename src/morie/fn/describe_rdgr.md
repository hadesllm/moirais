# rdgr - Ridge regression (L2 regularised OLS)

## WHAT IT DOES

Fits a linear regression with an L2 (squared-coefficients) penalty on
the slope estimates. Shrinks coefficients toward zero, which reduces
variance at the cost of small bias - useful when predictors are
collinear or when n is not much larger than p.

## WHEN TO USE

- Multicollinear predictors (high VIF; see `vif`).
- p approaching n (high-dimensional regression).
- You want stable estimates rather than unbiased ones.
- As a baseline before LASSO (`lasr`) when you don't need sparsity.

## WHEN NOT TO USE

- Low-dimensional, well-conditioned data - plain OLS.
- You need automatic variable selection (zero coefficients) - use `lasr`.
- Mixed selection + shrinkage - use `elnetr`.

## ASSUMPTIONS

- Linear relationship between X and y.
- Errors approximately uncorrelated (no strong autocorrelation).
- Predictors typically standardized first (so the L2 penalty is fair
  across columns of different scales).

## FORMULA

```
beta_ridge = argmin || y - X*beta ||^2 + alpha * || beta ||^2
```

Closed form: beta = (X'X + alpha*I)^-1 * X'*y. As alpha -> 0 reduces
to OLS; as alpha -> infinity all coefficients shrink to zero.

## INPUTS / OUTPUTS

```
rdgr(X, y, alpha=1.0, fit_intercept=True) -> RichResult
  X              predictor matrix (n x p)
  y              response (n,)
  alpha          L2 strength (>= 0)
  fit_intercept  whether to include and centre an intercept
  returns        coefficient table, R^2, intercept, alpha.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import rdgr
>>> r = rdgr([[1,2],[3,4],[5,6],[7,8]], [1,2,3,4], alpha=1.0)
>>> r["coef"]
[0.36..., 0.36...]
```

## COMMON MISTAKES

- Not standardizing X before fitting - L2 penalty unfairly hits the
  large-scale columns.
- Picking alpha by hand - prefer cross-validated RidgeCV when alpha
  isn't theoretically motivated.
- Ridge coefficients can never reach zero - if you want selection use
  lasr or elnetr.

## REFERENCES

- Hoerl & Kennard (1970). Ridge regression: biased estimation for
  nonorthogonal problems.
- Hastie, Tibshirani & Friedman (2009) Elements of Statistical
  Learning, ch.3.
