# elnetr - Elastic net (mixed L1/L2)

## WHAT IT DOES

Combines the L1 penalty of LASSO with the L2 penalty of Ridge. The
mixing parameter l1_ratio controls how much each contributes:
l1_ratio=0 is pure Ridge, l1_ratio=1 is pure Lasso. Elastic net
selects variables AND shrinks them, while being more stable than
pure Lasso under collinearity.

## WHEN TO USE

- Highly collinear predictors AND you want some sparsity.
- Lasso alone gives unstable solutions (same problem solves to
  different non-zero subsets across resamples).
- Genome-wide / high-dimensional applications where groups of
  correlated predictors should enter or exit together.

## WHEN NOT TO USE

- Low-dim, low-collinearity - plain OLS is simpler.
- Predictor groups truly need group sparsity - use group lasso.

## ASSUMPTIONS

- Linear relationship; standardized predictors.
- Errors uncorrelated.

## FORMULA

```
beta = argmin || y - X*beta ||^2 + alpha * [(1-rho)/2 ||beta||^2 + rho ||beta||_1]
```

## INPUTS / OUTPUTS

```
elnetr(X, y, alpha=1.0, l1_ratio=0.5, fit_intercept=True, max_iter=10000)
  alpha       overall regularisation strength
  l1_ratio    rho in [0, 1]; 0=Ridge, 1=Lasso, 0.5=balanced (default)
  returns     coefficient table, R^2, nonzero count, l1_ratio interpretation.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import elnetr
>>> r = elnetr([[1,1],[2,2],[3,3],[4,4]], [1,2,3,4], alpha=0.1, l1_ratio=0.7)
>>> r["coef"]
```

## COMMON MISTAKES

- Setting l1_ratio to 0 or 1 and being surprised when behaviour
  matches Ridge or Lasso exactly - that's by design.
- Not cross-validating BOTH alpha and l1_ratio - they interact.

## REFERENCES

- Zou & Hastie (2005). Regularization and variable selection via the
  elastic net.
- Hastie et al. (2009) ch.3.
