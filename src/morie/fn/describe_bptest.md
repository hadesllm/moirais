# bptest - Breusch-Pagan test for heteroscedasticity

## WHAT IT DOES

Tests whether residual variance depends on a set of predictors X.
Regresses squared residuals on X; LM = n*R^2 ~ chi^2(p) under H0
of homoscedasticity.

## WHEN TO USE

- OLS diagnostic - check whether the equal-variance assumption holds.
- Pair with White's test for HC-robust SEs.

## WHEN NOT TO USE

- Non-Normal residuals - test is sensitive to non-Normality.
- Robust SE-based regression - heteroscedasticity already accounted for.

## ASSUMPTIONS

- Residuals from an OLS fit.
- X = predictors (sometimes a subset thought to drive variance).

## FORMULA

```
e^2 ~ alpha + beta*X    (auxiliary regression)
LM = n * R^2 ~ chi^2(rank(X))
```

## INPUTS / OUTPUTS

```
bptest(residuals, X) -> RichResult
  residuals    OLS residuals
  X            predictor matrix
  returns      .statistic (LM), df, .pvalue, R^2.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import bptest
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> X = rng.standard_normal((50, 2))
>>> e = X @ [0.3, 0.1] + 0.1*rng.standard_normal(50)
>>> bptest(e, X).pvalue
```

## COMMON MISTAKES

- Confusing variables that drive variance vs. variables that are
  in the original regression - they can differ.
- Testing too many predictors - power drops.

## REFERENCES

- Breusch & Pagan (1979).
