# waldjt - Wald joint test (linear restrictions)

## WHAT IT DOES

Tests a set of linear restrictions on regression coefficients
(beta_1 = beta_2 and beta_3 = 0, simultaneously). The Wald statistic
is asymptotically chi^2 under H0.

## WHEN TO USE

- Joint hypothesis testing in regression (e.g., are these k coefficients
  all zero together?).
- Testing nested model restrictions.
- Standard inference in GLMs and survival models.

## WHEN NOT TO USE

- Highly nonlinear restrictions - convert to a nonlinear test.
- Small samples - use exact / permutation alternatives.

## ASSUMPTIONS

- Asymptotic Normality of beta_hat.
- The covariance matrix Sigma is consistent.

## FORMULA

```
W = (R*beta - r)' * (R*Sigma*R')^-1 * (R*beta - r)
W ~ chi^2(rank(R)) under H0
```

## INPUTS / OUTPUTS

```
waldjt(beta, cov, R, r) -> RichResult
  beta     fitted coefficients
  cov      coefficient covariance matrix
  R, r     restriction matrix and target
  returns  .statistic (W), df, .pvalue.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import waldjt
>>> import numpy as np
>>> b = np.array([1.0, 2.0, 3.0])
>>> S = np.eye(3) * 0.1
>>> R = np.array([[1.0, 0, 0], [0, 1, -1]])
>>> r = np.array([0.5, 0])
>>> waldjt(b, S, R, r).pvalue
```

## COMMON MISTAKES

- Wrong restriction matrix R - the row count must equal rank.
- Using when Sigma is poorly estimated (small n) - test is anti-
  conservative.

## REFERENCES

- Wald (1943). Tests of statistical hypotheses concerning several
  parameters.
