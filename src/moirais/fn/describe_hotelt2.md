# hotelt2 - Hotelling T^2 (one-sample)

## WHAT IT DOES

Multivariate analog of the one-sample t^2 test. Tests whether a
mean vector mu equals a hypothesized vector mu_0. Aggregates the
distance from x_bar to mu_0 across multiple variables, accounting
for their correlations.

## WHEN TO USE

- Testing a vector of means against a multivariate null.
- Multiple-outcome studies where you want a single combined test
  rather than k separate t-tests with multiple-testing correction.

## WHEN NOT TO USE

- n <= p - the sample covariance is rank-deficient.
- Severe non-Normality - use a permutation alternative.
- You actually want individual marginal tests - use t-tests per
  outcome with FWER correction (`bonfer`, `holm`, `bhfdr`).

## ASSUMPTIONS

- Multivariate Normality.
- n > p (so the sample covariance is invertible).
- Independent observations.

## FORMULA

```
T^2 = n * (x_bar - mu_0)' * S^-1 * (x_bar - mu_0)
F = ((n - p) / (p * (n - 1))) * T^2 ~ F(p, n - p)
```

## INPUTS / OUTPUTS

```
hotelt2(X, mu0) -> RichResult
  X        n x p data matrix
  mu0      hypothesized mean vector (length p)
  returns  T^2, F statistic, df1=p, df2=n-p, .pvalue, mean diff norm.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import hotelt2
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> X = rng.standard_normal((30, 3))
>>> r = hotelt2(X, [0, 0, 0])
>>> r.pvalue
```

## COMMON MISTAKES

- Running with n <= p - the test isn't defined.
- Reporting T^2 without the F-equivalent and df - reviewers want both.

## REFERENCES

- Hotelling (1931). The generalization of Student's ratio.
- Hedderich et al. (2023) ch.10.
