# wald - Wald test for a single coefficient

## WHAT IT DOES

z- or chi-squared form of the Wald test for one coefficient against
a null value. Same as a one-coefficient t-test asymptotically;
appropriate for GLMs / MLE-based inference.

## WHEN TO USE

- Reporting per-coefficient significance in a GLM, logistic, etc.
- Quick check of a single null hypothesis given an estimate and SE.

## WHEN NOT TO USE

- Joint tests of several coefficients - use waldjt.
- Bayesian inference - use credible intervals (`hpdint`).

## ASSUMPTIONS

- Asymptotic Normality of estimate.
- SE is consistent.

## FORMULA

```
z = (estimate - theta_0) / SE
or W = z^2 ~ chi^2(1)
```

## INPUTS / OUTPUTS

```
wald(estimate, std_error, null_value=0.0, test="z") -> RichResult
  test  "z" (default Normal-approx) or "chi2"
  returns  .statistic, .pvalue, |z|.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import wald
>>> wald(0.6, 0.2).pvalue < 0.05
True
```

## COMMON MISTAKES

- Reporting Wald p-value alongside SE without showing the estimate.
- Confusing Wald with LRT - they're related but not identical for
  small samples.

## REFERENCES

- Wald (1943).
