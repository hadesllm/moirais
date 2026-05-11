# mlenrm - Maximum-likelihood Normal fit

## WHAT IT DOES

Estimates the location (mu) and scale (sigma) parameters of a Normal
distribution by maximum likelihood. Equivalent to: mu_hat = sample
mean, sigma_hat = sqrt(sum_i (x - mu)^2 / n) (1/n divisor, not 1/(n-1)).

## WHEN TO USE

- Need MLE-style parameter estimates rather than unbiased estimators.
- Likelihood-ratio tests (`lrtst`) involving the Normal model.
- AIC/BIC computation - use mlenrm to get the log-likelihood at MLE.

## WHEN NOT TO USE

- For "the spread of the data" - use sample SD with 1/(n-1) divisor;
  MLE is biased downward.
- Heavy-tailed or skewed data - the MLE assumes Normal; misuse here
  gives nonsensical estimates.

## ASSUMPTIONS

- Independent, identically-distributed observations.
- Normality.

## FORMULA

```
mu_hat = (1/n) sum_i x_i           (= sample mean)
sigma_hat^2 = (1/n) sum_i (x_i - mu_hat)^2
```

Note 1/n - this is the MLE; the unbiased estimator uses 1/(n-1).

## INPUTS / OUTPUTS

```
mlenrm(x) -> RichResult
  x        numeric sample (n >= 2)
  returns  mu, sigma (MLE), sigma (unbiased), log-likelihood at MLE, n.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import mlenrm
>>> import numpy as np
>>> r = mlenrm(np.random.default_rng(0).standard_normal(500))
>>> r["mu"], r["sigma"]
```

## COMMON MISTAKES

- Reporting MLE sigma as if it were the unbiased SD - it's biased
  downward, especially for small n.
- Using on non-Normal data and being surprised by skewed parameter
  estimates.

## REFERENCES

- Casella & Berger (2002) Statistical Inference.
