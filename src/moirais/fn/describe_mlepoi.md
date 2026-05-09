# mlepoi - Maximum-likelihood Poisson fit

## WHAT IT DOES

Estimates the rate lambda of a Poisson distribution by maximum
likelihood. The MLE is just the sample mean of the counts.

## WHEN TO USE

- Counts data (non-negative integers).
- Estimating rate parameter for a Poisson process.
- Simple GLM intercept-only model fitting.

## WHEN NOT TO USE

- Counts have variance much greater than the mean (overdispersion;
  see the Pearson chi^2/df warning) - use negative-binomial GLM.
- Truncated data (e.g., zero-counts excluded) - use truncated Poisson.
- Time-since-event data - exponential distribution, not Poisson.

## ASSUMPTIONS

- Counts are non-negative integers.
- Poisson assumes Variance = Mean. If Var > Mean, you have overdispersion.
- Independent observations.

## FORMULA

```
lambda_hat = (1/n) sum_i x_i
```

## INPUTS / OUTPUTS

```
mlepoi(counts) -> RichResult
  counts      non-negative integers
  returns     lambda, sample variance, var/mean ratio (overdispersion check).
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import mlepoi
>>> import numpy as np
>>> r = mlepoi(np.random.default_rng(0).poisson(3.0, size=200))
>>> r["lambda"]
2.94  # ish
```

## COMMON MISTAKES

- Treating overdispersed data as Poisson - parameter estimates are
  fine but standard errors are too small.
- Forgetting that Poisson assumes equal-mean-and-variance; check
  the var/mean ratio.

## REFERENCES

- Wooldridge (2010) Econometric Analysis of Cross Section and Panel
  Data.
- Weisburd et al. (2022) ch.6.
