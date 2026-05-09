# hpdint - Highest Posterior Density credible interval

## WHAT IT DOES

For a 1D posterior sample, finds the shortest interval containing the
specified credible mass. Always at least as short as the equal-tailed
interval, and includes the highest-density region.

## WHEN TO USE

- Reporting Bayesian inference for a single parameter.
- When the posterior may be skewed - HPD captures the high-density
  region rather than splitting tails equally.
- After MCMC sampling.

## WHEN NOT TO USE

- Multimodal posteriors - HPD may not be a single interval; report
  modes separately.
- Equal-tailed CI is preferred - that's a different choice.

## ASSUMPTIONS

- Samples are i.i.d. draws from the (joint) posterior.
- Sample size is large enough that the posterior is well-estimated
  (typically >= 1000 effective samples).

## FORMULA

For sorted samples a_(1) <= ... <= a_(n):
```
k = floor(cred * n)
HPD = (a_(i*), a_(i*+k))
```
where i* minimizes a_(i+k) - a_(i) over i.

## INPUTS / OUTPUTS

```
hpdint(samples, cred=0.95) -> RichResult
  samples    posterior sample (numpy array)
  cred       credible level in (0, 1)
  returns    (lo, hi) tuple, width, n samples, posterior mean & median.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import hpdint
>>> import numpy as np
>>> hpdint(np.random.default_rng(0).standard_normal(10000), 0.95)
```

## COMMON MISTAKES

- Reporting HPD on too few samples - effective sample size matters.
- Confusing HPD with confidence interval - they have different
  philosophical interpretations.

## REFERENCES

- Box & Tiao (1973) Bayesian Inference in Statistical Analysis.
