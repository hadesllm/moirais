# ksonebs - Kolmogorov-Smirnov test

## WHAT IT DOES

Compares the empirical CDF of a sample to either a theoretical
distribution (one-sample mode) or to another sample's CDF (two-sample
mode). The test statistic D is the maximum absolute distance between
CDFs.

## WHEN TO USE

- Goodness-of-fit to a fully-specified distribution (Normal with
  given mean/sd, exponential with given rate, etc.).
- Comparing two independent samples to test if they come from the
  same distribution.

## WHEN NOT TO USE

- Testing Normality with estimated mean/SD - the standard K-S is
  over-conservative; use Lilliefors correction or Shapiro-Wilk.
- Heavily discrete data - CDF distances are not invariant to ties.
- Comparing distribution location specifically - use t-test or
  Mann-Whitney for power.

## ASSUMPTIONS

- Continuous distribution(s).
- Independent observations.
- For one-sample: distribution parameters are NOT estimated from data
  (or the K-S is conservative).

## FORMULA

```
D_one = sup_x |F_n(x) - F_0(x)|
D_two = sup_x |F_1(x) - F_2(x)|
```

## INPUTS / OUTPUTS

```
ksonebs(x, cdf_or_y="norm", alternative="two-sided") -> RichResult
  x          numeric sample
  cdf_or_y   distribution name string, callable, or second sample
  returns    .statistic (D), .pvalue, mode (one/two-sample), n.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import ksonebs
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> r = ksonebs(rng.standard_normal(100), "norm")
>>> r.pvalue > 0.05
True
```

## COMMON MISTAKES

- Using one-sample K-S with estimated parameters (use Lilliefors
  correction).
- Forgetting K-S has low power against differences in scale or shape
  when locations are equal - it tests CDF difference broadly.

## REFERENCES

- Kolmogorov (1933); Smirnov (1948).
- Wilcox (2017) ch.5.
