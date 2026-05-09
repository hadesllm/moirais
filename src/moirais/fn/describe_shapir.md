# shapir — Shapiro-Wilk test for Normality

## WHAT IT DOES

Tests the null hypothesis that the sample comes from a Normal
distribution. The test statistic W is the squared correlation between
the sample's order statistics and the expected order statistics from
a Standard Normal — small W means the empirical CDF deviates from a
straight Normal Q-Q line.

Among Normality tests, Shapiro-Wilk has the best power against most
realistic non-Normal alternatives for n in roughly [5, 5000].

## WHEN TO USE

- Pre-checking the Normality assumption for a parametric test
  (paired t, ANOVA, OLS residuals).
- Sample size n in [3, 5000].
- You want a quick formal check to complement a Q-Q plot.

## WHEN NOT TO USE

- n > 5000: even tiny deviations from Normal yield "significant"
  p-values, leading to needless rejection of robust parametric tests.
  Switch to visual inspection or `anddrl` (Anderson-Darling).
- n < 3: not enough data to run.
- Discrete or rounded data — even truly-Normal data discretized to
  integers will fail.
- AS A REPLACEMENT FOR LOOKING AT A Q-Q PLOT — Shapiro tells you yes/no
  but not WHY (skew? heavy tails? bimodal?). Always pair with a plot.

## ASSUMPTIONS

- Independent, identically-distributed observations.
- Continuous distribution (no discrete or rounded data).

## FORMULA

```
W = (Σᵢ aᵢ x_(i))² / Σᵢ (xᵢ − x̄)²
```

where `x_(i)` are the order statistics and `aᵢ` are constants derived
from the expected values, variances, and covariances of order statistics
from a Standard Normal sample of size n. Under H₀, W is close to 1; far
from 1 means non-Normal.

## INPUTS / OUTPUTS

```
shapir(x) → RichResult
  x        numeric sample (3 ≤ n ≤ 5000 recommended)
  returns  .statistic (W), .pvalue, n, sample mean, sample SD,
           sample skewness.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import shapir
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> r = shapir(rng.standard_normal(200))
>>> r.pvalue > 0.05    # should usually fail to reject for true Normal
True
```

## COMMON MISTAKES

- Treating p > 0.05 as "PROOF the data is Normal" — it just means we
  can't reject. Power is finite.
- Treating p < 0.05 as a license to abandon parametric methods — the
  parametric tests are often robust to mild non-Normality (Welch's t,
  CLT, etc.). Visual inspection trumps the formal test.
- Running on non-i.i.d. data (e.g., time-series with autocorrelation).

## REFERENCES

- Shapiro, S. S. & Wilk, M. B. (1965). An analysis of variance test
  for normality (complete samples). Biometrika 52(3-4): 591-611.
- Wilcox (2017) Modern Statistics ch. 5.
- Wooditch et al. (2021) A Beginner's Guide … ch. 12 (ANOVA assumptions).
