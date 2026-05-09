# anddrl - Anderson-Darling test for Normality

## WHAT IT DOES

Tests the null hypothesis of Normality. More tail-sensitive than
Shapiro-Wilk, so it detects deviations in tails better. The test
statistic A^2 is compared to critical values that vary with significance
level (typical: 15%, 10%, 5%, 2.5%, 1%).

## WHEN TO USE

- Normality testing when tail behavior matters most.
- Pairing with Shapiro-Wilk for cross-validation.
- Larger samples where Shapiro-Wilk loses power (n > 5000).

## WHEN NOT TO USE

- As a sole criterion - always pair with Q-Q plot.
- Discrete or rounded data - violates the continuity assumption.

## ASSUMPTIONS

- Independent observations.
- Continuous distribution.
- Sample size n >= 8 for reliable critical values.

## FORMULA

```
A^2 = -n - (1/n) * sum_i (2i - 1) * [log(F(x_i)) + log(1 - F(x_(n-i+1)))]
```
where x_(i) is the i-th order statistic and F() is the hypothesized CDF.

## INPUTS / OUTPUTS

```
anddrl(x) -> RichResult
  x        numeric sample (n >= 8)
  returns  .statistic (A^2), table of critical values per significance.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import anddrl
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> anddrl(rng.standard_normal(300)).statistic
```

## COMMON MISTAKES

- Treating A^2 like a single p-value - it's a statistic to compare to
  multiple critical values.
- Using when n < 8 - critical values are not tabulated.

## REFERENCES

- Anderson & Darling (1952). Asymptotic theory of certain "goodness
  of fit" criteria based on stochastic processes.
- Wilcox (2017) ch.5.
