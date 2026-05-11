# wilcoxn — Wilcoxon signed-rank test

## WHAT IT DOES

Nonparametric alternative to the paired t-test. Tests whether the
median of paired differences (or the median of a one-sample) is zero.
Internally ranks the absolute differences, attaches signs, and computes
the sum of positive ranks.

## WHEN TO USE

- Paired observations (or single sample tested against zero).
- Continuous or ordinal outcome.
- Differences are skewed, heavy-tailed, or otherwise non-Normal.
- You want a robust paired test without assuming Normality.

## WHEN NOT TO USE

- Independent samples — use `manwhi`.
- Three-or-more conditions per subject — use Friedman's test.
- Differences truly Normal — `paired` has more power.

## ASSUMPTIONS

- The paired differences are continuous.
- The distribution of differences is symmetric around its median
  (this matters for the inference; the test is biased toward false
  significance when the distribution is heavily skewed).

## FORMULA

```
T+ = sum of ranks of positive differences (signed-rank statistic)
```

Under H0 (median diff = 0), T+ has a known distribution; for n>=20,
asymptotically Normal with mean n(n+1)/4.

## INPUTS / OUTPUTS

```
wilcoxn(x, y=None, alternative="two-sided") -> RichResult
  x        paired sample (or differences directly if y omitted)
  y        optional partner sample
  returns  .statistic (W), .pvalue, n, median of differences,
           alternative.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import wilcoxn
>>> r = wilcoxn([100, 95, 110, 105, 98], [90, 92, 100, 102, 95])
>>> r.pvalue < 0.05
True
```

## COMMON MISTAKES

- Confusing wilcoxn (paired) with manwhi (independent) — completely
  different tests.
- Assuming "nonparametric" means "no assumptions" — symmetry of the
  difference distribution still matters.
- Reporting a sign rank test on differences when the data are nominal.

## REFERENCES

- Wilcoxon, F. (1945). Individual comparisons by ranking methods.
- Wilcox (2017) Modern Statistics ch. 7.
