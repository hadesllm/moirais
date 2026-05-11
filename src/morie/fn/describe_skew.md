# skew - Sample skewness

## WHAT IT DOES

Measures asymmetry of a sample distribution. Negative = left tail
longer; positive = right tail longer; 0 = symmetric (Normal).

The default uses the bias-corrected Fisher-Pearson coefficient of
skewness g_1.

## WHEN TO USE

- Diagnostic before parametric tests that assume Normality.
- Reporting distributional shape alongside mean and SD.
- Choosing transformations (log/sqrt for right-skewed positive data).

## WHEN NOT TO USE

- Tiny samples (n<10): skew estimates are very noisy.
- Visual diagnostics (Q-Q plot, histogram) are usually more
  informative for small data.

## ASSUMPTIONS

- Numeric data, n >= 3.

## FORMULA

```
g_1 = m_3 / m_2^(3/2)            (raw moment-based)
G_1 = (n^2 / ((n-1)(n-2))) * g_1 (Fisher-Pearson, bias-corrected)
```
where m_k is the k-th central sample moment.

## INPUTS / OUTPUTS

```
skew(x, bias=False) -> RichResult
  x        numeric sample
  bias     False (default) = Fisher-Pearson; True = raw m3/m2^1.5
  returns  .statistic (g_1), shape label, direction, n.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import skew
>>> skew([1, 2, 3, 4, 5, 6, 7, 8, 9, 100]).statistic   # right-skewed
3.1
```

## COMMON MISTAKES

- Treating |g_1| < 1 as "Normal" — it just rules out severe asymmetry;
  the data could still be heavy-tailed (kurt) or bimodal.
- Using on small samples — sampling variance of skew is large.
- Forgetting to interpret direction (positive = right tail longer).

## REFERENCES

- Wilcox (2017) ch.5; Hedderich et al. (2023) ch.4.
