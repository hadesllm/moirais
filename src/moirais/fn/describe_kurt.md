# kurt - Sample kurtosis (excess)

## WHAT IT DOES

Measures the heaviness of a distribution's tails. Default returns
Fisher's excess kurtosis (kurtosis - 3) so a Normal distribution has
g_2 = 0. Positive = heavier-than-Normal tails (leptokurtic); negative
= thinner (platykurtic).

## WHEN TO USE

- Diagnostic before parametric tests that assume Normality.
- Detecting outlier-prone or "fat-tailed" distributions.
- Pairing with skew (g_1) for full distributional shape characterisation.

## WHEN NOT TO USE

- Tiny samples (n<20): kurtosis estimates are very noisy.
- The visual diagnostic (Q-Q plot) is often more informative.

## ASSUMPTIONS

- Numeric data, n >= 4.

## FORMULA

```
g_2 = m_4 / m_2^2 - 3   (Fisher excess; Normal -> 0)
G_2 = bias-corrected version using n
```

## INPUTS / OUTPUTS

```
kurt(x, fisher=True, bias=False) -> RichResult
  x        numeric sample
  fisher   True = subtract 3 (excess); False = Pearson (Normal -> 3)
  bias     False = small-sample-corrected
  returns  .statistic (g_2), shape label, n.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import kurt
>>> kurt([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).statistic
-1.2  # platykurtic - thinner than Normal
```

## COMMON MISTAKES

- Mixing conventions between Fisher (Normal=0) and Pearson (Normal=3)
  - always check which one your tool uses.
- Treating kurt > 0 as proof of "outliers" - it just means heavier
  tails; could be a few extremes or many moderately-large values.

## REFERENCES

- Wilcox (2017) ch.5.
