# iqrng - Interquartile range

## WHAT IT DOES

Reports the spread of the middle 50% of the data: IQR = Q3 - Q1.
Robust to outliers (only depends on the 25th and 75th percentiles).

## WHEN TO USE

- Robust description of spread (alongside median).
- Box-plot construction (the box length = IQR; whiskers = Q1 - 1.5*IQR,
  Q3 + 1.5*IQR).
- Outlier detection (values outside the box-plot fences).

## WHEN NOT TO USE

- Very small samples (n < 4) - quartiles unstable.
- When you need a parametric spread (use SD or variance).

## ASSUMPTIONS

- Numeric data, n >= 4.
- Continuous (or fine-grained ordinal); ties don't break the test
  but reduce resolution.

## FORMULA

```
IQR = Q3 - Q1
```
where Q1, Q3 are 25th and 75th percentiles by linear interpolation.

## INPUTS / OUTPUTS

```
iqrng(x, method="linear") -> RichResult
  x        numeric sample
  method   percentile-interpolation method
  returns  .statistic (IQR), Q1, median (Q2), Q3, fences.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import iqrng
>>> iqrng([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).statistic
4.5
```

## COMMON MISTAKES

- Quoting IQR but reporting mean and SD - mismatched robust vs
  classical. Pair median + IQR for robust summary.
- Treating box-plot fences as "outlier" definitions - they're a
  visualization heuristic (1.5*IQR), not a formal test.

## REFERENCES

- Wilcox (2017) ch.4.
