# mad - Median absolute deviation

## WHAT IT DOES

Robust measure of variability: median of |x_i - median(x)|. Scaled
by 1.4826 (default) so that for Normal data, MAD ~= SD. This is
the most robust scale estimator for general use.

## WHEN TO USE

- Robust alternative to SD when outliers are suspected.
- Outlier detection: |x_i - median| > k * MAD with k=2.5 or 3.
- Pairing with median to give a robust (location, scale) summary.

## WHEN NOT TO USE

- Truly Normal data without outliers: SD is more efficient.
- Very small samples (n<10): MAD's breakdown to 50% protection
  comes at low efficiency; consider trimmed mean (`trmean`) instead.

## ASSUMPTIONS

- Numeric data, n >= 2.

## FORMULA

```
raw MAD = median_i |x_i - median(x)|
scaled MAD = 1.4826 * raw MAD   (default; ~= SD under Normality)
```

## INPUTS / OUTPUTS

```
mad(x, scale="normal") -> RichResult
  x        numeric sample
  scale    "normal" (x1.4826), "raw" (no rescale), or float
  returns  .statistic (MAD), raw MAD, sample SD, MAD/SD ratio, median.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import mad
>>> mad([1, 2, 3, 4, 5]).statistic
1.4826
```

## COMMON MISTAKES

- Forgetting the 1.4826 rescaling - users sometimes silently get a
  much smaller spread than expected.
- Using MAD on data with strong asymmetry - it's a symmetric measure.

## REFERENCES

- Wilcox (2017) ch.5 (robust scale measures).
