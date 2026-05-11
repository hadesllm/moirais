# covar - Sample covariance

## WHAT IT DOES

Computes the unbiased sample covariance Cov(x, y) using the n-1
denominator. Covariance measures the joint linear variability of
two series; positive means they tend to move together, negative
means they move opposite.

## WHEN TO USE

- Building a covariance matrix for multivariate methods (PCA, LDA).
- Computing Pearson correlation manually: r = cov(x,y) / (sd_x * sd_y).
- Constructing variance of a sum: Var(x+y) = Var(x) + Var(y) + 2*Cov(x,y).

## WHEN NOT TO USE

- For interpretation alone — covariance has units; Pearson r is
  unit-free and easier to compare. Prefer correlation for reporting.
- When you want a robust measure — outliers heavily affect covariance.

## ASSUMPTIONS

- Both series are numeric and have the same length.
- At least 2 observations.
- Independence between pairs (within-pair correlation is fine).

## FORMULA

```
Cov(x, y) = (1 / (n - 1)) sum_i (x_i - mean(x)) * (y_i - mean(y))
```

## INPUTS / OUTPUTS

```
covar(x, y) -> RichResult
  x, y     numeric series of equal length
  returns  .statistic (cov), Pearson r normalised, SD(x), SD(y), n.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import covar
>>> covar([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]).statistic
5.0
```

## COMMON MISTAKES

- Treating covariance as a measure of strength — its scale depends
  on the units of x and y. Use `kentau` or `spearm` for non-Pearson
  alternatives.
- Forgetting Bessel's correction (n-1 vs n) when reporting alongside
  other software that uses n.

## REFERENCES

- Wooditch et al. (2021) ch.14; Wilcox (2017) ch.4.
