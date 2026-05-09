# spearm - Spearman's rho rank correlation

## WHAT IT DOES

Pearson correlation computed on the ranks. Robust to monotonic-but-
nonlinear relationships and to outliers (extreme values become
ordinary ranks).

## WHEN TO USE

- Two continuous or ordinal variables.
- Suspected monotonic but possibly nonlinear relationship.
- Outliers or non-Normal marginals.

## WHEN NOT TO USE

- Truly linear+Normal - Pearson r has more power.
- Heavy ties - kentau-b handles ties more cleanly.

## ASSUMPTIONS

- Independent pairs.
- Both variables ordinal+.
- Monotonic relationship (not necessarily linear).

## FORMULA

Pearson r computed on rank(x), rank(y).

## INPUTS / OUTPUTS

```
spearm(x, y) -> RichResult
  returns  .statistic (rho), .pvalue, strength, direction.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import spearm
>>> spearm([1,2,3,4,5], [1,4,9,16,25]).statistic
1.0  # perfect monotonic, even though nonlinear
```

## COMMON MISTAKES

- Using Spearman as a "test of association" without recognizing the
  monotonicity assumption - U-shaped relationship gives rho ~= 0.

## REFERENCES

- Spearman (1904).
