# tschpr - Tschuprow's T (categorical association)

## WHAT IT DOES

Chi-squared-based association measure for r x c contingency tables.
Range [0, 1]. Like Cramer's V but uses sqrt((r-1)(c-1)) denominator
- they agree for square tables.

## WHEN TO USE

- Reporting effect size after a chi-squared test of independence.
- Comparing association strength across tables of different shapes.
- Square (r=c) tables where you'd otherwise use Cramer's V.

## WHEN NOT TO USE

- Pure 2x2 tables - phi or odds ratio is more standard.
- Continuous variables - use Pearson, Spearman, or Kendall.

## ASSUMPTIONS

- Counts in each cell are non-negative integers.
- Expected counts under independence >= 5 in most cells (otherwise
  use Fisher exact).

## FORMULA

```
T = sqrt(chi^2 / (n * sqrt((r-1)(c-1))))
```

## INPUTS / OUTPUTS

```
tschpr(table) -> RichResult
  table    r x c contingency table
  returns  .statistic (T), strength label, chi^2, df, n.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import tschpr
>>> tschpr([[10,20],[30,40]]).statistic
0.067  # weak association in this 2x2
```

## COMMON MISTAKES

- Confusing Tschuprow T with Cramer V - they differ for non-square
  tables.
- Reporting T without n - it depends on sample size.

## REFERENCES

- Tschuprow (1925/1939).
