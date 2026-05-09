# mcnem - McNemar's test (paired binary)

## WHAT IT DOES

Tests whether the marginal proportions of a paired binary outcome
differ. Uses only the discordant cells: subjects whose outcome flips
between the two conditions.

## WHEN TO USE

- Paired binary outcomes (same subject before/after, twin studies,
  matched cases).
- 2x2 contingency table where rows and columns are the same pair of
  conditions/timepoints.

## WHEN NOT TO USE

- Independent binary outcomes - use fishex or chi-squared.
- More than two paired timepoints - use Cochran's Q.

## ASSUMPTIONS

- Pairs are independent.
- Outcome is binary in both conditions.
- Discordant cells (b + c) >= 10 for reliable chi-squared approximation;
  with fewer use exact mid-p binomial.

## FORMULA

```
chi^2 = (|b - c| - 0.5)^2 / (b + c)   (continuity-corrected)
```
where b = subjects with condition-1 success and condition-2 failure,
c = subjects with condition-1 failure and condition-2 success.

Under H0 (marginal proportions equal), chi^2 ~ chi-squared(1).

## INPUTS / OUTPUTS

```
mcnem(table, continuity=True) -> RichResult
  table       2x2 contingency table
  continuity  apply 0.5 continuity correction
  returns     .statistic, .pvalue, df=1, all four cells.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import mcnem
>>> r = mcnem([[10, 20], [5, 15]])  # 20 b's vs 5 c's
>>> r.pvalue
```

## COMMON MISTAKES

- Forgetting it's paired - independent 2x2 is fishex/chi-squared.
- Using when discordant total b+c is very small - switch to exact
  mid-p binomial test.
- Reporting chi-squared without continuity correction by default.

## REFERENCES

- McNemar, Q. (1947). Note on the sampling error of the difference
  between correlated proportions or percentages. Psychometrika.
- Hedderich et al. (2023) ch.7.
