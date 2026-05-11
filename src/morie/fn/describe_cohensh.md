# cohensh - Cohen's h (effect size for two proportions)

## WHAT IT DOES

Effect-size measure for the difference between two proportions, using
arcsine-transformed values. Stable across the full [0, 1] range
(unlike raw difference, which is noisier near 0.5).

## WHEN TO USE

- Comparing two proportions or rates.
- Power analysis for two-proportion z-tests.
- Reporting effect size alongside chi-squared / Fisher exact tests.

## WHEN NOT TO USE

- Continuous outcomes - use Cohen's d (`cohend`).
- Multiple proportions - generalize to a multivariate effect size.

## ASSUMPTIONS

- Both p1, p2 in [0, 1].
- Independent samples for the comparison.

## FORMULA

```
h = 2 * (asin(sqrt(p1)) - asin(sqrt(p2)))
```

Cohen's benchmarks: |h| = 0.2 small, 0.5 medium, 0.8 large.

## INPUTS / OUTPUTS

```
cohensh(p1, p2) -> RichResult
  p1, p2    proportions
  returns   .statistic (h), benchmark, raw difference.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import cohensh
>>> cohensh(0.5, 0.4).statistic   # 0.20 - small
0.201
```

## COMMON MISTAKES

- Treating p1 - p2 as the effect size - h is the standardized form
  preferred for power calculations.
- Confusing h with Cohen's d - they're not interchangeable.

## REFERENCES

- Cohen (1988) Statistical Power Analysis ch.6.
