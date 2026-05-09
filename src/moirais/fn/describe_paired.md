# paired — Paired t-test

## WHAT IT DOES

Tests whether the mean difference between matched pairs of observations
is non-zero. The test reduces to a one-sample t-test on the differences
`d_i = x_i − y_i`, exploiting within-pair correlation to gain power
over the two-sample alternative.

## WHEN TO USE

- Each x has a natural y partner: same subject pre/post, twin pairs,
  matched controls, repeated measures on the same unit.
- Continuous outcome.
- The differences `d_i` are approximately Normal (n large helps).

## WHEN NOT TO USE

- Independent samples (no pairing) — use `welcht`.
- Differences heavily skewed or with outliers — use `wilcoxn`
  (Wilcoxon signed-rank, the nonparametric paired test).
- Three or more time points / conditions per subject — use repeated-
  measures ANOVA or a mixed model.
- Counts — model directly with a Poisson/NB GLM.

## ASSUMPTIONS

- Pairs are independent of each other.
- The differences `d_i = x_i − y_i` are approximately Normal in the
  population, OR n is large enough for CLT.
- Within a pair, x and y can be correlated (and usually are — that's
  the point of pairing).

## FORMULA

```
d_i = x_i − y_i
t = mean(d) / (sd(d) / √n)
df = n − 1
```

Under H₀ (mean difference = 0), `t ~ Student's t(n − 1)`.

## INPUTS / OUTPUTS

```
paired(x, y) → RichResult
  x, y     paired numeric samples (same length)
  returns  .statistic (t), .df, .pvalue, n pairs, mean diff,
           SD of differences.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import paired
>>> r = paired([10, 12, 15, 9, 11], [12, 11, 17, 8, 10])
>>> print(r)        # full summary including parallel-trend caveat
>>> r.pvalue
0.85    # roughly
```

## COMMON MISTAKES

- Forgetting to pair: throwing both vectors into `welcht` ignores the
  within-pair correlation and loses power.
- Pairing on the wrong dimension: time-1 vs time-2 by subject is
  paired, but time-1 of subject A vs time-1 of subject B is not.
- Heavy skew: switch to `wilcoxn`. The paired-t becomes anti-conservative.
- Tiny n: with n<10 you should rely more heavily on visual diagnostics
  than on the p-value alone.

## REFERENCES

- Student (1908). The probable error of a mean. Biometrika.
- Wilcox (2017) Modern Statistics for the Social and Behavioral
  Sciences, ch. 6.
- Wooditch et al. (2021) A Beginner's Guide to Statistics for
  Criminology and Criminal Justice using R, ch. 11.
