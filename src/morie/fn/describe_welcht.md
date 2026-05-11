# welcht — Welch's two-sample t-test (unequal variances)

## WHAT IT DOES

Tests whether two independent samples have different population means
when their variances are NOT assumed to be equal. The test statistic
follows an approximate t-distribution with Satterthwaite-adjusted
degrees of freedom. This is the safest default two-sample t-test —
the equal-variance ("Student's") t-test breaks badly when variances
differ even modestly.

## WHEN TO USE

- Two independent groups (no pairing or matching).
- Continuous outcome.
- Group variances may differ (this is the safe default — even when
  they're equal, Welch's loses very little efficiency).
- Sample sizes can be unequal.

## WHEN NOT TO USE

- Paired observations — use `paired` instead.
- More than two groups — use `kwallis` (nonparametric) or one-way ANOVA.
- Non-numeric / ordinal outcome — use `manwhi` (Mann-Whitney).
- Heavily skewed with small n — use `manwhi` or `permpv` (permutation).
- Counts / Poisson data — use `glmpoi` (Poisson regression).

## ASSUMPTIONS

- Independence within and between groups.
- Each group's mean is approximately Normal in distribution OR the
  sample size is large enough for the Central Limit Theorem to make
  the sample mean approximately Normal.
- Variances may differ across groups (this is the *relaxation* from
  Student's t).

## FORMULA

```
t = (x̄₁ − x̄₂) / sqrt(s₁²/n₁ + s₂²/n₂)

df = (s₁²/n₁ + s₂²/n₂)² / [(s₁²/n₁)²/(n₁-1) + (s₂²/n₂)²/(n₂-1)]
```

`df` is the Welch-Satterthwaite adjustment — typically non-integer.
Under H₀ the statistic follows approximately Student's t(df).

## INPUTS / OUTPUTS

```
welcht(x, y) → RichResult
  x, y     array-like — two independent numeric samples
  returns  RichResult with .statistic (t), .df, .pvalue,
           plus extra_summary table of n, means, mean diff, SDs.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import welcht
>>> r = welcht([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
>>> r.pvalue
0.018  # ish
>>> r < 0.05      # comparison ops work — true since p < 0.05
True
>>> print(r)      # full multi-section summary
```

## COMMON MISTAKES

- Using equal-variance t-test on unequal-variance data → biased
  Type-I error rate. `welcht` is the safe default.
- Ignoring the small-sample warning — when n<5 in either group, the
  t-approximation is unreliable; switch to `permpv` or `manwhi`.
- Confusing "n>30 means CLT applies" — only true when no extreme
  skew or outliers; check first with `shapir` or visual diagnostics.
- Reporting p without effect size — pair with `cohend` or `hedgeg`.

## REFERENCES

- Welch, B. L. (1947). The generalization of "Student's" problem when
  several different population variances are involved. Biometrika.
- Wilcox, R. R. (2017) Modern Statistics for the Social and Behavioral
  Sciences, ch. 6.
- Hedderich, Sachs & Reynarowych (2023) Applied Statistics: Methods
  Using R, ch. 7.
