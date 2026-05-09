# kwallis — Kruskal-Wallis H-test

## WHAT IT DOES

Nonparametric counterpart of one-way ANOVA. Tests whether ≥3
independent groups come from the same distribution. Internally it
ranks the pooled sample and computes a chi-squared-distributed
statistic from the per-group rank sums.

When all groups share the same distribution shape (only location
shifts), it tests for equal medians. When shapes differ, it tests
stochastic equality.

## WHEN TO USE

- 3 or more independent groups (for 2 groups use `manwhi`).
- Continuous or ordinal outcome.
- Distributions skewed, heavy-tailed, or otherwise non-Normal.
- You want a robust alternative to one-way ANOVA.

## WHEN NOT TO USE

- Repeated measures / paired data — use the Friedman test.
- Two groups — use `manwhi`.
- All groups Normal with equal variance — one-way ANOVA has more power.
- Counts — Poisson GLM or negative-binomial GLM.

## ASSUMPTIONS

- Independence between observations (and between groups).
- Outcome is at least ordinal.
- Distributions have the same shape (only location shift) IF
  interpreting as a "test of medians".
- No Normality assumption.

## FORMULA

```
H = (12 / N(N+1)) × Σⱼ Rⱼ² / nⱼ − 3(N+1)
```

where `Rⱼ` is the rank-sum for group j, `nⱼ` is its size, and
`N = Σⱼ nⱼ`. Under H₀, `H ~ χ²(k − 1)` with k groups.

When ties are present, divide H by the tie-correction factor
`1 − Σ (tᵢ³ − tᵢ) / (N³ − N)`.

## INPUTS / OUTPUTS

```
kwallis(*groups) → RichResult
  groups   pass each group as a separate positional arg, e.g.
           kwallis(g1, g2, g3) — each array-like
  returns  .statistic (H), .df (k-1), .pvalue, k, n total,
           per-group medians.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import kwallis
>>> r = kwallis([1, 2, 3], [4, 5, 6], [7, 8, 9])
>>> print(r)
>>> r.pvalue < 0.05
True
```

## COMMON MISTAKES

- Calling significant K-W "evidence of different means" without
  checking shapes — when shapes differ, you can have significant H
  with identical means.
- Forgetting to follow up with pair-wise tests (`manwhi` with Holm
  correction `holm` or BH FDR `bhfdr`) to find WHICH groups differ.
- Using on small per-group n (<5) — chi² approximation poor; use
  permutation pair-wise comparisons via `permpv`.

## REFERENCES

- Kruskal, W. H. & Wallis, W. A. (1952). Use of ranks in one-criterion
  variance analysis. JASA.
- Wilcox (2017) Modern Statistics ch. 10.
- Wooditch et al. (2021) A Beginner's Guide … ch. 12 (Nonparametric ANOVA).
