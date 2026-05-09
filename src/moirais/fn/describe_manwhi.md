# manwhi — Mann-Whitney U test (Wilcoxon rank-sum)

## WHAT IT DOES

Nonparametric alternative to the two-sample t-test. Tests whether
P(X > Y) ≠ 0.5 — a stochastic-dominance form. Internally it ranks
the pooled sample and compares rank sums between groups; the U
statistic counts pair-wise wins.

When data are continuous and shapes of the two distributions are
identical except for a location shift, the test compares medians.
When shapes differ, it tests stochastic dominance — *not* equality
of medians.

## WHEN TO USE

- Two independent groups.
- Continuous or ordinal outcome.
- Distributions skewed, heavy-tailed, or otherwise non-Normal.
- You want a robust alternative to `welcht` that doesn't assume Normality.

## WHEN NOT TO USE

- Paired observations — use `wilcoxn` (signed-rank).
- Three or more groups — use `kwallis` (Kruskal-Wallis).
- Nominal categorical outcomes — use a chi-squared test (`fishex` or
  contingency-table chi²).

## ASSUMPTIONS

- Independence within and between groups.
- The two distributions have the same general shape (only location
  shift) IF interpreting as a median-difference test.
- No assumption about Normality, equal variances, or homoscedasticity.

## FORMULA

For each pair (xᵢ, yⱼ), count S = #(xᵢ > yⱼ) + 0.5 × #(xᵢ = yⱼ).
Then `U₁ = S` and `U₂ = n₁ × n₂ − U₁`. Report the smaller (or, in
SciPy's convention, the U corresponding to the chosen alternative).

Asymptotic Normal: `z = (U − μ_U) / σ_U` with `μ_U = n₁n₂/2` and
`σ_U² = n₁n₂(n₁+n₂+1)/12` (no-ties version).

## INPUTS / OUTPUTS

```
manwhi(x, y, alternative="two-sided") → RichResult
  x, y         independent numeric (or ordinal) samples
  alternative  "two-sided", "less", or "greater"
  returns      .statistic (U), .pvalue, n(x), n(y), Median(x), Median(y).
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import manwhi
>>> r = manwhi([1, 2, 3, 4, 5], [4, 5, 6, 7, 8])
>>> print(r)
>>> r.pvalue < 0.05
True
```

## COMMON MISTAKES

- Treating `manwhi` as a "test of medians" without checking equal
  shapes. When shapes differ, p < .05 means stochastic dominance, not
  necessarily a difference in medians.
- Using on paired data — switch to `wilcoxn`.
- Using on small samples without exact p-values — for n₁ + n₂ < 20,
  exact computation is more reliable than the asymptotic Normal.

## REFERENCES

- Mann, H. B. & Whitney, D. R. (1947). On a test of whether one of two
  random variables is stochastically larger than the other. AMS.
- Wilcox (2017) Modern Statistics ch. 7.
- Wooditch et al. (2021) A Beginner's Guide … ch. 11.
