# fishex - Fisher's exact test (2x2)

## WHAT IT DOES

Computes the exact hypergeometric p-value for a 2x2 contingency table
under the null of independence. Unlike chi-squared, this gives an exact
result regardless of sample size.

## WHEN TO USE

- 2x2 contingency table with small expected counts (<5 in any cell).
- When you want an exact p-value that does not depend on chi-square
  approximation.
- Rare-event studies, randomized clinical trial outcomes, etc.

## WHEN NOT TO USE

- Larger tables (R x C) - use chi-squared with continuity correction.
- Very large n: Fisher exact is exact but computationally fine; chi-
  squared is essentially equivalent.
- Paired binary data - use mcnem.

## ASSUMPTIONS

- Marginal totals are fixed (the test conditions on them).
- Independent observations.
- Each cell count is a non-negative integer.

## FORMULA

Probability of the observed table under H0:
```
P = (a+b)! (c+d)! (a+c)! (b+d)! / (n! a! b! c! d!)
```
The p-value is the sum of P over all tables at least as extreme as
the observed one.

## INPUTS / OUTPUTS

```
fishex(table, alternative="two-sided") -> RichResult
  table        2x2 contingency table
  alternative  "two-sided", "less", "greater"
  returns      .statistic (odds ratio), .pvalue, cell counts.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import fishex
>>> r = fishex([[8, 2], [1, 9]])
>>> r.pvalue < 0.05
True
```

## COMMON MISTAKES

- Confusing Fisher's exact with chi-squared - they agree asymptotically
  but Fisher is appropriate for small samples.
- Using on paired binary - mcnem is the right test.
- Misordering the table - rows = exposure, cols = outcome.

## REFERENCES

- Fisher, R. A. (1935). The Design of Experiments.
- Hedderich et al. (2023) ch.7.
