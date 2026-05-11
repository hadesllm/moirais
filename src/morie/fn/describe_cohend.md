# cohend — Cohen's d (two-sample mean-difference effect size)

## WHAT IT DOES

Reports the standardized difference between two group means as a
unit-free effect-size measure. Useful for meta-analysis, power
calculations, and reporting practical significance alongside p-values.

The d scale is interpretable across studies and outcomes — a
medium-effect-size in education research means roughly the same
thing as a medium-effect-size in pharmacology.

## WHEN TO USE

- Reporting effect size alongside any two-sample test (e.g. with
  `welcht`, `manwhi`).
- Power analysis (feed d to `powtt2` or `npowtt`).
- Meta-analysis when extracting/combining effect sizes.
- Comparing intervention strengths across different scales / outcomes.

## WHEN NOT TO USE

- Small samples (n_total < 50): d is upward-biased. Use `hedgeg`
  (Hedges' g) — small-sample correction.
- Highly unequal variances: d uses pooled SD which is itself biased
  in that case. Use `glassd` (Glass's Δ) — control SD only.
- Paired designs: convert to standardized mean difference of paired
  differences instead.
- Categorical outcomes — use `cohensh` (proportions) or odds ratio
  (`odds`) with appropriate effect-size scale.

## ASSUMPTIONS

- Two independent groups (no pairing).
- Approximately equal variances in the two groups (or use `glassd`).
- Continuous outcome.

## FORMULA

```
d = (x̄₁ − x̄₂) / s_pooled

s_pooled = √[ ((n₁ − 1)·s₁² + (n₂ − 1)·s₂²) / (n₁ + n₂ − 2) ]
```

Cohen's (1988) conventional benchmarks:
| |d| | label |
|------|-------|
| 0.0  | negligible |
| 0.2  | small |
| 0.5  | medium |
| 0.8  | large |

These are heuristics — domain context matters. In medical research
even d=0.1 can be clinically important.

## INPUTS / OUTPUTS

```
cohend(x1, x2) → RichResult
  x1, x2    independent numeric samples
  returns   .value / .statistic (d), benchmark label, group means,
            mean diff, pooled SD, both n's. Castable to float and
            comparable to scalars (e.g., `cohend(...) > 0.5`).
```

## WORKED EXAMPLE

```python
>>> from morie.fn import cohend
>>> r = cohend([95, 100, 105, 98, 102], [110, 115, 120, 113, 117])
>>> r.value          # the d scalar
-3.06   # ish — large effect
>>> abs(r) > 0.8     # comparison ops work — true since |d| > 0.8
True
>>> print(r)         # full multi-section summary with benchmark label
```

## COMMON MISTAKES

- Reporting d without n: |d| = 0.5 with n=10 vs n=1000 carries very
  different precision (CI width). Always pair with a CI or n.
- Using on small samples without correction — switch to `hedgeg`.
- Confusing d with Pearson r — they're different scales. d/sqrt(d²+4)
  ≈ r for an equal-n two-group comparison.
- Comparing |d| values across radically different domains as if they
  meant the same thing — Cohen's benchmarks are field-relative.

## REFERENCES

- Cohen, J. (1988). Statistical Power Analysis for the Behavioral
  Sciences (2nd ed.). Routledge.
- Weisburd et al. (2022) Advanced Statistics in Criminology and
  Criminal Justice (5th ed.), ch. 11 eq.11.1.
- Wilcox (2017) Modern Statistics ch. 5.
