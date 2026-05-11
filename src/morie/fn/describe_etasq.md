# etasq - Eta-squared (one-way ANOVA effect size)

## WHAT IT DOES

Reports the proportion of total variance in y attributable to group
membership. Range [0, 1]. Used for one-way ANOVA reporting.

## WHEN TO USE

- One-way ANOVA effect-size reporting.
- Comparison across studies with different sample sizes.
- Power analysis for ANOVA designs.

## WHEN NOT TO USE

- Small samples - eta^2 is positively biased; use omega^2 (`omeg2`).
- Complex factorial designs - generalize to partial-eta^2.

## ASSUMPTIONS

- One factor, k groups.
- ANOVA assumptions: independence, Normality within groups, equal
  variances.

## FORMULA

```
eta^2 = SS_between / SS_total
```

## INPUTS / OUTPUTS

```
etasq(ss_between, ss_total) -> RichResult
  ss_between   between-group sum of squares
  ss_total     total sum of squares
  returns      .statistic (eta^2).
```

## WORKED EXAMPLE

```python
>>> from morie.fn import etasq
>>> etasq(40, 100).statistic
0.4  # 40% of variance explained by groups
```

## COMMON MISTAKES

- Reporting eta^2 without n - it's biased upward, especially for
  small n.
- Comparing eta^2 across designs with different df ratios -
  partial-eta^2 standardizes for that.

## REFERENCES

- Cohen (1973). Eta-squared and partial eta-squared.
