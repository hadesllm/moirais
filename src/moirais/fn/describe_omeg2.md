# omeg2 - Omega-squared (less biased than eta^2)

## WHAT IT DOES

Like eta^2 but with a small-sample correction. Less upward-biased,
especially for small n and many groups.

## WHEN TO USE

- ANOVA effect-size reporting, especially for small samples.
- Meta-analysis aggregating across studies.
- Always preferred over eta^2 unless legacy reasons demand eta^2.

## WHEN NOT TO USE

- Very large samples - omega^2 ~ eta^2.
- Mixed-design ANOVA - need partial omega^2 variants.

## ASSUMPTIONS

- One factor, k groups.
- Normality and equal variances within groups.

## FORMULA

```
omega^2 = (SS_between - df_between * MS_within) / (SS_total + MS_within)
```

## INPUTS / OUTPUTS

```
omeg2(ss_between, df_between, ms_within, ss_total) -> RichResult
  ss_between, df_between   between-group SS and df
  ms_within                within-group MS (= SS_within / df_within)
  ss_total                 total SS
  returns                  .statistic (omega^2).
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import omeg2
>>> omeg2(40, 2, 5, 100).statistic
0.286  # less than eta^2 = 0.40 from same data
```

## COMMON MISTAKES

- Computing with wrong df - omega^2 needs df_between AND ms_within.
- Treating omega^2 = eta^2 - they diverge for small samples.

## REFERENCES

- Hays (1963). Statistics for Psychologists.
- Olejnik & Algina (2003).
