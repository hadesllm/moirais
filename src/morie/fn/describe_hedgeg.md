# hedgeg - Hedges' g (bias-corrected Cohen's d)

## WHAT IT DOES

Applies a small-sample correction to Cohen's d. The naive d is biased
upward for small n; g multiplies by a J factor that approaches 1 as
n grows.

## WHEN TO USE

- Reporting d for small total samples (n_total < 50).
- Meta-analysis where small studies need de-biased effects.
- Always preferred over plain d unless you have a specific reason
  to use d (e.g., legacy comparison).

## WHEN NOT TO USE

- Large samples (n_total >> 100) - the correction is negligible.

## ASSUMPTIONS

- Same as Cohen's d: independent groups, approximately Normal,
  approximately equal variances.

## FORMULA

```
g = d * J
J = 1 - 3 / (4(n1 + n2) - 9)
```

J is approximately 1 for n1+n2 large; for n1+n2=10 it's about 0.91.

## INPUTS / OUTPUTS

```
hedgeg(cohens_d, n1, n2) -> RichResult
  returns  .statistic (g), J factor, df.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import hedgeg
>>> hedgeg(0.5, 30, 30).statistic
0.494  # ish - barely-corrected
```

## COMMON MISTAKES

- Reporting d for small samples - prefer g.
- Computing g with wrong J factor (some references use a different
  formula).

## REFERENCES

- Hedges (1981); Hedges & Olkin (1985) Statistical Methods for
  Meta-Analysis.
