# glassd - Glass's Δ effect size

## WHAT IT DOES

Like Cohen's d but standardizes by the control group's SD only,
rather than the pooled SD. Useful when treatment can change variance
in addition to mean.

## WHEN TO USE

- Treatment expected to also change variability (e.g., a teaching
  intervention that helps weaker students more than stronger).
- Comparing experiments where the control SD is the natural reference.

## WHEN NOT TO USE

- Variances clearly equal - Cohen's d / Hedges' g use more data.
- Control group is small (sd_c poorly estimated) - prefer pooled
  unless you have a strong reason.

## ASSUMPTIONS

- Independent groups.
- Control SD is meaningful as the reference scale.

## FORMULA

```
Glass's Δ = (mean_treated - mean_control) / sd_control
```

## INPUTS / OUTPUTS

```
glassd(treated, control) -> RichResult
  treated, control   numeric samples
  returns            .statistic (Δ), control SD as reference.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import glassd
>>> glassd([3, 4, 5, 6, 7], [1, 2, 3, 4, 5]).statistic
1.265  # using control SD only
```

## COMMON MISTAKES

- Standardizing by treated SD instead - that's a different quantity.
- Reporting Glass's Δ without explaining why you chose it (most
  readers expect Cohen's d).

## REFERENCES

- Glass (1976). Primary, secondary, and meta-analysis of research.
