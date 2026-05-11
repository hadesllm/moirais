# powtt2 - Power for two-sample t-test

## WHAT IT DOES

Computes the probability of correctly rejecting H0 when the true
effect size is d, for a given two-sample t-test setup. Power = 1 -
beta = the probability of NOT making a Type II error.

## WHEN TO USE

- A priori power analysis before designing a study.
- Post hoc to determine if your study had enough power to detect
  reasonable effects.
- Comparing alternative sample size allocations.

## WHEN NOT TO USE

- For Welch t-test - this assumes equal variances.
- Paired or one-sample - use the appropriate power calculation.
- Need to plan for non-Normal data - use simulation-based power.

## ASSUMPTIONS

- Two-sample t with equal variances.
- d is a true effect size of interest (you specify it).

## FORMULA

Power = 1 - F(t_crit; df, ncp) + F(-t_crit; df, ncp)
where F is the noncentral t CDF, t_crit = t_{1-alpha/2; df},
df = n1 + n2 - 2, ncp = d * sqrt(n1*n2/(n1+n2)).

## INPUTS / OUTPUTS

```
powtt2(d, n1, n2, alpha=0.05) -> RichResult
  d              true effect size (Cohen's d)
  n1, n2         per-group sample sizes
  alpha          significance level
  returns        power, beta = 1-power, df.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import powtt2
>>> powtt2(0.5, 30, 30).statistic
0.48  # ~50% chance to detect d=0.5 with n=30 per group
```

## COMMON MISTAKES

- Reporting "low power" without specifying for what effect - power
  depends on d.
- Using to "justify" a non-significant result post hoc - power
  analysis is for design, not interpretation.

## REFERENCES

- Cohen (1988). Statistical Power Analysis for the Behavioral Sciences.
