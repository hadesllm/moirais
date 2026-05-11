# diffd - Difference-in-Differences (canonical 2x2)

## WHAT IT DOES

Computes the canonical DD estimate from four group means:
treated_post - treated_pre - (control_post - control_pre). Identifies
the treatment effect under the parallel-trends assumption.

## WHEN TO USE

- Quasi-experimental program evaluation with one treated group and
  one control group, observed before and after an intervention.
- Natural experiments (policy changes, plant openings, etc.).
- A simple summary alongside a regression DD with full controls.

## WHEN NOT TO USE

- More than two time periods - use a panel data DD with two-way
  fixed effects.
- Heterogeneous treatment effects - use Callaway-Sant'Anna or related
  estimators.
- Parallel-trends assumption clearly violated - DD is biased; use
  synthetic control or matching.

## ASSUMPTIONS

- Parallel trends: control's pre-to-post change is a valid
  counterfactual for what treated would have done absent treatment.
- No spillovers from treated to control.
- No anticipation effects (treated didn't change behavior in
  anticipation of treatment).

## FORMULA

```
DD = (Y_treated_post - Y_treated_pre) - (Y_control_post - Y_control_pre)
```

## INPUTS / OUTPUTS

```
diffd(y_treated_pre, y_treated_post, y_control_pre, y_control_post)
  returns   DD estimate, both group changes, parallel-trends warning.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import diffd
>>> diffd(10, 15, 5, 7).statistic
3  # treated gained 5, control gained 2, net effect = 3
```

## COMMON MISTAKES

- Not testing the parallel-trends assumption - plot the pre-period
  trends.
- Reporting DD without standard errors - use a regression to get SE.
- Assuming DD is causal without addressing spillovers.

## REFERENCES

- Card & Krueger (1994). Minimum Wages and Employment.
- Imbens & Wooldridge (2009) NBER lecture notes.
