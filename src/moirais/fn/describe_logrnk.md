# logrnk - Log-rank test (two-group survival)

## WHAT IT DOES

Tests whether two survival distributions differ. At each event time,
counts observed vs. expected events under H0 (same distribution); the
sum is chi-squared distributed.

## WHEN TO USE

- Comparing two groups' time-to-event distributions.
- Censoring present (uses all available data).
- Accompanies Kaplan-Meier visualization.

## WHEN NOT TO USE

- More than two groups - use stratified log-rank or Wilcoxon-Gehan.
- Need to adjust for covariates - Cox PH.
- Hazards cross between groups - log-rank loses power; use a
  weighted variant or restricted-mean-survival comparison.

## ASSUMPTIONS

- Proportional hazards (failure rates have constant ratio across time).
- Non-informative censoring.
- Independence of observations.

## FORMULA

```
chi^2 = (O - E)^2 / V
```
where O is observed events in group 1, E is expected under H0, V is
the variance of O - all summed over event times.

## INPUTS / OUTPUTS

```
logrnk(times1, events1, times2, events2) -> RichResult
  times*, events*  per-group time + event vectors
  returns          .statistic, .pvalue, df=1, observed/expected per group.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import logrnk
>>> r = logrnk([5, 8, 12], [1, 1, 1], [10, 20, 30], [1, 1, 0])
>>> r.pvalue
```

## COMMON MISTAKES

- Treating significant log-rank as evidence of "constant hazard
  ratio" - it requires proportional hazards but doesn't quantify.
- Using when proportional-hazards assumption is violated - look for
  crossing K-M curves first.

## REFERENCES

- Mantel (1966). Evaluation of survival data and two new rank order
  statistics arising in its consideration.
