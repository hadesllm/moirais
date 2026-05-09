# kmsurv - Kaplan-Meier survival estimator

## WHAT IT DOES

Estimates the survival function S(t) = P(T > t) from right-censored
time-to-event data. Produces a step function that drops at each event
time.

## WHEN TO USE

- Time-to-event outcome (deaths, failures, recidivism, etc.).
- Right-censoring present (some subjects exit study before event).
- Visualizing or summarizing survival distribution.

## WHEN NOT TO USE

- Need to adjust for covariates - use Cox proportional hazards.
- Left or interval censoring - use specialized estimators.
- All subjects observed to event - just use the empirical CDF.

## ASSUMPTIONS

- Censoring is non-informative (uncorrelated with event hazard).
- Event times are continuous (ties handled by the estimator).
- All subjects share a common survival distribution (use logrnk to
  test for differences across groups).

## FORMULA

```
S(t) = product over t_k <= t of (1 - d_k / n_k)
```
where d_k = events at t_k, n_k = subjects at risk just before t_k.

## INPUTS / OUTPUTS

```
kmsurv(times, events) -> RichResult
  times    observation times (event or censoring)
  events   1 if event, 0 if censored
  returns  step-function table + median survival + n events/censored.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import kmsurv
>>> r = kmsurv([5, 8, 12, 15, 20], [1, 1, 0, 1, 0])
>>> print(r)        # full step-function table
```

## COMMON MISTAKES

- Treating Kaplan-Meier as if it required no assumptions - non-
  informative censoring is non-trivial in observational data.
- Reading the median survival time as a point estimate without
  reporting its CI.
- Using when you really wanted Cox PH for covariate adjustment.

## REFERENCES

- Kaplan & Meier (1958). Nonparametric estimation from incomplete
  observations. JASA.
- Wilcox (2017) ch.10.
