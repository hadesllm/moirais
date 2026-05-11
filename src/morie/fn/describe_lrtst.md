# lrtst - Likelihood ratio test (nested models)

## WHAT IT DOES

Tests whether a richer model (more parameters) fits significantly
better than a nested simpler model. -2*(LL_red - LL_full) follows
chi^2(df_diff) under H0.

## WHEN TO USE

- Comparing nested GLMs (logistic, Poisson, Gamma).
- Testing whether to include a variable / interaction term.
- Testing nested random-effect specifications.

## WHEN NOT TO USE

- Non-nested models - use AIC/BIC.
- Heavily violated regularity conditions (e.g., parameters on
  boundary) - chi^2 approximation breaks.

## ASSUMPTIONS

- Both models fit by maximum likelihood on the same data.
- Reduced model is nested inside the full.
- Sample size is large enough for chi^2 approximation.

## FORMULA

```
LR = -2 (LL_reduced - LL_full)
df = number of additional parameters in full
LR ~ chi^2(df) under H0 (reduced is correct)
```

## INPUTS / OUTPUTS

```
lrtst(loglik_full, loglik_reduced, df_diff) -> RichResult
  returns  .statistic (LR), df, .pvalue, LL difference, extra params.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import lrtst
>>> lrtst(-150, -160, 2).pvalue
< 0.01
```

## COMMON MISTAKES

- Comparing non-nested models - use AIC/BIC.
- Forgetting df_diff = number of constraints relaxed - usually new
  parameters added.
- LL_full < LL_reduced - means the more complex model fits worse, a
  red flag of a problem.

## REFERENCES

- Wilks (1938). The large-sample distribution of the likelihood ratio
  for testing composite hypotheses.
