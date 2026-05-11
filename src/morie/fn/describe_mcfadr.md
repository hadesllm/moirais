# mcfadr - McFadden pseudo-R^2

## WHAT IT DOES

Reports a pseudo-R^2 for binary/categorical regression models that
quantifies improvement over an intercept-only null. Range [0, 1].
McFadden's own benchmarks: 0.2-0.4 = excellent fit, NOT comparable
to OLS R^2.

## WHEN TO USE

- Reporting fit of logistic, multinomial, or Poisson regression.
- Comparing nested logistic regressions.

## WHEN NOT TO USE

- OLS regression - use plain R^2.
- Comparing across non-nested specifications - use AIC/BIC.
- Headline reporting alongside OLS R^2 - the scales aren't comparable.

## ASSUMPTIONS

- Both models estimated by maximum likelihood.
- Reduced (null) model is the intercept-only logistic regression.

## FORMULA

```
R^2_McFadden = 1 - LL_full / LL_null
```

## INPUTS / OUTPUTS

```
mcfadr(ll_full, ll_null) -> RichResult
  ll_full    log-likelihood with predictors
  ll_null    log-likelihood of intercept-only
  returns    .statistic (R^2), benchmark label, LL ratio.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import mcfadr
>>> mcfadr(-150.0, -200.0).statistic
0.25
```

## COMMON MISTAKES

- Comparing McFadden R^2 to OLS R^2 - completely different scales.
- Reporting it as a percentage of variance explained - it's NOT
  variance-explained semantics.
- Using when LL_full > LL_null (nominal R^2 > 1) - check parameterization.

## REFERENCES

- McFadden (1974). Conditional logit analysis of qualitative choice
  behavior.
- Weisburd et al. (2022) ch.4.
