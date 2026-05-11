# dwtest - Durbin-Watson test for autocorrelated residuals

## WHAT IT DOES

Tests whether OLS residuals exhibit serial autocorrelation. The DW
statistic ranges 0-4: ~2 means none, <2 means positive autocorrelation,
>2 means negative.

## WHEN TO USE

- Time-series regression diagnostic.
- Pair with Ljung-Box (`ljbox`) for cross-validation.
- Standard reporting for any time-ordered residuals.

## WHEN NOT TO USE

- Cross-sectional data with no natural order.
- Lagged dependent variable as predictor - DW is biased; use
  Durbin's h instead.

## ASSUMPTIONS

- Residuals from a regression with intercept, no lagged dependent
  variable.
- Errors are AR(1) under the alternative.

## FORMULA

```
DW = sum_t (e_t - e_{t-1})^2 / sum_t e_t^2
```

## INPUTS / OUTPUTS

```
dwtest(residuals) -> RichResult
  returns  .statistic (DW), verdict label, n.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import dwtest
>>> dwtest([0.1, -0.2, 0.3, -0.4, 0.5, -0.6]).statistic
3.13  # ish - negative autocorrelation
```

## COMMON MISTAKES

- Using DW critical values when n is small or there are many
  predictors - tabulated values matter.
- Treating DW = 2 as proof of no autocorrelation - it's a test
  statistic, not a binary indicator.

## REFERENCES

- Durbin & Watson (1950, 1951, 1971).
