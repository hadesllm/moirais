# ljbox - Ljung-Box Q-statistic for white noise

## WHAT IT DOES

Tests whether a series (typically residuals) shows any serial
autocorrelation across multiple lags simultaneously. Asymptotic
chi^2(lags) under H0 of white noise.

## WHEN TO USE

- Time-series model diagnostic (after fitting ARIMA/GARCH).
- Catch-all serial-correlation test alongside Durbin-Watson.

## WHEN NOT TO USE

- Sample size < 30 - asymptotic chi^2 approximation poor.
- Heteroscedastic series - use a robust variant.

## ASSUMPTIONS

- Residuals are stationary.
- Sample size is large.

## FORMULA

```
Q = n(n+2) sum_{k=1}^h rho_hat(k)^2 / (n - k)
Q ~ chi^2(h) under H0
```

## INPUTS / OUTPUTS

```
ljbox(residuals, lags=10) -> RichResult
  returns  .statistic (Q), df, .pvalue, max |rho|.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import ljbox
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> ljbox(rng.standard_normal(200), 5).pvalue > 0.05
True
```

## COMMON MISTAKES

- Picking too many lags (h close to n) - lose power.
- Using on too-few residuals.

## REFERENCES

- Ljung & Box (1978). On a measure of lack of fit in time series models.
