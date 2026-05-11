# rsq - Coefficient of determination R^2

## WHAT IT DOES

Reports the proportion of variance in y_true explained by predictions
y_pred. Range (-inf, 1]; 1 = perfect, 0 = no better than predicting
the mean, negative = worse than mean.

## WHEN TO USE

- Reporting OLS / regression model fit on the SAME data used to fit.
- Comparing predictions against held-out true values.
- Diagnostic of how much variance the model captures.

## WHEN NOT TO USE

- Logistic / count regression - use mcfadr (McFadden pseudo-R^2).
- Comparing models with different numbers of parameters - use AIC/BIC
  or adjusted R^2.
- Out-of-sample evaluation alone - pair with cross-validation.

## ASSUMPTIONS

- y_true and y_pred have same length.
- y_true is not constant (else SS_total = 0, undefined).

## FORMULA

```
R^2 = 1 - SS_res / SS_tot
SS_res = sum_i (y_i - yhat_i)^2
SS_tot = sum_i (y_i - mean(y))^2
```

## INPUTS / OUTPUTS

```
rsq(y_true, y_pred) -> RichResult
  returns  .statistic (R^2), benchmark, SS values, n.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import rsq
>>> rsq([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]).statistic
1.0
```

## COMMON MISTAKES

- Reporting R^2 in-sample for a fit and out-of-sample for evaluation
  - they're different quantities; label clearly.
- Using R^2 for nonlinear models - it can decrease as you add
  predictors.
- Negative R^2 indicates the model is worse than mean - check parameterisation.

## REFERENCES

- Weisburd et al. (2022) ch.2; Wooditch et al. (2021) ch.15.
