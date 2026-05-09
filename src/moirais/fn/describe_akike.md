# akike - Akaike Information Criterion (AIC)

## WHAT IT DOES

Reports a model-fit index that trades likelihood against complexity.
Lower AIC = better trade-off. Used for nested-or-non-nested model
comparison.

## WHEN TO USE

- Comparing competing models on the same data.
- Selecting the best of several non-nested specifications.
- Reporting alongside log-likelihood and BIC.

## WHEN NOT TO USE

- Comparing models on different datasets - AIC is not transferable.
- When you specifically want sparser models for large n - prefer BIC
  (`bayic`).

## ASSUMPTIONS

- Models are estimated by maximum likelihood (or pseudo-ML).
- Sample sizes are equal across compared models.

## FORMULA

```
AIC = 2k - 2 log(L)
```
where k = number of estimated parameters, L = maximum likelihood.

## INPUTS / OUTPUTS

```
akike(loglik, k) -> RichResult
  loglik    log-likelihood of fitted model (negative is fine)
  k         number of parameters
  returns   .statistic (AIC), penalty 2k.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import akike
>>> akike(-150, 5).statistic
310.0
```

## COMMON MISTAKES

- Comparing AICs across models with different sample sizes.
- Reporting AIC alone - always show k and log L.
- Confusing AIC with AICc (small-sample-corrected) - they diverge for n<40.

## REFERENCES

- Akaike (1974). A new look at the statistical model identification.
- Burnham & Anderson (2002) Model Selection and Multimodel Inference.
