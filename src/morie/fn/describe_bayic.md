# bayic - Bayesian (Schwarz) Information Criterion (BIC)

## WHAT IT DOES

Like AIC, but penalises complexity more severely (penalty grows with
log(n) instead of 2). Lower = better. BIC favours sparser models for
n >= 8.

## WHEN TO USE

- Comparing competing models.
- When you want stronger sparsity preference than AIC.
- Approximating log Bayes factors (`bayesf`) under flat priors.

## WHEN NOT TO USE

- Tiny samples (n<10): penalty hardly differs from AIC.
- Models with unconstrained parameter spaces (priors must be proper
  for the BIC approximation to log Bayes factor).

## ASSUMPTIONS

- ML estimation.
- Same data across compared models.

## FORMULA

```
BIC = k log(n) - 2 log(L)
```

## INPUTS / OUTPUTS

```
bayic(loglik, k, n) -> RichResult
  loglik    log-likelihood
  k         parameters
  n         sample size
  returns   .statistic (BIC), AIC for context, penalty difference.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import bayic
>>> bayic(-150, 5, 100).statistic
323.025  # ish
```

## COMMON MISTAKES

- Reporting BIC differences without converting them to log Bayes
  factors when the audience expects evidence weights.
- Using BIC and AIC interchangeably - they prefer different models
  for n>=8.

## REFERENCES

- Schwarz (1978). Estimating the dimension of a model.
