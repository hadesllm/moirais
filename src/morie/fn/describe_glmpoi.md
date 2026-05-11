# glmpoi - Poisson regression (GLM)

## WHAT IT DOES

Fits a generalized linear model with a log-link and Poisson outcome:
log(E[Y|X]) = X*beta. Coefficients are interpreted on the log-mean
scale; exp(coef) is the multiplicative rate ratio for a one-unit
increase in the predictor.

## WHEN TO USE

- Count outcomes (events, accidents, deaths, recidivism counts).
- Rate-style data (events per unit exposure, with offset).
- Variance approximately equal to mean (no overdispersion).

## WHEN NOT TO USE

- Overdispersion (Variance > Mean substantially) - use negative-
  binomial GLM. Watch the Pearson chi^2/df ratio in the result.
- Zero-inflated counts - use a zero-inflated Poisson or hurdle model.
- Continuous outcomes - use OLS or a Gamma/inverse-Gaussian GLM.

## ASSUMPTIONS

- Counts y_i ~ Poisson(mu_i) with log(mu_i) = X_i' * beta.
- Independent observations.
- Equidispersion: Var(Y|X) = E(Y|X).

## FORMULA

```
log(mu_i) = beta_0 + beta_1 * x_i1 + ... + beta_p * x_ip
log L = sum_i [y_i * log(mu_i) - mu_i - log(y_i!)]
```

## INPUTS / OUTPUTS

```
glmpoi(X, y, add_intercept=True) -> RichResult
  X            n x p predictor matrix
  y            non-negative integer counts
  returns      coefficient table with sig codes, AIC, deviance, Pearson
               chi^2/df overdispersion check, fitted values.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import glmpoi
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> X = rng.standard_normal((100, 2))
>>> y = rng.poisson(np.exp(X @ [0.3, 0.5]))
>>> r = glmpoi(X, y)
>>> print(r)   # full coefficient table + sig codes + chi^2/df
```

## COMMON MISTAKES

- Ignoring the overdispersion warning and reporting too-small SEs.
- Interpreting coefficients additively - they're multiplicative on the
  rate scale: exp(beta) = rate ratio.
- Forgetting an offset for exposure (e.g., person-years).

## REFERENCES

- McCullagh & Nelder (1989) Generalized Linear Models.
- Weisburd et al. (2022) ch.6.
