# bayesf - Bayes factor (BIC approximation)

## WHAT IT DOES

Computes the approximate Bayes factor BF_10 from BIC values: how
many times more likely is the data under H1 than under H0? Uses
Schwarz's BIC approximation, valid as a quick rule under flat priors.

## WHEN TO USE

- Quick model comparison without explicit prior elicitation.
- Reporting evidence weights alongside p-values.
- Choosing between nested or non-nested model specifications.

## WHEN NOT TO USE

- True Bayes factor with informative priors - use full marginal
  likelihood computation.
- BIC fails for some models (e.g., mixture models with unidentified
  components) - the approximation breaks.

## ASSUMPTIONS

- Each model fit by maximum likelihood.
- Priors are roughly flat - the BIC approximation breaks for highly
  informative priors.
- Sample sizes equal across compared models.

## FORMULA

```
BF_10 ~= exp(-(BIC_1 - BIC_0) / 2)
```

Jeffreys/Kass-Raftery scale:
```
BF > 100         decisive
BF 30 to 100     very strong
BF 10 to 30      strong
BF 3 to 10       substantial
BF 1 to 3        weak
BF < 1           favors H0 (mirrored)
```

## INPUTS / OUTPUTS

```
bayesf(loglik_h1, loglik_h0, k_h1, k_h0, n) -> RichResult
  returns   BF_10, log10(BF), strength label, BIC values, k's.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import bayesf
>>> bayesf(-100, -110, 5, 3, 100).statistic
220.27  # ish - decisive in favor of H1
```

## COMMON MISTAKES

- Treating BF_10 as p-value-like - it's an evidence ratio, not a
  probability.
- Reading BF=1.5 as "strong evidence" - it's barely worth mentioning.

## REFERENCES

- Kass & Raftery (1995). Bayes factors.
- Schwarz (1978).
