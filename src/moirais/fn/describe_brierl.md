# brierl - Brier score

## WHAT IT DOES

Mean squared error of probabilistic predictions vs observed binary
outcomes. Lower is better; 0 is perfect calibration. Decomposes into
reliability + resolution + uncertainty.

## WHEN TO USE

- Evaluating probabilistic classifiers (logistic regression, Bayes
  nets, deep nets with sigmoid output).
- Reporting alongside AUROC for fuller picture.
- Comparing calibrated vs uncalibrated predictions.

## WHEN NOT TO USE

- Multi-class - use multi-class Brier or log-loss.
- Pure ranking matters more than probability - use AUROC.

## ASSUMPTIONS

- y_true is 0/1.
- p_pred is a probability in [0, 1].

## FORMULA

```
BS = (1/n) sum_i (p_i - y_i)^2
```

## INPUTS / OUTPUTS

```
brierl(p_pred, y_true) -> RichResult
  returns  .statistic (BS), baseline (predict base rate), skill score.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import brierl
>>> brierl([0.7, 0.3, 0.8, 0.1], [1, 0, 1, 0]).statistic
```

## COMMON MISTAKES

- Reporting BS alone - pair with skill score (vs base-rate baseline)
  to interpret.
- Confusing Brier with log-loss - log-loss penalizes confident
  wrongness more harshly.

## REFERENCES

- Brier (1950). Verification of forecasts expressed in terms of
  probability.
