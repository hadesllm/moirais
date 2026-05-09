# logloss - Cross-entropy / binary log loss

## WHAT IT DOES

Mean negative log-likelihood for binary predictions. Penalizes
confident wrong predictions much more than Brier score. Lower is
better; 0 is perfect.

## WHEN TO USE

- Training-loss reporting for logistic regression and binary classifiers.
- Comparing models on the same dataset.
- Calibration assessment when extreme errors matter most.

## WHEN NOT TO USE

- Multi-class - use categorical cross-entropy.
- Robust to outliers needed - log-loss is sensitive to confident wrong.

## ASSUMPTIONS

- y_true is 0/1.
- p_pred is in (0, 1) - clipped to [eps, 1-eps] internally.

## FORMULA

```
LL = -(1/n) sum_i [y_i log(p_i) + (1-y_i) log(1-p_i)]
```

## INPUTS / OUTPUTS

```
logloss(p_pred, y_true, eps=1e-15) -> RichResult
  returns  .statistic (LL), baseline LL, skill score.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import logloss
>>> logloss([0.7, 0.3, 0.8, 0.1], [1, 0, 1, 0]).statistic
```

## COMMON MISTAKES

- Allowing p_pred = 0 or 1 - log(0) = -infinity. Always clip.
- Using log-loss alone when calibration AND ranking matter -
  pair with AUROC + Brier.

## REFERENCES

- Cover & Thomas (2006) Elements of Information Theory ch.2.
