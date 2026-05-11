# aurroc - Area under ROC curve

## WHAT IT DOES

Quantifies the ability of a continuous score to rank positive cases
above negative cases. AUROC = P(score(positive) > score(negative)).

## WHEN TO USE

- Binary classification model (logistic regression, classifier).
- Reporting model performance independent of threshold choice.
- Comparing classifiers on the same dataset.

## WHEN NOT TO USE

- Class imbalance very severe - AUROC can be optimistic; use precision-
  recall AUC instead.
- Multi-class - use macro/micro-averaged AUROC.
- Calibration matters more than ranking - use brierl or logloss.

## ASSUMPTIONS

- Binary outcome.
- Score is continuous and rank-orderable.
- y_true encodes positive class as 1.

## FORMULA

```
AUROC = integral over thresholds of TPR(t) * (-dFPR(t))
```
Equivalent to the Mann-Whitney U / (n_pos * n_neg).

## INPUTS / OUTPUTS

```
aurroc(y_true, score) -> RichResult
  y_true   binary labels (0/1)
  score    continuous decision scores
  returns  .statistic (AUROC in [0, 1]), benchmark, n positive/negative.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import aurroc
>>> r = aurroc([0, 0, 0, 1, 1, 1], [0.1, 0.3, 0.5, 0.6, 0.7, 0.9])
>>> r.statistic   # 1.0 - perfect ranking
```

## COMMON MISTAKES

- Reporting AUROC without confidence interval - sample variance
  matters, especially with small or imbalanced n.
- Using AUROC as a sole metric when calibration is also important.
- Confusing AUROC=0.5 (chance) with AUROC=0 (perfectly inverted) -
  the latter means you should flip your model's predictions.

## REFERENCES

- Hanley & McNeil (1982). The meaning and use of the area under a
  receiver operating characteristic curve. Radiology.
- Hastie, Tibshirani & Friedman (2009) ch.9.
