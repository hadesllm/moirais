# mcc - Matthews correlation coefficient

## WHAT IT DOES

Balanced quality measure for binary classification, even with imbalanced
classes. Range [-1, 1]: +1 perfect, 0 chance, -1 inversely correlated.

## WHEN TO USE

- Imbalanced binary classification (e.g., 95% negative class).
- Reporting alongside accuracy when accuracy alone is misleading.
- Multi-class MCC also exists (for >2 classes).

## WHEN NOT TO USE

- Probabilistic predictions matter (use Brier / log-loss / AUROC).
- Sample size very small - MCC is highly variable.

## ASSUMPTIONS

- Binary labels.
- Confusion matrix counts are non-negative integers.

## FORMULA

```
MCC = (TP * TN - FP * FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

## INPUTS / OUTPUTS

```
mcc(tp, tn, fp, fn) -> RichResult
  returns  .statistic (MCC), accuracy for context, all four cells.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import mcc
>>> mcc(50, 40, 5, 5).statistic
0.798
```

## COMMON MISTAKES

- Treating MCC as a probability - it's a correlation.
- Reporting MCC alone in heavily imbalanced data without also
  showing precision/recall.

## REFERENCES

- Matthews (1975). Comparison of the predicted and observed secondary
  structure of T4 phage lysozyme.
