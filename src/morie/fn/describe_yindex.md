# yindex - Youden's J / informedness

## WHAT IT DOES

Reports the difference between true-positive rate and false-positive
rate for a single threshold. Range [-1, 1]; 1 = perfect, 0 = chance.
Maximized over thresholds gives the optimal cutoff under equal mis-
classification costs.

## WHEN TO USE

- Selecting an operating threshold for a continuous classifier.
- Reporting alongside AUROC.
- Quick informedness summary for a single decision rule.

## WHEN NOT TO USE

- Cost ratio matters (FP and FN have different costs) - compute a
  cost-weighted utility instead.
- Continuous decision needed - use AUROC.

## ASSUMPTIONS

- Binary classification at a single threshold.
- TPR and FPR are estimable from your data.

## FORMULA

```
J = TPR - FPR
```

## INPUTS / OUTPUTS

```
yindex(tpr, fpr) -> RichResult
  returns  .statistic (J), TPR, FPR, specificity = 1 - FPR.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import yindex
>>> yindex(0.8, 0.2).statistic
0.6
```

## COMMON MISTAKES

- Picking a "best" threshold by maximizing J on training data and
  reporting performance on the same data - this is optimistic; use
  cross-validation.
- Treating J as a percentage - it's a difference of rates, not a
  rate.

## REFERENCES

- Youden (1950). Index for rating diagnostic tests.
