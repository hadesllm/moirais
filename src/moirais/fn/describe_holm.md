# holm - Holm-Bonferroni step-down rejection

## WHAT IT DOES

Sorts p-values from smallest to largest and rejects each in sequence
against alpha/(m - k) thresholds. Once any test fails to be rejected,
all larger p-values automatically fail. Strictly more powerful than
plain Bonferroni while giving the same FWER guarantee.

## WHEN TO USE

- Multiple-testing scenarios where you want FWER control.
- m anywhere from 5 to thousands.
- Almost always preferred over plain Bonferroni for the same control.

## WHEN NOT TO USE

- You'd rather control FDR than FWER (use `bhfdr`).
- Tests have known logical structure - use hierarchical procedures.

## ASSUMPTIONS

- Same as Bonferroni; valid under independence or positive dependence.

## FORMULA

For sorted p-values p_(1) <= p_(2) <= ... <= p_(m):
```
reject test (k) if p_(k) <= alpha / (m - k + 1)
```
Once any test fails, all subsequent fail by definition.

## INPUTS / OUTPUTS

```
holm(p_values, alpha=0.05) -> RichResult
  p_values   list of test p-values
  alpha      FWER target
  returns    per-test reject/fail decisions, table sorted by p,
             count of rejections.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import holm
>>> holm([0.001, 0.04, 0.05, 0.001], 0.05)["rejects"]
[True, False, False, True]
```

## COMMON MISTAKES

- Forgetting Holm is sequential - one failure stops all larger ones.
- Reporting BH FDR results as if they were FWER-controlled.

## REFERENCES

- Holm (1979). A simple sequentially rejective multiple test procedure.
