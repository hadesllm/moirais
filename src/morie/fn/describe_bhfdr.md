# bhfdr - Benjamini-Hochberg false discovery rate

## WHAT IT DOES

Controls the expected proportion of false discoveries (FDR) among
rejected nulls. More powerful than FWER procedures (Bonferroni, Holm)
- you reject more tests at the cost of allowing some Type I errors.

## WHEN TO USE

- Many tests (m large), where FWER control is too strict.
- High-throughput screening (genomics, neuroimaging, A/B test sweeps).
- When you can tolerate some false positives in the rejection set.

## WHEN NOT TO USE

- Few tests with strong consequences for any false positive - use
  Holm.
- You need a strict guarantee that no Type I error has occurred -
  FWER procedures are appropriate.

## ASSUMPTIONS

- BH is valid under independence or positive regression dependence.
- For arbitrary dependence, use the Benjamini-Yekutieli variant.

## FORMULA

For sorted p-values p_(1) <= ... <= p_(m):
```
k* = max k where p_(k) <= k * alpha / m
reject all tests with rank <= k*
```

## INPUTS / OUTPUTS

```
bhfdr(p_values, alpha=0.05) -> RichResult
  p_values   list of test p-values
  alpha      target FDR (typically 0.05 or 0.10)
  returns    per-test reject/fail, sorted-by-p table, threshold k*alpha/m.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import bhfdr
>>> bhfdr([0.001, 0.04, 0.045, 0.5], 0.05)["rejects"]
[True, False, False, False]
```

## COMMON MISTAKES

- Treating FDR as if it were FWER - they have different
  guarantees. FDR allows ~alpha-fraction false positives in the
  rejected set.
- Mis-applying when tests have severe negative dependence - use BY
  variant.

## REFERENCES

- Benjamini & Hochberg (1995). Controlling the false discovery rate.
