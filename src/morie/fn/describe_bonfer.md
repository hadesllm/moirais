# bonfer - Bonferroni multiple-testing correction

## WHAT IT DOES

Computes the per-test alpha you should use when running m comparisons
to control the familywise Type I error rate (FWER) at a target level.
Simplest, most conservative correction.

## WHEN TO USE

- Small number of comparisons (m < 20).
- You want a strict FWER guarantee.
- You're not sure which test will be significant a priori.

## WHEN NOT TO USE

- Large m (>= 100): Bonferroni becomes too conservative; use Holm
  step-down (`holm`) or Benjamini-Hochberg FDR (`bhfdr`) for more power.
- You can rank tests a priori - use a hierarchical scheme.

## ASSUMPTIONS

- Tests are roughly independent (or positively correlated; Bonferroni
  is still valid but loses power).

## FORMULA

```
alpha' = alpha / m
```

Reject test_i if p_i < alpha'.

## INPUTS / OUTPUTS

```
bonfer(alpha, m) -> RichResult
  alpha   target FWER (typically 0.05)
  m       number of tests
  returns .statistic = corrected per-test threshold.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import bonfer
>>> bonfer(0.05, 4).statistic
0.0125
```

## COMMON MISTAKES

- Applying Bonferroni to dependent tests as if it were optimal -
  Holm or BH are usually better.
- Reporting raw p-values without FWER control after running 20+
  tests.

## REFERENCES

- Bonferroni (1936). Wilcox (2017) ch.10.
