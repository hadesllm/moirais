# kentau - Kendall's tau-b correlation

## WHAT IT DOES

Nonparametric correlation between two ordinal/continuous series. Tau-b
counts concordant vs discordant pairs and adjusts for ties in either
variable. Range [-1, 1].

## WHEN TO USE

- Two ordinal or continuous variables.
- Ties present.
- Robust nonparametric correlation that doesn't assume Normality or
  even monotonic linearity.

## WHEN NOT TO USE

- Pure linear-Normal data - Pearson r has more power.
- For monotone but not necessarily linear relationships, Spearman
  rho is also commonly used.

## ASSUMPTIONS

- Pairs are independent.
- Both variables ordinal+.
- Ties handled by tau-b (tau-a doesn't and is biased with ties).

## FORMULA

tau-b = (n_C - n_D) / sqrt((n_C + n_D + n_X) * (n_C + n_D + n_Y))

## INPUTS / OUTPUTS

```
kentau(x, y) -> RichResult
  returns  .statistic (tau-b), .pvalue, n pairs, strength, direction.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import kentau
>>> kentau([1,2,3,4,5], [2,1,4,3,5]).statistic
0.6  # ish
```

## COMMON MISTAKES

- Using tau-a with ties (biased) - use tau-b.
- Comparing tau magnitude to Pearson r magnitude as if they were
  on the same scale - tau is consistently smaller.

## REFERENCES

- Kendall (1938).
