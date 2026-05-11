# gkgam - Goodman-Kruskal gamma

## WHAT IT DOES

Symmetric measure of ordinal association. Uses only concordant and
discordant pairs (ignores ties entirely). Range [-1, 1].

## WHEN TO USE

- Two ordinal variables with potentially many ties.
- Symmetric measure where ties shouldn't penalise concordance.
- Comparing categorical/ordinal-treatment-vs-ordinal-outcome studies.

## WHEN NOT TO USE

- Continuous data - use Pearson, Spearman, or Kendall.
- Need a measure that handles ties via penalty (use kentau-b instead).

## ASSUMPTIONS

- Independent pairs.
- Both ordinal+.

## FORMULA

```
gamma = (n_C - n_D) / (n_C + n_D)
```

## INPUTS / OUTPUTS

```
gkgam(x, y) -> RichResult
  returns  .statistic (gamma), concordant/discordant/tied counts, total pairs.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import gkgam
>>> gkgam([1,2,3,4,5], [1,2,3,4,5]).statistic
1.0
```

## COMMON MISTAKES

- Tied-pair-heavy data: gamma can look strong (|gamma| ~ 1) when most
  pairs are tied; check the tied-pair count.
- Comparing gamma magnitude across studies with different tie rates.

## REFERENCES

- Goodman & Kruskal (1954).
