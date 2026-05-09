# somerd - Somers' D (asymmetric ordinal-ordinal)

## WHAT IT DOES

Asymmetric measure of ordinal-ordinal association. Treats x as
predictor and y as outcome - counts pair-wise wins on y conditional
on x's order, normalised by pairs untied on x.

## WHEN TO USE

- One ordinal variable predicts another and you want a directional
  measure.
- Rank-based concordance for binary classification (related to AUROC).
- Survival analysis: Somers' D is a common discrimination measure.

## WHEN NOT TO USE

- Symmetric measure of association needed - use kentau or spearm.
- Two continuous variables with linear relationship - Pearson is
  fine.

## ASSUMPTIONS

- Independent pairs.
- Both ordinal+.

## FORMULA

```
D(Y|X) = (n_C - n_D) / (n_C + n_D + n_Y)
```
where n_Y = pairs tied on Y (but not on X).

## INPUTS / OUTPUTS

```
somerd(x, y) -> RichResult
  returns  .statistic (D), .pvalue, table of cell counts.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import somerd
>>> somerd([1,2,3,4,5], [1,2,3,4,5]).statistic
1.0
```

## COMMON MISTAKES

- Treating Somers' D as symmetric - swap predictor/outcome and you
  get a different number.
- Confusing Somers' D with Goodman-Kruskal gamma - both ignore some
  ties but differently.

## REFERENCES

- Somers (1962).
