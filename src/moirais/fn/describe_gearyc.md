# gearyc - Geary's C (alternative spatial autocorrelation)

## WHAT IT DOES

Like Moran's I but uses squared differences instead of cross-
products. Range typically [0, 2]; null expectation is 1. Values near
0 indicate strong positive autocorrelation, near 2 strong negative.

## WHEN TO USE

- Same situations as Moran's I.
- Cross-validating Moran's I (the two should usually agree).
- Local-difference-sensitive applications - Geary's C emphasizes
  local outliers.

## WHEN NOT TO USE

- Same exclusions as Moran's I.

## ASSUMPTIONS

- Same as Moran's I.

## FORMULA

```
C = ((n - 1) / (2 S_0)) * sum_{ij} w_{ij} (x_i - x_j)^2
                          / sum_i (x_i - x_bar)^2
```

## INPUTS / OUTPUTS

```
gearyc(x, W) -> RichResult
  returns  .statistic (C), pattern label.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import gearyc
>>> import numpy as np
>>> W = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=float)
>>> gearyc([1, 2, 3], W).statistic
```

## COMMON MISTAKES

- Confusing Moran I and Geary C interpretations - they're inversely
  related (high I corresponds to low C).

## REFERENCES

- Geary (1954). Cliff & Ord (1973).
