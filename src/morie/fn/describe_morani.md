# morani - Moran's I (spatial autocorrelation)

## WHAT IT DOES

Tests whether values at nearby locations are more similar than expected
under random arrangement. Positive I means clustering (similar values
cluster together); near-zero means random; negative means dispersion
(checkerboard pattern).

## WHEN TO USE

- Geo-located data: crime rates, disease incidence, environmental
  measures, etc.
- Detecting spatial clusters before fitting non-spatial models.
- Pairing with Geary's C for cross-validation.

## WHEN NOT TO USE

- Need local rather than global autocorrelation - use Local Moran's I
  (LISA).
- Time-series autocorrelation - use Durbin-Watson (`dwtest`) or
  Ljung-Box (`ljbox`).

## ASSUMPTIONS

- Spatial weights matrix W is non-negative and meaningful (typically
  row-standardized).
- Independent observations after accounting for spatial structure.

## FORMULA

```
I = (n / S_0) * sum_{ij} w_{ij} (x_i - x_bar)(x_j - x_bar)
                / sum_i (x_i - x_bar)^2
```

where S_0 = sum of weights. Null expectation E[I] = -1/(n-1).

## INPUTS / OUTPUTS

```
morani(x, W) -> RichResult
  x      length-n attribute vector
  W      n x n spatial weights
  returns .statistic (I), null expectation, pattern label.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import morani
>>> import numpy as np
>>> W = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=float)
>>> morani([1, 2, 3], W).statistic
```

## COMMON MISTAKES

- Using a binary contiguity W when the natural weights are inverse-
  distance - choice of W matters.
- Reporting I without a null-distribution test - p-values come from
  permutation or analytic approximation.

## REFERENCES

- Moran (1950). Notes on continuous stochastic phenomena.
- Cliff & Ord (1973).
