# mahalan - Mahalanobis distance

## WHAT IT DOES

Distance between a point and a distribution's centroid, accounting
for variable scales and correlations. A standardized "how far is x
from mu, given the covariance structure?".

## WHEN TO USE

- Multivariate outlier detection.
- Anomaly scoring in high-dimensional data.
- Comparing observations against a multivariate Normal reference.

## WHEN NOT TO USE

- Highly non-Normal data - use a robust covariance estimator.
- Singular covariance matrix - use pinv (we do, with a warning), but
  interpret with extreme caution.
- Categorical data - distance is not meaningful.

## ASSUMPTIONS

- Covariance matrix Sigma is positive-definite (or pinv is acceptable).
- Underlying distribution is approximately multivariate Normal for
  the chi^2 cutoff to be valid.

## FORMULA

```
D = sqrt((x - mu)' * Sigma^-1 * (x - mu))
```
Squared Mahalanobis distance is chi^2(p) under multivariate Normality.

## INPUTS / OUTPUTS

```
mahalan(x, mu, cov) -> RichResult
  x        observation (length p)
  mu       centroid (length p)
  cov      p x p covariance matrix
  returns  distance, squared distance, dimension, condition number,
           chi^2 cutoff comparison.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import mahalan
>>> import numpy as np
>>> mahalan([1, 2], [0, 0], np.eye(2))["distance"]
2.236  # = sqrt(5)
```

## COMMON MISTAKES

- Using a sample covariance matrix that's nearly singular - get a
  near-infinite Mahalanobis distance.
- Treating a single observation's distance as a p-value - it's not;
  compare against the chi^2(p) reference distribution.

## REFERENCES

- Mahalanobis (1936). Wilcox (2017) ch.5.
