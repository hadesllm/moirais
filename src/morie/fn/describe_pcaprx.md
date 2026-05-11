# pcaprx - Principal Component Analysis

## WHAT IT DOES

Decomposes a data matrix X into orthogonal components ranked by the
variance they capture. Used for dimensionality reduction, noise
removal, and exploratory visualization.

## WHEN TO USE

- High-dimensional data you want to reduce for visualization.
- Decorrelating features for downstream modelling.
- Detecting redundant or collinear predictors.
- Exploratory factor analysis (PCA is a common starting point).

## WHEN NOT TO USE

- Categorical data - use multiple correspondence analysis.
- Components must be interpretable as human concepts - use rotated
  factor analysis instead.
- Sparse high-dim data - use sparse PCA.

## ASSUMPTIONS

- Numerical data.
- Linear structure - PCA captures linear correlations only. For
  nonlinear, use kernel PCA or autoencoders.
- Standardized predictors (otherwise the high-variance columns
  dominate the components).

## FORMULA

X = U Sigma V'   (SVD of centered/standardized X)

Components are columns of V; eigenvalues of cov(X) are diag(Sigma)^2 / (n-1).

## INPUTS / OUTPUTS

```
pcaprx(X, n_components=None, standardize=True) -> RichResult
  X              data matrix (n x p)
  n_components   how many to retain (default: all)
  standardize    z-score columns first (recommended)
  returns        component loadings, scores, variance per component,
                 cumulative variance, n_for_80, n_for_95.
```

## WORKED EXAMPLE

```python
>>> from morie.fn import pcaprx
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> X = rng.standard_normal((50, 5))
>>> r = pcaprx(X)
>>> print(r)   # variance-explained table
```

## COMMON MISTAKES

- Not standardizing - one variable on a 0-1000 scale dominates a
  variable on 0-1 scale.
- Treating PC1 as "the most important variable" - it's a linear
  combination; interpret loadings to understand what it means.
- Selecting components by eyeballing the scree plot when CV-based
  selection is more principled.

## REFERENCES

- Pearson (1901); Hotelling (1933).
- Hastie et al. (2009) ch.14; Wilcox (2017) ch.13.
