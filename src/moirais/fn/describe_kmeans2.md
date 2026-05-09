# kmeans2 - K-means clustering

## WHAT IT DOES

Partition n observations into k clusters that minimize within-cluster
sum-of-squares. Iteratively: assign each point to nearest centroid,
recompute centroids as the mean of assigned points, repeat to
convergence.

## WHEN TO USE

- Exploratory clustering when you suspect k roughly-spherical groups.
- Pre-processing for downstream modelling (cluster-as-feature).
- Vector quantization (lossy compression of high-dim points to k codes).

## WHEN NOT TO USE

- Clusters are non-convex or different sizes/densities - k-means
  prefers equal spherical clusters. Use DBSCAN or HDBSCAN.
- You don't know k - use silhouette analysis or gap statistic to
  choose, or use a method that selects k automatically (DBSCAN).
- Categorical or mixed data - use k-modes or hierarchical clustering.

## ASSUMPTIONS

- Euclidean distance is meaningful.
- Clusters are roughly equally-sized and roughly spherical.
- Features standardized (otherwise large-scale features dominate).

## FORMULA

Minimize total within-cluster sum-of-squares (inertia):
```
J = sum over clusters c of sum over points x in c of ||x - mu_c||^2
```

## INPUTS / OUTPUTS

```
kmeans2(X, n_clusters=3, n_init=10, random_state=42) -> RichResult
  X            data matrix
  n_clusters   k
  n_init       number of restarts (best result kept; protects against
               local minima)
  returns      cluster labels, centroids, inertia, cluster-size table.
```

## WORKED EXAMPLE

```python
>>> from moirais.fn import kmeans2
>>> import numpy as np
>>> X = np.vstack([
...     np.random.default_rng(0).standard_normal((20, 2)),
...     np.random.default_rng(1).standard_normal((20, 2)) + [5, 5],
... ])
>>> r = kmeans2(X, n_clusters=2)
>>> print(r)
```

## COMMON MISTAKES

- Picking k=3 because "that's a good number" - choose by inertia/elbow
  or silhouette.
- Running n_init=1 and getting unlucky - keep n_init=10 default.
- Treating cluster numbers as ordered - they're nominal; cluster 0 vs
  cluster 1 has no inherent meaning.

## REFERENCES

- MacQueen (1967); Lloyd (1957/1982).
- Hastie et al. (2009) ch.14.
