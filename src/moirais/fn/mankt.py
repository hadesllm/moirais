# moirais.fn — function file (hadesllm/moirais)
"""Mantel test for spatial matrix correlation."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def mantel_test(
    dist_x: np.ndarray,
    dist_y: np.ndarray,
    n_perm: int = 999,
    seed: int | None = None,
) -> SpatialResult:
    r"""
    Mantel test for correlation between two distance/dissimilarity matrices.

    Computes the Pearson correlation between the lower-triangular elements of
    two symmetric distance matrices and assesses significance via permutation.

    .. math::

        r_M = \frac{\sum_{i<j}(X_{ij} - \bar{X})(Y_{ij} - \bar{Y})}
              {\sqrt{\sum_{i<j}(X_{ij}-\bar{X})^2 \sum_{i<j}(Y_{ij}-\bar{Y})^2}}

    Parameters
    ----------
    dist_x : np.ndarray
        (n, n) symmetric distance matrix for the first variable.
    dist_y : np.ndarray
        (n, n) symmetric distance matrix for the second variable.
    n_perm : int
        Number of permutations for significance testing (default 999).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    SpatialResult
        statistic = Mantel r, p_value from permutation test, extra has
        ``perm_distribution`` array.

    References
    ----------
    Mantel N (1967). The detection of disease clustering and a generalized
    regression approach. *Cancer Research*, 27(2), 209--220.

    Legendre P, Legendre L (2012). *Numerical Ecology*. 3rd English ed.
    Elsevier. Chapter 10.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.spatial.distance import squareform, pdist
    >>> coords = np.random.default_rng(42).uniform(0, 100, (20, 2))
    >>> vals = np.random.default_rng(42).normal(0, 1, 20)
    >>> geo = squareform(pdist(coords))
    >>> attr = squareform(pdist(vals.reshape(-1, 1)))
    >>> res = mantel_test(geo, attr, n_perm=499, seed=42)
    >>> isinstance(res.statistic, float)
    True
    """
    dist_x = np.asarray(dist_x, dtype=np.float64)
    dist_y = np.asarray(dist_y, dtype=np.float64)
    n = dist_x.shape[0]
    if dist_x.shape != (n, n) or dist_y.shape != (n, n):
        raise ValueError("Both matrices must be square and same size.")

    idx = np.tril_indices(n, k=-1)
    x_vec = dist_x[idx]
    y_vec = dist_y[idx]

    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        a_c = a - a.mean()
        b_c = b - b.mean()
        denom = np.sqrt(np.sum(a_c**2) * np.sum(b_c**2))
        if denom == 0.0:
            return 0.0
        return float(np.sum(a_c * b_c) / denom)

    obs_r = _pearson(x_vec, y_vec)

    rng = np.random.default_rng(seed)
    perm_r = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(n)
        perm_mat = dist_y[np.ix_(perm, perm)]
        perm_r[k] = _pearson(x_vec, perm_mat[idx])

    p_value = float((np.sum(perm_r >= obs_r) + 1) / (n_perm + 1))

    return SpatialResult(
        name="mantel_test",
        statistic=obs_r,
        p_value=p_value,
        extra={"n": n, "n_perm": n_perm, "perm_distribution": perm_r},
    )


mankt = mantel_test


def cheatsheet() -> str:
    return "mantel_test({}) -> Mantel test for spatial matrix correlation."
