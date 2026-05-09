"""Spatiotemporal heterogeneity test."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def spacetime_heterogeneity(
    values: np.ndarray,
    W: np.ndarray,
    n_perm: int = 499,
    seed: int | None = None,
) -> SpatialResult:
    r"""
    Test for spatiotemporal heterogeneity in a panel dataset.

    Evaluates whether the spatial autocorrelation structure (measured by
    Moran's I) is stable across time periods. The test statistic is the
    variance of per-period Moran's I values:

    .. math::

        H = \frac{1}{T}\sum_{t=1}^{T}(I_t - \bar{I})^2

    Significance is assessed via permutation: rows of the panel are randomly
    permuted and the test statistic recomputed.

    Parameters
    ----------
    values : np.ndarray
        (n, T) panel data.
    W : np.ndarray
        (n, n) spatial weight matrix.
    n_perm : int
        Number of permutations (default 499).
    seed : int or None
        Random seed.

    Returns
    -------
    SpatialResult
        statistic = H (variance of per-period Moran's I), p_value from
        permutation test, local_values = (T,) per-period I values.

    References
    ----------
    Levin SA (1992). The problem of pattern and scale in ecology.
    *Ecology*, 73(6), 1943--1967. doi:10.2307/1941447

    Rey SJ, Janikas MV (2006). STARS: Space-Time Analysis of Regional
    Systems. *Geographical Analysis*, 38(1), 67--86.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n, T = 20, 8
    >>> W = np.ones((n, n)) / (n - 1)
    >>> np.fill_diagonal(W, 0)
    >>> vals = rng.normal(0, 1, (n, T))
    >>> res = spacetime_heterogeneity(vals, W, n_perm=199, seed=42)
    >>> res.statistic >= 0
    True
    """
    values = np.asarray(values, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    n, T = values.shape
    if W.shape != (n, n):
        raise ValueError("W must be (n, n).")

    def _moran_i(col: np.ndarray) -> float:
        z = col - col.mean()
        denom = float(z @ z)
        if denom < 1e-15:
            return 0.0
        return float(z @ W @ z / denom)

    def _het_stat(v: np.ndarray) -> tuple[float, np.ndarray]:
        morans = np.array([_moran_i(v[:, tt]) for tt in range(v.shape[1])])
        return float(np.var(morans, ddof=0)), morans

    obs_H, obs_morans = _het_stat(values)

    rng = np.random.default_rng(seed)
    perm_H = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(n)
        perm_H[k] = _het_stat(values[perm, :])[0]

    p_value = float((np.sum(perm_H >= obs_H) + 1) / (n_perm + 1))

    return SpatialResult(
        name="spacetime_heterogeneity",
        statistic=obs_H,
        p_value=p_value,
        local_values=obs_morans,
        extra={"n": n, "T": T, "n_perm": n_perm, "mean_moran": float(np.mean(obs_morans))},
    )


sthet = spacetime_heterogeneity


def cheatsheet() -> str:
    return "spacetime_heterogeneity({}) -> Spatiotemporal heterogeneity test."
