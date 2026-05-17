"""OLS with spatial diagnostics on residuals."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def ols_spatial_diagnostics(
    Z: np.ndarray,
    X: np.ndarray,
    coords: np.ndarray,
    k: int = 5,
) -> SpatialResult:
    r"""OLS regression with Moran's I test on residuals.

    Parameters
    ----------
    Z : np.ndarray
        Response vector, shape ``(n,)``.
    X : np.ndarray
        Design matrix, shape ``(n, p)``.
    coords : np.ndarray
        Observation coordinates, shape ``(n, 2)``.
    k : int
        Number of nearest neighbours for weight matrix.

    Returns
    -------
    SpatialResult
        ``statistic`` is Moran's I of OLS residuals.
        ``extra`` has ``beta``, ``residuals``, ``r_squared``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 6.

    .. epigraph::

    """
    Z = np.asarray(Z, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    n = len(Z)

    beta = np.linalg.lstsq(X, Z, rcond=None)[0]
    resid = Z - X @ beta
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((Z - Z.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    W = np.zeros((n, n))
    for i in range(n):
        idx = np.argsort(dist[i])[1 : k + 1]
        W[i, idx] = 1.0
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    W = W / rs

    e = resid - resid.mean()
    num = float(n * (e @ W @ e))
    den = float(W.sum() * np.sum(e**2))
    moran_i = num / den if den != 0 else 0.0

    return SpatialResult(
        name="ols_spatial_diagnostics",
        statistic=moran_i,
        p_value=None,
        extra={"beta": beta, "residuals": resid, "r_squared": r2},
    )


sgols = ols_spatial_diagnostics


def cheatsheet() -> str:
    return "ols_spatial_diagnostics({}) -> OLS with spatial diagnostics on residuals."
