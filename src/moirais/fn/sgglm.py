"""Spatial GLM (Poisson) with residual diagnostics."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def spatial_glm_poisson(
    counts: np.ndarray,
    X: np.ndarray,
    coords: np.ndarray,
    k: int = 5,
) -> SpatialResult:
    r"""Poisson GLM with Moran test on deviance residuals.

    Parameters
    ----------
    counts : np.ndarray
        Count response, shape ``(n,)``.
    X : np.ndarray
        Design matrix, shape ``(n, p)``.
    coords : np.ndarray
        Observation coordinates, shape ``(n, 2)``.
    k : int
        KNN for weights.

    Returns
    -------
    SpatialResult
        ``statistic`` is Moran's I of deviance residuals.
        ``extra`` has ``beta``, ``fitted``, ``deviance_residuals``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 6.

    .. epigraph::

        "We are not our mistakes." -- Aloy, Horizon
    """
    y = np.asarray(counts, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    n = len(y)

    beta = np.zeros(X.shape[1])
    for _ in range(25):
        eta = X @ beta
        mu = np.exp(eta)
        mu = np.maximum(mu, 1e-10)
        W_diag = mu
        z = eta + (y - mu) / mu
        XtWX = X.T * W_diag @ X
        XtWz = X.T @ (W_diag * z)
        try:
            beta = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            break

    mu = np.exp(X @ beta)
    mu = np.maximum(mu, 1e-10)
    sign = np.sign(y - mu)
    dev_resid = sign * np.sqrt(2 * np.abs(y * np.log(np.maximum(y, 1e-10) / mu) - (y - mu)))

    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    Wm = np.zeros((n, n))
    for i in range(n):
        idx = np.argsort(dist[i])[1 : k + 1]
        Wm[i, idx] = 1.0
    rs = Wm.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    Wm = Wm / rs

    e = dev_resid - dev_resid.mean()
    ss = np.sum(e**2)
    S0 = Wm.sum()
    moran_i = float(n * (e @ Wm @ e) / (S0 * ss)) if ss > 0 and S0 > 0 else 0.0

    return SpatialResult(
        name="spatial_glm_poisson",
        statistic=moran_i,
        p_value=None,
        extra={"beta": beta, "fitted": mu, "deviance_residuals": dev_resid},
    )


sgglm = spatial_glm_poisson


def cheatsheet() -> str:
    return "spatial_glm_poisson({}) -> Spatial GLM (Poisson) with residual diagnostics."
