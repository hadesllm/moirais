"""Spatial logistic regression with diagnostics."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def spatial_logistic(
    binary_Y: np.ndarray,
    X: np.ndarray,
    coords: np.ndarray,
    k: int = 5,
) -> SpatialResult:
    r"""Logistic regression with spatial autocorrelation check.

    Parameters
    ----------
    binary_Y : np.ndarray
        Binary response, shape ``(n,)``.
    X : np.ndarray
        Design matrix, shape ``(n, p)``.
    coords : np.ndarray
        Observation coordinates, shape ``(n, 2)``.
    k : int
        KNN for weights.

    Returns
    -------
    SpatialResult
        ``statistic`` is Moran's I of Pearson residuals.
        ``extra`` has ``beta``, ``fitted_prob``, ``pearson_residuals``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 6.

    .. epigraph::

        "All of this happened because people refused to see." -- Aloy, Horizon
    """
    y = np.asarray(binary_Y, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    n = len(y)

    beta = np.zeros(X.shape[1])
    for _ in range(25):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        W_diag = mu * (1 - mu)
        z = eta + (y - mu) / W_diag
        XtWX = X.T * W_diag @ X
        XtWz = X.T @ (W_diag * z)
        try:
            beta = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            break

    mu = 1.0 / (1.0 + np.exp(-(X @ beta)))
    mu = np.clip(mu, 1e-10, 1 - 1e-10)
    pearson = (y - mu) / np.sqrt(mu * (1 - mu))

    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    Wm = np.zeros((n, n))
    for i in range(n):
        idx = np.argsort(dist[i])[1 : k + 1]
        Wm[i, idx] = 1.0
    rs = Wm.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    Wm = Wm / rs

    e = pearson - pearson.mean()
    ss = np.sum(e**2)
    S0 = Wm.sum()
    moran_i = float(n * (e @ Wm @ e) / (S0 * ss)) if ss > 0 and S0 > 0 else 0.0

    return SpatialResult(
        name="spatial_logistic",
        statistic=moran_i,
        p_value=None,
        extra={"beta": beta, "fitted_prob": mu, "pearson_residuals": pearson},
    )


sglgt = spatial_logistic


def cheatsheet() -> str:
    return "spatial_logistic({}) -> Spatial logistic regression with diagnostics."
