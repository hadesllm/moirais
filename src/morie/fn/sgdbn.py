"""Spatial Durbin model (SDM)."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def spatial_durbin_model(
    Z: np.ndarray,
    X: np.ndarray,
    W: np.ndarray,
) -> SpatialResult:
    r"""Fit a Spatial Durbin Model.

    .. math::

        Z = \rho W Z + X \beta + W X \theta + \varepsilon

    Parameters
    ----------
    Z : np.ndarray
        Response, shape ``(n,)``.
    X : np.ndarray
        Covariates, shape ``(n, p)``.
    W : np.ndarray
        Spatial weights, shape ``(n, n)``.

    Returns
    -------
    SpatialResult
        ``statistic`` is estimated :math:`\rho`.
        ``extra`` has ``beta``, ``theta``, ``residuals``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 6.

    .. epigraph::

        "Close your heart to it." -- Kratos, God of War
    """
    Z = np.asarray(Z, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    n = len(Z)
    I = np.eye(n)

    WX = W @ X
    X_aug = np.column_stack([X, WX])

    rho_grid = np.linspace(-0.9, 0.9, 50)
    best_ll = -np.inf
    best_rho = 0.0
    best_coef = None
    best_resid = None

    for rho in rho_grid:
        A = I - rho * W
        Zy = A @ Z
        coef = np.linalg.lstsq(X_aug, Zy, rcond=None)[0]
        resid = Zy - X_aug @ coef
        sigma2 = np.sum(resid**2) / n
        if sigma2 <= 0:
            continue
        sign, logdet = np.linalg.slogdet(A)
        if sign <= 0:
            continue
        ll = -0.5 * n * np.log(2 * np.pi * sigma2) + logdet - 0.5 * n
        if ll > best_ll:
            best_ll = ll
            best_rho = rho
            best_coef = coef
            best_resid = resid

    p = X.shape[1]
    if best_coef is None:
        best_coef = np.zeros(2 * p)
        best_resid = Z.copy()

    return SpatialResult(
        name="spatial_durbin_model",
        statistic=float(best_rho),
        p_value=None,
        extra={
            "beta": best_coef[:p],
            "theta": best_coef[p:],
            "residuals": best_resid,
            "log_likelihood": float(best_ll),
        },
    )


sgdbn = spatial_durbin_model


def cheatsheet() -> str:
    return "spatial_durbin_model({}) -> Spatial Durbin model (SDM)."
