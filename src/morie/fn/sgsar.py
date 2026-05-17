"""Spatial autoregressive lag model (SAR)."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def sar_lag_model(
    Z: np.ndarray,
    X: np.ndarray,
    W: np.ndarray,
    rho_init: float = 0.0,
) -> SpatialResult:
    r"""Fit a spatial lag model via concentrated log-likelihood.

    .. math::

        Z = \rho W Z + X\beta + \varepsilon

    Uses grid search over :math:`\rho`.

    Parameters
    ----------
    Z : np.ndarray
        Response, shape ``(n,)``.
    X : np.ndarray
        Design matrix, shape ``(n, p)``.
    W : np.ndarray
        Row-standardized spatial weights, shape ``(n, n)``.
    rho_init : float
        Initial value (not used; grid search).

    Returns
    -------
    SpatialResult
        ``statistic`` is estimated :math:`\rho`.
        ``extra`` has ``beta``, ``residuals``, ``log_likelihood``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 6.

    .. epigraph::

    """
    Z = np.asarray(Z, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    n = len(Z)

    eigs = np.real(np.linalg.eigvals(W))
    rho_min = 1.0 / min(eigs.min(), -0.01) + 0.01
    rho_max = 1.0 / max(eigs.max(), 0.01) - 0.01
    rho_grid = np.linspace(max(rho_min, -0.99), min(rho_max, 0.99), 50)

    best_ll = -np.inf
    best_rho = 0.0
    best_beta = None
    best_resid = None

    I = np.eye(n)
    for rho in rho_grid:
        A = I - rho * W
        Zy = A @ Z
        beta = np.linalg.lstsq(X, Zy, rcond=None)[0]
        resid = Zy - X @ beta
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
            best_beta = beta
            best_resid = resid

    if best_beta is None:
        best_beta = np.linalg.lstsq(X, Z, rcond=None)[0]
        best_resid = Z - X @ best_beta

    return SpatialResult(
        name="sar_lag_model",
        statistic=float(best_rho),
        p_value=None,
        extra={
            "beta": best_beta,
            "residuals": best_resid,
            "log_likelihood": float(best_ll),
        },
    )


sgsar = sar_lag_model


def cheatsheet() -> str:
    return "sar_lag_model({}) -> Spatial autoregressive lag model (SAR)."
