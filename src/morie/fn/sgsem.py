"""Spatial error model (SEM)."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def sem_error_model(
    Z: np.ndarray,
    X: np.ndarray,
    W: np.ndarray,
) -> SpatialResult:
    r"""Fit a spatial error model via concentrated likelihood.

    .. math::

        Z = X\beta + u, \quad u = \lambda W u + \varepsilon

    Parameters
    ----------
    Z : np.ndarray
        Response, shape ``(n,)``.
    X : np.ndarray
        Design matrix, shape ``(n, p)``.
    W : np.ndarray
        Spatial weights matrix, shape ``(n, n)``.

    Returns
    -------
    SpatialResult
        ``statistic`` is estimated :math:`\lambda`.
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
    I = np.eye(n)

    lam_grid = np.linspace(-0.9, 0.9, 50)
    best_ll = -np.inf
    best_lam = 0.0
    best_beta = None
    best_resid = None

    for lam in lam_grid:
        B = I - lam * W
        Zs = B @ Z
        Xs = B @ X
        beta = np.linalg.lstsq(Xs, Zs, rcond=None)[0]
        resid = Zs - Xs @ beta
        sigma2 = np.sum(resid**2) / n
        if sigma2 <= 0:
            continue
        sign, logdet = np.linalg.slogdet(B)
        if sign <= 0:
            continue
        ll = -0.5 * n * np.log(2 * np.pi * sigma2) + logdet - 0.5 * n
        if ll > best_ll:
            best_ll = ll
            best_lam = lam
            best_beta = beta
            best_resid = resid

    if best_beta is None:
        best_beta = np.linalg.lstsq(X, Z, rcond=None)[0]
        best_resid = Z - X @ best_beta

    return SpatialResult(
        name="sem_error_model",
        statistic=float(best_lam),
        p_value=None,
        extra={
            "beta": best_beta,
            "residuals": best_resid,
            "log_likelihood": float(best_ll),
        },
    )


sgsem = sem_error_model


def cheatsheet() -> str:
    return "sem_error_model({}) -> Spatial error model (SEM)."
