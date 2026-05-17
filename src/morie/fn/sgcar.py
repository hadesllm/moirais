"""Conditional autoregressive (CAR) model."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def conditional_autoregressive(
    Z: np.ndarray,
    W: np.ndarray,
    X: np.ndarray | None = None,
) -> SpatialResult:
    r"""Fit a CAR model via iterative GLS.

    .. math::

        Z = X\beta + \phi, \quad
        \phi | \phi_{-i} \sim N\bigl(\rho \sum_j w_{ij}\phi_j,\;\tau^2\bigr)

    Parameters
    ----------
    Z : np.ndarray
        Response, shape ``(n,)``.
    W : np.ndarray
        Adjacency weights, shape ``(n, n)``.
    X : np.ndarray, optional
        Covariates. Defaults to intercept.

    Returns
    -------
    SpatialResult
        ``statistic`` is estimated :math:`\rho`.
        ``extra`` has ``beta``, ``tau2``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 6.

    .. epigraph::

    """
    Z = np.asarray(Z, dtype=np.float64).ravel()
    W = np.asarray(W, dtype=np.float64)
    n = len(Z)
    if X is None:
        X = np.ones((n, 1))
    else:
        X = np.asarray(X, dtype=np.float64)

    D = np.diag(W.sum(axis=1))
    I = np.eye(n)

    best_ll = -np.inf
    best_rho = 0.0
    best_beta = None
    best_tau2 = 1.0

    for rho in np.linspace(0.01, 0.99, 30):
        Q = D - rho * W
        try:
            Q_inv = np.linalg.inv(Q)
        except np.linalg.LinAlgError:
            continue
        XtQX = X.T @ Q @ X
        beta = np.linalg.solve(XtQX, X.T @ Q @ Z)
        resid = Z - X @ beta
        tau2 = float(resid @ Q @ resid / n)
        if tau2 <= 0:
            continue
        sign, logdet = np.linalg.slogdet(Q)
        if sign <= 0:
            continue
        ll = 0.5 * logdet - 0.5 * n * np.log(tau2) - 0.5 * n
        if ll > best_ll:
            best_ll = ll
            best_rho = rho
            best_beta = beta
            best_tau2 = tau2

    if best_beta is None:
        best_beta = np.linalg.lstsq(X, Z, rcond=None)[0]

    return SpatialResult(
        name="conditional_autoregressive",
        statistic=float(best_rho),
        p_value=None,
        extra={"beta": best_beta, "tau2": best_tau2},
    )


sgcar = conditional_autoregressive


def cheatsheet() -> str:
    return "conditional_autoregressive({}) -> Conditional autoregressive (CAR) model."
