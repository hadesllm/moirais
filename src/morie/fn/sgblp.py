"""Best linear unbiased predictor (BLUP) for spatial data."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def blup_spatial(
    Z: np.ndarray,
    coords: np.ndarray,
    X: np.ndarray,
    cov_matrix: np.ndarray,
    target: np.ndarray | None = None,
    target_X: np.ndarray | None = None,
    target_cov: np.ndarray | None = None,
) -> SpatialResult:
    r"""Spatial BLUP prediction.

    .. math::

        \hat{Z}(s_0) = \mathbf{x}_0^T \hat{\beta}_{GLS}
        + \mathbf{c}_0^T \mathbf{C}^{-1}(\mathbf{Z} - \mathbf{X}\hat{\beta}_{GLS})

    Parameters
    ----------
    Z : np.ndarray
        Observed values, shape ``(n,)``.
    coords : np.ndarray
        Observation coordinates, shape ``(n, 2)``.
    X : np.ndarray
        Design matrix, shape ``(n, p)``.
    cov_matrix : np.ndarray
        Covariance matrix, shape ``(n, n)``.
    target : np.ndarray, optional
        Prediction location, shape ``(2,)``.
    target_X : np.ndarray, optional
        Covariates at target, shape ``(p,)``.
    target_cov : np.ndarray, optional
        Covariance between target and observations, shape ``(n,)``.

    Returns
    -------
    SpatialResult
        ``statistic`` is prediction; ``extra`` has ``beta_gls``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.

    .. epigraph::

    """
    Z = np.asarray(Z, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    C = np.asarray(cov_matrix, dtype=np.float64)
    n = len(Z)

    C_inv = np.linalg.inv(C)
    XtCiX = X.T @ C_inv @ X
    beta_gls = np.linalg.solve(XtCiX, X.T @ C_inv @ Z)
    residuals = Z - X @ beta_gls

    if target_X is not None and target_cov is not None:
        x0 = np.asarray(target_X, dtype=np.float64)
        c0 = np.asarray(target_cov, dtype=np.float64)
        pred = float(x0 @ beta_gls + c0 @ C_inv @ residuals)
    else:
        pred = float(beta_gls[0]) if len(beta_gls) > 0 else 0.0

    return SpatialResult(
        name="blup_spatial",
        statistic=pred,
        p_value=None,
        extra={"beta_gls": beta_gls},
    )


sgblp = blup_spatial


def cheatsheet() -> str:
    return "blup_spatial({}) -> Best linear unbiased predictor (BLUP) for spatial data."
