"""Generalized least squares for spatial data."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def gls_spatial(
    Z: np.ndarray,
    X: np.ndarray,
    cov_matrix: np.ndarray,
) -> SpatialResult:
    r"""GLS estimation with known spatial covariance.

    .. math::

        \hat{\beta}_{GLS} = (\mathbf{X}^T \mathbf{V}^{-1} \mathbf{X})^{-1}
        \mathbf{X}^T \mathbf{V}^{-1} \mathbf{Z}

    Parameters
    ----------
    Z : np.ndarray
        Response vector, shape ``(n,)``.
    X : np.ndarray
        Design matrix, shape ``(n, p)``.
    cov_matrix : np.ndarray
        Covariance matrix :math:`\mathbf{V}`, shape ``(n, n)``.

    Returns
    -------
    SpatialResult
        ``statistic`` is the first coefficient.
        ``extra`` has ``beta``, ``beta_var``, ``residuals``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 6.

    .. epigraph::

        "Do not be sorry. Be better." -- Kratos, God of War
    """
    Z = np.asarray(Z, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    V = np.asarray(cov_matrix, dtype=np.float64)

    V_inv = np.linalg.inv(V)
    XtVi = X.T @ V_inv
    beta = np.linalg.solve(XtVi @ X, XtVi @ Z)
    beta_var = np.linalg.inv(XtVi @ X)
    resid = Z - X @ beta

    return SpatialResult(
        name="gls_spatial",
        statistic=float(beta[0]),
        p_value=None,
        extra={
            "beta": beta,
            "beta_var": beta_var,
            "residuals": resid,
        },
    )


sggls = gls_spatial


def cheatsheet() -> str:
    return "gls_spatial({}) -> Generalized least squares for spatial data."
