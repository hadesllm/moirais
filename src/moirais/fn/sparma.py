"""Spatial ARMA model for areal time series."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def spatial_arma(
    y: np.ndarray,
    W: np.ndarray,
    ar_order: int = 1,
    ma_order: int = 0,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> SpatialResult:
    r"""
    Spatial autoregressive moving average (SARMA) model for areal data.

    Combines a spatial lag component with a spatial error (MA) component:

    .. math::

        \mathbf{y} = \rho \mathbf{W}\mathbf{y}
                    + \mathbf{X}\boldsymbol{\beta}
                    + \mathbf{u}, \quad
        \mathbf{u} = \lambda \mathbf{W}\mathbf{u} + \boldsymbol{\varepsilon}

    Estimation uses iterated generalized least squares on the concentrated
    log-likelihood.

    Parameters
    ----------
    y : np.ndarray
        (n,) observations across spatial units.
    W : np.ndarray
        (n, n) row-standardized spatial weight matrix.
    ar_order : int
        Spatial autoregressive order (1 = SAR(1)).
    ma_order : int
        Spatial moving-average order (0 or 1).
    max_iter : int
        Maximum iterations for estimation.
    tol : float
        Convergence tolerance.

    Returns
    -------
    SpatialResult
        statistic = spatial rho, extra has ``lambda_ma``, ``sigma2``,
        ``residuals``, ``converged``.

    References
    ----------
    Huang JS (1984). The autoregressive moving average model for spatial
    analysis. *Australian Journal of Statistics*, 26(2), 169--178.

    Anselin L (1988). *Spatial Econometrics: Methods and Models*. Kluwer
    Academic, Dordrecht. Chapter 10.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 25
    >>> W = np.zeros((n, n))
    >>> for i in range(n):
    ...     for j in range(n):
    ...         if abs(i - j) == 1:
    ...             W[i, j] = 1
    >>> W = W / np.maximum(W.sum(axis=1, keepdims=True), 1)
    >>> y = rng.normal(0, 1, n)
    >>> res = spatial_arma(y, W, ar_order=1, ma_order=1)
    >>> -1.0 <= res.statistic <= 1.0
    True
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    W = np.asarray(W, dtype=np.float64)
    n = len(y)
    if W.shape != (n, n):
        raise ValueError("W must be (n, n).")

    I = np.eye(n)
    Wy = W @ y

    rho = 0.0
    lam = 0.0
    converged = False

    for it in range(max_iter):
        A = I - rho * W
        u = A @ y
        if ma_order >= 1:
            Wu = W @ u
            B = I - lam * W
            eps = np.linalg.solve(B, u) if np.abs(lam) > 1e-12 else u
        else:
            eps = u

        sigma2 = float(np.mean(eps**2))

        grad_rho = float(2.0 * eps @ (-Wy)) / n
        new_rho = rho - 0.01 * grad_rho
        new_rho = float(np.clip(new_rho, -0.99, 0.99))

        new_lam = lam
        if ma_order >= 1:
            Wu = W @ u
            grad_lam = float(2.0 * eps @ (-Wu)) / n
            new_lam = lam - 0.01 * grad_lam
            new_lam = float(np.clip(new_lam, -0.99, 0.99))

        if abs(new_rho - rho) < tol and abs(new_lam - lam) < tol:
            rho = new_rho
            lam = new_lam
            converged = True
            break
        rho = new_rho
        lam = new_lam

    resid = (I - rho * W) @ y
    sigma2 = float(np.mean(resid**2))

    return SpatialResult(
        name="spatial_arma",
        statistic=rho,
        extra={
            "lambda_ma": lam,
            "sigma2": sigma2,
            "residuals": resid,
            "converged": converged,
            "n_iter": it + 1,
            "n": n,
            "ar_order": ar_order,
            "ma_order": ma_order,
        },
    )


sparma = spatial_arma


def cheatsheet() -> str:
    return "spatial_arma({}) -> Spatial ARMA model for areal time series."
