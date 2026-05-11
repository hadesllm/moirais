"""Spatial Durbin model estimation."""

import numpy as np
from scipy import optimize

from ._containers import DescriptiveResult


def spatial_durbin(y: np.ndarray, X: np.ndarray, W: np.ndarray) -> DescriptiveResult:
    """
    Estimate a Spatial Durbin Model (SDM).

    .. math::

        y = \\rho W y + X \\beta + W X \\theta + \\varepsilon

    Includes both spatially lagged dependent and independent variables.

    :param y: (n,) dependent variable.
    :param X: (n, k) explanatory variables.
    :param W: (n, n) row-standardised spatial weights matrix.
    :return: DescriptiveResult with rho and coefficients.

    References
    ----------
    LeSage J, Pace RK (2009). Introduction to Spatial Econometrics.
    CRC Press.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    n = len(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    WX = W @ X
    X_full = np.column_stack([X, WX])

    def neg_loglik(rho):
        y_star = y - rho * W @ y
        beta = np.linalg.lstsq(X_full, y_star, rcond=None)[0]
        e = y_star - X_full @ beta
        sigma2 = np.dot(e, e) / n
        ll = -n / 2 * np.log(2 * np.pi * sigma2) - n / 2
        sign, logdet = np.linalg.slogdet(np.eye(n) - rho * W)
        if sign > 0:
            ll += logdet
        return -ll

    res = optimize.minimize_scalar(neg_loglik, bounds=(-0.99, 0.99), method="bounded")
    rho_hat = float(res.x)
    y_star = y - rho_hat * W @ y
    beta = np.linalg.lstsq(X_full, y_star, rcond=None)[0]
    e = y_star - X_full @ beta
    sigma2 = float(np.dot(e, e) / n)
    k = X.shape[1]
    return DescriptiveResult(
        name="spatial_durbin",
        value=rho_hat,
        extra={
            "beta": beta[:k],
            "theta": beta[k:],
            "rho": rho_hat,
            "sigma2": sigma2,
            "loglik": float(-res.fun),
            "n": n,
        },
    )


spdur = spatial_durbin


def cheatsheet() -> str:
    return "spatial_durbin({}) -> Spatial Durbin model estimation."
