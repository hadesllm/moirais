"""Spatial lag model (SLM) estimation."""

import numpy as np
from scipy import optimize

from ._containers import DescriptiveResult


def spatial_lag(y: np.ndarray, X: np.ndarray, W: np.ndarray) -> DescriptiveResult:
    """
    Estimate a spatial lag model (SLM) via concentrated MLE.

    .. math::

        y = \\rho W y + X \\beta + \\varepsilon

    :param y: (n,) dependent variable.
    :param X: (n, k) explanatory variables.
    :param W: (n, n) row-standardised spatial weights matrix.
    :return: DescriptiveResult with coefficients, rho, loglik in extra.
    :raises ValueError: If dimensions mismatch.

    References
    ----------
    Anselin L (1988). Spatial Econometrics: Methods and Models.
    Kluwer Academic Publishers.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    n = len(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if W.shape != (n, n) or X.shape[0] != n:
        raise ValueError("Dimension mismatch.")
    eigenvalues_W = np.real(np.linalg.eigvals(W))
    rho_min, rho_max = 1.0 / min(eigenvalues_W.min(), -0.01), 1.0 / max(eigenvalues_W.max(), 0.01)
    rho_min, rho_max = max(rho_min, -0.99), min(rho_max, 0.99)

    def neg_loglik(rho):
        y_star = y - rho * W @ y
        beta = np.linalg.lstsq(X, y_star, rcond=None)[0]
        e = y_star - X @ beta
        sigma2 = np.dot(e, e) / n
        ll = -n / 2 * np.log(2 * np.pi * sigma2) - np.sum(e**2) / (2 * sigma2)
        sign, logdet = np.linalg.slogdet(np.eye(n) - rho * W)
        if sign > 0:
            ll += logdet
        return -ll

    res = optimize.minimize_scalar(neg_loglik, bounds=(rho_min, rho_max), method="bounded")
    rho_hat = float(res.x)
    y_star = y - rho_hat * W @ y
    beta = np.linalg.lstsq(X, y_star, rcond=None)[0]
    e = y_star - X @ beta
    sigma2 = float(np.dot(e, e) / n)
    return DescriptiveResult(
        name="spatial_lag",
        value=rho_hat,
        extra={
            "coefficients": beta,
            "rho": rho_hat,
            "sigma2": sigma2,
            "loglik": float(-res.fun),
            "n": n,
            "k": X.shape[1],
        },
    )


splag = spatial_lag


def cheatsheet() -> str:
    return "spatial_lag({}) -> Spatial lag model (SLM) estimation."
