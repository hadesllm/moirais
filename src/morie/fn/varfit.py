"""Vector autoregression via OLS."""

import numpy as np

from ._containers import DescriptiveResult


def var_fit(Y, p=1):
    """
    Fit a VAR(p) model via equation-by-equation OLS.

    :param Y: (T, k) multivariate time series.
    :param p: Lag order.
    :return: DescriptiveResult with coefficient matrix, residuals, AIC.
    :raises ValueError: If T <= k*p.

    References
    ----------
    Lütkepohl H (2005). New Introduction to Multiple Time Series Analysis.
    Springer.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    T, k = Y.shape
    if k * p + 1 >= T:
        raise ValueError(f"Need T > k*p+1, got T={T}, k*p+1={k * p + 1}")

    Z = np.column_stack([Y[p - i - 1 : T - i - 1] for i in range(p)])
    Z = np.column_stack([np.ones(T - p), Z])
    y = Y[p:]
    beta = np.linalg.lstsq(Z, y, rcond=None)[0]
    resid = y - Z @ beta
    n = T - p
    sigma = resid.T @ resid / n
    log_det = np.linalg.slogdet(sigma)[1]
    aic = log_det + 2 * p * k**2 / n

    return DescriptiveResult(
        name="var_fit",
        value=float(aic),
        extra={
            "coefficients": beta,
            "residuals": resid,
            "sigma": sigma,
            "aic": float(aic),
            "p": p,
            "k": k,
        },
    )


def cheatsheet() -> str:
    return "var_fit({}) -> Vector autoregression via OLS."
