# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""
AR(p) model fitting via Yule-Walker equations.

Fits autoregressive model of order p to univariate time series using the
Yule-Walker estimator. Provides AR coefficients, variance, and PACF.

Category: TimeSeries
"""

import numpy as np


def arfit(y, p=1, demean=True):
    r"""Fit AR(p) model via Yule-Walker method.

    Parameters
    ----------
    y : array-like
        Univariate time series, shape (n,).
    p : int, optional
        Order of autoregression. Default 1.
    demean : bool, optional
        If True, subtract mean before fitting. Default True.

    Returns
    -------
    TimeSeriesResult
        Fields: ar_coeff (array), sigma2 (float), acf (array), pacf (array),
        aic (float), bic (float), n (int).

    Notes
    -----
    Yule-Walker equations:

    .. math::
        \\rho(k) = \\sum_{j=1}^{p} \\phi_j \\rho(k-j), \\quad k=1,\\ldots,p

    where \\phi are AR coefficients and \\rho is ACF.

    References
    ----------
    Brockwell, P. J., & Davis, R. A. (2016). Introduction to Time Series
    and Forecasting (3rd ed.). Springer. Section 3.3.
    """
    from ._containers import TimeSeriesResult

    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")

    n = len(y)
    if n <= p:
        raise ValueError(f"Need n > p; got n={n}, p={p}")

    if demean:
        y = y - np.mean(y)

    # Compute sample autocovariance
    acov = np.array([np.mean(y[:-k] * y[k:]) if k > 0 else np.mean(y**2)
                     for k in range(p + 1)])

    # Yule-Walker equations: R * phi = r
    R = np.array([[acov[abs(i-j)] for j in range(p)] for i in range(p)])
    r = acov[1:p+1]

    # Solve for AR coefficients
    try:
        phi = np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        # Fallback to least-squares if singular
        phi = np.linalg.lstsq(R, r, rcond=None)[0]

    # Residual variance
    sigma2 = acov[0] - np.dot(phi, r)
    sigma2 = float(np.maximum(sigma2, 1e-10))  # Ensure positive

    # AIC and BIC
    aic = n * np.log(sigma2) + 2 * p
    bic = n * np.log(sigma2) + p * np.log(n)

    # Compute ACF and PACF for diagnostics
    acf_vals = acov / acov[0]

    # PACF via Yule-Walker recursion (Durbin-Levinson)
    pacf_vals = np.zeros(p + 1)
    pacf_vals[0] = 1.0
    pacf_vals[1] = phi[0]

    if p > 1:
        phi_temp = np.copy(phi)
        for k in range(2, p + 1):
            numerator = acf_vals[k] - np.sum(phi_temp[:k-1] * acf_vals[k-1:0:-1])
            denominator = 1.0 - np.sum(phi_temp[:k-1]**2)
            if abs(denominator) > 1e-10:
                pacf_k = numerator / denominator
                pacf_vals[k] = pacf_k
                phi_old = phi_temp[:k-1].copy()
                phi_temp[k-1] = pacf_k
                phi_temp[:k-1] -= pacf_k * phi_old[::-1]

    return TimeSeriesResult(
        name=short,
        values=phi.copy(),
        extra={
            "ar_coeff": phi.copy(),
            "sigma2": sigma2,
            "acf": acf_vals[:p+1].copy(),
            "pacf": pacf_vals.copy(),
            "aic": float(aic),
            "bic": float(bic),
            "n": n,
            "p": p,
        },
    )


short = "arfit"
alias = "ar_yulewalker"
quote = "If I have seen further it is by standing on the shoulders of giants. -- Isaac Newton"
__all__ = ["arfit"]


def cheatsheet() -> str:
    return "arfit(y, p=1) -> AR(p) Yule-Walker fit with σ², AIC, BIC"
