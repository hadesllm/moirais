# morie.fn — function file (hadesllm/morie)
"""
MA(q) model fitting via method of moments.

Fits moving-average model of order q using the method of moments or
conditional maximum likelihood (CML).

Category: TimeSeries
"""

import numpy as np
from scipy.optimize import minimize


def mafit(y, q=1, method="cml"):
    """Fit MA(q) model.

    Parameters
    ----------
    y : array-like
        Univariate time series, shape (n,).
    q : int, optional
        Order of moving average. Default 1.
    method : str, optional
        "cml" (conditional maximum likelihood) or "mle" (exact MLE).
        Default "cml".

    Returns
    -------
    TimeSeriesResult
        Fields: ma_coeff (array), sigma2 (float), acf (array), loglik (float), n (int).

    References
    ----------
    Brockwell, P. J., & Davis, R. A. (2016). Introduction to Time Series
    and Forecasting (3rd ed.). Springer. Section 3.4.
    """
    from ._containers import TimeSeriesResult

    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")

    n = len(y)
    if n <= q:
        raise ValueError(f"Need n > q; got n={n}, q={q}")

    y = y - np.mean(y)

    # Compute sample ACF
    acov = np.array([np.mean(y[:-k] * y[k:]) if k > 0 else np.mean(y**2)
                     for k in range(q + 1)])
    rho = acov / acov[0]

    # Initial estimate from method of moments
    # For MA(q): rho(k) = -theta_k + sum_j theta_j*theta_{j+k} / (1 + sum theta_j^2)
    # Use first-lag approximation for initial guess
    theta_init = np.zeros(q)
    theta_init[0] = -rho[1] if abs(rho[1]) < 1 else 0.5 * np.sign(rho[1])

    def cml_loglik(theta, y, q):
        """Conditional log-likelihood for MA(q)."""
        sigma2 = np.mean(y**2)
        eps = np.copy(y)
        for t in range(q, len(y)):
            eps[t] = y[t] - np.sum(theta * eps[t-q:t][::-1])
        return -np.sum(eps[q:]**2) / (2 * sigma2)

    if method == "cml":
        result = minimize(lambda th: -cml_loglik(th, y, q), theta_init,
                         method="BFGS")
        theta = result.x
        loglik = -result.fun
    else:
        raise ValueError(f"Unknown method: {method}")

    # Residual variance via backsubstitution
    eps = np.copy(y)
    for t in range(q, len(y)):
        eps[t] = y[t] - np.sum(theta * eps[t-q:t][::-1])
    sigma2 = np.mean(eps[q:]**2)

    # ACF of MA(q)
    acf_ma = np.zeros(q + 1)
    acf_ma[0] = 1.0
    denominator = 1 + np.sum(theta**2)
    for k in range(1, q + 1):
        acf_ma[k] = (-theta[k-1] + np.sum(theta[:k-1] * theta[k-1::-1])) / denominator

    return TimeSeriesResult(
        name=short,
        values=theta.copy(),
        extra={
            "ma_coeff": theta.copy(),
            "sigma2": float(sigma2),
            "acf": acf_ma.copy(),
            "loglik": float(loglik),
            "n": n,
            "q": q,
        },
    )


short = "mafit"
alias = "ma_fitting"
quote = "Life moves pretty fast. -- Ferris Bueller"
__all__ = ["mafit"]


def cheatsheet() -> str:
    return "mafit(y, q=1) -> MA(q) fit via conditional likelihood"
