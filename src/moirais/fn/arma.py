# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""ARIMA model fitting. 'The happiness of your life depends upon the quality of your thoughts. — Marcus Aurelius' -- Maz Kanata"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from ._containers import TimeSeriesResult


def arima_fit(x, *, p: int = 1, d: int = 0, q: int = 0) -> TimeSeriesResult:
    """Fit ARIMA(p,d,q) model via conditional MLE.

    Parameters
    ----------
    x : array-like
        Time series observations.
    p : int
        AR order. Default 1.
    d : int
        Differencing order. Default 0.
    q : int
        MA order. Default 0.

    Returns
    -------
    TimeSeriesResult
    """
    x = np.asarray(x, dtype=float)
    # Differencing
    for _ in range(d):
        x = np.diff(x)
    n = len(x)
    if n < p + q + 2:
        raise ValueError("Not enough observations")
    mean_x = x.mean()
    x_c = x - mean_x

    def neg_log_lik(params):
        ar = params[:p]
        ma = params[p : p + q]
        sigma2 = params[-1] ** 2 + 1e-10
        resid = np.zeros(n)
        for t in range(max(p, q), n):
            pred = sum(ar[i] * x_c[t - i - 1] for i in range(p))
            pred += sum(ma[j] * resid[t - j - 1] for j in range(min(q, t)))
            resid[t] = x_c[t] - pred
        return 0.5 * n * np.log(2 * np.pi * sigma2) + 0.5 * np.sum(resid**2) / sigma2

    init = np.zeros(p + q + 1)
    init[-1] = np.std(x_c)
    res = optimize.minimize(neg_log_lik, init, method="L-BFGS-B")
    params = res.x
    ar_coefs = params[:p].tolist()
    ma_coefs = params[p : p + q].tolist()
    sigma = abs(params[-1])
    return TimeSeriesResult(
        name=f"ARIMA({p},{d},{q})",
        values=np.array(ar_coefs + ma_coefs),
        extra={
            "ar": ar_coefs,
            "ma": ma_coefs,
            "sigma": float(sigma),
            "aic": float(2 * res.fun + 2 * (p + q + 1)),
            "n": n,
            "converged": res.success,
        },
    )


arma = arima_fit


def cheatsheet() -> str:
    return "arima_fit({}) -> ARIMA model fitting. 'The happiness of your life depends upon the quality of your thoughts. — Marcus Aurelius' -- Maz Kana"
