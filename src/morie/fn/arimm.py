# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""ARIMA(p,d,q) model -- differencing + ARMA."""

import numpy as np
from scipy import optimize

from ._containers import DescriptiveResult


def arima_fit(y: np.ndarray, p: int = 1, d: int = 1, q: int = 0) -> DescriptiveResult:
    r"""
    Fit an ARIMA(p,d,q) model via differencing then conditional MLE.

    .. math::

        \\Delta^d y_t = c + \\sum_{i=1}^{p} \\phi_i \\Delta^d y_{t-i}
        + \\sum_{j=1}^{q} \\theta_j \\varepsilon_{t-j} + \\varepsilon_t

    :param y: 1-D time series.
    :param p: AR order. Default 1.
    :param d: Differencing order. Default 1.
    :param q: MA order. Default 0.
    :return: DescriptiveResult with phi, theta, sigma2, AIC, BIC.
    :raises ValueError: If series too short.

    References
    ----------
    Box G.E.P. & Jenkins G.M. (1970). Time Series Analysis:
    Forecasting and Control. Holden-Day.
    """
    y = np.asarray(y, dtype=float).ravel()
    yd = y.copy()
    for _ in range(d):
        yd = np.diff(yd)
    n = len(yd)
    m = max(p, q, 1)
    if n < m + 10:
        raise ValueError(f"Need at least {m + 10} post-differencing obs, got {n}.")
    mu = float(yd.mean())
    yc = yd - mu

    def neg_ll(params):
        phi = params[:p]
        theta = params[p : p + q] if q > 0 else np.array([])
        eps = np.zeros(n)
        for t in range(m, n):
            pred = 0.0
            for i in range(p):
                pred += phi[i] * yc[t - 1 - i]
            for j in range(q):
                pred += theta[j] * eps[t - 1 - j]
            eps[t] = yc[t] - pred
        ss = np.sum(eps[m:] ** 2)
        T = n - m
        sig2 = ss / T
        if sig2 <= 0:
            return 1e10
        return 0.5 * T * np.log(2 * np.pi * sig2) + ss / (2 * sig2)

    x0 = np.zeros(p + q)
    res = optimize.minimize(neg_ll, x0, method="Nelder-Mead", options={"maxiter": 5000})
    phi = res.x[:p].tolist()
    theta = res.x[p : p + q].tolist() if q > 0 else []
    k = p + q + 1
    T = n - m
    aic = 2 * res.fun + 2 * k
    bic = 2 * res.fun + k * np.log(T)
    eps = np.zeros(n)
    for t in range(m, n):
        pred = 0.0
        for i in range(p):
            pred += res.x[i] * yc[t - 1 - i]
        for j in range(q):
            pred += res.x[p + j] * eps[t - 1 - j]
        eps[t] = yc[t] - pred
    sigma2 = float(np.sum(eps[m:] ** 2) / T)
    return DescriptiveResult(
        name="arima_fit",
        value=sigma2,
        extra={
            "phi": phi,
            "theta": theta,
            "d": d,
            "p": p,
            "q": q,
            "mean_diff": mu,
            "sigma2": sigma2,
            "aic": float(aic),
            "bic": float(bic),
            "loglik": float(-res.fun),
            "n_original": len(y),
            "n_diff": n,
        },
    )


arimm = arima_fit


def cheatsheet() -> str:
    return "arima_fit({}) -> ARIMA(p,d,q) model via differencing + conditional MLE."
