# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""ARIMA(p,d,q) fitting via conditional MLE."""

import numpy as np
from scipy import optimize

from ._containers import DescriptiveResult


def arima_fit(y: np.ndarray, order: tuple[int, int, int] = (1, 0, 0)) -> DescriptiveResult:
    """
    Fit an ARIMA(p,d,q) model via conditional maximum likelihood.

    Differences the series *d* times, then fits AR(p) + MA(q) by
    minimising the conditional sum of squared residuals.

    :param y: (n,) time series.
    :param order: (p, d, q) tuple.
    :return: DescriptiveResult with AR/MA coefficients, sigma2, AIC.
    :raises ValueError: If series too short for specified order.

    References
    ----------
    Box GEP, Jenkins GM, Reinsel GC (2015). Time Series Analysis.
    5th ed. Wiley.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    p, d, q = order
    yd = y.copy()
    for _ in range(d):
        yd = np.diff(yd)
    n = len(yd)
    if n <= p + q + 1:
        raise ValueError("Series too short for specified order.")
    start = max(p, q)

    def residuals(params):
        ar = params[:p]
        ma = params[p : p + q]
        c = params[-1]
        e = np.zeros(n)
        for t in range(start, n):
            pred = c
            for i in range(p):
                pred += ar[i] * yd[t - i - 1]
            for j in range(q):
                pred += ma[j] * e[t - j - 1]
            e[t] = yd[t] - pred
        return e[start:]

    def objective(params):
        e = residuals(params)
        return float(np.sum(e**2))

    x0 = np.zeros(p + q + 1)
    x0[-1] = float(yd.mean())
    res = optimize.minimize(objective, x0, method="L-BFGS-B")
    params = res.x
    e = residuals(params)
    sigma2 = float(np.sum(e**2) / len(e))
    k_total = p + q + 2
    aic = len(e) * np.log(sigma2) + 2 * k_total
    return DescriptiveResult(
        name="arima",
        value=sigma2,
        extra={
            "ar": params[:p].tolist(),
            "ma": params[p : p + q].tolist(),
            "intercept": float(params[-1]),
            "sigma2": sigma2,
            "aic": float(aic),
            "order": order,
            "n": n,
            "residuals": e,
        },
    )


arima = arima_fit


def cheatsheet() -> str:
    return "arima_fit({}) -> ARIMA(p,d,q) fitting via conditional MLE."
