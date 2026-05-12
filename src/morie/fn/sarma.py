# morie.fn -- function file (hadesllm/morie)
"""Seasonal ARMA model."""

import numpy as np
from scipy import optimize

from ._containers import DescriptiveResult


def seasonal_arma(
    y: np.ndarray, order: tuple[int, int] = (1, 0), seasonal_order: tuple[int, int, int] = (1, 0, 12)
) -> DescriptiveResult:
    """
    Fit a seasonal ARMA model via conditional least squares.

    Combines non-seasonal AR(p)/MA(q) with seasonal AR(P)/MA(Q)
    at the specified period.

    :param y: (n,) time series.
    :param order: (p, q) non-seasonal order.
    :param seasonal_order: (P, Q, s) seasonal order and period.
    :return: DescriptiveResult with coefficients and residuals.

    References
    ----------
    Hyndman RJ, Athanasopoulos G (2021). Forecasting: Principles and
    Practice. 3rd ed. OTexts.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    p, q = order
    P, Q, s = seasonal_order
    n = len(y)
    start = max(p, q, P * s, Q * s)
    if n <= start + 1:
        raise ValueError("Series too short for specified order.")

    def objective(params):
        ar = params[:p]
        ma = params[p : p + q]
        sar = params[p + q : p + q + P]
        sma = params[p + q + P : p + q + P + Q]
        c = params[-1]
        e = np.zeros(n)
        for t in range(start, n):
            pred = c
            for i in range(p):
                pred += ar[i] * y[t - i - 1]
            for j in range(q):
                pred += ma[j] * e[t - j - 1]
            for i in range(P):
                pred += sar[i] * y[t - (i + 1) * s]
            for j in range(Q):
                pred += sma[j] * e[t - (j + 1) * s]
            e[t] = y[t] - pred
        return float(np.sum(e[start:] ** 2))

    n_params = p + q + P + Q + 1
    x0 = np.zeros(n_params)
    x0[-1] = float(y.mean())
    res = optimize.minimize(objective, x0, method="L-BFGS-B")
    params = res.x
    sigma2 = res.fun / (n - start)
    return DescriptiveResult(
        name="seasonal_arma",
        value=float(sigma2),
        extra={
            "ar": params[:p].tolist(),
            "ma": params[p : p + q].tolist(),
            "sar": params[p + q : p + q + P].tolist(),
            "sma": params[p + q + P : p + q + P + Q].tolist(),
            "intercept": float(params[-1]),
            "sigma2": float(sigma2),
            "order": order,
            "seasonal_order": seasonal_order,
            "n": n,
        },
    )


sarma = seasonal_arma


def cheatsheet() -> str:
    return "seasonal_arma({}) -> Seasonal ARMA model."
