# morie.fn -- function file (hadesllm/morie)
"""Seasonal ARIMA (SARIMA) model."""

import numpy as np
from scipy import optimize

from ._containers import DescriptiveResult


def sarima_fit(
    y: np.ndarray,
    p: int = 1,
    d: int = 1,
    q: int = 0,
    P: int = 1,
    D: int = 1,
    Q: int = 0,
    m: int = 12,
) -> DescriptiveResult:
    """
    Fit a SARIMA(p,d,q)(P,D,Q)[m] model.

    Applies regular differencing *d* times, seasonal differencing *D*
    times at lag *m*, then fits an ARMA with both regular and seasonal
    AR/MA terms via conditional MLE.

    :param y: 1-D time series.
    :param p: Non-seasonal AR order. Default 1.
    :param d: Non-seasonal differencing. Default 1.
    :param q: Non-seasonal MA order. Default 0.
    :param P: Seasonal AR order. Default 1.
    :param D: Seasonal differencing. Default 1.
    :param Q: Seasonal MA order. Default 0.
    :param m: Seasonal period. Default 12.
    :return: DescriptiveResult with coefficients and diagnostics.
    :raises ValueError: If series too short.

    References
    ----------
    Box G.E.P., Jenkins G.M. & Reinsel G.C. (2015). Time Series
    Analysis, 5th ed. Wiley.
    """
    y = np.asarray(y, dtype=float).ravel()
    yd = y.copy()
    for _ in range(d):
        yd = np.diff(yd)
    for _ in range(D):
        yd = yd[m:] - yd[:-m]
    n = len(yd)
    start = max(p, q, P * m, Q * m, 1)
    if n < start + 10:
        raise ValueError(f"Need more observations after differencing, got {n}.")
    mu = float(yd.mean())
    yc = yd - mu
    n_params = p + q + P + Q

    def neg_ll(params):
        phi = params[:p]
        theta = params[p : p + q] if q > 0 else np.array([])
        Phi_s = params[p + q : p + q + P]
        Theta_s = params[p + q + P : p + q + P + Q] if Q > 0 else np.array([])
        eps = np.zeros(n)
        for t in range(start, n):
            pred = 0.0
            for i in range(p):
                pred += phi[i] * yc[t - 1 - i]
            for i in range(P):
                idx = t - m * (i + 1)
                if idx >= 0:
                    pred += Phi_s[i] * yc[idx]
            for j in range(q):
                pred += theta[j] * eps[t - 1 - j]
            for j in range(Q):
                idx = t - m * (j + 1)
                if idx >= 0:
                    pred += Theta_s[j] * eps[idx]
            eps[t] = yc[t] - pred
        ss = np.sum(eps[start:] ** 2)
        T = n - start
        sig2 = ss / T
        if sig2 <= 0:
            return 1e10
        return 0.5 * T * np.log(2 * np.pi * sig2) + ss / (2 * sig2)

    x0 = np.zeros(n_params)
    res = optimize.minimize(neg_ll, x0, method="Nelder-Mead", options={"maxiter": 10000})
    T = n - start
    k = n_params + 1
    aic = 2 * res.fun + 2 * k
    bic = 2 * res.fun + k * np.log(T) if T > 0 else float("nan")
    return DescriptiveResult(
        name="sarima_fit",
        value=float(res.fun),
        extra={
            "phi": res.x[:p].tolist(),
            "theta": res.x[p : p + q].tolist() if q > 0 else [],
            "Phi": res.x[p + q : p + q + P].tolist(),
            "Theta": res.x[p + q + P :].tolist() if Q > 0 else [],
            "order": (p, d, q),
            "seasonal_order": (P, D, Q, m),
            "aic": float(aic),
            "bic": float(bic),
            "n_original": len(y),
            "n_diff": n,
        },
    )


sarim = sarima_fit


def cheatsheet() -> str:
    return "sarima_fit({}) -> Seasonal ARIMA (SARIMA) model."
