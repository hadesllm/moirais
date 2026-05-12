# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""ARMA(p,q) model fitting via conditional MLE."""

import numpy as np
from scipy import optimize

from ._containers import DescriptiveResult


def arma_fit(y: np.ndarray, p: int = 1, q: int = 1) -> DescriptiveResult:
    r"""
    Fit an ARMA(p,q) model via conditional maximum likelihood.

    .. math::

        y_t = c + \\sum_{i=1}^{p} \\phi_i y_{t-i}
        + \\sum_{j=1}^{q} \\theta_j \\varepsilon_{t-j} + \\varepsilon_t

    :param y: 1-D time series.
    :param p: AR order. Default 1.
    :param q: MA order. Default 1.
    :return: DescriptiveResult with phi, theta, sigma2.
    :raises ValueError: If series too short.

    References
    ----------
    Box G.E.P. & Jenkins G.M. (1970). Time Series Analysis:
    Forecasting and Control. Holden-Day.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    m = max(p, q)
    if n < m + 10:
        raise ValueError(f"Need at least {m + 10} observations, got {n}.")
    mu = float(y.mean())
    yc = y - mu

    def neg_loglik(params):
        phi = params[:p]
        theta = params[p : p + q]
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
    x0[:p] = 0.1
    res = optimize.minimize(neg_loglik, x0, method="Nelder-Mead", options={"maxiter": 5000})
    phi = res.x[:p]
    theta = res.x[p : p + q]
    eps = np.zeros(n)
    for t in range(m, n):
        pred = 0.0
        for i in range(p):
            pred += phi[i] * yc[t - 1 - i]
        for j in range(q):
            pred += theta[j] * eps[t - 1 - j]
        eps[t] = yc[t] - pred
    sigma2 = float(np.sum(eps[m:] ** 2) / (n - m))
    return DescriptiveResult(
        name="arma_fit",
        value=sigma2,
        extra={
            "phi": phi.tolist(),
            "theta": theta.tolist(),
            "mean": mu,
            "sigma2": sigma2,
            "residuals": eps,
            "p": p,
            "q": q,
            "n": n,
            "loglik": float(-res.fun),
        },
    )


armam = arma_fit


def cheatsheet() -> str:
    return "arma_fit({}) -> ARMA(p,q) model via conditional MLE."
