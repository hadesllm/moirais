# morie.fn — function file (hadesllm/morie)
"""ARCH-in-Mean (ARCH-M) — risk premium proportional to conditional volatility."""
from __future__ import annotations

import numpy as np
from scipy import optimize

from ._richresult import RichResult

__all__ = ["arch_in_mean"]


def arch_in_mean(x):
    r"""Fit an ARCH(1)-in-mean model with Gaussian innovations.

    .. math::

        y_t = \mu + \delta\,\sigma_t + \epsilon_t,\quad
        \sigma_t^2 = \omega + \alpha\,\epsilon_{t-1}^2,
        \quad \epsilon_t \sim \mathcal{N}(0, \sigma_t^2).

    Parameters
    ----------
    x : array-like
        Return series.

    Returns
    -------
    RichResult
        keys: ``mu``, ``delta``, ``omega``, ``alpha``, ``loglik``, ``n``,
        ``conditional_variance``, ``method``.

    References
    ----------
    Engle RF, Lilien DM, Robins RP (1987). Estimating Time-Varying Risk
    Premia in the Term Structure: The ARCH-M Model. *Econometrica*
    55(2), 391-407.
    """
    y = np.asarray(x, dtype=float).ravel()
    n = y.size
    if n < 20:
        raise ValueError(f"Need at least 20 observations, got {n}.")

    def neg_ll(p):
        mu, delta, omega, alpha = p
        if omega <= 0 or alpha < 0 or alpha >= 0.999:
            return 1e10
        s2 = np.zeros(n)
        s2[0] = max(np.var(y), 1e-10)
        eps = np.zeros(n)
        eps[0] = y[0] - mu - delta * np.sqrt(s2[0])
        for t in range(1, n):
            s2[t] = omega + alpha * eps[t - 1] ** 2
            s2[t] = max(s2[t], 1e-12)
            eps[t] = y[t] - mu - delta * np.sqrt(s2[t])
        return 0.5 * np.sum(np.log(2 * np.pi * s2) + eps ** 2 / s2)

    var_y = float(np.var(y))
    fit = optimize.minimize(
        neg_ll,
        [float(np.mean(y)), 0.0, var_y * 0.5, 0.2],
        bounds=[(-10, 10), (-10, 10), (1e-8, var_y * 10), (1e-8, 0.999)],
        method="L-BFGS-B",
    )
    mu, delta, omega, alpha = fit.x
    s2 = np.zeros(n)
    s2[0] = var_y
    eps = np.zeros(n)
    eps[0] = y[0] - mu - delta * np.sqrt(s2[0])
    for t in range(1, n):
        s2[t] = omega + alpha * eps[t - 1] ** 2
        eps[t] = y[t] - mu - delta * np.sqrt(s2[t])
    return RichResult(payload={
        "mu": float(mu), "delta": float(delta),
        "omega": float(omega), "alpha": float(alpha),
        "loglik": float(-fit.fun),
        "conditional_variance": s2,
        "n": int(n),
        "method": "ARCH(1)-in-mean Gaussian MLE (numpy)",
    })


def cheatsheet():
    return "archm: ARCH-in-mean risk premium (Engle, Lilien & Robins 1987)."
