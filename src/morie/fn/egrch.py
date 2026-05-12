# morie.fn — function file (hadesllm/morie)
"""EGARCH(1,1) asymmetric volatility model.

Wraps `arch.arch_model(..., vol='EGARCH')` when available; falls back to
a pure-numpy MLE on Gaussian innovations so the function is usable
without the optional `arch` dependency.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize

from ._richresult import RichResult

__all__ = ["egarch_model"]


def egarch_model(x):
    r"""Fit an EGARCH(1,1) model to financial returns.

    .. math::

        \log(\sigma_t^2) = \omega + \alpha\,g(z_{t-1}) + \beta \log(\sigma_{t-1}^2),
        \quad g(z) = \theta z + \gamma(|z|-E|z|)

    Parameters
    ----------
    x : array-like
        Return series (recommended demeaned).

    Returns
    -------
    RichResult
        keys: ``omega``, ``alpha``, ``beta``, ``gamma``, ``theta``,
        ``loglik``, ``n``, ``conditional_variance``, ``method``.

    References
    ----------
    Nelson DB (1991). Conditional Heteroskedasticity in Asset Returns: A
    New Approach. *Econometrica* 59(2), 347-370.
    """
    r = np.asarray(x, dtype=float).ravel()
    r = r - r.mean()
    n = r.size
    if n < 20:
        raise ValueError(f"Need at least 20 observations, got {n}.")

    # Try the canonical `arch` package first; fall back to pure NumPy MLE.
    try:
        from arch import arch_model
        m = arch_model(r, mean="Zero", vol="EGARCH", p=1, o=1, q=1, dist="normal")
        fit = m.fit(disp="off", show_warning=False)
        params = fit.params
        return RichResult(payload={
            "omega": float(params.get("omega", np.nan)),
            "alpha": float(params.get("alpha[1]", np.nan)),
            "gamma": float(params.get("gamma[1]", np.nan)),
            "beta":  float(params.get("beta[1]",  np.nan)),
            "theta": float(params.get("gamma[1]", np.nan)),  # alias
            "loglik": float(fit.loglikelihood),
            "n": int(n),
            "conditional_variance": np.asarray(fit.conditional_volatility) ** 2,
            "method": "EGARCH(1,1) via arch.arch_model",
        })
    except Exception:
        pass

    # Pure-NumPy MLE fallback ------------------------------------------------
    EZ = np.sqrt(2.0 / np.pi)  # E|Z| for standard normal

    def neg_ll(p):
        omega, alpha, gamma, beta = p
        if abs(beta) >= 1.0:
            return 1e10
        log_s2 = np.zeros(n)
        log_s2[0] = np.log(np.var(r) + 1e-12)
        for t in range(1, n):
            z = r[t - 1] / np.sqrt(np.exp(log_s2[t - 1]) + 1e-12)
            log_s2[t] = omega + beta * log_s2[t - 1] + alpha * (np.abs(z) - EZ) + gamma * z
        s2 = np.exp(log_s2)
        ll = -0.5 * np.sum(np.log(2 * np.pi * s2) + r ** 2 / s2)
        return -ll

    x0 = [0.0, 0.1, 0.0, 0.9]
    bnds = [(-5, 5), (-1.0, 1.0), (-1.0, 1.0), (-0.999, 0.999)]
    fit = optimize.minimize(neg_ll, x0, bounds=bnds, method="L-BFGS-B")
    omega, alpha, gamma, beta = fit.x
    log_s2 = np.zeros(n)
    log_s2[0] = np.log(np.var(r) + 1e-12)
    for t in range(1, n):
        z = r[t - 1] / np.sqrt(np.exp(log_s2[t - 1]) + 1e-12)
        log_s2[t] = omega + beta * log_s2[t - 1] + alpha * (np.abs(z) - EZ) + gamma * z
    return RichResult(payload={
        "omega": float(omega),
        "alpha": float(alpha),
        "gamma": float(gamma),
        "beta":  float(beta),
        "theta": float(gamma),
        "loglik": float(-fit.fun),
        "n": int(n),
        "conditional_variance": np.exp(log_s2),
        "method": "EGARCH(1,1) Gaussian MLE (numpy)",
    })


# CANONICAL TEST -------------------------------------------------------------
# rng = np.random.default_rng(0); n=200
# eps = rng.standard_normal(n); s2 = np.zeros(n); r = np.zeros(n)
# omega, alpha, gamma, beta = -0.1, 0.1, -0.05, 0.95
# s2[0] = 1.0
# for t in range(1, n):
#     z = r[t-1]/np.sqrt(s2[t-1]); g = alpha*(abs(z)-np.sqrt(2/np.pi)) + gamma*z
#     s2[t] = np.exp(omega + beta*np.log(s2[t-1]) + g); r[t] = np.sqrt(s2[t])*eps[t]
# egarch_model(r)  # -> beta near 0.95


def cheatsheet():
    return "egrch: EGARCH(1,1) asymmetric volatility (Nelson 1991)."
