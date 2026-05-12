# morie.fn — function file (hadesllm/morie)
"""TGARCH / GJR-GARCH(1,1) — threshold GARCH with asymmetric news impact."""
from __future__ import annotations

import numpy as np
from scipy import optimize

from ._richresult import RichResult

__all__ = ["tgarch_model"]


def tgarch_model(x):
    r"""Fit a Glosten-Jagannathan-Runkle GARCH(1,1) model.

    .. math::

        \sigma_t^2 = \omega + (\alpha + \gamma I_{t-1})\,\epsilon_{t-1}^2
                     + \beta\,\sigma_{t-1}^2,
        \quad I_{t-1} = 1\{\epsilon_{t-1} < 0\}.

    Parameters
    ----------
    x : array-like
        Return series.

    Returns
    -------
    RichResult
        keys: ``omega``, ``alpha``, ``gamma``, ``beta``, ``persistence``,
        ``loglik``, ``conditional_variance``, ``n``, ``method``.

    References
    ----------
    Glosten LR, Jagannathan R, Runkle DE (1993). On the Relation
    Between the Expected Value and the Volatility of the Nominal
    Excess Return on Stocks. *J. Finance* 48(5), 1779-1801.
    """
    r = np.asarray(x, dtype=float).ravel()
    r = r - r.mean()
    n = r.size
    if n < 20:
        raise ValueError(f"Need at least 20 observations, got {n}.")

    try:
        from arch import arch_model
        m = arch_model(r, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="normal")
        fit = m.fit(disp="off", show_warning=False)
        params = fit.params
        omega = float(params.get("omega", np.nan))
        alpha = float(params.get("alpha[1]", np.nan))
        gamma = float(params.get("gamma[1]", np.nan))
        beta  = float(params.get("beta[1]",  np.nan))
        return RichResult(payload={
            "omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta,
            "persistence": alpha + 0.5 * gamma + beta,
            "loglik": float(fit.loglikelihood),
            "conditional_variance": np.asarray(fit.conditional_volatility) ** 2,
            "n": int(n),
            "method": "GJR-GARCH(1,1) via arch.arch_model",
        })
    except Exception:
        pass

    def neg_ll(p):
        omega, alpha, gamma, beta = p
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + 0.5 * gamma + beta >= 1:
            return 1e10
        s2 = np.zeros(n)
        s2[0] = np.var(r) + 1e-10
        for t in range(1, n):
            I = 1.0 if r[t - 1] < 0 else 0.0
            s2[t] = omega + (alpha + gamma * I) * r[t - 1] ** 2 + beta * s2[t - 1]
            s2[t] = max(s2[t], 1e-12)
        return 0.5 * np.sum(np.log(2 * np.pi * s2) + r ** 2 / s2)

    var_r = float(np.var(r))
    fit = optimize.minimize(
        neg_ll,
        [var_r * 0.05, 0.05, 0.05, 0.85],
        bounds=[(1e-8, var_r * 10), (1e-8, 0.5), (-0.5, 0.999), (1e-8, 0.999)],
        method="L-BFGS-B",
    )
    omega, alpha, gamma, beta = fit.x
    s2 = np.zeros(n)
    s2[0] = var_r
    for t in range(1, n):
        I = 1.0 if r[t - 1] < 0 else 0.0
        s2[t] = omega + (alpha + gamma * I) * r[t - 1] ** 2 + beta * s2[t - 1]
    return RichResult(payload={
        "omega": float(omega), "alpha": float(alpha),
        "gamma": float(gamma), "beta": float(beta),
        "persistence": float(alpha + 0.5 * gamma + beta),
        "loglik": float(-fit.fun),
        "conditional_variance": s2,
        "n": int(n),
        "method": "GJR-GARCH(1,1) Gaussian MLE (numpy)",
    })


# CANONICAL TEST: r ~ AR(1)/GJR-GARCH innovations, n=200; gamma should
# come out > 0 reflecting leverage effect.


def cheatsheet():
    return "tgrch: GJR-GARCH(1,1) threshold GARCH (Glosten et al. 1993)."
