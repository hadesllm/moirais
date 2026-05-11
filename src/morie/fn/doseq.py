# morie.fn — function file (hadesllm/morie)
"""Dose-response analysis (logistic/probit)."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize

from ._containers import ESRes


def dose_response(doses: list[float] | np.ndarray, responses: list[int] | np.ndarray, totals: list[int] | np.ndarray, link: str = "logit", confidence: float = 0.95, cdf=None) -> ESRes:
    """Dose-response analysis via logistic or probit regression.

    Fits P(response) = link^{-1}(alpha + beta * dose).

    Parameters
    ----------
    doses : array-like of float
        Dose levels.
    responses : array-like of int
        Number responding at each dose.
    totals : array-like of int
        Total subjects at each dose.
    link : str, default 'logit'
        Link function: 'logit' or 'probit'.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    ESRes
        estimate is LD50 (dose at 50% response).

    References
    ----------
    Finney, D. J. (1971). *Probit Analysis*, 3rd ed. Cambridge
    University Press.
    """
    x = np.asarray(doses, dtype=float)
    r = np.asarray(responses, dtype=float)
    n = np.asarray(totals, dtype=float)

    if len(x) != len(r) or len(x) != len(n):
        raise ValueError("All arrays must match")
    if len(x) < 2:
        raise ValueError("Need at least 2 dose levels")

    p = r / n

    if link == "logit":
        inv_link = lambda z: 1 / (1 + np.exp(-z))
    elif link == "probit":
        inv_link = lambda z: stats.norm.cdf(z)
    else:
        raise ValueError("link must be 'logit' or 'probit'")

    def neg_ll(params):
        a, b = params
        eta = a + b * x
        mu = inv_link(eta)
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        ll = np.sum(r * np.log(mu) + (n - r) * np.log(1 - mu))
        return -ll

    res = minimize(neg_ll, [0.0, 0.1], method="Nelder-Mead")
    alpha, beta = res.x

    ld50 = -alpha / beta if abs(beta) > 1e-10 else np.inf

    try:
        from scipy.optimize import approx_fprime
        hess_inv = np.linalg.inv(
            np.array([
                [approx_fprime([alpha, beta], lambda p: neg_ll(p), 1e-5)[i]
                 for i in range(2)]
            ]).reshape(1, 2).repeat(2, axis=0)
        )
        se_beta = np.sqrt(abs(hess_inv[1, 1]))
    except Exception:
        se_beta = np.nan

    z = stats.norm.ppf((1 + confidence) / 2)

    return ESRes(
        measure="dose_response",
        estimate=float(ld50),
        extra={
            "alpha": float(alpha),
            "beta": float(beta),
            "se_beta": float(se_beta),
            "link": link,
            "ld50": float(ld50),
            "converged": res.success,
        },
    )


doseq = dose_response


def cheatsheet() -> str:
    return "dose_response({}) -> Dose-response analysis (logistic/probit)."
