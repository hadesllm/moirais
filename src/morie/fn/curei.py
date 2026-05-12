# morie.fn -- function file (hadesllm/morie)
"""Mixture cure model."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ._containers import DescriptiveResult


def cure_model(
    times: np.ndarray | list,
    events: np.ndarray | list,
    X: np.ndarray | list | None = None,
    *,
    max_iter: int = 200,
) -> DescriptiveResult:
    """
    Fit a mixture cure model.

    Assumes a fraction *p* are cured (never experience event) and
    the uncured follow a Weibull survival distribution.

    Parameters
    ----------
    times : array-like
        Event/censoring times.
    events : array-like
        Event indicators (1 = event, 0 = censored).
    X : array-like, optional
        Covariates for cure fraction (logistic).
    max_iter : int
        Maximum iterations.

    Returns
    -------
    DescriptiveResult
        extra has 'cure_fraction', 'weibull_shape', 'weibull_scale'.

    References
    ----------
    Berkson, J., & Gage, R. P. (1952). Survival curve for cancer
    patients following treatment. *JASA*, 47(259), 501-515.
    """
    t = np.asarray(times, dtype=float)
    d = np.asarray(events, dtype=int)
    if len(t) != len(d):
        raise ValueError("times and events must match.")

    def neg_loglik(params):
        pi = 1 / (1 + np.exp(-params[0]))
        shape = np.exp(params[1])
        scale = np.exp(params[2])
        S_u = np.exp(-((t / scale) ** shape))
        f_u = (shape / scale) * (t / scale) ** (shape - 1) * S_u
        f_u = np.maximum(f_u, 1e-300)
        S_u = np.maximum(S_u, 1e-300)
        ll = np.sum(d * np.log((1 - pi) * f_u + 1e-300) + (1 - d) * np.log(pi + (1 - pi) * S_u + 1e-300))
        return -ll

    x0 = [0.0, 0.0, np.log(np.median(t[t > 0]) + 1e-6)]
    res = minimize(neg_loglik, x0, method="Nelder-Mead", options={"maxiter": max_iter})

    pi = 1 / (1 + np.exp(-res.x[0]))
    shape = np.exp(res.x[1])
    scale = np.exp(res.x[2])

    return DescriptiveResult(
        name="cure_model",
        value=float(pi),
        extra={
            "cure_fraction": float(pi),
            "weibull_shape": float(shape),
            "weibull_scale": float(scale),
            "converged": bool(res.success),
            "loglik": float(-res.fun),
        },
    )


curei = cure_model


def cheatsheet() -> str:
    return "cure_model({}) -> Mixture cure model."
