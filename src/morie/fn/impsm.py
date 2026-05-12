# morie.fn -- function file (hadesllm/morie)
"""Importance sampling estimator (Geweke 1989).

Given samples ``x_i ~ q`` and target measure ``p``, estimates
``E_p[h(X)] ~= (1/N) sum_i w_i h(x_i)`` where ``w_i = p(x_i)/q(x_i)``.
Both **unnormalised** (works when p is known up to constant) and
**self-normalised** ratio estimators are returned, plus the effective
sample size ``ESS = (sum w)^2 / sum(w^2)``.
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["importance_sampling"]


def importance_sampling(x, h=None, p=None, q=None):
    """Importance-sampling Monte-Carlo estimator.

    Parameters
    ----------
    x : array-like
        Samples drawn from proposal ``q``.
    h : callable
        Integrand ``h(x) -> scalar``.  Default ``h(x)=x``.
    p : callable
        Target density ``p(x) -> scalar``.  Default standard normal.
    q : callable
        Proposal density ``q(x) -> scalar``.  Default standard normal
        (yields unweighted mean as sanity check).

    Returns
    -------
    RichResult: estimate, estimate_sn, se, ess, n, method.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 1:
        return RichResult(payload={"estimate": float("nan"), "n": 0,
                                   "method": "Importance sampling (empty)"})
    if h is None:
        h = lambda z: z
    if p is None:
        p = lambda z: np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    if q is None:
        q = lambda z: np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    hx = np.asarray(h(x), dtype=float)
    px = np.asarray(p(x), dtype=float)
    qx = np.asarray(q(x), dtype=float)
    w = px / qx
    est = float(np.mean(w * hx))            # unnormalised
    est_sn = float(np.sum(w * hx) / np.sum(w))  # self-normalised
    var = float(np.var(w * hx, ddof=1) / n)
    se = float(np.sqrt(var))
    ess = float(np.sum(w) ** 2 / np.sum(w ** 2))
    return RichResult(payload={
        "estimate": est, "estimate_sn": est_sn, "se": se,
        "ess": ess, "n": int(n),
        "method": "Importance sampling (Geweke 1989)",
    })


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> x = rng.standard_normal(2000)
# >>> # E[X^2] under standard normal = 1
# >>> res = importance_sampling(x, h=lambda z: z**2)
# >>> assert abs(res["estimate"] - 1.0) < 0.2


def cheatsheet():
    return "impsm(x, h, p, q): importance sampling estimate + ESS."
