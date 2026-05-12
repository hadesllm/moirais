# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Antithetic variates Monte-Carlo (Hammersley & Morton 1956).

For uniform draws ``U_i in (0,1)`` and integrand ``f``, the antithetic
estimator is::

    theta_AV = (1/N) sum_i  ( f(U_i) + f(1 - U_i) ) / 2

which is unbiased and has variance no larger than crude MC when ``f`` is
monotone in U.  Returns variance-reduction ratio vs the crude estimator.
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["antithetic_variates"]


def antithetic_variates(x=None, f=None, N: int = 1000, seed: int = 42):
    """Antithetic-variate Monte Carlo.

    Parameters
    ----------
    x : array-like, optional
        Pre-drawn U(0,1) samples.  If None, ``N`` random U(0,1)s are drawn.
    f : callable
        Integrand on (0,1) -> scalar.  Default ``f(u)=u`` (true E=0.5).
    N : int
        Used when ``x`` is None.
    seed : int

    Returns
    -------
    RichResult: estimate (antithetic), estimate_crude, se, var_ratio,
    n, method.
    """
    if f is None:
        f = lambda u: u
    rng = np.random.default_rng(seed)
    if x is None:
        u = rng.uniform(0.0, 1.0, size=int(N))
    else:
        u = np.asarray(x, dtype=float).ravel()
        # ensure values are in (0,1): if not, rescale via empirical rank
        if (u.min() < 0) or (u.max() > 1):
            ranks = u.argsort().argsort() + 1
            u = ranks / (u.size + 1.0)
    n = u.size
    fu = np.asarray(f(u), dtype=float)
    fu_anti = np.asarray(f(1.0 - u), dtype=float)
    paired = 0.5 * (fu + fu_anti)
    est_av = float(paired.mean())
    se_av = float(paired.std(ddof=1) / np.sqrt(n))
    est_crude = float(fu.mean())
    var_crude = float(fu.var(ddof=1) / n)
    var_av = se_av ** 2
    ratio = float(var_av / var_crude) if var_crude > 0 else float("nan")
    return RichResult(payload={
        "estimate": est_av, "estimate_crude": est_crude,
        "se": se_av, "var_ratio_av_over_crude": ratio,
        "n_pairs": int(n),
        "method": "Antithetic variates (Hammersley & Morton 1956)",
    })


# CANONICAL TEST
# >>> # E_U[U] = 0.5
# >>> res = antithetic_variates(N=2000, seed=0)
# >>> assert abs(res["estimate"] - 0.5) < 0.05
# >>> # For monotone f(u)=u, antithetic variance is exactly 0
# >>> assert res["se"] < 1e-9


def cheatsheet():
    return "antth(x or N, f): antithetic-variate MC estimator + var ratio."
