# morie.fn — function file (hadesllm/morie)
"""Return-level estimation for GEV-fitted block maxima (Coles 2001, Ch. 3).

For block-maxima with GEV(mu, sigma, xi), the level exceeded once every
``T`` blocks is

    xi != 0:  z_T = mu - (sigma / xi) * (1 - (-log(1 - 1/T))^{-xi})
    xi == 0:  z_T = mu - sigma * log(-log(1 - 1/T))

Delta-method SEs require the GEV information matrix (from
:func:`morie.fn.extvm`).
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult
from . import extvm as _extvm

__all__ = ["return_level"]


def return_level(x, return_period: float = 100.0):
    """Estimate the ``T``-block return level from GEV-fitted maxima.

    Parameters
    ----------
    x : array-like
        Block maxima sample.
    return_period : float
        Return period ``T`` in block units (default 100).

    Returns
    -------
    RichResult: z, se, return_period, mu, sigma, xi, n, method.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 5:
        return RichResult(payload={"estimate": float("nan"),
                                   "n": int(x.size),
                                   "method": "Return level (n<5)"})
    fit = _extvm.extreme_value_gev(x)
    mu = float(fit["mu"]); sigma = float(fit["sigma"]); xi = float(fit["xi"])
    p = 1.0 / return_period
    yp = -np.log(1 - p)
    if abs(xi) < 1e-6:
        z = mu - sigma * np.log(yp)
        # delta-method derivatives wrt (mu, sigma, xi)
        d_mu = 1.0
        d_sig = -np.log(yp)
        d_xi = 0.5 * sigma * np.log(yp) ** 2  # Coles eq. (3.4) limit
    else:
        z = mu - (sigma / xi) * (1 - yp ** (-xi))
        d_mu = 1.0
        d_sig = -(1 / xi) * (1 - yp ** (-xi))
        d_xi = (sigma / xi ** 2) * (1 - yp ** (-xi)) \
            - (sigma / xi) * yp ** (-xi) * np.log(yp)
    se_mu = float(fit.get("se_mu", float("nan")))
    se_sig = float(fit.get("se_sigma", float("nan")))
    se_xi = float(fit.get("se_xi", float("nan")))
    # diagonal-only approximation if cov off-diagonals unavailable
    var_z = (d_mu * se_mu) ** 2 + (d_sig * se_sig) ** 2 + (d_xi * se_xi) ** 2
    se = float(np.sqrt(max(0.0, var_z)))
    return RichResult(payload={
        "z": float(z), "estimate": float(z), "se": se,
        "return_period": float(return_period),
        "mu": mu, "sigma": sigma, "xi": xi,
        "n": int(x.size),
        "method": "Return level (Coles 2001)",
    })


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> x = rng.gumbel(loc=10.0, scale=2.0, size=500)
# >>> res = return_level(x, return_period=100)
# >>> # Gumbel 100-yr level: 10 - 2*log(-log(0.99)) ~= 19.2
# >>> assert 16 < res["z"] < 23


def cheatsheet():
    return "retlv(x, return_period=100): GEV return-level estimate + SE."
