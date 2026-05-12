# morie.fn — function file (hadesllm/morie)
"""Copula parameter estimation (Gaussian / Clayton / Gumbel).

Implements method-of-moments (or rank-based) fits via Kendall's tau
inversion (Nelsen 2006, *An Introduction to Copulas*, 2nd ed.):

* Gaussian:  rho   = sin(pi * tau / 2)
* Clayton:   theta = 2 tau / (1 - tau)
* Gumbel:    theta = 1 / (1 - tau)

All three families have closed-form tau-to-parameter maps.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["copula_estimation"]


def copula_estimation(x, y, family: str = "gaussian"):
    """Estimate copula parameter from rank correlation.

    Parameters
    ----------
    x, y : array-like
        Marginal samples.
    family : {"gaussian", "clayton", "gumbel"}

    Returns
    -------
    RichResult: estimate (the parameter), kendall_tau, family,
    n, method, plus pseudo-observations u, v (empirical CDFs).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(x.size, y.size)
    if n < 3:
        return RichResult(payload={"estimate": float("nan"), "n": int(n),
                                   "method": f"copula-{family} (n<3)"})
    tau, _ = stats.kendalltau(x[:n], y[:n])
    family = family.lower()
    if family == "gaussian":
        theta = float(np.sin(np.pi * tau / 2))
    elif family == "clayton":
        theta = float(2 * tau / (1 - tau)) if tau < 1 else float("inf")
    elif family == "gumbel":
        theta = float(1.0 / (1 - tau)) if tau < 1 else float("inf")
    else:
        raise ValueError(f"Unknown copula family {family!r}; "
                         "use gaussian / clayton / gumbel.")
    # empirical pseudo-observations
    u = (stats.rankdata(x[:n]) - 0.5) / n
    v = (stats.rankdata(y[:n]) - 0.5) / n
    se = float(np.sqrt((1 - tau ** 2) / n))
    return RichResult(payload={
        "estimate": theta, "kendall_tau": float(tau),
        "se_tau": se, "u": u, "v": v, "family": family,
        "n": int(n),
        "method": f"Copula {family} (rank-based; Nelsen 2006)",
    })


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> # bivariate normal with rho=0.7
# >>> z = rng.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], 500)
# >>> res = copula_estimation(z[:,0], z[:,1], family="gaussian")
# >>> assert abs(res["estimate"] - 0.7) < 0.1


def cheatsheet():
    return "copul(x, y, family='gaussian'|'clayton'|'gumbel'): copula param."
