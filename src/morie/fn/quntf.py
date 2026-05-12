# morie.fn -- function file (hadesllm/morie)
"""Nonparametric sample quantile function (Parzen 1979).

Returns the right-continuous empirical quantile function ``Q(tau) =
inf{x : F_n(x) >= tau}`` evaluated at user-supplied probabilities.

Also computes per-quantile SEs via the asymptotic density-quantile
formula

    var(Q(tau)) ~= tau(1-tau) / (n f(Q(tau))^2)

with ``f`` estimated by a Gaussian kernel (Silverman's rule bandwidth).
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["quantile_function"]


def quantile_function(x, taus=None):
    """Nonparametric quantile estimation with asymptotic SEs.

    Parameters
    ----------
    x : array-like
        Sample.
    taus : array-like, optional
        Probabilities in (0, 1).  Default = (0.1, 0.25, 0.5, 0.75, 0.9).

    Returns
    -------
    RichResult: taus, quantiles, se, n, method.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        return RichResult(payload={"estimate": float("nan"), "n": int(n),
                                   "method": "Quantile fn (n<2)"})
    if taus is None:
        taus = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
    taus = np.asarray(taus, dtype=float)
    q = np.quantile(x, taus, method="linear")
    # Silverman's rule for density at each q
    sd = x.std(ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    h = 1.06 * min(sd, iqr / 1.34) * n ** (-0.2) if iqr > 0 else 1.06 * sd * n ** (-0.2)
    if h <= 0:
        h = sd if sd > 0 else 1.0
    # kernel density estimate at each quantile
    fhat = np.array([
        np.mean(np.exp(-0.5 * ((x - qi) / h) ** 2) / (h * np.sqrt(2 * np.pi)))
        for qi in q
    ])
    se = np.sqrt(taus * (1 - taus) / (n * fhat ** 2))
    return RichResult(payload={
        "taus": taus, "quantiles": q, "se": se,
        "estimate": float(q[len(q) // 2]),
        "bandwidth": float(h), "n": int(n),
        "method": "Empirical quantile function (Parzen 1979)",
    })


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> x = rng.normal(0, 1, 1000)
# >>> res = quantile_function(x, taus=[0.5])
# >>> assert abs(res["quantiles"][0]) < 0.1


def cheatsheet():
    return "quntf(x, taus=...): empirical quantile function + SEs."
