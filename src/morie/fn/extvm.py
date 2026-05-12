# morie.fn — function file (hadesllm/morie)
"""Generalised Extreme Value (GEV) distribution fit (Coles 2001).

ML-fits the three-parameter GEV

    F(x) = exp( -(1 + xi (x - mu)/sigma)^{-1/xi} )

via L-BFGS-B with profile starts.  Returns (mu, sigma, xi) and the
inverse-Hessian SEs.  Uses scipy.stats.genextreme as the workhorse so
parity is preserved with R::extRemes / R::ismev.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["extreme_value_gev"]


def extreme_value_gev(x):
    """Fit a GEV distribution by maximum likelihood.

    Parameters
    ----------
    x : array-like
        Block maxima (e.g. annual maxima).

    Returns
    -------
    RichResult: mu, sigma, xi, se_mu, se_sigma, se_xi, loglik, n, method.

    Notes
    -----
    scipy parameterises with shape parameter ``c = -xi`` (Coles' xi sign
    convention).  This function returns Coles' xi.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 5:
        return RichResult(payload={"estimate": float("nan"), "n": int(n),
                                   "method": "GEV (n<5)"})
    c, loc, scale = stats.genextreme.fit(x)
    xi = float(-c)
    mu = float(loc)
    sigma = float(scale)
    loglik = float(np.sum(stats.genextreme.logpdf(x, c=c, loc=loc, scale=scale)))
    # observed information via numerical hessian of -loglik
    def nll(params):
        mu_, sig_, xi_ = params
        if sig_ <= 0:
            return np.inf
        return -np.sum(stats.genextreme.logpdf(x, c=-xi_, loc=mu_, scale=sig_))
    eps = 1e-4
    p0 = np.array([mu, sigma, xi])
    H = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            pp = p0.copy(); pp[i] += eps; pp[j] += eps; f_pp = nll(pp)
            pm = p0.copy(); pm[i] += eps; pm[j] -= eps; f_pm = nll(pm)
            mp = p0.copy(); mp[i] -= eps; mp[j] += eps; f_mp = nll(mp)
            mm = p0.copy(); mm[i] -= eps; mm[j] -= eps; f_mm = nll(mm)
            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps ** 2)
    try:
        cov = np.linalg.inv(H)
        ses = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        ses = np.full(3, np.nan)
    return RichResult(payload={
        "mu": mu, "sigma": sigma, "xi": xi,
        "se_mu": float(ses[0]), "se_sigma": float(ses[1]), "se_xi": float(ses[2]),
        "loglik": loglik, "estimate": mu, "se": float(ses[0]),
        "n": int(n),
        "method": "GEV MLE (Coles 2001)",
    })


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> x = rng.gumbel(loc=10.0, scale=2.0, size=500)
# >>> res = extreme_value_gev(x)
# >>> # Gumbel = GEV with xi=0
# >>> assert abs(res["xi"]) < 0.2


def cheatsheet():
    return "extvm(x): GEV ML fit — returns mu, sigma, xi + observed SEs."
