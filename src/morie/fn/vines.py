"""Vine copula — D-vine pair-copula construction (Aas, Czado, Frigessi &
Bakken 2009).

Implements a Gaussian D-vine: factorises ``f(x_1, ..., x_d)`` as a
product of bivariate Gaussian pair-copulas along D-vine edges:

    c(u_1,...,u_d) = prod_{j=1}^{d-1} prod_{i=1}^{d-j}
                       c_{i,i+j | i+1:i+j-1}( F(u_i|*), F(u_{i+j}|*) )

For simplicity (and tractable identification at moderate d), all
pair-copulas are Gaussian; conditional CDFs use the Gaussian rosenblatt
update.  Returns the partial-correlation matrix and the joint
log-likelihood.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["vine_copula"]


def vine_copula(x):
    """Fit a Gaussian D-vine.

    Parameters
    ----------
    x : (n, d) array-like
        Multivariate sample.

    Returns
    -------
    RichResult: partial_corr (d-by-d matrix), loglik, n, d, method.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 2:
        return RichResult(payload={"estimate": float("nan"),
                                   "method": "vine copula (need 2-D, n>=3)"})
    n, d = x.shape
    # pseudo-observations
    u = (np.argsort(np.argsort(x, axis=0), axis=0) + 1) / (n + 1)
    z = stats.norm.ppf(u)
    # full Gaussian correlation matrix
    R = np.corrcoef(z, rowvar=False)
    # partial-correlation D-vine: standard recursion
    P = np.eye(d)
    for j in range(1, d):
        for i in range(d - j):
            if j == 1:
                P[i, i + j] = R[i, i + j]
            else:
                # Yule's partial correlation recursion
                cond = list(range(i + 1, i + j))
                idx = [i, i + j] + cond
                sub = R[np.ix_(idx, idx)]
                # partial corr of (0,1) given (2..)
                inv = np.linalg.pinv(sub)
                pc = -inv[0, 1] / np.sqrt(inv[0, 0] * inv[1, 1])
                P[i, i + j] = pc
            P[i + j, i] = P[i, i + j]
    # joint Gaussian log-likelihood under R
    sign, logdet = np.linalg.slogdet(R)
    if sign <= 0:
        loglik = float("nan")
    else:
        # 2 * loglik = -n * (logdet + tr(R^-1 (z'z/n) ) - d)  (for centred z)
        S = (z.T @ z) / n
        try:
            R_inv = np.linalg.inv(R)
            loglik = -0.5 * n * (logdet + float(np.trace(R_inv @ S)))
        except Exception:
            loglik = float("nan")
    return RichResult(payload={
        "partial_corr": P, "R": R, "loglik": float(loglik),
        "estimate": float(np.mean(np.abs(P[np.triu_indices(d, k=1)]))),
        "n": int(n), "d": int(d),
        "method": "Gaussian D-vine copula (Aas et al. 2009)",
    })


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> Sigma = np.array([[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]])
# >>> z = rng.multivariate_normal([0, 0, 0], Sigma, 500)
# >>> res = vine_copula(z)
# >>> assert res["d"] == 3
# >>> assert np.isfinite(res["loglik"])


def cheatsheet():
    return "vines(X): Gaussian D-vine partial-corr matrix + log-likelihood."
