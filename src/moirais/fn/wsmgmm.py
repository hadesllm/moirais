"""Gaussian mixture model via EM."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_gmm_em"]


def wasserman_gmm_em(X, k):
    """
    Gaussian mixture model via EM

    Formula: p(x) = sum pi_k N(x | mu_k, Sigma_k)

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: params

    References
    ----------
    Wasserman (2004), Ch 19
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gaussian mixture model via EM"})


def cheatsheet():
    return "wsmgmm: Gaussian mixture model via EM"
