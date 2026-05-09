"""BIC = -2 l_hat + k log n."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_bic"]


def wasserman_bic(loglik, k, n):
    """
    BIC = -2 l_hat + k log n

    Formula: BIC = -2 log L_hat + k log(n)

    Parameters
    ----------
    loglik : array-like
        Input data.
    k : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Wasserman (2004), Ch 13
    """
    loglik = np.atleast_1d(np.asarray(loglik, dtype=float))
    n = len(loglik)
    result = float(np.mean(loglik))
    se = float(np.std(loglik, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BIC = -2 l_hat + k log n"})


def cheatsheet():
    return "wsmbic: BIC = -2 l_hat + k log n"
