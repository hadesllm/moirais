"""AIC = -2 l_hat + 2 k."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_aic"]


def wasserman_aic(loglik, k):
    """
    AIC = -2 l_hat + 2 k

    Formula: AIC = -2 log L_hat + 2 k

    Parameters
    ----------
    loglik : array-like
        Input data.
    k : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AIC = -2 l_hat + 2 k"})


def cheatsheet():
    return "wsmaic: AIC = -2 l_hat + 2 k"
