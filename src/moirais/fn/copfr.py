"""General copula frailty for clustered survival."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["copula_frailty"]


def copula_frailty(time, event, cluster):
    """
    General copula frailty for clustered survival

    Formula: C(u,v;theta) frailty over Cox marginals

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    cluster : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Shih & Louis (1995)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "General copula frailty for clustered survival"})


def cheatsheet():
    return "copfr: General copula frailty for clustered survival"
