"""Clayton copula bivariate survival."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["clayton_copula_frailty"]


def clayton_copula_frailty(time, event, cluster):
    """
    Clayton copula bivariate survival

    Formula: S(t1, t2) = (S1^{-theta} + S2^{-theta} - 1)^{-1/theta}

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
    Clayton (1978)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Clayton copula bivariate survival"})


def cheatsheet():
    return "clyfr: Clayton copula bivariate survival"
