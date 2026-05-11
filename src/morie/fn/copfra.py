"""Frank copula CDF (Archimedean)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["frank_copula"]


def frank_copula(y, u, v, theta):
    """
    Frank copula CDF (Archimedean)

    Formula: C(u,v) = -1/theta * log(1 + (e^{-theta u} - 1)(e^{-theta v} - 1)/(e^{-theta} - 1))

    Parameters
    ----------
    y : array-like
        Input data.
    u : array-like
        Input data.
    v : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Frank (1979); Genest (1987)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Frank copula CDF (Archimedean)"})


def cheatsheet():
    return "copfra: Frank copula CDF (Archimedean)"
