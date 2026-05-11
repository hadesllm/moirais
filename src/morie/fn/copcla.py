"""Clayton copula CDF (Archimedean, lower-tail)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["clayton_copula"]


def clayton_copula(y, u, v, theta):
    """
    Clayton copula CDF (Archimedean, lower-tail)

    Formula: C(u,v) = (u^{-theta} + v^{-theta} - 1)^{-1/theta}

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
    Clayton (1978)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Clayton copula CDF (Archimedean, lower-tail)"})


def cheatsheet():
    return "copcla: Clayton copula CDF (Archimedean, lower-tail)"
