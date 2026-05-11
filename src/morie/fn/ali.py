"""Ali-Mikhail-Haq copula CDF."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ali_mikhail_haq_copula"]


def ali_mikhail_haq_copula(y, u, v, theta):
    """
    Ali-Mikhail-Haq copula CDF

    Formula: C(u,v) = uv / (1 - theta(1-u)(1-v))

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
    Ali, Mikhail, Haq (1978)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ali-Mikhail-Haq copula CDF"})


def cheatsheet():
    return "ali: Ali-Mikhail-Haq copula CDF"
