"""Chinese Restaurant Process — exchangeable partition prior."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["chinese_restaurant_process"]


def chinese_restaurant_process(n, alpha):
    """
    Chinese Restaurant Process — exchangeable partition prior

    Formula: P(z_i=k) = n_k/(n-1+alpha); P(new) = alpha/(n-1+alpha)

    Parameters
    ----------
    n : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Aldous (1985); Pitman (2006)
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Chinese Restaurant Process — exchangeable partition prior"})


def cheatsheet():
    return "dpcrp: Chinese Restaurant Process — exchangeable partition prior"
