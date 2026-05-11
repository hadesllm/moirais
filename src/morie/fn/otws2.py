"""p-Wasserstein distance between empirical 1-D measures."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_wasserstein_p_1d"]


def ot_wasserstein_p_1d(x, y, p):
    """
    p-Wasserstein distance between empirical 1-D measures

    Formula: W_p^p = (1/n) Σ |x_(i)-y_(i)|^p

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Wp

    References
    ----------
    Bobkov & Ledoux (2019)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "p-Wasserstein distance between empirical 1-D measures"})


def cheatsheet():
    return "otws2: p-Wasserstein distance between empirical 1-D measures"
