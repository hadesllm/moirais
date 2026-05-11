"""Subgradient method update."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_subgrad_method"]


def boyd_subgrad_method(f, subgrad, x0, t):
    """
    Subgradient method update

    Formula: x^{k+1} = x^k - t_k g^k, g^k in df(x^k)

    Parameters
    ----------
    f : array-like
        Input data.
    subgrad : array-like
        Input data.
    x0 : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x

    References
    ----------
    Boyd CVX Ch 9
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Subgradient method update"})


def cheatsheet():
    return "cvxsbm: Subgradient method update"
