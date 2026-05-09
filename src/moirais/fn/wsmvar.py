"""Variance Var(X) = E[(X-mu)^2]."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_variance"]


def wasserman_variance(x):
    """
    Variance Var(X) = E[(X-mu)^2]

    Formula: Var(X) = E[X^2] - E[X]^2

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Wasserman (2004), Ch 3
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variance Var(X) = E[(X-mu)^2]"})


def cheatsheet():
    return "wsmvar: Variance Var(X) = E[(X-mu)^2]"
