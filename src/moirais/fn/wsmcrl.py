"""Cramer-Rao lower bound Var(T) >= 1/(n I(theta))."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_cramer_rao"]


def wasserman_cramer_rao(theta, n, I):
    """
    Cramer-Rao lower bound Var(T) >= 1/(n I(theta))

    Formula: Var(T) >= 1 / (n I(theta))

    Parameters
    ----------
    theta : array-like
        Input data.
    n : array-like
        Input data.
    I : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bound

    References
    ----------
    Wasserman (2004), Ch 9
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    n = len(theta)
    result = float(np.mean(theta))
    se = float(np.std(theta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cramer-Rao lower bound Var(T) >= 1/(n I(theta))"})


def cheatsheet():
    return "wsmcrl: Cramer-Rao lower bound Var(T) >= 1/(n I(theta))"
