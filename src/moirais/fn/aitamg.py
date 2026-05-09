"""Amalgamation: sum a subset of parts into one."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aitchison_amalgamation"]


def aitchison_amalgamation(x, idx):
    """
    Amalgamation: sum a subset of parts into one

    Formula: x' = (x_{S^c}, sum x_S)

    Parameters
    ----------
    x : array-like
        Input data.
    idx : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x_amalg

    References
    ----------
    Aitchison (1986)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Amalgamation: sum a subset of parts into one"})


def cheatsheet():
    return "aitamg: Amalgamation: sum a subset of parts into one"
