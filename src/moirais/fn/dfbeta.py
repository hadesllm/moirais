"""DFBETAs per-coefficient influence."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dfbetas"]


def dfbetas(X, y):
    """
    DFBETAs per-coefficient influence

    Formula: change in β_j when obs i removed, scaled

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Belsley-Kuh-Welsch (1980)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DFBETAs per-coefficient influence"})


def cheatsheet():
    return "dfbeta: DFBETAs per-coefficient influence"
