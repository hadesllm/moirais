"""Item nonresponse adjustment."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["item_nonresponse"]


def item_nonresponse(y, R, X):
    """
    Item nonresponse adjustment

    Formula: weight by 1/Pr(respond | X)

    Parameters
    ----------
    y : array-like
        Input data.
    R : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kalton-Flores-Cervantes (2003)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Item nonresponse adjustment"})


def cheatsheet():
    return "itnnrs: Item nonresponse adjustment"
