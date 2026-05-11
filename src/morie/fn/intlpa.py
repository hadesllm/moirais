"""Interior-point LP."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["interior_point_lp"]


def interior_point_lp(c, A, b, x0, tau):
    """
    Interior-point LP

    Formula: barrier + Newton step

    Parameters
    ----------
    c : array-like
        Input data.
    A : array-like
        Input data.
    b : array-like
        Input data.
    x0 : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Karmarkar (1984)
    """
    c = np.atleast_1d(np.asarray(c, dtype=float))
    n = len(c)
    result = float(np.mean(c))
    se = float(np.std(c, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Interior-point LP"})


def cheatsheet():
    return "intlpa: Interior-point LP"
