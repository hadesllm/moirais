"""Robinson unification."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["unification"]


def unification(t1, t2):
    """
    Robinson unification

    Formula: compute MGU of two terms

    Parameters
    ----------
    t1 : array-like
        Input data.
    t2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robinson (1965)
    """
    t1 = np.atleast_1d(np.asarray(t1, dtype=float))
    n = len(t1)
    result = float(np.mean(t1))
    se = float(np.std(t1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Robinson unification"})


def cheatsheet():
    return "unifAlg: Robinson unification"
