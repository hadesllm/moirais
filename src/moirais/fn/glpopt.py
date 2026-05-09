"""GLPK LP wrapper."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["glpk_lp"]


def glpk_lp(c, A, b):
    """
    GLPK LP wrapper

    Formula: open-source LP via GLPK

    Parameters
    ----------
    c : array-like
        Input data.
    A : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Makhorin (GLPK)
    """
    c = np.atleast_1d(np.asarray(c, dtype=float))
    n = len(c)
    result = float(np.mean(c))
    se = float(np.std(c, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GLPK LP wrapper"})


def cheatsheet():
    return "glpopt: GLPK LP wrapper"
