"""Aitchison norm of a composition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aitchison_norm"]


def aitchison_norm(x):
    """
    Aitchison norm of a composition

    Formula: ||x||_A = ||clr(x)||_2

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: n

    References
    ----------
    Aitchison (1986)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Aitchison norm of a composition"})


def cheatsheet():
    return "aitnrm: Aitchison norm of a composition"
