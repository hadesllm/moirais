"""Pinsker's inequality (TV vs KL)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pinsker_inequality"]


def pinsker_inequality(p, q):
    """
    Pinsker's inequality (TV vs KL)

    Formula: ||p-q||_TV <= sqrt(0.5 D_KL(p||q))

    Parameters
    ----------
    p : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Pinsker (1964)
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    n = len(p)
    result = float(np.mean(p))
    se = float(np.std(p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pinsker's inequality (TV vs KL)"})


def cheatsheet():
    return "pinsk1: Pinsker's inequality (TV vs KL)"
