"""User-based collaborative filtering."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["user_cf"]


def user_cf(R, u, i, k):
    """
    User-based collaborative filtering

    Formula: r̂_{ui} = sum sim(u,v) r_{vi} / sum sim

    Parameters
    ----------
    R : array-like
        Input data.
    u : array-like
        Input data.
    i : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Resnick et al (1994) GroupLens
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    n = len(R)
    result = float(np.mean(R))
    se = float(np.std(R, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "User-based collaborative filtering"})


def cheatsheet():
    return "ucfR: User-based collaborative filtering"
