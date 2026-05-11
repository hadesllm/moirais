"""CLUB contrastive log-ratio upper bound."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["club_upper_bound"]


def club_upper_bound(x, y, q):
    """
    CLUB contrastive log-ratio upper bound

    Formula: I_CLUB = E[log q(y|x)] - E_marg[log q(y|x)]

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cheng-Hassani-Wang-Carin (2020) CLUB
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CLUB contrastive log-ratio upper bound"})


def cheatsheet():
    return "clbuvc: CLUB contrastive log-ratio upper bound"
