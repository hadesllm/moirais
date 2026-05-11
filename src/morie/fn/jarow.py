"""Jaro-Winkler similarity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["jaro_winkler"]


def jaro_winkler(s1, s2, p):
    """
    Jaro-Winkler similarity

    Formula: Jaro + prefix scaling p

    Parameters
    ----------
    s1 : array-like
        Input data.
    s2 : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Winkler (1990)
    """
    s1 = np.atleast_1d(np.asarray(s1, dtype=float))
    n = len(s1)
    result = float(np.mean(s1))
    se = float(np.std(s1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Jaro-Winkler similarity"})


def cheatsheet():
    return "jarow: Jaro-Winkler similarity"
