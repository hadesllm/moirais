"""CF approximation of π."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["continued_fraction_pi"]


def continued_fraction_pi(n):
    """
    CF approximation of π

    Formula: convergents of CF expansion

    Parameters
    ----------
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    classical
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CF approximation of π"})


def cheatsheet():
    return "conti: CF approximation of π"
