"""Cold-start handling."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cold_start_user"]


def cold_start_user(user, mode):
    """
    Cold-start handling

    Formula: fallback to popular / content / metadata

    Parameters
    ----------
    user : array-like
        Input data.
    mode : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schein et al (2002)
    """
    user = np.atleast_1d(np.asarray(user, dtype=float))
    n = len(user)
    result = float(np.mean(user))
    se = float(np.std(user, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cold-start handling"})


def cheatsheet():
    return "colE: Cold-start handling"
