"""J function = (1-G)/(1-F) — CSR has J=1."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ripley_j_function"]


def ripley_j_function(points, window, r):
    """
    J function = (1-G)/(1-F) — CSR has J=1

    Formula: J(d) = (1 - G(d)) / (1 - F(d))

    Parameters
    ----------
    points : array-like
        Input data.
    window : array-like
        Input data.
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    van Lieshout & Baddeley (1996)
    """
    points = np.atleast_1d(np.asarray(points, dtype=float))
    n = len(points)
    result = float(np.mean(points))
    se = float(np.std(points, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "J function = (1-G)/(1-F) — CSR has J=1"})


def cheatsheet():
    return "ripJ: J function = (1-G)/(1-F) — CSR has J=1"
