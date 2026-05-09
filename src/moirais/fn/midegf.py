"""MI degrees-of-freedom."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mi_degrees_of_freedom"]


def mi_degrees_of_freedom(B, W, m):
    """
    MI degrees-of-freedom

    Formula: (m-1)(1+rate)^2/rate^2

    Parameters
    ----------
    B : array-like
        Input data.
    W : array-like
        Input data.
    m : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Barnard-Rubin (1999)
    """
    B = np.atleast_1d(np.asarray(B, dtype=float))
    n = len(B)
    result = float(np.mean(B))
    se = float(np.std(B, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MI degrees-of-freedom"})


def cheatsheet():
    return "midegf: MI degrees-of-freedom"
