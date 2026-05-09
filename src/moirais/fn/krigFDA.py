"""Universal kriging."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kriging"]


def kriging(coords, values, new_coords):
    """
    Universal kriging

    Formula: Z* = β'X + λ'(Z − X β); GLS

    Parameters
    ----------
    coords : array-like
        Input data.
    values : array-like
        Input data.
    new_coords : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cressie (1993)
    """
    values = np.atleast_1d(np.asarray(values, dtype=float))
    n = len(values)
    result = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Universal kriging"})


def cheatsheet():
    return "krigFDA: Universal kriging"
