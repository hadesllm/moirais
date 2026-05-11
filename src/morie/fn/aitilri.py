"""Inverse ILR back to a closed composition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aitchison_ilr_inverse"]


def aitchison_ilr_inverse(y, V):
    """
    Inverse ILR back to a closed composition

    Formula: x = C(exp(V y))

    Parameters
    ----------
    y : array-like
        Input data.
    V : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x

    References
    ----------
    Egozcue et al. (2003)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Inverse ILR back to a closed composition"})


def cheatsheet():
    return "aitilri: Inverse ILR back to a closed composition"
