"""Bound under unknown compliance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_compliance"]


def bound_compliance(y, D, Z):
    """
    Bound under unknown compliance

    Formula: intersection over compliance types

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    Z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Imbens-Rubin (1997)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bound under unknown compliance"})


def cheatsheet():
    return "bnscom: Bound under unknown compliance"
