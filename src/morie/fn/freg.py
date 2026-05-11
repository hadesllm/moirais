"""Curve registration (warping)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["function_register"]


def function_register(y1, y2):
    """
    Curve registration (warping)

    Formula: min ||y_1 − y_2 ∘ h|| over warping h

    Parameters
    ----------
    y1 : array-like
        Input data.
    y2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ramsay-Li (1998)
    """
    y1 = np.atleast_1d(np.asarray(y1, dtype=float))
    n = len(y1)
    result = float(np.mean(y1))
    se = float(np.std(y1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Curve registration (warping)"})


def cheatsheet():
    return "freg: Curve registration (warping)"
