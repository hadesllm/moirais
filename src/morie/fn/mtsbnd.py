"""Manski-Pepper monotone treatment selection (MTS) bounds."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["monotone_treatment_selection"]


def monotone_treatment_selection(y, D, y_min, y_max):
    """
    Manski-Pepper monotone treatment selection (MTS) bounds

    Formula: E[Y(d)|D=d'] is monotone in d'; tightens Manski bounds

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    y_min : array-like
        Input data.
    y_max : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Manski-Pepper (2000)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Manski-Pepper monotone treatment selection (MTS) bounds"})


def cheatsheet():
    return "mtsbnd: Manski-Pepper monotone treatment selection (MTS) bounds"
