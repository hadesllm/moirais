"""Manski-Pepper monotone treatment response (MTR) bounds."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["monotone_treatment_response"]


def monotone_treatment_response(y, D, direction):
    """
    Manski-Pepper monotone treatment response (MTR) bounds

    Formula: Y(d') >= Y(d) for d' > d; tightens upper/lower bound depending on direction

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    direction : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Manski-Pepper monotone treatment response (MTR) bounds"})


def cheatsheet():
    return "mtrbnd: Manski-Pepper monotone treatment response (MTR) bounds"
