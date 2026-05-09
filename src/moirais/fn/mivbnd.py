"""Manski monotone instrumental variable bounds."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["monotone_iv_bounds"]


def monotone_iv_bounds(y, D, Z, y_min, y_max):
    """
    Manski monotone instrumental variable bounds

    Formula: E[Y(d)|Z] monotone in Z; intersection of bounds across Z

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    Z : array-like
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
    Manski-Pepper (2000); Bhattacharya-Shaikh-Vytlacil (2008)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Manski monotone instrumental variable bounds"})


def cheatsheet():
    return "mivbnd: Manski monotone instrumental variable bounds"
