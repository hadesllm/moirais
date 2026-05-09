"""Galbraith plot z_i vs 1/se_i."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_galbraith"]


def ma_galbraith(yi, se_i):
    """
    Galbraith plot z_i vs 1/se_i

    Formula: z_i = y_i/se_i; weighted regression slope = θ̂

    Parameters
    ----------
    yi : array-like
        Input data.
    se_i : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: z, x, slope

    References
    ----------
    Galbraith (1988)
    """
    yi = np.atleast_1d(np.asarray(yi, dtype=float))
    n = len(yi)
    result = float(np.mean(yi))
    se = float(np.std(yi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Galbraith plot z_i vs 1/se_i"})


def cheatsheet():
    return "magal: Galbraith plot z_i vs 1/se_i"
