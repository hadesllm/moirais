"""Empty space function F(r)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["empty_space_function"]


def empty_space_function(coords, r_grid, window):
    """
    Empty space function F(r)

    Formula: P(NN distance from random point <= r)

    Parameters
    ----------
    coords : array-like
        Input data.
    r_grid : array-like
        Input data.
    window : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Baddeley-Turner (2005)
    """
    coords = np.atleast_1d(np.asarray(coords, dtype=float))
    n = len(coords)
    result = float(np.mean(coords))
    se = float(np.std(coords, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Empty space function F(r)"})


def cheatsheet():
    return "empfun: Empty space function F(r)"
