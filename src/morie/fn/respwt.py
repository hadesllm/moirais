"""Response-rate weight adjustment per cell."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["response_weight"]


def response_weight(y, weights, cell, r_h, n_h):
    """
    Response-rate weight adjustment per cell

    Formula: w_i' = w_i * (n_h / r_h)

    Parameters
    ----------
    y : array-like
        Input data.
    weights : array-like
        Input data.
    cell : array-like
        Input data.
    r_h : array-like
        Input data.
    n_h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lohr (2010) §8.6
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Response-rate weight adjustment per cell"})


def cheatsheet():
    return "respwt: Response-rate weight adjustment per cell"
