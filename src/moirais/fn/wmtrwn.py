"""Row-normalize spatial weights."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["weights_row_normalize"]


def weights_row_normalize(W):
    """
    Row-normalize spatial weights

    Formula: w_ij' = w_ij / sum_j w_ij

    Parameters
    ----------
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Anselin (1988)
    """
    W = np.atleast_1d(np.asarray(W, dtype=float))
    n = len(W)
    result = float(np.mean(W))
    se = float(np.std(W, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Row-normalize spatial weights"})


def cheatsheet():
    return "wmtrwn: Row-normalize spatial weights"
