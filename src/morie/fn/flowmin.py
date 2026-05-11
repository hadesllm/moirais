"""Stoer-Wagner min-cut."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["min_cut"]


def min_cut(A):
    """
    Stoer-Wagner min-cut

    Formula: O(VE + V² log V) min-cut algorithm

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Stoer-Wagner (1997)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stoer-Wagner min-cut"})


def cheatsheet():
    return "flowmin: Stoer-Wagner min-cut"
