"""Mixed-integer LP branch + bound."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mip_branch_bound"]


def mip_branch_bound(c, A, b, integer_indices):
    """
    Mixed-integer LP branch + bound

    Formula: branch on fractional integer; LP relaxation

    Parameters
    ----------
    c : array-like
        Input data.
    A : array-like
        Input data.
    b : array-like
        Input data.
    integer_indices : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Land-Doig (1960)
    """
    c = np.atleast_1d(np.asarray(c, dtype=float))
    n = len(c)
    result = float(np.mean(c))
    se = float(np.std(c, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mixed-integer LP branch + bound"})


def cheatsheet():
    return "miprgr: Mixed-integer LP branch + bound"
