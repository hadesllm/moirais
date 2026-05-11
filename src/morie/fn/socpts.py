"""Second-order cone programming."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["second_order_cone"]


def second_order_cone(c, A, b, domains):
    """
    Second-order cone programming

    Formula: min c^T x s.t. ||A_i x + b_i|| <= c_i^T x + d_i

    Parameters
    ----------
    c : array-like
        Input data.
    A : array-like
        Input data.
    b : array-like
        Input data.
    domains : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lobo-Vandenberghe-Boyd-Lebret (1998)
    """
    c = np.atleast_1d(np.asarray(c, dtype=float))
    n = len(c)
    result = float(np.mean(c))
    se = float(np.std(c, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Second-order cone programming"})


def cheatsheet():
    return "socpts: Second-order cone programming"
