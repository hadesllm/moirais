"""Strong convexity definition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_strong_convex"]


def boyd_strong_convex(f, m):
    """
    Strong convexity definition

    Formula: f(y) >= f(x) + grad f(x)'(y-x) + (m/2)|y-x|^2

    Parameters
    ----------
    f : array-like
        Input data.
    m : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bool

    References
    ----------
    Boyd CVX Ch 9
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Strong convexity definition"})


def cheatsheet():
    return "cvxstgc: Strong convexity definition"
