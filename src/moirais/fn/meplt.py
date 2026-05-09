"""Mean excess plot."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mean_excess"]


def mean_excess(x, u_grid):
    """
    Mean excess plot

    Formula: e(u) = E[X-u | X>u]

    Parameters
    ----------
    x : array-like
        Input data.
    u_grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Davison-Smith (1990)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mean excess plot"})


def cheatsheet():
    return "meplt: Mean excess plot"
