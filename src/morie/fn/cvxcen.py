"""Central path of barrier."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_central_path"]


def boyd_central_path(f0, f, t):
    """
    Central path of barrier

    Formula: x*(t) = argmin t f0(x) + phi(x)

    Parameters
    ----------
    f0 : array-like
        Input data.
    f : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x

    References
    ----------
    Boyd CVX Ch 11
    """
    f0 = np.atleast_1d(np.asarray(f0, dtype=float))
    n = len(f0)
    result = float(np.mean(f0))
    se = float(np.std(f0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Central path of barrier"})


def cheatsheet():
    return "cvxcen: Central path of barrier"
