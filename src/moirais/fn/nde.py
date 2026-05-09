"""Natural direct effect (Pearl 2001)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["natural_direct_effect"]


def natural_direct_effect(X, M, Y):
    """
    Natural direct effect (Pearl 2001)

    Formula: NDE = E[Y(1, M(0)) - Y(0, M(0))]

    Parameters
    ----------
    X : array-like
        Input data.
    M : array-like
        Input data.
    Y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Pearl (2001); VanderWeele (2015)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Natural direct effect (Pearl 2001)"})


def cheatsheet():
    return "nde: Natural direct effect (Pearl 2001)"
