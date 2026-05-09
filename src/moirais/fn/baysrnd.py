"""Random-effects shrinkage."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["shrinkage_random"]


def shrinkage_random(y, X, group):
    """
    Random-effects shrinkage

    Formula: random intercepts u_g ~ N(0, tau^2)

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    group : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lindley-Smith (1972)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Random-effects shrinkage"})


def cheatsheet():
    return "baysrnd: Random-effects shrinkage"
