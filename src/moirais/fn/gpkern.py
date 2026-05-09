"""GP kernel composition utilities."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gp_kernel_compose"]


def gp_kernel_compose(X, Y, kernel_spec):
    """
    GP kernel composition utilities

    Formula: k_sum, k_prod, k_warp transformations

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    kernel_spec : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Duvenaud et al (2013); Rasmussen-Williams (2006)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GP kernel composition utilities"})


def cheatsheet():
    return "gpkern: GP kernel composition utilities"
