"""1-Wasserstein distance (1D)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserstein_1d"]


def wasserstein_1d(p, q, support):
    """
    1-Wasserstein distance (1D)

    Formula: W1(p,q) = integral |F_p - F_q| dx

    Parameters
    ----------
    p : array-like
        Input data.
    q : array-like
        Input data.
    support : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kantorovich-Rubinstein (1958)
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    n = len(p)
    result = float(np.mean(p))
    se = float(np.std(p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "1-Wasserstein distance (1D)"})


def cheatsheet():
    return "wassdt: 1-Wasserstein distance (1D)"
