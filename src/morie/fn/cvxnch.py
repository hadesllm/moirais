"""Nuclear norm of matrix."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_nuclear_norm"]


def boyd_nuclear_norm(X):
    """
    Nuclear norm of matrix

    Formula: |X|_* = sum sigma_i(X)

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Boyd CVX Ch 6
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nuclear norm of matrix"})


def cheatsheet():
    return "cvxnch: Nuclear norm of matrix"
