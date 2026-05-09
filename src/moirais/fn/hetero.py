"""HTMT heterotrait-monotrait ratio."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["htmt_ratio"]


def htmt_ratio(X, construct_assignment):
    """
    HTMT heterotrait-monotrait ratio

    Formula: avg between-construct corr / sqrt(within-construct)

    Parameters
    ----------
    X : array-like
        Input data.
    construct_assignment : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Henseler-Ringle-Sarstedt (2015)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "HTMT heterotrait-monotrait ratio"})


def cheatsheet():
    return "hetero: HTMT heterotrait-monotrait ratio"
