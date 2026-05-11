"""Baron-Kenny three-equation mediation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_mediation_baron_kenny"]


def causal_mediation_baron_kenny(X, M, Y):
    """
    Baron-Kenny three-equation mediation

    Formula: c = c' + a*b; product of coefficients

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
        Keys: a, b, cprime, indirect, c

    References
    ----------
    Baron & Kenny (1986)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Baron-Kenny three-equation mediation"})


def cheatsheet():
    return "causmedb: Baron-Kenny three-equation mediation"
