"""CTT item difficulty."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ctt_difficulty"]


def ctt_difficulty(X):
    """
    CTT item difficulty

    Formula: p_j = mean(X_j)

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Nunnally-Bernstein (1994)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CTT item difficulty"})


def cheatsheet():
    return "cttdif: CTT item difficulty"
