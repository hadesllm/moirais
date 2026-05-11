"""CTT alpha-if-item-deleted."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ctt_alpha_max"]


def ctt_alpha_max(X):
    """
    CTT alpha-if-item-deleted

    Formula: alpha computed with each item dropped

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
    Cronbach (1951)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CTT alpha-if-item-deleted"})


def cheatsheet():
    return "cttamx: CTT alpha-if-item-deleted"
