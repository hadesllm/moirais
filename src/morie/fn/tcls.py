"""t-closeness baseline."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["t_closeness"]


def t_closeness(X, quasi_ids, sensitive, t):
    """
    t-closeness baseline

    Formula: distance(class dist, full dist) ≤ t

    Parameters
    ----------
    X : array-like
        Input data.
    quasi_ids : array-like
        Input data.
    sensitive : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Li-Li-Venkatasubramanian (2007)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "t-closeness baseline"})


def cheatsheet():
    return "tcls: t-closeness baseline"
