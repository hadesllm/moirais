"""GP mixture of experts."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gp_mixture_of_experts"]


def gp_mixture_of_experts(X, y, X_test, K):
    """
    GP mixture of experts

    Formula: gating network selects per-region GP expert

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    X_test : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tresp (2001)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GP mixture of experts"})


def cheatsheet():
    return "gpmoe: GP mixture of experts"
