"""Cronbach's alpha (CTT)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ctt_alpha_classic"]


def ctt_alpha_classic(X):
    """
    Cronbach's alpha (CTT)

    Formula: alpha = k/(k-1) (1 - sum sigma_i^2 / sigma_T^2)

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cronbach's alpha (CTT)"})


def cheatsheet():
    return "ctta1c: Cronbach's alpha (CTT)"
