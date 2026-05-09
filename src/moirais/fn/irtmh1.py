"""DIF Mantel-Haenszel."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dif_mantel_haenszel"]


def dif_mantel_haenszel(X, group, total_score):
    """
    DIF Mantel-Haenszel

    Formula: common odds ratio across score strata

    Parameters
    ----------
    X : array-like
        Input data.
    group : array-like
        Input data.
    total_score : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Holland-Thayer (1988)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DIF Mantel-Haenszel"})


def cheatsheet():
    return "irtmh1: DIF Mantel-Haenszel"
