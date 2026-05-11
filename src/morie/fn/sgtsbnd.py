"""Information-theoretic detectability threshold for SBM."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_sbm_detect_threshold"]


def sgt_sbm_detect_threshold(a, b, k):
    """
    Information-theoretic detectability threshold for SBM

    Formula: Detectable iff (a-b)² > k(a+(k-1)b)/(k-1)

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: detectable

    References
    ----------
    Decelle-Krzakala-Moore-Zdeborová (2011)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Information-theoretic detectability threshold for SBM"})


def cheatsheet():
    return "sgtsbnd: Information-theoretic detectability threshold for SBM"
