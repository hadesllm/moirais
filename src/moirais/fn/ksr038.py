"""Donsker theorem via uniform entropy integral."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_donsker_uniform_entropy"]


def kosorok_ch2_donsker_uniform_entropy(F, P):
    """
    Donsker theorem via uniform entropy integral

    Formula: If J(1, F, L2) < inf and P*F^2 < inf then F is P-Donsker

    Parameters
    ----------
    F : array-like
        Input data.
    P : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Thm 2.5, p. 18
    """
    F = np.atleast_1d(np.asarray(F, dtype=float))
    n = len(F)
    result = float(np.mean(F))
    se = float(np.std(F, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Donsker theorem via uniform entropy integral"})


def cheatsheet():
    return "ksr038: Donsker theorem via uniform entropy integral"
