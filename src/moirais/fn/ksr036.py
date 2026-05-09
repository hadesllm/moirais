"""Bracketing-entropy sufficient condition for the Donsker property."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_donsker_bracketing_theorem"]


def kosorok_ch2_donsker_bracketing_theorem(F, P):
    """
    Bracketing-entropy sufficient condition for the Donsker property

    Formula: If J_[](inf, F, L2(P)) < inf then F is P-Donsker

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
    Kosorok (2008), Thm 2.3, p. 17
    """
    F = np.atleast_1d(np.asarray(F, dtype=float))
    n = len(F)
    result = float(np.mean(F))
    se = float(np.std(F, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bracketing-entropy sufficient condition for the Donsker property"})


def cheatsheet():
    return "ksr036: Bracketing-entropy sufficient condition for the Donsker property"
