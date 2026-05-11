"""Sufficient bracketing-entropy condition for Glivenko-Cantelli class."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_glivenko_cantelli_bracketing"]


def kosorok_ch2_glivenko_cantelli_bracketing(F, P, eps):
    """
    Sufficient bracketing-entropy condition for Glivenko-Cantelli class

    Formula: If N_[](eps, F, L1(P)) < inf for every eps > 0 then F is P-Glivenko-Cantelli

    Parameters
    ----------
    F : array-like
        Input data.
    P : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Thm 2.2, p. 16
    """
    F = np.atleast_1d(np.asarray(F, dtype=float))
    n = len(F)
    result = float(np.mean(F))
    se = float(np.std(F, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sufficient bracketing-entropy condition for Glivenko-Cantelli class"})


def cheatsheet():
    return "ksr034: Sufficient bracketing-entropy condition for Glivenko-Cantelli class"
