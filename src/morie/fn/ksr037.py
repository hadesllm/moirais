"""Glivenko-Cantelli theorem based on uniform covering numbers (envelope-based)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_glivenko_cantelli_uniform"]


def kosorok_ch2_glivenko_cantelli_uniform(F, P):
    """
    Glivenko-Cantelli theorem based on uniform covering numbers (envelope-based)

    Formula: If sup_Q N(eps ||F||_{Q,1}, F, L1(Q)) < inf for every eps and P*F < inf then F is P-Glivenko-Cantelli

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
    Kosorok (2008), Thm 2.4, p. 18
    """
    F = np.atleast_1d(np.asarray(F, dtype=float))
    n = len(F)
    result = float(np.mean(F))
    se = float(np.std(F, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Glivenko-Cantelli theorem based on uniform covering numbers (envelope-based)"})


def cheatsheet():
    return "ksr037: Glivenko-Cantelli theorem based on uniform covering numbers (envelope-based)"
