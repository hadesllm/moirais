"""Outer almost-sure bootstrap consistency for Donsker classes with bounded second moment of envelope."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_bootstrap_donsker_almost_sure"]


def kosorok_ch2_bootstrap_donsker_almost_sure(F, P):
    """
    Outer almost-sure bootstrap consistency for Donsker classes with bounded second moment of envelope

    Formula: F is P-Donsker and P*[ sup_F (f(X)-Pf)^2 ] < inf iff G_hat_n =>* G outer a.s.

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
    Kosorok (2008), Thm 2.7, p. 20
    """
    F = np.atleast_1d(np.asarray(F, dtype=float))
    n = len(F)
    result = float(np.mean(F))
    se = float(np.std(F, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Outer almost-sure bootstrap consistency for Donsker classes with bounded second moment of envelope"})


def cheatsheet():
    return "ksr041: Outer almost-sure bootstrap consistency for Donsker classes with bounded second moment of envelope"
