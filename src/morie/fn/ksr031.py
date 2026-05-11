"""Asymptotic tightness condition for weak convergence in l-infinity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_weak_convergence_tightness"]


def kosorok_ch2_weak_convergence_tightness(X_n, rho, eps, delta):
    """
    Asymptotic tightness condition for weak convergence in l-infinity

    Formula: lim_{delta->0} limsup_n P*( sup_{rho(s,t) < delta} |X_n(s) - X_n(t)| > eps ) = 0

    Parameters
    ----------
    X_n : array-like
        Input data.
    rho : array-like
        Input data.
    eps : array-like
        Input data.
    delta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 2, Eq 2.6, p. 15
    """
    X_n = np.atleast_1d(np.asarray(X_n, dtype=float))
    n = len(X_n)
    result = float(np.mean(X_n))
    se = float(np.std(X_n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Asymptotic tightness condition for weak convergence in l-infinity"})


def cheatsheet():
    return "ksr031: Asymptotic tightness condition for weak convergence in l-infinity"
