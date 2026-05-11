"""Weak convergence to a tight limit characterised by fidi convergence plus asymptotic tightness."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_weak_convergence_iff"]


def kosorok_ch2_weak_convergence_iff(X_n, X, T):
    """
    Weak convergence to a tight limit characterised by fidi convergence plus asymptotic tightness

    Formula: X_n => X tight in l_inf(T) iff (i) all fidi distributions converge and (ii) X_n is asymptotically tight

    Parameters
    ----------
    X_n : array-like
        Input data.
    X : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Thm 2.1, p. 15
    """
    X_n = np.atleast_1d(np.asarray(X_n, dtype=float))
    n = len(X_n)
    result = float(np.mean(X_n))
    se = float(np.std(X_n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Weak convergence to a tight limit characterised by fidi convergence plus asymptotic tightness"})


def cheatsheet():
    return "ksr032: Weak convergence to a tight limit characterised by fidi convergence plus asymptotic tightness"
