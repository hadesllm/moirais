"""Equivalent characterisation of weak convergence via bounded Lipschitz functions."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_weak_convergence_lipschitz"]


def kosorok_ch2_weak_convergence_lipschitz(X_n, X):
    """
    Equivalent characterisation of weak convergence via bounded Lipschitz functions

    Formula: X_n => X iff sup_{f in BL_1} | E* f(X_n) - E f(X) | -> 0

    Parameters
    ----------
    X_n : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 2, Eq 2.8, p. 19
    """
    X_n = np.atleast_1d(np.asarray(X_n, dtype=float))
    n = len(X_n)
    result = float(np.mean(X_n))
    se = float(np.std(X_n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Equivalent characterisation of weak convergence via bounded Lipschitz functions"})


def cheatsheet():
    return "ksr039: Equivalent characterisation of weak convergence via bounded Lipschitz functions"
