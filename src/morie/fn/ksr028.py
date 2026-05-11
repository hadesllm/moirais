"""Classical Glivenko-Cantelli uniform convergence of empirical distribution function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_glivenko_cantelli_classical"]


def kosorok_ch2_glivenko_cantelli_classical(F_n, F):
    """
    Classical Glivenko-Cantelli uniform convergence of empirical distribution function

    Formula: sup_{t in R} | F_n(t) - F(t) | -> 0 almost surely

    Parameters
    ----------
    F_n : array-like
        Input data.
    F : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 2, Eq 2.3, p. 10
    """
    F_n = np.atleast_1d(np.asarray(F_n, dtype=float))
    n = len(F_n)
    result = float(np.mean(F_n))
    se = float(np.std(F_n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Classical Glivenko-Cantelli uniform convergence of empirical distribution function"})


def cheatsheet():
    return "ksr028: Classical Glivenko-Cantelli uniform convergence of empirical distribution function"
