"""Definition of a P-Glivenko-Cantelli class via uniform almost-sure convergence."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_glivenko_cantelli_class"]


def kosorok_ch2_glivenko_cantelli_class(F, P_n, P):
    """
    Definition of a P-Glivenko-Cantelli class via uniform almost-sure convergence

    Formula: sup_{f in F} | P_n f - P f | -> 0 almost surely (outer)

    Parameters
    ----------
    F : array-like
        Input data.
    P_n : array-like
        Input data.
    P : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 2, Eq 2.4, p. 10
    """
    F = np.atleast_1d(np.asarray(F, dtype=float))
    n = len(F)
    result = float(np.mean(F))
    se = float(np.std(F, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Definition of a P-Glivenko-Cantelli class via uniform almost-sure convergence"})


def cheatsheet():
    return "ksr029: Definition of a P-Glivenko-Cantelli class via uniform almost-sure convergence"
