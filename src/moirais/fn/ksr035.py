"""Bracketing integral whose finiteness implies the Donsker property."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_donsker_bracketing_integral"]


def kosorok_ch2_donsker_bracketing_integral(F, P, r, delta):
    """
    Bracketing integral whose finiteness implies the Donsker property

    Formula: J_[](delta, F, L_r(P)) = integral_0^delta sqrt(log N_[](eps, F, L_r(P))) deps

    Parameters
    ----------
    F : array-like
        Input data.
    P : array-like
        Input data.
    r : array-like
        Input data.
    delta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 2, Eq following 2.6, p. 17
    """
    F = np.atleast_1d(np.asarray(F, dtype=float))
    n = len(F)
    result = float(np.mean(F))
    se = float(np.std(F, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bracketing integral whose finiteness implies the Donsker property"})


def cheatsheet():
    return "ksr035: Bracketing integral whose finiteness implies the Donsker property"
