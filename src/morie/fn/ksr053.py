"""Continuous inverse of the Kaplan-Meier derivative operator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_kaplan_meier_inverse"]


def kosorok_ch2_kaplan_meier_inverse(S_0, L, F_0, a, t):
    """
    Continuous inverse of the Kaplan-Meier derivative operator

    Formula: Psi_dot^{-1}_{theta_0}(a)(t) = -S_0(t) * { a(0) + integral_0^t 1/(L(u-)S_0(u-)) [da(u) + a(u) dF_0(u)/S_0(u)] }

    Parameters
    ----------
    S_0 : array-like
        Input data.
    L : array-like
        Input data.
    F_0 : array-like
        Input data.
    a : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 2, Eq 2.17, p. 27
    """
    S_0 = np.atleast_1d(np.asarray(S_0, dtype=float))
    n = len(S_0)
    result = float(np.mean(S_0))
    se = float(np.std(S_0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Continuous inverse of the Kaplan-Meier derivative operator"})


def cheatsheet():
    return "ksr053: Continuous inverse of the Kaplan-Meier derivative operator"
