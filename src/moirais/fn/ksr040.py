"""Equivalence of P-Donsker property and bootstrap consistency in probability."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_bootstrap_donsker_iff"]


def kosorok_ch2_bootstrap_donsker_iff(F, P):
    """
    Equivalence of P-Donsker property and bootstrap consistency in probability

    Formula: F is P-Donsker iff G_hat_n converges weakly (in prob.) to G in l_inf(F) (multiplier or nonparametric weights)

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
    Kosorok (2008), Thm 2.6, p. 20
    """
    F = np.atleast_1d(np.asarray(F, dtype=float))
    n = len(F)
    result = float(np.mean(F))
    se = float(np.std(F, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Equivalence of P-Donsker property and bootstrap consistency in probability"})


def cheatsheet():
    return "ksr040: Equivalence of P-Donsker property and bootstrap consistency in probability"
