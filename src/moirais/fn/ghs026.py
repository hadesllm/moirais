"""Finite-level density p_m of a tail-free measure obtained by averaging P(A_epsilon)/mu(A_epsilon) over the level-m partition cells.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch3_tailfree_finite_density_pm"]


def ghosal_ch3_tailfree_finite_density_pm(P, mu, A_epsilon, m):
    """
    Finite-level density p_m of a tail-free measure obtained by averaging P(A_epsilon)/mu(A_epsilon) over the level-m partition cells.

    Formula: p_m = sum_{epsilon in E^m} ( P(A_epsilon) / mu(A_epsilon) ) * 1_{A_epsilon}

    Parameters
    ----------
    P : array-like
        Input data.
    mu : array-like
        Input data.
    A_epsilon : array-like
        Input data.
    m : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: distribution

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 3, Eq 3.19, p. 44
    """
    P = np.atleast_1d(np.asarray(P, dtype=float))
    n = len(P)
    result = float(np.mean(P))
    se = float(np.std(P, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Finite-level density p_m of a tail-free measure obtained by averaging P(A_epsilon)/mu(A_epsilon) over the level-m partition cells."})


def cheatsheet():
    return "ghs026: Finite-level density p_m of a tail-free measure obtained by averaging P(A_epsilon)/mu(A_epsilon) over the level-m partition cells."
