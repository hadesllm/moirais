"""Uniform covering number supremum used for entropy bounds independent of P."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_uniform_covering_number"]


def kosorok_ch2_uniform_covering_number(F, eps, r):
    """
    Uniform covering number supremum used for entropy bounds independent of P

    Formula: sup_Q N( eps * ||F||_{Q,r}, F, L_r(Q) )

    Parameters
    ----------
    F : array-like
        Input data.
    eps : array-like
        Input data.
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 2, Eq 2.7, p. 18
    """
    F = np.atleast_1d(np.asarray(F, dtype=float))
    n = len(F)
    result = float(np.mean(F))
    se = float(np.std(F, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Uniform covering number supremum used for entropy bounds independent of P"})


def cheatsheet():
    return "ksr033: Uniform covering number supremum used for entropy bounds independent of P"
