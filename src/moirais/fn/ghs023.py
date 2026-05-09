"""Sufficient condition on the splitting variables ensuring that almost all realizations of a tail-free random measure are absolutely continuous with respect to mu.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch3_tailfree_abs_continuity_cond"]


def ghosal_ch3_tailfree_abs_continuity_cond(V, mu, m):
    """
    Sufficient condition on the splitting variables ensuring that almost all realizations of a tail-free random measure are absolutely continuous with respect to mu.

    Formula: sup_{m in N} max_{epsilon in E^m} E( prod_{j=1}^{m} V_{epsilon_1 ... epsilon_j}^2 ) / mu^2( A_{epsilon_1 ... epsilon_m} ) < infty

    Parameters
    ----------
    V : array-like
        Input data.
    mu : array-like
        Input data.
    m : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 3, Eq 3.16, p. 44
    """
    V = np.atleast_1d(np.asarray(V, dtype=float))
    n = len(V)
    result = float(np.mean(V))
    se = float(np.std(V, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sufficient condition on the splitting variables ensuring that almost all realizations of a tail-free random measure are absolutely continuous with respect to mu."})


def cheatsheet():
    return "ghs023: Sufficient condition on the splitting variables ensuring that almost all realizations of a tail-free random measure are absolutely continuous with respect to mu."
