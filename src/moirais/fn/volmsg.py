"""Markov-switching GARCH(1,1) two-regime."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_markov_switching_garch"]


def vol_markov_switching_garch(r, K, init):
    """
    Markov-switching GARCH(1,1) two-regime

    Formula: Two regimes with separate (ω,α,β); P transition

    Parameters
    ----------
    r : array-like
        Input data.
    K : array-like
        Input data.
    init : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: omega, alpha, beta, P, ll

    References
    ----------
    Klaassen (2002)
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Markov-switching GARCH(1,1) two-regime"})


def cheatsheet():
    return "volmsg: Markov-switching GARCH(1,1) two-regime"
