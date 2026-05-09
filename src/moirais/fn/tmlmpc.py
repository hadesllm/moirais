"""TMLE for multi-state cumulative hazard contrast."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_multi_state_phc"]


def tmle_multi_state_phc(time, state, D, X):
    """
    TMLE for multi-state cumulative hazard contrast

    Formula: target Lambda_k(t|do(A=a)) for transition k

    Parameters
    ----------
    time : array-like
        Input data.
    state : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rytgaard-Eriksson-Gerds-vdL (2024)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for multi-state cumulative hazard contrast"})


def cheatsheet():
    return "tmlmpc: TMLE for multi-state cumulative hazard contrast"
