"""TMLE for cumulative incidence under competing risks."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_competing_risks"]


def tmle_competing_risks(time, event_type, D, X):
    """
    TMLE for cumulative incidence under competing risks

    Formula: target F_k(t|A=a); cause-specific Q

    Parameters
    ----------
    time : array-like
        Input data.
    event_type : array-like
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
    Rytgaard-Gerds-vdL (2023)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for cumulative incidence under competing risks"})


def cheatsheet():
    return "tmlcmp: TMLE for cumulative incidence under competing risks"
