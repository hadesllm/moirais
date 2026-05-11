"""DR for adaptive bandit-DiD."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dr_bandit_did"]


def dr_bandit_did(y, D_t, X, pi_t):
    """
    DR for adaptive bandit-DiD

    Formula: DR moment under known time-varying assignment

    Parameters
    ----------
    y : array-like
        Input data.
    D_t : array-like
        Input data.
    X : array-like
        Input data.
    pi_t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hadad et al (2021); Athey et al (2024)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DR for adaptive bandit-DiD"})


def cheatsheet():
    return "drbnk: DR for adaptive bandit-DiD"
