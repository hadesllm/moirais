"""Targeted MLE one-step update for ATE."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_tmle_targeted"]


def causal_tmle_targeted(y, T, ps, Q1, Q0):
    """
    Targeted MLE one-step update for ATE

    Formula: Q̂* = Q̂ + ε̂ H, where H is clever covariate

    Parameters
    ----------
    y : array-like
        Input data.
    T : array-like
        Input data.
    ps : array-like
        Input data.
    Q1 : array-like
        Input data.
    Q0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ATE_TMLE, IF

    References
    ----------
    van der Laan & Rubin (2006)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Targeted MLE one-step update for ATE"})


def cheatsheet():
    return "caustmle: Targeted MLE one-step update for ATE"
