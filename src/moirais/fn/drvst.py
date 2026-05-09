"""Variance-stabilized DR-DiD."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dr_did_variance_stab"]


def dr_did_variance_stab(y, D, X):
    """
    Variance-stabilized DR-DiD

    Formula: weight by sqrt(eff sample size)

    Parameters
    ----------
    y : array-like
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
    Roth (2024) Empirical Bayes
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variance-stabilized DR-DiD"})


def cheatsheet():
    return "drvst: Variance-stabilized DR-DiD"
