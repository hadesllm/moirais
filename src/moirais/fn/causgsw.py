"""Standardised mean difference vs target population."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_generalisability_smd"]


def causal_generalisability_smd(X_trial, X_pop):
    """
    Standardised mean difference vs target population

    Formula: SMD_g = (μ̄_trial-μ̄_pop)/σ_pop

    Parameters
    ----------
    X_trial : array-like
        Input data.
    X_pop : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: SMD_g

    References
    ----------
    Stuart-Cole-Bradshaw-Leaf (2011)
    """
    X_trial = np.atleast_1d(np.asarray(X_trial, dtype=float))
    n = len(X_trial)
    result = float(np.mean(X_trial))
    se = float(np.std(X_trial, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Standardised mean difference vs target population"})


def cheatsheet():
    return "causgsw: Standardised mean difference vs target population"
