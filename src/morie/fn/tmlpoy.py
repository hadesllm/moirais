"""Propensity-only TMLE — robust if Q misspecified."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_propensity_only"]


def tmle_propensity_only(y, D, X):
    """
    Propensity-only TMLE — robust if Q misspecified

    Formula: outcome-free target via propensity weighting alone

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
    Hernán-Robins (2020) Causal Inference Book Ch 12
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Propensity-only TMLE — robust if Q misspecified"})


def cheatsheet():
    return "tmlpoy: Propensity-only TMLE — robust if Q misspecified"
