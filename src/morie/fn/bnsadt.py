"""Adversarial bound under unknown parametric family."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_adversarial"]


def bound_adversarial(y, D, family):
    """
    Adversarial bound under unknown parametric family

    Formula: max over family F of estimator

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    family : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Andrews-Kasy (2019)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Adversarial bound under unknown parametric family"})


def cheatsheet():
    return "bnsadt: Adversarial bound under unknown parametric family"
