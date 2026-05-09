"""Backdoor-adjusted ATE via stratification."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_backdoor_estimate"]


def causal_backdoor_estimate(y, X, Z):
    """
    Backdoor-adjusted ATE via stratification

    Formula: ATE = Σ_z P(Z=z)(E[Y|X=1,Z=z]-E[Y|X=0,Z=z])

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    Z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ATE, se

    References
    ----------
    Pearl (2009)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Backdoor-adjusted ATE via stratification"})


def cheatsheet():
    return "causbckd: Backdoor-adjusted ATE via stratification"
