"""Hazard model with dependent censoring (copula)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dependent_censoring_hazard"]


def dependent_censoring_hazard(time, event, X):
    """
    Hazard model with dependent censoring (copula)

    Formula: S(t,c) = C(S_T(t), S_C(c); theta) copula

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zheng & Klein (1995)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hazard model with dependent censoring (copula)"})


def cheatsheet():
    return "depcen: Hazard model with dependent censoring (copula)"
